"""
Phase 4 - Training Loop: Huấn luyện Hierarchical DQN-PPO Agent
================================================================
Chia 2 giai đoạn:
  1. Train DQN Head (Strategic): Agent học KHI NÀO mua/bán (Buy/Hold/Sell)
     - Freeze PPO Actor + Critic
     - Allocation mặc định 50% khi PPO chưa train
  2. Train PPO Actor (Tactical): Agent học BAO NHIÊU % vốn (continuous allocation)
     - Freeze Encoder + DQN Head
     - DQN policy đã fix (deterministic argmax)
Cuối cùng: Backtest hierarchical trên val + test set.

CHECKPOINT/RESUME:
  - Tự động lưu checkpoint mỗi N episodes
  - Khi chạy lại, tự động phát hiện và tiếp tục từ checkpoint cuối
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import advanced_metrics as am
from phase3_trading_env import HybridTradingEnv
from phase4_hybrid_agent import (
    HybridAgent, ReplayBuffer, PPOContinuousRolloutBuffer
)

# Checkpoint interval: lưu mỗi N episodes
CHECKPOINT_INTERVAL = 25
CHECKPOINT_DIR = config.MODELS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_epsilon(step, total_steps=config.DQN_EPSILON_DECAY_STEPS):
    """Tính epsilon theo linear decay schedule."""
    eps = config.DQN_EPSILON_START - (
        config.DQN_EPSILON_START - config.DQN_EPSILON_END
    ) * min(step / total_steps, 1.0)
    return eps


# =============================================================
# CHECKPOINT UTILITIES
# =============================================================
def _safe_torch_load(path, map_location=None):
    """
    Load checkpoint tương thích cả PyTorch cũ (< 1.13) và mới (>= 2.0).
    PyTorch >= 2.0 mặc định weights_only=True → lỗi nếu checkpoint chứa
    list/dict/string. Cần weights_only=False cho checkpoint đầy đủ.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 không có parameter weights_only
        return torch.load(path, map_location=map_location)


def save_checkpoint(phase, agent, optimizer, episode, global_step,
                    episode_rewards, target_agent=None, extra=None):
    """Lưu checkpoint đầy đủ để có thể resume training."""
    ckpt = {
        "phase": phase,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "episode_rewards": episode_rewards,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "hierarchical_v2",  # Đánh dấu phiên bản kiến trúc
    }
    if target_agent is not None:
        ckpt["target_agent_state_dict"] = target_agent.state_dict()
    if extra:
        ckpt.update(extra)

    path_latest = CHECKPOINT_DIR / f"checkpoint_{phase}_latest.pth"
    path_numbered = CHECKPOINT_DIR / f"checkpoint_{phase}_ep{episode+1:04d}.pth"

    torch.save(ckpt, path_latest)
    torch.save(ckpt, path_numbered)

    print(f"    💾 Checkpoint saved: {path_numbered.name} "
          f"(ep={episode+1}, steps={global_step})")

    # Giữ tối đa 5 checkpoint gần nhất, xóa cũ
    existing = sorted(CHECKPOINT_DIR.glob(f"checkpoint_{phase}_ep*.pth"))
    if len(existing) > 5:
        for old in existing[:-5]:
            old.unlink()


def load_checkpoint(phase, agent, optimizer, target_agent=None):
    """
    Load checkpoint gần nhất nếu có.
    Tự động phát hiện checkpoint incompatible (kiến trúc cũ) và xóa.
    """
    path_latest = CHECKPOINT_DIR / f"checkpoint_{phase}_latest.pth"

    if not path_latest.exists():
        return 0, 0, []

    print(f"\n  🔄 Phát hiện checkpoint: {path_latest.name}")

    try:
        ckpt = _safe_torch_load(path_latest, map_location=config.DEVICE)

        # Kiểm tra phiên bản kiến trúc — nếu không có tag hoặc tag cũ → incompatible
        if ckpt.get("architecture") != "hierarchical_v2":
            print("  ⚠️ Checkpoint từ kiến trúc CŨ (flat agent). Xóa và train lại.")
            _cleanup_old_checkpoints(phase)
            return 0, 0, []

        agent.load_state_dict(ckpt["agent_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if target_agent is not None and "target_agent_state_dict" in ckpt:
            target_agent.load_state_dict(ckpt["target_agent_state_dict"])

    except (RuntimeError, KeyError) as e:
        # RuntimeError: state_dict key mismatch (kiến trúc khác)
        # KeyError: checkpoint thiếu trường bắt buộc
        print(f"  ⚠️ Checkpoint INCOMPATIBLE: {e}")
        print("  → Xóa checkpoint cũ và train lại từ đầu.")
        _cleanup_old_checkpoints(phase)
        return 0, 0, []

    start_episode = ckpt["episode"] + 1
    global_step = ckpt["global_step"]
    episode_rewards = ckpt["episode_rewards"]

    print(f"  ✔ Resume từ episode {start_episode}/{ckpt.get('phase', phase).upper()}")
    print(f"    Steps đã train: {global_step:,}")
    print(f"    Saved at: {ckpt.get('timestamp', 'N/A')}")
    if episode_rewards:
        print(f"    Last avg reward: {np.mean(episode_rewards[-50:]):.2f}")

    return start_episode, global_step, episode_rewards


def _cleanup_old_checkpoints(phase):
    """Xóa tất cả checkpoint của một phase."""
    for ckpt_path in CHECKPOINT_DIR.glob(f"checkpoint_{phase}_*.pth"):
        ckpt_path.unlink()
        print(f"    🗑️ Đã xóa: {ckpt_path.name}")


# =============================================================
# GIAI ĐOẠN 1: TRAIN DQN HEAD — Strategic (Buy/Hold/Sell)
# =============================================================
def train_dqn_phase(agent, env):
    """
    Train Shared Encoder + DQN Head: học KHI NÀO nên Buy/Hold/Sell.
    PPO Actor + Critic bị freeze hoàn toàn.
    Allocation mặc định = 50% (PPO chưa train).
    """
    print("\n" + "=" * 60)
    print("  GIAI ĐOẠN 1: TRAIN DQN HEAD (Buy/Hold/Sell)")
    print(f"  Allocation mặc định: {config.DQN_DEFAULT_FRACTION*100:.0f}%")
    print("=" * 60)

    device = config.DEVICE

    # Freeze PPO heads
    for param in agent.ppo_actor.parameters():
        param.requires_grad = False
    for param in agent.ppo_critic.parameters():
        param.requires_grad = False

    # Target network cho Double DQN
    target_agent = copy.deepcopy(agent)
    target_agent.eval()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=config.DQN_LR,
    )

    # Thử load checkpoint
    start_episode, global_step, episode_rewards = load_checkpoint(
        "dqn", agent, optimizer, target_agent
    )

    if start_episode >= config.DQN_EPISODES:
        print(f"  ✔ DQN Phase đã hoàn tất ({config.DQN_EPISODES} episodes). Bỏ qua.")
        _unfreeze_ppo(agent)
        return agent

    replay_buffer = ReplayBuffer()
    criterion = nn.SmoothL1Loss()
    start_time = time.time()

    for episode in range(start_episode, config.DQN_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            epsilon = get_epsilon(global_step)
            state_tensor = (torch.tensor(state, dtype=torch.float32)
                            .unsqueeze(0).to(device))

            # DQN chọn direction, allocation cố định 50%
            direction = agent.select_direction_dqn(state_tensor, epsilon)
            fraction = config.DQN_DEFAULT_FRACTION

            # Step với hierarchical action
            next_state, reward, terminated, truncated, info = \
                env.step_hierarchical(direction, fraction)
            done = terminated or truncated

            replay_buffer.push(state, direction, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            global_step += 1

            # Train trên mini-batch
            if len(replay_buffer) >= config.DQN_BATCH_SIZE:
                _train_dqn_batch(agent, target_agent, replay_buffer,
                                 optimizer, criterion, device)

            if global_step % config.DQN_TARGET_UPDATE_FREQ == 0:
                target_agent.load_state_dict(agent.state_dict())

        episode_rewards.append(total_reward)
        metrics = env.get_performance_metrics()

        if (episode + 1) % 10 == 0 or episode == start_episode:
            _print_dqn_progress(episode, start_episode, episode_rewards,
                                global_step, metrics, start_time)

        # Lưu checkpoint định kỳ
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint("dqn", agent, optimizer, episode,
                            global_step, episode_rewards, target_agent)

    # Lưu checkpoint cuối cùng
    save_checkpoint("dqn", agent, optimizer, config.DQN_EPISODES - 1,
                    global_step, episode_rewards, target_agent)

    _unfreeze_ppo(agent)

    total_time = time.time() - start_time
    print(f"  ✔ DQN Phase hoàn tất. Steps: {global_step:,} | "
          f"Time: {total_time/60:.1f}m")
    return agent


def _train_dqn_batch(agent, target_agent, replay_buffer,
                     optimizer, criterion, device):
    """Train DQN trên 1 mini-batch từ replay buffer."""
    batch_s, batch_a, batch_r, batch_ns, batch_d = \
        replay_buffer.sample(config.DQN_BATCH_SIZE)

    states_t = torch.tensor(batch_s, dtype=torch.float32).to(device)
    actions_t = torch.tensor(batch_a, dtype=torch.long).to(device)
    rewards_t = torch.tensor(batch_r, dtype=torch.float32).to(device)
    next_states_t = torch.tensor(batch_ns, dtype=torch.float32).to(device)
    dones_t = torch.tensor(batch_d, dtype=torch.float32).to(device)

    # Double DQN: agent chọn action, target đánh giá Q-value
    q_values = agent.get_q_values(states_t)
    q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = agent.get_q_values(next_states_t).argmax(dim=1)
        next_q = target_agent.get_q_values(next_states_t)
        next_q_selected = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards_t + config.DQN_GAMMA * next_q_selected * (1 - dones_t)

    loss = criterion(q_selected, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()


def _print_dqn_progress(episode, start_episode, episode_rewards,
                        global_step, metrics, start_time):
    """In progress DQN training."""
    avg_reward = np.mean(episode_rewards[-50:])
    elapsed = time.time() - start_time
    eps_done = episode - start_episode + 1
    eps_per_min = eps_done / (elapsed / 60) if elapsed > 0 else 0
    eta_min = ((config.DQN_EPISODES - episode - 1) / eps_per_min
               if eps_per_min > 0 else 0)
    print(f"  Episode {episode+1:>4}/{config.DQN_EPISODES} | "
          f"Reward: {episode_rewards[-1]:>8.2f} | "
          f"Avg50: {avg_reward:>8.2f} | "
          f"ε: {get_epsilon(global_step):.3f} | "
          f"Return: {metrics.get('total_return', 0)*100:.1f}% | "
          f"ETA: {eta_min:.0f}m")


def _unfreeze_ppo(agent):
    """Unfreeze PPO heads."""
    for param in agent.ppo_actor.parameters():
        param.requires_grad = True
    for param in agent.ppo_critic.parameters():
        param.requires_grad = True


# =============================================================
# GIAI ĐOẠN 2: TRAIN PPO ACTOR — Tactical (Continuous Allocation)
# =============================================================
def train_ppo_phase(agent, env):
    """
    Train PPO Actor (continuous allocation) + PPO Critic.
    Freeze Shared Encoder + DQN Head.
    DQN policy đã fix (deterministic argmax Q) → PPO chỉ tối ưu allocation.
    """
    print("\n" + "=" * 60)
    print("  GIAI ĐOẠN 2: TRAIN PPO ACTOR (Continuous Allocation)")
    print("  DQN policy: FROZEN (deterministic)")
    print("=" * 60)

    device = config.DEVICE

    # Freeze Encoder + DQN
    for param in agent.encoder.parameters():
        param.requires_grad = False
    for param in agent.dqn_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=config.PPO_LR,
    )

    # Thử load checkpoint
    start_episode, _, episode_rewards = load_checkpoint("ppo", agent, optimizer)

    if start_episode >= config.PPO_EPISODES:
        print(f"  ✔ PPO Phase đã hoàn tất ({config.PPO_EPISODES} episodes). Bỏ qua.")
        _unfreeze_all(agent)
        return agent

    rollout = PPOContinuousRolloutBuffer()
    start_time = time.time()

    for episode in range(start_episode, config.PPO_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        rollout.clear()

        while not done:
            state_tensor = (torch.tensor(state, dtype=torch.float32)
                            .unsqueeze(0).to(device))

            # DQN chọn direction (deterministic — đã fix policy)
            direction = agent.select_direction_dqn(state_tensor, epsilon=0.0)

            if direction == config.DIRECTION_HOLD:
                # Hold → không cần PPO allocation
                fraction = 0.0
                log_prob = 0.0
                value = agent.get_value(state_tensor).item()
            else:
                # Buy/Sell → PPO chọn allocation
                fraction, log_prob, value = \
                    agent.select_allocation_ppo(state_tensor)

            next_state, reward, terminated, truncated, info = \
                env.step_hierarchical(direction, fraction)
            done = terminated or truncated

            rollout.add(state, fraction, log_prob, reward, value, float(done))
            state = next_state
            total_reward += reward

        # Tính GAE
        with torch.no_grad():
            last_state = (torch.tensor(state, dtype=torch.float32)
                          .unsqueeze(0).to(device))
            last_value = agent.get_value(last_state).item()

        advantages, returns = rollout.compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update (chỉ cập nhật PPO Actor + Critic)
        _ppo_update(agent, rollout, advantages, returns, optimizer, device)

        episode_rewards.append(total_reward)
        metrics = env.get_performance_metrics()

        if (episode + 1) % 10 == 0 or episode == start_episode:
            _print_ppo_progress(episode, start_episode, episode_rewards,
                                metrics, start_time)

        # Lưu checkpoint định kỳ
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint("ppo", agent, optimizer, episode,
                            0, episode_rewards)

    # Lưu checkpoint cuối cùng
    save_checkpoint("ppo", agent, optimizer, config.PPO_EPISODES - 1,
                    0, episode_rewards)

    _unfreeze_all(agent)

    total_time = time.time() - start_time
    print(f"  ✔ PPO Phase hoàn tất. Time: {total_time/60:.1f}m")
    return agent


def _ppo_update(agent, rollout, advantages, returns, optimizer, device):
    """Thực hiện PPO clipped surrogate update cho continuous allocation."""
    for _ in range(config.PPO_EPOCHS_PER_UPDATE):
        for batch in rollout.get_batches(advantages, returns):
            b_states, b_fractions, b_old_log_probs, b_advantages, b_returns = batch
            b_states = b_states.to(device)
            b_fractions = b_fractions.to(device)
            b_old_log_probs = b_old_log_probs.to(device)
            b_advantages = b_advantages.to(device)
            b_returns = b_returns.to(device)

            # Clamp fractions vào (eps, 1-eps) để tránh log(0) trong Beta
            b_fractions = b_fractions.clamp(1e-6, 1 - 1e-6)

            # Evaluate allocation với gradient
            new_log_probs, entropy, values = \
                agent.evaluate_allocation(b_states, b_fractions)

            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - config.PPO_CLIP_RATIO,
                1.0 + config.PPO_CLIP_RATIO,
            ) * b_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, b_returns)

            loss = (
                policy_loss
                + config.PPO_VALUE_COEFF * value_loss
                - config.PPO_ENTROPY_COEFF * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()


def _print_ppo_progress(episode, start_episode, episode_rewards,
                        metrics, start_time):
    """In progress PPO training."""
    avg_reward = np.mean(episode_rewards[-50:])
    elapsed = time.time() - start_time
    eps_done = episode - start_episode + 1
    eps_per_min = eps_done / (elapsed / 60) if elapsed > 0 else 0
    eta_min = ((config.PPO_EPISODES - episode - 1) / eps_per_min
               if eps_per_min > 0 else 0)
    print(f"  Episode {episode+1:>4}/{config.PPO_EPISODES} | "
          f"Reward: {episode_rewards[-1]:>8.2f} | "
          f"Avg50: {avg_reward:>8.2f} | "
          f"Return: {metrics.get('total_return', 0)*100:.1f}% | "
          f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
          f"ETA: {eta_min:.0f}m")


def _unfreeze_all(agent):
    """Unfreeze toàn bộ parameters."""
    for param in agent.parameters():
        param.requires_grad = True


# =============================================================
# BACKTEST + ĐÁNH GIÁ (Hierarchical Inference)
# =============================================================
def evaluate_agent(agent, split="test"):
    """
    Chạy backtest hierarchical trên tập đánh giá.
    DQN chọn direction → PPO chọn allocation → step_hierarchical().
    """
    print(f"\n  --- BACKTEST HIERARCHICAL trên tập {split.upper()} ---")
    device = config.DEVICE
    agent.eval()

    env = HybridTradingEnv(split=split)
    state, _ = env.reset()
    done = False

    # Tracking direction distribution
    direction_counts = {0: 0, 1: 0, 2: 0}
    allocation_history = []

    while not done:
        state_tensor = (torch.tensor(state, dtype=torch.float32)
                        .unsqueeze(0).to(device))

        # Hierarchical inference: DQN → direction, PPO → allocation
        direction, fraction = agent.select_hierarchical_action(
            state_tensor, epsilon=0.0
        )
        direction_counts[direction] += 1
        if direction != config.DIRECTION_HOLD:
            allocation_history.append(fraction)

        state, _, terminated, truncated, _ = \
            env.step_hierarchical(direction, fraction)
        done = terminated or truncated

    # === ADVANCED METRICS ===
    adv_metrics = am.compute_agent_advanced_metrics(
        env.trade_history, env.initial_balance
    )
    print(am.format_agent_metrics_table(adv_metrics, label=f"Agent ({split.upper()})"))

    # Direction distribution
    total_steps = sum(direction_counts.values())
    dir_names = {0: "Buy", 1: "Hold", 2: "Sell"}
    print(f"\n  📊 Direction Distribution:")
    for d, count in direction_counts.items():
        pct = count / total_steps * 100
        print(f"    {dir_names[d]:>4}: {count:>5} ({pct:.1f}%)")

    # Allocation statistics
    if allocation_history:
        alloc = np.array(allocation_history)
        print(f"\n  📊 Allocation Statistics (khi Buy/Sell):")
        print(f"    Mean:   {alloc.mean()*100:.1f}%")
        print(f"    Median: {np.median(alloc)*100:.1f}%")
        print(f"    Std:    {alloc.std()*100:.1f}%")
        print(f"    Min:    {alloc.min()*100:.1f}%")
        print(f"    Max:    {alloc.max()*100:.1f}%")

    # So sánh với Buy & Hold
    prices = np.load(config.STATES_DIR / f"{split}_prices.npy")
    bh_return = (prices[-1] - prices[0]) / prices[0]
    print(f"\n  Buy & Hold     : {bh_return*100:>8.2f}%")
    print(f"  Agent vs B&H   : "
          f"{(adv_metrics['total_return'] - bh_return)*100:>+8.2f}%")

    # Lưu metrics
    all_metrics = {
        **adv_metrics,
        "buy_hold_return": float(bh_return),
        "direction_distribution": {dir_names[k]: v
                                   for k, v in direction_counts.items()},
        "allocation_mean": float(np.mean(allocation_history))
        if allocation_history else 0.0,
        "allocation_std": float(np.std(allocation_history))
        if allocation_history else 0.0,
    }
    metrics_path = config.METRICS_DIR / f"agent_backtest_{split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    return adv_metrics


def run():
    """Chạy toàn bộ Phase 4: Train DQN → Train PPO → Evaluate."""
    print("\n" + "=" * 60)
    print("  PHASE 4: HUẤN LUYỆN HIERARCHICAL DQN-PPO AGENT")
    print("  DQN = Strategic (Buy/Hold/Sell)")
    print("  PPO = Tactical (Continuous Allocation)")
    print("  Hỗ trợ Resume từ Checkpoint")
    print("=" * 60)

    device = config.DEVICE

    # === QUICK MODE: Override hyperparameters để test nhanh ===
    if config.QUICK_MODE:
        config.DQN_EPISODES = config.QUICK_DQN_EPISODES
        config.PPO_EPISODES = config.QUICK_PPO_EPISODES
        config.DQN_EPSILON_DECAY_STEPS = config.QUICK_EPSILON_DECAY
        print(f"\n  ⚡ QUICK MODE = ON")
        print(f"  → Symbols: {config.QUICK_SYMBOLS}")
        print(f"  → DQN episodes: {config.DQN_EPISODES}")
        print(f"  → PPO episodes: {config.PPO_EPISODES}")
        print(f"  → Epsilon decay: {config.DQN_EPSILON_DECAY_STEPS:,}")
    else:
        print(f"\n  🏋️ FULL MODE")
        print(f"  → Symbols: all {len(config.SYMBOLS)} mã")
        print(f"  → DQN episodes: {config.DQN_EPISODES}")
        print(f"  → PPO episodes: {config.PPO_EPISODES}")

    print(f"  Device: {device}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Checkpoint interval: mỗi {CHECKPOINT_INTERVAL} episodes")

    # Kiểm tra observations tồn tại
    obs_path = config.STATES_DIR / "train_observations.npy"
    if not obs_path.exists():
        print("  [ERROR] Chưa có train_observations.npy! Chạy Phase 3 trước.")
        return

    # Tạo environment (Quick Mode filter symbols tự động)
    if config.QUICK_MODE:
        env_train = HybridTradingEnv(split="train")
        # Filter chỉ giữ lại symbols trong QUICK_SYMBOLS
        symbols = env_train.symbols
        mask = np.isin(symbols, config.QUICK_SYMBOLS)
        env_train.observations = env_train.observations[mask]
        env_train.prices = env_train.prices[mask]
        env_train.symbols = symbols[mask]
        env_train.n_steps = len(env_train.observations)
        print(f"  Quick filter: {env_train.n_steps} steps "
              f"(từ {len(mask)} → {mask.sum()})")
    else:
        env_train = HybridTradingEnv(split="train")

    state_dim = env_train.state_dim
    print(f"  State dim: {state_dim}")
    print(f"  Train steps: {env_train.n_steps}")

    # Tạo agent
    agent = HybridAgent(state_dim).to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent parameters: {total_params:,}")
    print(f"  DQN directions: {config.DQN_NUM_DIRECTIONS} (Buy/Hold/Sell)")
    print(f"  PPO allocation: continuous Beta(α,β) ∈ (0, 1)")

    # Liệt kê checkpoints hiện có
    existing_ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pth"))
    if existing_ckpts:
        print(f"\n  📂 Checkpoints hiện có ({len(existing_ckpts)}):")
        for ckpt_path in existing_ckpts[-5:]:
            try:
                ckpt_info = _safe_torch_load(ckpt_path, map_location="cpu")
                ep = ckpt_info.get("episode", "?")
                phase = ckpt_info.get("phase", "?")
                arch = ckpt_info.get("architecture", "legacy")
                ts = ckpt_info.get("timestamp", "?")
                ep_display = ep + 1 if isinstance(ep, int) else ep
                print(f"    {ckpt_path.name}: phase={phase} "
                      f"ep={ep_display} arch={arch} ({ts})")
            except Exception:
                print(f"    {ckpt_path.name}: ⚠️ không đọc được")
    else:
        print("\n  📂 Không có checkpoint nào. Bắt đầu từ đầu.")

    # ==========================================
    # Giai đoạn 1: DQN (Strategic)
    # ==========================================
    dqn_done_path = config.MODELS_DIR / "agent_after_dqn.pth"
    dqn_checkpoint_exists = \
        (CHECKPOINT_DIR / "checkpoint_dqn_latest.pth").exists()

    if dqn_done_path.exists() and not dqn_checkpoint_exists:
        print(f"\n  ✔ DQN đã hoàn tất trước đó. Load từ {dqn_done_path.name}")
        try:
            state_dict = _safe_torch_load(dqn_done_path, map_location=device)
            agent.load_state_dict(state_dict)
        except (RuntimeError, Exception) as e:
            print(f"  ⚠️ Model DQN cũ INCOMPATIBLE: {e}")
            print("  → Xóa model cũ và train lại từ đầu.")
            dqn_done_path.unlink(missing_ok=True)
            agent = train_dqn_phase(agent, env_train)
            torch.save(agent.state_dict(), dqn_done_path)
            print(f"  ✔ DQN model saved: {dqn_done_path}")
    else:
        agent = train_dqn_phase(agent, env_train)
        torch.save(agent.state_dict(), dqn_done_path)
        print(f"  ✔ DQN model saved: {dqn_done_path}")

        # Dọn checkpoint DQN sau khi hoàn tất
        for ckpt in CHECKPOINT_DIR.glob("checkpoint_dqn_*.pth"):
            ckpt.unlink()
        print("  🗑️ Đã xóa checkpoint DQN (phase hoàn tất)")

    # ==========================================
    # Giai đoạn 2: PPO (Tactical)
    # ==========================================
    agent = train_ppo_phase(agent, env_train)

    # Lưu model cuối cùng
    final_path = config.MODELS_DIR / "hybrid_agent_final.pth"
    torch.save(agent.state_dict(), final_path)
    print(f"\n  ✔ Agent đã lưu: {final_path}")

    # Dọn checkpoint PPO sau khi hoàn tất
    for ckpt in CHECKPOINT_DIR.glob("checkpoint_ppo_*.pth"):
        ckpt.unlink()
    print("  🗑️ Đã xóa checkpoint PPO (phase hoàn tất)")

    # Đánh giá
    evaluate_agent(agent, "val")
    evaluate_agent(agent, "test")

    print(f"\n{'='*60}")
    print("  ✔ PHASE 4 HOÀN TẤT")
    print(f"{'='*60}")


# =============================================================
# PER-SYMBOL TRAINING — Mỗi mã train agent riêng
# =============================================================
def run_per_symbol():
    """Train riêng từng mã cổ phiếu. Tránh conflict giữa các thị trường."""
    print("\n" + "=" * 60)
    print("  PHASE 4: PER-SYMBOL HIERARCHICAL DQN-PPO")
    print("  Mỗi mã train agent riêng biệt")
    print("=" * 60)

    device = config.DEVICE

    # Override episodes cho per-symbol
    config.DQN_EPISODES = config.PER_SYMBOL_DQN_EPISODES
    config.PPO_EPISODES = config.PER_SYMBOL_PPO_EPISODES
    config.DQN_EPSILON_DECAY_STEPS = config.PER_SYMBOL_EPSILON_DECAY

    # Tìm symbols có data thực tế
    train_symbols = np.load(
        config.STATES_DIR / "train_symbols.npy", allow_pickle=True
    )
    available_symbols = sorted(set(train_symbols))
    print(f"  Symbols có data: {available_symbols}")
    print(f"  DQN episodes/mã: {config.DQN_EPISODES}")
    print(f"  PPO episodes/mã: {config.PPO_EPISODES}")
    print(f"  Device: {device}")

    # Thư mục lưu per-symbol
    per_symbol_dir = config.MODELS_DIR / "per_symbol"
    per_symbol_dir.mkdir(exist_ok=True)

    all_results = {}
    total_start = time.time()

    for i, symbol in enumerate(available_symbols):
        sym_start = time.time()
        sym_dir = per_symbol_dir / symbol
        sym_dir.mkdir(exist_ok=True)
        sym_ckpt_dir = sym_dir / "checkpoints"
        sym_ckpt_dir.mkdir(exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(available_symbols)}] Training: {symbol}")
        print(f"{'─' * 60}")

        # Kiểm tra nếu đã train xong
        final_model = sym_dir / "hybrid_agent_final.pth"
        if final_model.exists():
            print(f"  ✔ {symbol} đã train xong. Skip.")
            # Load model để evaluate
            env_tmp = HybridTradingEnv(split="train", symbol_filter=symbol)
            agent = HybridAgent(env_tmp.state_dim).to(device)
            try:
                state_dict = _safe_torch_load(final_model, map_location=device)
                agent.load_state_dict(state_dict)
                val_metrics = _evaluate_symbol(agent, symbol, "val", sym_dir)
                test_metrics = _evaluate_symbol(agent, symbol, "test", sym_dir)
                all_results[symbol] = {
                    "val": val_metrics, "test": test_metrics
                }
            except Exception as e:
                print(f"  ⚠️ Load failed: {e}. Sẽ train lại.")
                final_model.unlink()
            else:
                continue

        # Tạo env filtered
        env_train = HybridTradingEnv(split="train", symbol_filter=symbol)
        print(f"  Train steps: {env_train.n_steps}")
        print(f"  Penalty scale: {env_train.penalty_scale:.3f}")

        # Tạo agent mới cho mỗi symbol
        state_dim = env_train.state_dim
        agent = HybridAgent(state_dim).to(device)

        # Override checkpoint dir cho symbol này
        global CHECKPOINT_DIR
        old_ckpt_dir = CHECKPOINT_DIR
        CHECKPOINT_DIR = sym_ckpt_dir

        # === DQN Phase ===
        dqn_path = sym_dir / "agent_after_dqn.pth"
        dqn_ckpt = sym_ckpt_dir / "checkpoint_dqn_latest.pth"

        if dqn_path.exists() and not dqn_ckpt.exists():
            print(f"  ✔ DQN đã xong. Load model.")
            state_dict = _safe_torch_load(dqn_path, map_location=device)
            agent.load_state_dict(state_dict)
        else:
            agent = train_dqn_phase(agent, env_train)
            torch.save(agent.state_dict(), dqn_path)
            # Dọn checkpoint DQN
            for ckpt in sym_ckpt_dir.glob("checkpoint_dqn_*.pth"):
                ckpt.unlink()
            print(f"  ✔ DQN saved: {dqn_path.name}")

        # === PPO Phase ===
        agent = train_ppo_phase(agent, env_train)
        torch.save(agent.state_dict(), final_model)
        # Dọn checkpoint PPO
        for ckpt in sym_ckpt_dir.glob("checkpoint_ppo_*.pth"):
            ckpt.unlink()
        print(f"  ✔ Model saved: {final_model}")

        # Restore checkpoint dir
        CHECKPOINT_DIR = old_ckpt_dir

        # === Evaluate ===
        val_metrics = _evaluate_symbol(agent, symbol, "val", sym_dir)
        test_metrics = _evaluate_symbol(agent, symbol, "test", sym_dir)
        all_results[symbol] = {"val": val_metrics, "test": test_metrics}

        elapsed = time.time() - sym_start
        print(f"  ⏱️ {symbol} hoàn tất: {elapsed/60:.1f} phút")

    # === AGGREGATE RESULTS ===
    total_elapsed = time.time() - total_start
    _print_aggregate_results(all_results, total_elapsed)

    # Lưu aggregate metrics
    agg_path = config.METRICS_DIR / "agent_per_symbol_summary.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  📊 Summary saved: {agg_path}")


def _evaluate_symbol(agent, symbol, split, sym_dir):
    """Evaluate agent trên 1 symbol cụ thể."""
    device = config.DEVICE
    agent.eval()

    env = HybridTradingEnv(split=split, symbol_filter=symbol)
    state, _ = env.reset()
    done = False

    direction_counts = {0: 0, 1: 0, 2: 0}
    allocation_history = []

    while not done:
        state_tensor = (torch.tensor(state, dtype=torch.float32)
                        .unsqueeze(0).to(device))
        direction, fraction = agent.select_hierarchical_action(
            state_tensor, epsilon=0.0
        )
        direction_counts[direction] += 1
        if direction != config.DIRECTION_HOLD:
            allocation_history.append(fraction)

        state, _, terminated, truncated, _ = \
            env.step_hierarchical(direction, fraction)
        done = terminated or truncated

    # Compute metrics
    adv_metrics = am.compute_agent_advanced_metrics(
        env.trade_history, env.initial_balance
    )

    # Direction info
    total = sum(direction_counts.values())
    dir_names = {0: "Buy", 1: "Hold", 2: "Sell"}
    dir_pct = {dir_names[k]: f"{v/total*100:.1f}%"
               for k, v in direction_counts.items()}

    # Allocation stats
    alloc_mean = float(np.mean(allocation_history)) if allocation_history else 0
    alloc_std = float(np.std(allocation_history)) if allocation_history else 0

    result = {
        **adv_metrics,
        "direction": dir_pct,
        "allocation_mean": alloc_mean,
        "allocation_std": alloc_std,
    }

    # Lưu metrics per symbol
    metrics_path = sym_dir / f"backtest_{split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print tóm tắt
    print(f"    {split.upper():4s}: Return={adv_metrics['total_return']*100:+.1f}% "
          f"| WR={adv_metrics['win_rate']*100:.0f}% "
          f"| MaxDD={adv_metrics['max_drawdown']*100:.0f}% "
          f"| Sharpe={adv_metrics['sharpe_ratio']:.2f} "
          f"| Buy={dir_pct['Buy']} Hold={dir_pct['Hold']} Sell={dir_pct['Sell']}")

    return result


def _print_aggregate_results(all_results, total_elapsed):
    """In bảng tổng hợp kết quả per-symbol."""
    print(f"\n{'=' * 80}")
    print("  TỔNG HỢP KẾT QUẢ PER-SYMBOL TRAINING")
    print(f"{'=' * 80}")
    print(f"  {'Symbol':<8} {'Val Ret':>8} {'Test Ret':>9} {'Val WR':>7} "
          f"{'Test WR':>8} {'Val DD':>7} {'Test DD':>8} {'Test Sharpe':>12}")
    print(f"  {'─' * 72}")

    val_returns, test_returns = [], []
    for symbol, data in sorted(all_results.items()):
        v, t = data.get("val", {}), data.get("test", {})
        vr = v.get("total_return", 0) * 100
        tr = t.get("total_return", 0) * 100
        vw = v.get("win_rate", 0) * 100
        tw = t.get("win_rate", 0) * 100
        vd = v.get("max_drawdown", 0) * 100
        td = t.get("max_drawdown", 0) * 100
        ts = t.get("sharpe_ratio", 0)
        val_returns.append(vr)
        test_returns.append(tr)
        print(f"  {symbol:<8} {vr:>+7.1f}% {tr:>+8.1f}% {vw:>6.1f}% "
              f"{tw:>7.1f}% {vd:>6.1f}% {td:>7.1f}% {ts:>11.3f}")

    print(f"  {'─' * 72}")
    if val_returns:
        print(f"  {'AVG':<8} {np.mean(val_returns):>+7.1f}% "
              f"{np.mean(test_returns):>+8.1f}%")
    print(f"\n  ⏱️ Tổng thời gian: {total_elapsed/60:.0f} phút")


if __name__ == "__main__":
    if config.PER_SYMBOL_MODE:
        run_per_symbol()
    else:
        run()
