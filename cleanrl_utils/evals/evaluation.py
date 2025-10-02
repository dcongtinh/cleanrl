from typing import Callable
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/adsl/Workspace/cleanrl')
import gymnasium as gym
import torch
import numpy as np
import random
import cleanrl_utils.fourier as fourier

def compute_fft_smoothness(action_series, sampling_frequency = 0.002):
    """
    action_series: numpy array of shape (T,) or (T, d) where T is timesteps
    sampling_frequency: scalar, e.g., 30 Hz
    Returns:
        Sm: smoothness score (lower is smoother)
    """
    action_series = np.atleast_2d(action_series)  # Ensure shape (T, d)
    if action_series.shape[0] < action_series.shape[1]:
        action_series = action_series.T  # Ensure time is first axis

    T, d = action_series.shape
    Sm_total = 0

    for dim in range(d):
        x = action_series[:, dim]
        X = np.fft.fft(x)
        M = np.abs(X[:T // 2])      # Amplitudes (1-sided)
        f = np.fft.fftfreq(T, d=1/sampling_frequency)[:T // 2]  # Frequencies

        Sm = (2 / (T * sampling_frequency)) * np.sum(M * f)
        Sm_total += Sm

    return Sm_total / d  # Mean across all dimensions

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    algo: str = "SAC",
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    seed: int = 0,
):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    # agent = Model(envs).to(device)
    agent = torch.load(model_path, map_location=device, weights_only=False)
    agent.eval()

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []

    actionss = []
    eps_lens = []
    exploration_noise = 0.1
    all_actions = []

    smooth_errors = []

    last_action = None
    current_error = []

    while len(episodic_returns) < eval_episodes:
        if algo == "SAC":
            actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        else:
            actions = agent(torch.Tensor(obs).to(device))
            actions += torch.normal(0, agent.action_scale * exploration_noise)
            actions = actions.detach().cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # Track action difference for smoothness
        if last_action is not None:
            diff = np.abs(actions - last_action)
            current_error.append(diff)
        last_action = actions

        all_actions.append(actions)


        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")

                episodic_returns += [info["episode"]["r"]]

                actionss.append(np.stack(all_actions))
                eps_lens.append(len(all_actions))

                all_actions = []

                # Compute smooth metric for the episode
                if current_error:
                    episode_smoothness = np.mean(current_error)
                    smooth_errors.append(episode_smoothness)
                else:
                    smooth_errors.append(0.0)  # No movement = perfectly smooth (or undefined)

                # Reset for next episode
                current_error = []
                last_action = None


        obs = next_obs
    
    print(f'Eval over {eval_episodes} episodes: mean reward {np.mean(episodic_returns):.6f} ± {np.std(episodic_returns):.6f}')
    print(f'Eval over {eval_episodes} episodes: mean smooth {np.mean(smooth_errors)*100:.6f} ± {np.std(smooth_errors)*100:.6f}')

    actionss = np.array(actionss)


    smooths, smooths2 = [], []
    for action_index in range(envs.single_action_space.shape[0]):
        freqs, amplitudes = fourier.from_actions(actionss[:, :, 0, action_index], eps_lens)
        smooth = fourier.smoothness(amplitudes)
        smooths.append(smooth)
        # print(f"action {action_index} smoothness: {smooth}")
    print(f'Eval over {eval_episodes} episodes: mean smoot2 {np.mean(smooths)*100:.6f} ± {np.std(smooths)*100:.6f}')

    for ep in range(eval_episodes):
        smooth = compute_fft_smoothness(actionss[ep, :, 0,])
        smooths2.append(smooth)
        # print(f"Ep {ep} smoothness: {smooth}")
    
    print(f'Eval over {eval_episodes} episodes: mean smooth3 {np.mean(smooths2):.6f} ± {np.std(smooths2):.6f}')
    return episodic_returns, smooth_errors, smooths, smooths2


if __name__ == "__main__":

    algo = 'DDPG'
    model = '-4gradcaps-'
    env = 'walker-run-v0'

    if algo == 'SAC':
        from cleanrl.sac_continuous_action import Actor, make_env
    elif algo == 'TD3':
        from cleanrl.td3_continuous_action import Actor, make_env
    elif algo == 'DDPG':
        from cleanrl.ddpg_continuous_action import Actor, make_env

    SEED = 0
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    model_path = "/home/adsl/Workspace/cleanrl/cleanrl/runs/DDPG/4gradcaps/DDPG-4gradcaps-lmd0.1-reg_coeff1.0-walker-run-v0-seed1-2025_10_02_10h59m27s__minmax/weights/best_reward-720.2479_smooth-31.8339.w"
    for seed in range(3, 4):
        evaluate(
            model_path,
            make_env,
            "dm_control/walker-run-v0",
            algo=algo,
            eval_episodes=10,
            run_name=f"eval",
            device="cuda",
            capture_video=False,
            seed=seed,
        )
