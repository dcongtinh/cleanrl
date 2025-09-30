from typing import Callable
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/adsl/Workspace/cleanrl')
import gymnasium as gym
import torch
import numpy as np
import random

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    seed: int = 0,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    # agent = Model(envs).to(device)
    agent = torch.load(model_path, map_location=device, weights_only=False)
    agent.eval()

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    print(np.mean(episodic_returns), np.std(episodic_returns))
    return episodic_returns


if __name__ == "__main__":

    from cleanrl.sac_continuous_action import Actor, make_env
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    model_path = "/home/adsl/Workspace/cleanrl/cleanrl/runs/SAC/SAC-vanilla-lmd0.1-walker-run-v0-seed1-2025_09_29_12h08m30s/weights/best_reward-780.8646_smooth-37.0630.w"
    evaluate(
        model_path,
        make_env,
        "dm_control/walker-run-v0",
        eval_episodes=10,
        run_name=f"eval",
        device="cuda",
        capture_video=False,
        seed=SEED
    )
