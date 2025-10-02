# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings('ignore')  # hides all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import time
from datetime import datetime
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
import math
from tqdm.auto import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    optimize_memory_usage: bool = False
    """toggle for optmizing memory usage (OMU) in the replay buffer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""

    algo: str = "DDPG"
    """the name of the algorithm"""
    smoothness: str = "vanilla"
    """the type of smoothness regularization (vanilla, caps, gradcaps, 2gradcaps, 3gradcaps, 4gradcaps)"""
    lmd: float = 0.1
    """the coefficient of the smoothness regularization"""
    reg_coeff: float = 1.0
    """the weight of the smoothness regularization loss"""
    notes: str = ""
    """notes about the experiment"""


LOG_STD_MAX = 2
LOG_STD_MIN = -5
eval_episodes = 10  # number of episodes to run for evaluation
eval_every = 2500  # how often to evaluate the policy
log_every = 1000  # how often to log results to tensorboard
best_return = -10000
best_smooth = 10000

try:
    args = tyro.cli(Args)
    print(args)
    dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    run_name = f"{args.algo}-{args.smoothness}-lmd{args.lmd}-reg_coeff{args.reg_coeff}-{args.env_id.split('/')[-1]}-seed{args.seed}-{dt}__minmax"
except:
    pass


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def evaluate(agent):
    agent.eval()
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, args.capture_video, run_name)])

    obs, _ = envs.reset(seed=0)
    episodic_returns = []
    smooth_errors = []

    last_action = None
    current_error = []

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = agent(torch.Tensor(obs).to(device))
            actions += torch.normal(0, agent.action_scale * args.exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # Track action difference for smoothness
        if last_action is not None:
            diff = np.abs(actions - last_action)
            current_error.append(diff)
        last_action = actions

        next_obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]

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
    agent.train()
    return episodic_returns, smooth_errors
    

if __name__ == "__main__":

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    print()
    print(run_name)
    project_dir = f"runs/{args.algo}/{args.smoothness}/{run_name}"
    os.makedirs(f"{project_dir}/weights", exist_ok=True)
    os.system(f"cp {os.path.basename(__file__)} {project_dir}/")
    os.system(f"cp ../cleanrl_utils/buffers.py {project_dir}/")

    writer = SummaryWriter(project_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32

    def get_order(s: str) -> int:
        if s == "gradcaps":
            return 1
        if match := re.fullmatch(r'(\d+)gradcaps', s):
            return int(match.group(1))
        return 0

    caps_order = get_order(args.smoothness) #int(args.smoothness[0])+1
    print("caps_order", caps_order)
    # lmd = 0.1
    loss_weights = []
    for n in range(0, caps_order+1):
        loss_weights.append((1 - args.lmd)*np.power(args.lmd, n)) # lmd-0.05
    print("loss_weights", loss_weights)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        optimize_memory_usage=args.optimize_memory_usage,
        handle_timeout_termination=False,
        history_len=caps_order
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in (pbar := tqdm(range(args.total_timesteps))):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                def actor_action(obs):
                    current_actions = actor(obs)
                    # current_actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    # current_actions = current_actions.clamp(envs.single_action_space.low[0], envs.single_action_space.high[0])
                    # current_actions = current_actions.clip(envs.single_action_space.low, envs.single_action_space.high)
                    return current_actions

                current_actions = actor(data.observations)
                actor_loss = -qf1(data.observations, current_actions).mean()

                # SMOOTHNESS PART
                caps_temporal_loss = 0

                if (args.smoothness == 'caps'):
                    next_actions = actor_action(data.next_observations)

                    diff_normed = (next_actions - current_actions)
                    caps_temporal_loss = torch.nn.functional.mse_loss(diff_normed, torch.zeros_like(diff_normed))

                    # If you want to normalize by the batch size
                    caps_temporal_loss = caps_temporal_loss / (current_actions.size(0))
                elif (args.smoothness == 'gradcaps'):
                    prev_actions = actor_action(data.prev_observations[0]) # t = -1
                    next_actions = actor_action(data.next_observations)
                    diff_prev = current_actions - prev_actions
                    diff_next = next_actions - current_actions

                    displacement = torch.abs(next_actions - prev_actions).detach()

                    grad_diff = diff_next - diff_prev
                    norm_scaler = 1.0 / (displacement + 0.00001)

                    norm_scaler = torch.tanh(norm_scaler)

                    grad_normed = grad_diff * norm_scaler

                    grad_loss = torch.nn.functional.mse_loss(grad_normed, torch.zeros_like(grad_normed))

                    caps_temporal_loss = (args.reg_coeff * grad_loss) / (current_actions.size(0))
                elif ('gradcaps' in args.smoothness):
                    def get_action(t):
                        if t == 1:
                            next_actions = actor_action(data.next_observations)
                            return next_actions
                        if t == 0:
                            return current_actions # current_actions

                        actions = actor_action(data.prev_observations[abs(t)-1])
                        return actions
                    action_chain = [get_action(1-i) for i in range(0, caps_order+2)]

                    minmax = None
                    for act_idx in range(*envs.single_action_space.shape):
                        actions_t = None
                        for t_ in range(len(action_chain)):
                            if actions_t is not None:
                                actions_t = torch.vstack((actions_t, action_chain[t_][:, act_idx]))
                            else:
                                actions_t = torch.unsqueeze(action_chain[t_][:, act_idx], 0)

                        actions_t = torch.reshape(actions_t, (actions_t.shape[1], actions_t.shape[0]))
                        mx = torch.max(actions_t, axis=1).values
                        mn = torch.min(actions_t, axis=1).values
                        if minmax is not None:
                            minmax = torch.vstack((minmax, mx-mn))
                        else:
                            minmax = torch.unsqueeze(mx-mn, 0)

                    minmax = torch.reshape(minmax, (minmax.shape[1], minmax.shape[0]))
                    displacement = torch.abs(minmax).detach() # minmax_norm

                    norm_scaler = 1.0 / (displacement + 0.00001)
                    norm_scaler = torch.tanh(norm_scaler)
                    
                    def grad_diff(n):
                        result = 0.0
                        for k in range(n + 2):
                            sign = (-1) ** k
                            binom = math.comb(n+1, k)
                            result += sign * binom * action_chain[k]

                        return result

                    grad_normed = 0.0
                    for i in range(1, caps_order+1):
                        grad_normed += loss_weights[i-1] * grad_diff(i)

                    grad_normed *= norm_scaler
                    grad_loss = torch.nn.functional.mse_loss(grad_normed, torch.zeros_like(grad_normed))
                    caps_temporal_loss = (args.reg_coeff * grad_loss) / (current_actions.size(0))

                actor_loss += caps_temporal_loss
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if (global_step > args.learning_starts) and (global_step % eval_every == 0):
            returns, smooths = evaluate(actor)
            
            avg_return, std_return = np.mean(returns), np.std(returns)
            avg_smooth, std_smooth = np.mean(smooths), np.std(smooths)

            writer.add_scalar("eval/ep_reward", avg_return, global_step)
            writer.add_scalar("eval/ep_reward_std", std_return, global_step)
            writer.add_scalar("eval/ep_reward_upper", np.max(returns), global_step)
            writer.add_scalar("eval/ep_reward_lower", np.min(returns), global_step)
            writer.add_scalar("eval/smooth_error", avg_smooth, global_step)
            writer.add_scalar("eval/smooth_error_std", std_smooth, global_step)
            writer.add_scalar("eval/smooth_error_upper", np.max(smooths), global_step)
            writer.add_scalar("eval/smooth_error_lower", np.min(smooths), global_step)

            # save model
            if (avg_return >= best_return):
                if (avg_return == best_return and avg_smooth < best_smooth) or (avg_return > best_return):
                    best_return = avg_return
                    best_smooth = avg_smooth

                    weights_path = f"{project_dir}/weights/best_reward-{best_return:.4f}_smooth-{best_smooth*100:.4f}.w"
                    os.system(f"rm -f {project_dir}/weights/*")
                    torch.save(actor, weights_path)
                best_return = avg_return
                writer.add_scalar("eval/best_reward", best_return, 0)
                writer.add_scalar("eval/best_smooth", best_smooth, 0)

            perc = int(global_step / args.total_timesteps * 100)
            # print(f"Env step {global_step:8d} / {args.total_timesteps} ({perc:2d}%)  Avg Episode Reward {avg_return:10.3f} ± {std_return:5.3f}; {avg_smooth*100:10.3f} ± {std_smooth*100:5.3f}")
            # print(f"Env step {global_step:8d} / {args.total_timesteps} ({perc:2d}%) Best Episode Reward {best_return:10.3f} ± {0:05.3f}; {best_smooth*100:10.3f} ± {0:05.3f}")
            # print()

            pbar.set_description(f"Step {global_step:8d}: {avg_return:10.3f} ± {std_return:5.3f}; {avg_smooth*100:10.3f} ± {std_smooth*100:5.3f} | Best = {best_return:5.3f} ± {0:05.3f}; {best_smooth*100:10.3f} ± {0:05.3f} ###")

    envs.close()
    writer.close()
