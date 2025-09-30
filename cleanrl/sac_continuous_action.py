# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
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
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    algo: str = "SAC"
    """the name of the algorithm"""
    smoothness: str = "vanilla"
    """the type of smoothness regularization (vanilla, caps, gradcaps, 2gradcaps, 3gradcaps, 4gradcaps)"""
    lmd: float = 0.1
    """the coefficient of the smoothness regularization"""


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
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
eval_episodes = 10  # number of episodes to run for evaluation
eval_every = 2500  # how often to evaluate the policy
log_every = 1000  # how often to log results to tensorboard
best_return = -10000
best_smooth = 10000

args = tyro.cli(Args)
print(args)
dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
run_name = f"{args.algo}-{args.smoothness}-lmd{args.lmd}-{args.env_id.split('/')[-1]}-seed{args.seed}-{dt}__notOMU"

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

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
            actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    def get_order(s: str) -> int:
        if s == "gradcaps":
            return 1
        if match := re.fullmatch(r'(\d+)gradcaps', s):
            return int(match.group(1))
        return 0

    caps_order = get_order(args.smoothness) #int(args.smoothness[0])+1
    print("caps_order", caps_order)
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
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    def get_action(t):
                        if t == 1:
                            next_actions, _, _ = actor.get_action(data.next_observations)
                            return next_actions
                        if t == 0:
                            return pi # current_actions

                        actions, _, _ = actor.get_action(data.prev_observations[abs(t)-1])
                        return actions


                    def gradient(t, n):
                        if n == 0:
                            return get_action(t)

                        return gradient(t, n-1) - gradient(t-1, n-1)

                    # SMOOTHNESS PART
                    caps_temporal_loss = 0
                    if (args.smoothness == 'caps'):
                        next_actions, _, _ = actor.get_action(data.next_observations)

                        diff_normed = (next_actions - pi)
                        caps_temporal_loss = torch.nn.functional.mse_loss(diff_normed, torch.zeros_like(diff_normed))

                        # If you want to normalize by the batch size
                        caps_temporal_loss = caps_temporal_loss / (pi.size(0))
                    elif (args.smoothness == 'gradcaps'):
                        prev_actions, _, _ = actor.get_action(data.prev_observations[0]) # t = -1
                        next_actions, _, _ = actor.get_action(data.next_observations)
                        diff_prev = pi - prev_actions
                        diff_next = next_actions - pi

                        displacement = torch.abs(next_actions - prev_actions).detach()

                        grad_diff = diff_next - diff_prev
                        norm_scaler = 1.0 / (displacement + 0.00001)

                        norm_scaler = torch.tanh(norm_scaler)

                        grad_normed = grad_diff * norm_scaler

                        grad_loss = torch.nn.functional.mse_loss(grad_normed, torch.zeros_like(grad_normed))

                        caps_temporal_loss = (grad_loss) / (pi.size(0))
                    elif ('gradcaps' in args.smoothness):
                        grad_normed, min_order = 0.0, 2

                        # lmd = 0.1
                        loss_weights = []
                        for n in range(0, caps_order+1):
                            loss_weights.append((1 - args.lmd)*np.power(args.lmd, n)) # lmd-0.05

                        for idx, order in enumerate(range(min_order, caps_order+2)):
                            grad_prev = gradient(0, order-1)
                            grad_next = gradient(1, order-1)
                            grad_diff = grad_prev - grad_next # default

                            if order == 1:
                                norm_scaler = 1
                            else:
                                # minmax_norm
                                action_chain = [get_action(1-i) for i in range(0, caps_order+2)]
                                minmax = None

                                for act_idx in range(*envs.single_action_space.shape):
                                    actions_t = None
                                    for t_ in range(len(action_chain)):
                                        # print(action_chain[t_].shape)
                                        if actions_t is not None:
                                            actions_t = torch.vstack((actions_t, action_chain[t_][:, act_idx]))
                                        else:
                                            actions_t = torch.unsqueeze(action_chain[t_][:, act_idx], 0)

                                    actions_t = torch.reshape(actions_t, (actions_t.shape[1], actions_t.shape[0]))
                                    # print(actions_t.shape)
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
                            # print('loss_weights[idx]', loss_weights[idx])
                            # print('grad_diff', grad_diff.shape)
                            # print('norm_scaler', norm_scaler.shape)
                            grad_normed += loss_weights[idx] * torch.abs(grad_diff * (norm_scaler))

                        grad_loss = torch.nn.functional.mse_loss(grad_normed, torch.zeros_like(grad_normed))
                        caps_temporal_loss = (grad_loss) / (pi.size(0))

                    actor_loss += caps_temporal_loss

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % log_every == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        
        if (global_step > args.learning_starts) and (global_step % eval_every == 0):
            returns, smooths = evaluate(actor)
            
            avg_return, std_return = np.mean(returns), np.std(returns)
            avg_smooth, std_smooth = np.mean(smooths), np.std(smooths)

            writer.add_scalar("eval/ep_reward", avg_return, global_step)
            writer.add_scalar("eval/ep_reward_upper", np.max(returns), global_step)
            writer.add_scalar("eval/ep_reward_lower", np.min(returns), global_step)
            writer.add_scalar("eval/smooth_error", avg_smooth, global_step)
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
            print(run_name)
            print(f"Env step {global_step:8d} / {args.total_timesteps} ({perc:2d}%)  Avg Episode Reward {avg_return:10.3f} ± {std_return:5.3f}; {avg_smooth*100:10.3f} ± {std_smooth*100:5.3f}")
            print(f"Env step {global_step:8d} / {args.total_timesteps} ({perc:2d}%) Best Episode Reward {best_return:10.3f} ± {0:05.3f}; {best_smooth*100:10.3f} ± {0:05.3f}")
            print()

    envs.close()
    writer.close()
