import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
# import sys
# sys.path.append('/Users/wangez/Downloads/RL/gym-solitaire') 
# sys.path.append('../gym-solitaire') 

from PegSolitaire_Env.envs import obs_to_board, board_to_obs
import matplotlib as mpl
import matplotlib.pyplot as plt

LR = 0.001
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY_PC = 50
GAMMA = 1
# NUM_EPISODES = 20000
NUM_EPISODES = 20
PATH = "pegsolitaire.pt"


def make_linear_decay_schedule(start_val, end_val, decay_pc):
    range_val = end_val - start_val
    decay_episodes = NUM_EPISODES * decay_pc / 100
    per_episode_delta = range_val / decay_episodes

    def linear_decay_schedule(episode):
        if episode > decay_episodes:
            return end_val
        else:
            return start_val + per_episode_delta * episode

    return linear_decay_schedule


def fst(pair):
    return pair[0]


def snd(pair):
    return pair[1]


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        layers = [
            nn.Linear(in_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def evaluate_valid_actions(net, s):
    current_board = obs_to_board(s)

    def evaluate_valid_action(a):
        new_board = current_board.make_move(a)
        s2 = board_to_obs(new_board)
        return s2

    valid_actions = current_board.valid_actions()
    s2s = list(map(evaluate_valid_action, valid_actions))

    with torch.no_grad():
        x = torch.tensor(s2s)
        s2_values = net(x).squeeze(-1).numpy()
        s2_value_a_pairs = list(zip(s2_values, valid_actions))

    return s2_value_a_pairs


def make_policy(net):
    def pi(s, epsilon=0):
        s2_value_a_pairs = evaluate_valid_actions(net, s)
        if random.random() < epsilon:
            return random.choice(s2_value_a_pairs)
        else:
            best_pair = s2_value_a_pairs[0]
            for pair in s2_value_a_pairs[1:]:
                if fst(pair) > fst(best_pair):
                    best_pair = pair
            return best_pair

    return pi


def show_plots(final_rewards, final_rewards_ma):
    mpl.rcParams['toolbar'] = 'None'
    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.plot(final_rewards, linewidth=0.5)
    plt.ylabel('Final Reward')
    plt.xlabel('Episodes')
    plt.subplot(212)
    plt.plot([0] * 100 + final_rewards_ma, linewidth=0.5)
    plt.ylabel('Final Reward (moving average)')
    plt.xlabel('Episodes')
    plt.show()


def train(env, pi, net, loss_fn, opt):
    final_rewards = []
    final_rewards_ma = []
    best_final_reward = -np.inf
    best_final_reward_ma = -np.inf
    epsilon_decay_schedule = make_linear_decay_schedule(EPSILON_START, EPSILON_END, EPSILON_DECAY_PC)
    for episode in range(NUM_EPISODES):
        epsilon = epsilon_decay_schedule(episode)
        s,info = env.reset()
        while True:
            # Use the policy to choose an action. Also, return the current
            # estimate of the value of the next state that the chosen action
            # leads to.
            s2_value, a = pi(s, epsilon)

            # Step the environment using the chosen action returning the next
            # state, the reward, the terminaited/truncated flag and the info dict (which we
            # ignore).
            s2, r, terminiated, truncated, _ = env.step(a)

            # Get the current estimate of the value of the current state.
            s_value = net(torch.tensor(s)).squeeze()

            # Calculate the TD target value of the current state based on the
            # reward received from the environment plus the discounted estimate
            # of the next state. If the terminated/truncated flag is set, use the reward only.
            s_value_target = r + (1 - terminiated) * GAMMA * s2_value

            # Calculate the loss.
            target = torch.tensor(s_value_target, dtype=torch.float32)
            loss = loss_fn(s_value, target)

            # Back propagate the loss.
            opt.zero_grad()
            loss.backward()
            opt.step()

            if terminiated:
                # Update stats.
                final_rewards.append(r)
                if r > best_final_reward:
                    best_final_reward = r
                final_reward_ma = -np.inf
                if len(final_rewards) >= 100:
                    final_reward_ma = np.mean(final_rewards[-100:])
                    final_rewards_ma.append(final_reward_ma)
                    if final_reward_ma > best_final_reward_ma:
                        best_final_reward_ma = final_reward_ma
                print(f"episode: {episode:5}; "
                      f"epsilon: {epsilon:.3f}; "
                      f"final reward (best): {r:3} ({best_final_reward:3}); "
                      f"final reward ma (best): {final_reward_ma:8.3f} ({best_final_reward_ma:8.3f})")

                # If the moving average final reward looks good enough,
                # save the trained model and return.
                if final_reward_ma >= 50:
                    show_plots(final_rewards, final_rewards_ma)
                    print(f"saving trained model to {PATH}")
                    torch.save(net.state_dict(), PATH)
                    return

                break

            s = s2
    show_plots(final_rewards, final_rewards_ma)


def play(env, pi):
    actions = []
    s,info = env.reset()
    while True:
        _, a = pi(s)
        actions.append(a)
        s2, _, terminiated, truncated, _ = env.step(a)
        if terminiated:
            print(f"actions: {actions}")
            env.render()
            break
        s = s2

def play_with_reward(env, pi):
    episode_reward = 0
    actions = []
    s,info = env.reset()
    while True:
        _, a = pi(s)
        actions.append(a)
        s2, reward, terminiated, truncated, _ = env.step(a)
        episode_reward += reward
        if terminiated:
            print(f"actions: {actions}")
            print(f"episode_reward: {episode_reward}")
            env.render()
            break
        s = s2
    
    return episode_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--play", action="store_true",
                        help="load a previously trained model and play a single episode")
    parser.add_argument("-v", "--verbosity", action="count",
                        help="increase output verbosity")
    args = parser.parse_args()

    env = gym.make('pegsolitaire-v0',render_mode='human')

    in_dim = env.observation_space.shape[0]
    net = Net(in_dim)
    pi = make_policy(net)

    if args.play:
        net.eval()
        net.load_state_dict(torch.load(PATH))
        play(env, pi)
    else:
        net.train()
        loss_fn = torch.nn.functional.mse_loss
        opt = optim.Adam(net.parameters(), lr=LR)
        train(env, pi, net, loss_fn, opt)

    # Sample code to run play_with_reward function
    # for _ in range(num_episodes):
    #     episode_reward = play_with_reward(env, pi)
    #     total_rewards.append(episode_reward)

    #     average_reward = np.mean(total_rewards)

    #     print(f"Test Results over {num_episodes} episodes:")
    #     print(f"Average Reward: {average_reward:.3f}")
    #     print(f"Standard Deviation: {np.std(total_rewards):.3f}")
if __name__ == "__main__":
    main()
