# import all the relevant packages
# pip install numpy gymnasium[classic_control]
import gymnasium as gym
import numpy as np

# tuneables
GAMMA = 0.99
ALPHA = 0.01

EPISODES = 2000

# create the CartPole environment
env = gym.make('CartPole-v1', render_mode="human")

# Initialize linear softmax policy weights and bias:
# π(a | s) = softmax(weights.T • state + bias)[a]
# state dimension = 4 (x, x_dot, theta, theta_dot), actions = 2
weights = np.zeros((4, 2))
bias = np.zeros(2)

def action_probs(st):
    """Return the probability of each action given a continuous state."""
    logits = np.dot(st, weights) + bias
    # subtract max for numerical stability
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)

def policy(st):
    # sample an action from the policy distribution (exploration is built in)
    probs = action_probs(st)
    return int(np.random.choice(2, p=probs)), probs

def discounted_returns(rewards):
    """Compute G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... for every step."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + GAMMA * running
        returns[t] = running
    return returns

scores = []

for e in range(EPISODES):
    # reset the environment
    state, _ = env.reset()

    # reset done
    done = False
    total_reward = 0

    # policy gradient is Monte Carlo: collect a full episode before updating
    states, actions, probs_list, rewards = [], [], [], []

    while not done:
        action, probs = policy(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        states.append(state)
        actions.append(action)
        probs_list.append(probs)
        rewards.append(reward)

        state = new_state

    # normalize returns to act as a baseline and reduce variance
    returns = discounted_returns(rewards)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    # REINFORCE update (gradient ascent on expected return):
    # ∇_w log π(a | s) = outer(state, onehot(a) - π(· | s)),  ∇_b log π(a | s) = onehot(a) - π(· | s)
    grad_w = np.zeros_like(weights)
    grad_b = np.zeros_like(bias)
    for st, ac, probs, g in zip(states, actions, probs_list, returns):
        dlog = -probs
        dlog[ac] += 1.0
        grad_w += np.outer(st, dlog) * g
        grad_b += dlog * g

    weights += ALPHA * grad_w
    bias += ALPHA * grad_b

    scores.append(total_reward)

    if (e + 1) % 100 == 0:
        recent_avg = np.mean(scores[-50:])
        print(f"Episode {e + 1:4d} | Last 50 Avg Score: {recent_avg:5.1f}")

env.close()

# display learned model parameters
print("\nTrained Policy Weights (4 x 2):")
print(weights)
print("Trained Policy Biases (2,):")
print(bias)
