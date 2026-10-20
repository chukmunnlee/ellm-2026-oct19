# import all the relevant packages
# pip install numpy gymnasium[classic_control]
import gymnasium as gym
import numpy as np

# tuneables
GAMMA = 0.99
ALPHA = 0.001
DECAY = 0.002
MIN_PROB = 0.02
MAX_PROB = 1.0

EPISODES = 2000

# create the CartPole environment
env = gym.make('CartPole-v1', render_mode="human")

# Initialize linear function approximator weights and bias:
# Q(s, a) = weights[:, a] • state + bias[a]
# state dimension = 4 (x, x_dot, theta, theta_dot), actions = 2
weights = np.zeros((4, 2))
bias = np.zeros(2)

def predict_q_values(st):
    """Predict Q-values for all actions given a continuous state."""
    return np.dot(st, weights) + bias

def decay(ep):
    return MIN_PROB + (MAX_PROB - MIN_PROB) * np.exp(-ep * DECAY)

def policy(st, eps):
    if np.random.random() > eps:
        # take the max action predicted by the function
        return int(np.argmax(predict_q_values(st)))

    # take a random action
    return np.random.randint(0, 2)

def q_value(st, ac):
    return predict_q_values(st)[ac]

def q_max_value(st):
    return np.amax(predict_q_values(st))

scores = []

for e in range(EPISODES):
    # reset the environment
    eps = decay(e)
    state, _ = env.reset()
    action = policy(state, eps)

    # reset done
    done = False
    total_reward = 0

    while not done:
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            # penalize early failures to steer the weights away from falling
            target_reward = -1.0 if terminated and total_reward < 500 else reward
            td_target = target_reward
        else:
            # one step reward + prediction
            next_action = policy(new_state, eps)
            predicted_future_reward = q_max_value(new_state)
            td_target = reward + (GAMMA * predicted_future_reward)

        # TD error
        current_prediction = q_value(state, action)
        td_error = td_target - current_prediction

        # Gradient descent update for linear function approximation:
        # ∇_w Q(s, a) = state,  ∇_b Q(s, a) = 1
        weights[:, action] += ALPHA * td_error * state
        bias[action] += ALPHA * td_error

        if not done:
            action = next_action
            state = new_state

    scores.append(total_reward)

    if (e + 1) % 100 == 0:
        recent_avg = np.mean(scores[-50:])
        print(f"Episode {e + 1:4d} | Last 50 Avg Score: {recent_avg:5.1f} | Epsilon: {eps:.3f}")

env.close()

# display learned model parameters
print("\nTrained Linear Model Weights (4 x 2):")
print(weights)
print("Trained Model Biases (2,):")
print(bias)
