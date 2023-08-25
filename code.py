import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio

import tensorflow as tf
from tensorflow.keras import layers

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

def nn_model(s_size, a_size, h_size):
    inputs = layers.Input(shape=(s_size,))
    x = layers.Dense(h_size, activation='relu')(inputs)
    outputs = layers.Dense(a_size, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class Policy:
    def __init__(self, s_size, a_size, h_size):
        tf.keras.backend.clear_session()
        self.model = nn_model(s_size, a_size, h_size)

    def act(self, state):
        state = tf.constant(state, dtype=tf.float32)
        if len(tf.constant(state).shape) == 1:
            state = tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0)
        probs = self.model(state)
        action = tf.random.categorical(tf.math.log(probs), 1).numpy().flatten().item()
        return action

def loss_function(returns, probs):
    log_prob = tf.math.log(probs)
    loss = - returns * log_prob
    return tf.reduce_mean(loss)

def train_step(states, actions, returns):
    with tf.GradientTape() as tape:
        probs = policy.model(states)
        probs = tf.gather_nd(
                        probs,
                        tf.stack([tf.range(len(actions), dtype=tf.int32), tf.cast(actions, dtype=tf.int32)], axis=1)
                    )
        # return returns, probs
        loss = loss_function(returns, probs)
    grads = tape.gradient(loss, policy.model.trainable_weights)
    optimizer.apply_gradients(zip(grads, policy.model.trainable_weights))
    return loss

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    losses = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        rewards = []
        states = []
        actions = []

        state = env.reset()[0]
        states.append(state)
        # Line 4 of pseudocode
        for t in range(max_t):
            action = policy.act(state)
            actions.append(action)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            states.append(state)
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = tf.constant(returns)
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

        # Line 7:
        policy_loss = []
        states = tf.constant(states, dtype=tf.float32)
        loss = train_step(states, actions, returns)

        # return returns, probs
        losses.append(loss)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores, losses

policy = Policy(s_size, a_size, 16)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
n_training_episodes = 3000
n_evaluation_episodes = 10
max_t = 1000
gamma = 1.0
print_every = 100

scores, losses = reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every)

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards_ep = 0

    for step in range(max_steps):
        action = policy.act(state)
        new_state, reward, done, info, _ = env.step(action)
        total_rewards_ep += reward

        if done:
            break
        state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

evaluate_agent(eval_env,
               max_t,
               n_evaluation_episodes,
               policy)
