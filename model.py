import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.mu_layer = tf.keras.layers.Dense(action_dim)
        self.std_layer = tf.keras.layers.Dense(action_dim, activation='softplus')  # Std must be positive

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mu = self.mu_layer(x)  # Mean of the action distribution
        std = self.std_layer(x)  # Standard deviation of the action distribution
        return mu, std

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value_layer = tf.keras.layers.Dense(1)  # Output single scalar value

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        value = self.value_layer(x)  # State value prediction
        return value


class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2, actor_lr=0.0003, critic_lr=0.001,
                 train_epochs=10, batch_size=64, entropy_coeff=0.01, lambda_adv=0.95, checkpoint_dir="checkpoints"):
        # Actor-Critic models
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.lambda_adv = lambda_adv
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_action(self, state):
        """
        Sample an action from the policy given the state.
        """

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, std = self.actor(state_tensor)
        dist = tfp.distributions.Normal(mu, std)
        std+=1e-14

        # Sample action
        action = dist.sample()  # Sample from the distribution (plus epsilon to avoid nans)
        log_prob = dist.log_prob(action)  # Log probability of the action

        # Apply squashing function to restrict action to [-1, 1] (if needed)
        action = tf.tanh(action)
        
        # Adjust the log probability to account for the squashing function
        log_prob -= tf.reduce_sum(tf.math.log(1 - tf.tanh(action)**2 + 1e-6), axis=-1)

        return action.numpy(), log_prob.numpy()

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        """
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            advantage = delta + self.gamma * self.lambda_adv * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return np.array(advantages)

    def update(self, states, actions, log_probs, returns, advantages):
        """
        Update the actor and critic using PPO loss functions.
        """
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        for _ in range(self.train_epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                # Actor loss
                mu, std = self.actor(states)
                dist = tfp.distributions.Normal(mu, std)
                
                # Calculate new log probabilities
                new_log_probs = dist.log_prob(actions)
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                # Expand advantages to (batch_size, num_actions, 1)
                advantages = tf.reshape(advantages, (-1, 1, 1))

                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                
                # Add entropy bonus for exploration
                entropy = tf.reduce_mean(dist.entropy())
                actor_loss -= self.entropy_coeff * entropy

                # Critic loss (mean squared error)
                values = tf.squeeze(self.critic(states), axis=-1)
                critic_loss = tf.reduce_mean((returns - values) ** 2)

            # Apply gradients
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        #print("Actor gradients:", [tf.reduce_sum(tf.abs(g)) for g in actor_grads if g is not None])
        #print("Critic gradients:", [tf.reduce_sum(tf.abs(g)) for g in critic_grads if g is not None])

    def save(self):
        """
        Save the weights of the actor and critic networks.
        """
        actor_path = os.path.join(self.checkpoint_dir, "actor_weights.h5")
        critic_path = os.path.join(self.checkpoint_dir, "critic_weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Agent weights saved to {self.checkpoint_dir}")

    def load(self):
        """
        Load the weights of the actor and critic networks.
        """
        actor_path = os.path.join(self.checkpoint_dir, "actor_weights.h5")
        critic_path = os.path.join(self.checkpoint_dir, "critic_weights.h5")

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Agent weights loaded from {self.checkpoint_dir}")
        else:
            print("No saved weights found. Starting from scratch.")


def train(env, agent, max_episodes=500, print_info=False):
    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        episode_reward = 0

        iters = 0

        # Collect trajectory
        done = False
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
            episode_reward += reward
            iters+=1

        # Compute returns and advantages
        values = agent.critic(tf.convert_to_tensor(states + [state], dtype=tf.float32)).numpy().flatten()
        returns = []
        g = 0
        for r, d in zip(rewards[::-1], dones[::-1]):
            g = r + agent.gamma * g * (1 - d)
            returns.insert(0, g)

        advantages = agent.compute_advantages(rewards, values, dones)
        returns = np.array(returns)

        # Update PPO agent
        agent.update(states, actions, log_probs, returns, advantages)

        # Log progress
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if print_info:
            print(f"Dir: {env.simulation.rocket.get_dir()}, Throttle: {env.simulation.rocket.get_throttle()}")
            print(f"Number of iterations: {iters}, Apogee: {env.simulation.rocket.get_apogee()}")
