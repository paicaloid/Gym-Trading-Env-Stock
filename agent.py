import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorNetwork, CriticNetwork


class PPOExperience:
    '''
    initializes the PPOExperience object with empty lists for storing experiences.
    '''
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        '''
        generates batches of experiences from the stored lists for training the PPO algorithm
        It first calculates the number of states (n_states) and
        creates an array of indices from 0 to n_states.
        It then shuffles the indices randomly and splits them into batches of size batch_size.
        The method returns the states, actions, log probabilities,
        values, rewards, dones, and batches in numpy arrays.
        '''
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_experience(self, state, action, log_probs, values, reward, done):
        '''
        stores a new experience in the appropriate list
        '''
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_experience(self):
        '''
        clears all stored experiences
        '''
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []


class Agent:
    def __init__(
                self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                n_epochs=10, chkpt_dir='models/'
                ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        # self.memory = PPOMemory(batch_size)
        self.experience = PPOExperience(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.experience.store_experience(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        '''
        - Trains the agent for a number of epochs by generating batches of experiences
            from the PPOExperience buffer.
        - calculates advantages for each experience using Generalized Advantage
            Estimation (GAE).
        - calculates the actor and critic losses for each batch and  performs a backpropagation step
        to update the actor and critic networks
        - clears the PPOExperience buffer to start anew in the next training iteration
        '''
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.experience.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(
                                                    prob_ratio,
                                                    1-self.policy_clip,
                                                    1+self.policy_clip
                                                    )
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(
                                                weighted_probs,
                                                weighted_clipped_probs
                                                )
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                              returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

        self.experience.clear_experience()
