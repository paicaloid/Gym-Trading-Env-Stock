from keras.layers import Dense, Flatten
from tensorflow import keras


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")
        self.flatten = Flatten()
        self.fc3 = Dense(n_actions, activation="softmax")

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def get_config(self):
        config = super(ActorNetwork, self).get_config()
        config.update(
            {
                "n_actions": self.fc3.units,
                "fc1_dims": self.fc1.units,
                "fc2_dims": self.fc2.units,
            }
        )
        return config


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")
        self.flatten = Flatten()
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        q = self.q(x)

        return q

    def get_config(self):
        config = super(CriticNetwork, self).get_config()
        config.update({"fc1_dims": self.fc1.units, "fc2_dims": self.fc2.units})
        return config
