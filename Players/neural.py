import numpy as np
import tensorflow as tf
import math
import queue

import overflow_queue

class neural:
        @staticmethod
        def indexFromState(state):
                state_index = 0
                value = 1
                #~decode from base three
                for v in state:
                        state_index += value * v
                        value *= 3
                return state_index

        DEFAULT_CONFIG = {
                "batch_size": 1000, # batch_size == ML batch size; <= history size
                "history_size": 10000, # number of "Q-rows" to keep around
                "steps_per_update": 1000, # only do a TF-step once for every this-may new observations
                "train_delay": 3000,
                "learning_rate": 0.0001,
        }

        def __init__(self, state_shape, num_actions, config=None):
                if config is None:
                        config = dict()
                self._config = neural.DEFAULT_CONFIG.copy()
                self._config.update(config)

                self._history = overflow_queue.OverflowQueue(self._config["history_size"])

                tf.reset_default_graph()
                self._input = tf.placeholder(shape=(None,) + state_shape,dtype=tf.float32, name="inputs")
                net = self._input

                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, 20, activation=tf.nn.leaky_relu)

                net = tf.layers.dense(net, num_actions, name="outputs", activation=tf.nn.leaky_relu)
                self._computedQ = net
                self._targetQ = tf.placeholder(shape=[None, num_actions],dtype=tf.float32, name="targetQ")

                self._loss = tf.losses.mean_squared_error(self._computedQ, self._targetQ)
                trainer = tf.train.GradientDescentOptimizer(learning_rate=self._config["learning_rate"])
                self._updateModel = trainer.minimize(self._loss)

                init = tf.global_variables_initializer()
                self._sess = tf.Session()
                self._sess.run(init)

        def computeQState(self, state):
                return self._sess.run(self._computedQ, feed_dict={self._input:[state]})[0]

        def updateQState(self, cStep, state, qValues):
                self._history.addOne((state, qValues))
                if cStep % self._config["steps_per_update"] != 0 or len(self._history) < self._config["train_delay"]:
                        return

                data = self._history.sample(self._config["batch_size"])

                inputs = list(map(lambda tup: tup[0], data))
                outputs = list(map(lambda tup: tup[1], data))

                #inputs = list(map(lambda state: self._input_vals[neural.indexFromState(state)], inputs))

                #import pdb; pdb.set_trace()
                self._sess.run(self._updateModel,feed_dict={self._input:inputs,self._targetQ:outputs})

        def close(self):
                self._sess.close()
