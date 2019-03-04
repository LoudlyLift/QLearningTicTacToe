import numpy as np
import tensorflow as tf
import math

class neural:
        NUM_STATES=int(math.pow(3,9)) # hardcoded for TTT because it's only
                                      # temporary until we switch to TF based
                                      # qtables
        @staticmethod
        def indexFromState(state):
                state_index = 0
                value = 1
                #~decode from base three
                for v in state:
                        state_index += value * v
                        value *= 3
                return state_index

        def __init__(self, state_shape, num_actions):
                tf.reset_default_graph()
                self._input = tf.placeholder(shape=[1,neural.NUM_STATES],dtype=tf.float32)
                net = self._input

                #net = tf.layers.dense(net, 20)

                net = tf.layers.dense(net, num_actions)
                self._computedQ = net
                self._predict = tf.argmax(self._computedQ,1)

                self._targetQ = tf.placeholder(shape=[1,num_actions],dtype=tf.float32)
                loss = tf.reduce_sum(tf.square(self._targetQ - self._computedQ))
                trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                self._updateModel = trainer.minimize(loss)

                init = tf.global_variables_initializer()
                self._sess = tf.Session()
                self._sess.run(init)

                eye = np.identity(neural.NUM_STATES)
                self._input_vals = [eye[index:index+1] for index in range(neural.NUM_STATES)]

        def computeQState(self, state):
                index = neural.indexFromState(state)
                inp = self._input_vals[index]
                return self._sess.run(self._computedQ, feed_dict={self._input:inp})[0]

        def updateQState(self, state, qValues):
                index = neural.indexFromState(state)
                inp = self._input_vals[index]
                self._sess.run(self._updateModel,feed_dict={self._input:inp,self._targetQ:[qValues]})

        def close(self):
                self._sess.close()
