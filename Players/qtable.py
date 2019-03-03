import numpy
import math

class qtable:
        NUM_STATES=int(math.pow(3,9)) # hardcoded for TTT because it's only
                                      # temporary until we switch to TF based
                                      # qtables

        def __init__(self, state_shape, num_actions):
                self._table = numpy.zeros((qtable.NUM_STATES, num_actions,))

        @staticmethod
        def indexFromState(state):
                state_index = 0
                value = 1
                #~decode from base three
                for v in state:
                        state_index += value * v
                        value *= 3
                return state_index

        def computeQState(self, state):
                index = qtable.indexFromState(state)
                return self._table[index]

        def updateQState(self, state, qValues):
                index = qtable.indexFromState(state)
                self._table[index] = qValues
