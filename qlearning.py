import numpy

class QEnv:
        def __init__(self):
                pass

        # returns state
        def reset(self):
                pass

        def get_random_move(self):
                pass

        # returns (state, reward, isDone)
        def step(self, action):
                pass

class Player:
        def __init__(self, ):
                pass

        #finds the "row" of the Q-table corresponding to the given state
        def computeQState(self, state):
                pass

class TFQLearning:
        def __init__(self, env, compute_randact, player, future_discount=.99):
                self._env = env
                self._player = player
                self._compute_randact = compute_randact
                self._future_discount = future_discount

        # runs count episodes.
        #Returns [ Σ(episode i's rewards) for i in range(count) ]
        def runEpisodes(self, count=1):
                reward_sums = []
                for ep_num in range(count):
                        state_old = self._env.reset()
                        reward_sum = 0
                        done = False
                        cStep = 0


                        while not done:
                                allActQs = self._player.computeQState(state_old)
                                if numpy.random.rand(1) < self._compute_randact(ep_num):
                                        act = self._env.get_random_move()
                                else:
                                        act = numpy.argmax(allActQs)
                                state_new,reward,done = self._env.step(act)
                                maxHypotheticalQ = max(self._player.computeQState(stateNew))
                                allActQ[act] = reward + future_discount * maxHypotheticalQ
                                self._player.updateQState(stateOld, allActQ)

                                reward_sum += reward
                                state_old = state_new
                                cStep += 1
                        reward_sums.append(reward_sum)
                return reward_sums
