import numpy

class TFQLearning:
        """env must define these methods:

            reset(self): resets env for a new game, and returns the starting
            state.

            getRandomMove(self): get a random move

            step(self, int): perform the specified move and return the tuple
                    (state_new,reward,done).

            getStateShape(): returns a tuple of numbers. Each value represents
            the length of the state matrix in the corresponding
            dimension. e.g. returning the value (2, 3, 4) indicates that state
            of this environment is represented by 24 integers and that those
            integers take the shape of a cube that has sides of length two,
            three, and four.

            getNumActions(): returns the number of actions that can be made at
            any given time

        compute_randact(episode_num): given the episode number, this computes
        probability with which a random move should be made instead of action
        chosen.

        cls_player must make an instance using
        cls_player(state_shape, num_actions). That instance must have these
        methods:

            computeQState(self, state): returns a list of the estimated value of
                taking each enumerated action. (i.e. the row of the QTable
                corresponding to state)

            updateQState(self, state, qValues): do the player's equivalent of
                updating state's row in the Q-Table to match it's new estimated
                values.

        """
        def __init__(self, env, compute_randact, cls_player, future_discount=.99):
                self._env = env
                state = self._env.reset()
                state_shape = numpy.asarray(state).shape
                self._player = cls_player(state_shape, self._env.getNumActions())
                self._compute_randact = compute_randact
                self._future_discount = future_discount

        # runs count episodes.
        #
        # Returns (player, [ Σ(episode i's rewards) for i in range(count) ])
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
                                        act = self._env.getRandomMove()
                                else:
                                        act = numpy.argmax(allActQs)
                                state_new,reward,done = self._env.step(act)
                                maxHypotheticalQ = max(self._player.computeQState(state_new))
                                allActQs[act] = reward + self._future_discount * maxHypotheticalQ
                                self._player.updateQState(state_old, allActQs)

                                reward_sum += reward
                                state_old = state_new
                                cStep += 1
                        if (ep_num % 100 == 0):
                                print("episode %5d: %f" % (ep_num, reward_sum))
                        reward_sums.append(reward_sum)
                return (self._player, reward_sums)
