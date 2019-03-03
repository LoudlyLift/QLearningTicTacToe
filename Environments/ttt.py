import numpy
import enum

class TTT:
        """A controller for a game of Tic Tac Toe.

        The two players are True and False. By default False plays first.

        Any indicies can be considered to be mapped to the three by three board
        in raster scan order. Note that Tic-Tac-Toe is highly symmetric, so
        there are many other mappings that are equally valid.

        0 1 2
        3 4 5
        6 7 8

        reset must be called before the start of the first (and each subsequent)
        game.
        """
        class IllegalMoveError(Exception):
                pass

        BOARD_SIZE = 9

        # If a player claims every index in one of the lists in ALL_WIN_PATTERS,
        # that player wins.
        #
        # NOTE: all patterns must contain exactly three indicies
        ALL_WIN_PATTERNS = [
                [0,1,2],[3,4,5],[6,7,8], # "horizontals"
                [0,3,6],[1,4,7],[2,5,8], # "verticals"
                [0,4,8],[2,4,6],         # diagonals
        ]

        # TODO: why can't WIN_PATTERNS be initialized using ALL_WIN_PATTERNS?

        # WIN_PATTERNS[i] is the list of all win patterns that require the
        # player claim indpex i of the game board
        WIN_PATTERNS = [
                list(filter(lambda pattern: i in pattern,
                            [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]])) for i in range(BOARD_SIZE)
        ]

        """Prepares for the start of a new game"""
        def reset(self, first_player=False):
                self._board = [None] * TTT.BOARD_SIZE
                self._next_player = first_player
                self._played_moves = 0
                self._isActive = True
                self._victor = None
                return self.getState()

        def __str__(self):
                return f"{self.__dict__}"

        def _declare_win(self, player):
                self._isActive = False
                self._victor = player

        def _declare_tie(self):
                self._isActive = False
                self._victor = None

        def makeMove(self, i):
                if not self._isActive: # should never happen
                        raise IllegalMoveError("Game has already ended")
                self._played_moves += 1
                if self._board[i] != None: # illegal moves are suicide
                        self._declare_win(not self._next_player)
                        return (self.getState(), -10, True)

                self._board[i] = self._next_player

                # check for victory
                for pattern in TTT.WIN_PATTERNS[i]:
                        values = list(map(lambda i: self._board[i], pattern))
                        if values[0] == values[1] and values[1] == values[2]:
                                self._declare_win(self._next_player)
                                return (self.getState(), 1, True)

                # check for stalemate
                if self._played_moves == TTT.BOARD_SIZE:
                        self._declare_tie()
                        return (self.getState(), 0.5, True)

                self._next_player = not self._next_player

                return (self.getState(), 0, False)
        step = makeMove

        def getRandomMove(self):
                assert(self._isActive)
                num_possible = 9 - self._played_moves
                assert(num_possible > 0)

                rand_move = numpy.random.randint(num_possible)
                move = 0

                while rand_move > 0:
                        move += 1
                        if self._board[move] is not None:
                                continue
                        rand_move -= 1
                return move

        """Returns True if the game has ended. False otherwise."""
        def isOver(self):
                return not self._isActive

        """Assuming that the game has ended, this function will return a value
        corresponding to the ending state of the game. A value of 'None' means
        that the game ended in a tie. A boolean value means that the
        corresponding player won.
        """
        def getVictor(self):
                assert(not self._isActive)

                return _victor


        """Assuming the game is not over, returns the player who is to play next"""
        def getNextPlayer(self):
                assert(self._isActive)
                return self._next_player

        def getNumActions(self):
                return TTT.BOARD_SIZE

        @staticmethod
        def playerToInt(player):
                if player == None:
                        return 0
                return player + 1

        """Returns a list of numbers that represents the state of the game
        """
        def getState(self):
                #nones  = [ self._board[i] == None  for i in range(TTT.BOARD_SIZE) ]
                #falses = [ self._board[i] == False for i in range(TTT.BOARD_SIZE) ]
                #trues  = [ self._board[i] == True  for i in range(TTT.BOARD_SIZE) ]
                #state = nones + falses + trues # state as booleans
                #state = map(int, state)
                #return state
                return list(map(TTT.playerToInt, self._board))

class TTT_vsRandoAI(TTT):
        def step(self, i):
                assert(self._isActive)
                plr = self._next_player

                #AI's move
                TTT.step(self, i)
                if self._isActive:
                        #opponent's move
                        TTT.makeMove(self, self.getRandomMove())
                if self._isActive:
                        return (self.getState(), 0, False)
                if self._victor is None:
                        pt = 0
                elif self._victor == plr:
                        pt = 1
                else:
                        pt = -1

                return (self.getState(), pt, True)
