import sys
sys.path.insert(0, "Environments")
sys.path.insert(0, "Players")

import qlearning
import ttt
import neural
import numpy as np

def comp_randact(episode):
        episode = max(episode, 1) # to avoid divide by zero
        odds = 10 / episode
        odds = max(odds, 0.01)
        odds = min(odds, 1)
        return odds

controller = ttt.TTT_vsRandoAI()
qlrn = qlearning.TFQLearning(controller, comp_randact, neural.neural, future_discount=0.5)
(player, results) = qlrn.runEpisodes(100000)

#Play vs human
controller = ttt.TTT()
done = False
while not done:
        state = controller.reset()
        controller.printBoard()
        while controller._isActive:
                moves = player.computeQState(state)
                move = np.argmax(moves)
                print(f"AI played move {move}")
                state, _, _ = controller.step(move)
                controller.printBoard()
                if controller._isActive:
                        move = int(input("Enter your move [0-8]: "))
                        state, _, _ = controller.step(move)
        assert(not controller._isActive)
        controller.printBoard()
        if controller._victor is None:
                print("The game is a tie")
        else:
                winner = "T" if controller._victor else "F"
                print(f"The winner is {winner}")
        response = input("Play again? (y/n): ")
        done = False if response == "y" else True
player.close()
