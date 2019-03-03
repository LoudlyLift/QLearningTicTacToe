import sys
sys.path.insert(0, "Environments")
sys.path.insert(0, "Players")

import qlearning
import ttt
import qtable

def comp_randact(episode):
        episode = max(episode, 1) # to avoid divide by zero
        odds = 100 / episode
        odds = max(odds, 0.00001)
        odds = min(odds, 1)
        return odds

qlrn = qlearning.TFQLearning(ttt.TTT(), comp_randact, qtable.qtable)
(player, results) = qlrn.runEpisodes(10000)
print("Fin.")
import pdb; pdb.set_trace()
