import sys
sys.path.insert(0, "Environments")
sys.path.insert(0, "Players")

import qlearning
import ttt
import neural

def comp_randact(episode):
        episode = max(episode, 1) # to avoid divide by zero
        odds = 10 / episode
        odds = max(odds, 0.001)
        odds = min(odds, 1)
        return odds

qlrn = qlearning.TFQLearning(ttt.TTT_vsRandoAI(), comp_randact, neural.neural, future_discount=0.75)
(player, results) = qlrn.runEpisodes(10000)
print("Fin.")
import pdb; pdb.set_trace()
player.close()
