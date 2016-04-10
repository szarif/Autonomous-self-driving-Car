import random
from collections import OrderedDict

import sys

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class QLearningAgent():

    valid_actions = ['left', 'right','forward', None]
    DEFAULT_Q_VALUE = 0

    def __init__(self, env):
        self.qValues = OrderedDict()

    def initializeQValues(self):
        for qValue in self.qValues.items():
            qValue = self.DEFAULT_Q_VALUE

    def chooseAction(self, state):
        # check current state with all possible actions and perform action with the highest qValue
        # if current state does not have a qValue for all actions perform action at random
        # with possiblity 1 - epsilon you might perform a random value

        epsilon = .2

        if random.random() > epsilon:
            #use our past experience to choose the next action

            max = -sys.maxsize - 1

            nextAction =  None
            equalActions = []

            for action in self.valid_actions:
                qValue = self.qValues( (state,action) )

                if qValue > max:
                    max = qValue
                    equalActions = [action]
                elif qValue == max:
                    #equal chance of choosing one action
                    equalActions.append(action)

                nextAction = random.choice(equalActions)
        else:
            nextAction = random.choice(self.valid_actions)

        return nextAction

    def doAction(self):
        state = None

        action = self.chooseAction(state)

        currentQValue = self.qValues((state,action))

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.updateQValueForStateActionPair(state, action, reward)


    def updateQValueForStateActionPair(self, state, action, reward, maxQDifference):
        # alpha is the learning rate
        alpha = .2
        currentQValue = self.qValues( (state,action) )
        updatedQValue = currentQValue + alpha[ reward + maxQDifference ]

