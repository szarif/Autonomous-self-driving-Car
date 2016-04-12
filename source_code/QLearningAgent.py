import random
from collections import OrderedDict

import sys

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class QLearningAgent(Agent):

    valid_actions = ['left', 'right','forward', None]
    DEFAULT_Q_VALUE = 0

    def __init__(self, env):
        self.qValues = OrderedDict()

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

                qValue = self.getQValueForStateActionPair(state, action)

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

    def getQValueForStateActionPair(self, state, action):
        if (state,action) not in self.qValues :
                    self.qValues[(state,action)] = self.DEFAULT_Q_VALUE

        return self.qValues[(state,action)]

    def doAction(self):
        state = self.getState()

        action = self.chooseAction(state)

        currentQValue = self.getQValueForStateActionPair(state, action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        nextState = self.getState()

        maxNextStateActionQValue = self.getStateActionMaxQValue(nextState)

        self.updateQValueForStateActionPair(state, action, reward , currentQValue, maxNextStateActionQValue)

    def getStateActionMaxQValue(self, state):
        max = self.DEFAULT_Q_VALUE

        for action in self.valid_actions:

                qValue = self.getQValueForStateActionPair(state, action)

                if qValue > max:
                    max = qValue

        return max


    def getState(self):
        inputs = self.env.sense(self)
        state = inputs['light'] + inputs['oncoming'] + inputs['left'] + inputs['right']
        return state

    def updateQValueForStateActionPair(self, state, action, reward, currentQValue, maxNextStateActionQValue):
        # alpha is the learning rate
        alpha = .2

        # gamma
        gamma = .2

        updatedQValue = (currentQValue + alpha * (reward + (gamma * maxNextStateActionQValue) - currentQValue))

        self.qValues[(state, action)] = updatedQValue



