import random
from collections import OrderedDict

import sys

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LAgent(Agent):

    valid_actions = ['left', 'right','forward', None]

    def __init__(self, env):
        self.qValues = OrderedDict()
        initializeQValues()
        # learning rate
        self.alpha = 0.2
        # probability of choosing random action
        # important for exploring all state spaces
        self.epsilon = 0.2
        # discount value
        self.gamma = 0.2

    def initializeQValues():
        lights = ['green', 'red']
        oncomingStates = ['forward', 'right', 'left', 'none']
        leftStates = ['forward', 'right', 'left', 'none']
        rightStates = ['forward', 'right', 'left', 'none']
        waypoints = ['forward', 'right', 'left', 'none']
        actions = ['forward', 'right', 'left', None]

        for light in lights:
            for oncomingState in oncomingStates:
                for leftState in leftStates:
                    for rightState in rightStates:
                        for waypoint in waypoints:
                            state = light + oncomingState + leftState + rightState + waypoint
                            for action in actions:
                                self.qValues[(state, action)] = 0

    def getAction(self, state):
        # check current state with all possible actions and perform action with the highest qValue
        # if current state does not have a qValue for all actions perform action at random
        # with possiblity 1 - epsilon you might perform a random value

        epsilon = self.epsilon

        # random.randint(a, b) returns random int N such that a <= N <= b
        # 0.2 chance of selecting random action
        # 0.8 chance of selecting action with highest Q-value
        if random.randint(1,10) > 10 * epsilon:

            # smallest negative number in Python
            maxValue= -sys.maxsize - 1

            # initalize next action
            nextAction =  None
            # list that stores actions with equal Q Values
            equalActions = []

            for action in self.valid_actions:
                # get qValue from table
                qValue = self.qValues[(state,action)]

                # populate list of equal actions
                if qValue > maxValue:
                    maxValue = qValue
                    # erase previous equalActions list and append current action
                    equalActions = [action]
                elif qValue == maxValue:
                    # add to list of equalActions
                    # used later for randomly selecting from equal actions
                    equalActions.append(action)
            # randomly select from list of equal actions
            nextAction = random.choice(equalActions)
        else:
            # select any random action
            nextAction = random.choice(self.valid_actions)

        return nextAction

    def getState(self):
        inputs = self.env.sense(self)

        light = inputs['light']

        oncoming = inputs['oncoming']
        if oncoming is None:
            oncoming = 'none'

        left = inputs['left']
        if left is None:
            left = 'none'

        right = inputs['right']
        if right is None:
            right = 'none'

        nextWaypoint = self.next_waypoint
        if self.next_waypoint is None:
            nextWaypoint = 'none'

        state = light + oncoming + left + right + nextWaypoint

        self.state = state

        return state

    def act(self):
        state = self.getState()

        action = self.getAction(state)

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

    def updateQValueForStateActionPair(self, state, action, reward, currentQValue, maxNextStateActionQValue):
        # alpha is the learning rate
        alpha = self.alpha

        # gamma
        gamma = self.gamma

        updatedQValue = (currentQValue + alpha * (reward + (gamma * maxNextStateActionQValue) - currentQValue))

        self.qValues[(state, action)] = updatedQValue