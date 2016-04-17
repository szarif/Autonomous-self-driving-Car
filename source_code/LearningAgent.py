import random
import sys

from collections import OrderedDict

from Environment import Environment
from Agent import Agent
from Planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    valid_actions = ['left', 'right','forward', None]
    DEFAULT_Q_VALUE = 0

    def __init__(self, env):
        super(LearningAgent).__init__()  # sets self.env = env, state = None, next_waypoint = None, and a default color

        self.qValues = OrderedDict()
        self.env = env;
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.state = None
        self.next_waypoint = None

        # for testing
        self.totalReward = 0
        self.totalActions = 1
        self.averageList = []

        self.negativeRewardCount = 0;

        self.initializeQValues()


    def initializeQValues(self):
        lights = ["green", "red"]
        oncomingStates = ["forward", "right", "left", "none"]
        leftStates = ["forward", "right", "left", "none"]
        rightStates = ["forward", "right", "left", "none"]
        waypoints = ["forward", "right", "left", "none"]
        actions = ["forward", "right", "left", None]

        for light in lights:
            for oncomingState in oncomingStates:
                for leftState in leftStates:
                    for rightState in rightStates:
                        for waypoint in waypoints:
                            state = light + oncomingState + leftState + rightState + waypoint
                            for action in actions:
                                self.qValues[(state, action)] = 0

    def get_next_waypoint(self):
        return self.next_waypoint

    def get_state(self):
        return self.state

    def reset(self, destination=None):
        print("total actions: " );
        print(self.totalActions);

        print("total negative rewards: " );
        print(self.negativeRewardCount);


        self.averageList.append(self.totalReward/self.totalActions)
        # print(self.totalReward/self.totalActions);
        self.planner.route_to(destination)
        self.totalReward = 0;
        self.totalActions = 1;
        self.negativeRewardCount = 0;
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
       # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        self.act()

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getAction(self, state):
        # check current state with all possible actions and perform action with the highest qValue
        # if current state does not have a qValue for all actions perform action at random
        # with possiblity 1 - epsilon you might perform a random value

        epsilon = self.eValues[state];

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

    def getQValueForStateActionPair(self, state, action):
        # if (state,action) not in self.qValues :
        #             self.qValues[(state,action)] = self.DEFAULT_Q_VALUE

        return self.qValues[(state,action)]

    def act(self):
        self.totalActions += 1;
        state = self.getState()

        action = self.getAction(state)

        #self.next_waypoint = action

        currentQValue = self.getQValueForStateActionPair(state, action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if (reward < 0): self.negativeRewardCount = self.negativeRewardCount + 1;

        self.totalReward += reward;

        nextState = self.getState()

        maxNextStateActionQValue = self.getStateActionMaxQValue(nextState)

        self.updateQValueForStateActionPair(state, action, reward , currentQValue, maxNextStateActionQValue)

    def getStateActionMaxQValue(self, state):
        maxValue = self.DEFAULT_Q_VALUE

        for action in self.valid_actions:

                qValue = self.getQValueForStateActionPair(state, action)

                if qValue > maxValue:
                    maxValue = qValue

        return maxValue


    def getState(self):
        inputs = self.env.sense(self)

        light = inputs['light']

        oncoming = inputs['oncoming']
        if oncoming is None:
            oncoming = "none"

        left = inputs['left']
        if left is None:
            left = "none"

        right = inputs['right']
        if right is None:
            right = "none"

        nextwaypoint = self.next_waypoint
        if self.next_waypoint is None:
            nextwaypoint = "none"

        state = light + oncoming + left + right + nextwaypoint

        self.state = state

        return state

    def updateQValueForStateActionPair(self, state, action, reward, currentQValue, maxNextStateActionQValue):
        # alpha is the learning rate
        alpha = .2

        # gamma
        gamma = .2

        updatedQValue = (currentQValue + alpha * (reward + (gamma * maxNextStateActionQValue) - currentQValue))

        self.qValues[(state, action)] = updatedQValue
        # print("q value: " )
        # print(self.qValues[(state, action)])
        # print("------")



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
