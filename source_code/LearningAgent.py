import random
import sys

from collections import OrderedDict

from environment import Environment
from Agent import Agent
from Planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    valid_actions = ['left', 'right','forward', None]
    DEFAULT_Q_VALUE = 0

    def __init__(self, env):
        super(LearningAgent).__init__()  # sets self.env = env, state = None, next_waypoint = None, and a default color

        self.numTrials = 1000;
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
        self.totalDeterministicRewardList = []
        self.SuccessfulTripsList = []

        self.trialCount = 0
        self.previousStateList = []

        # list containing the number of deterministic actions with negative rewards per trial
        self.deterministicNegativeActionList = []
        self.deterministicNegativeAction = 0
        # list holding integers 0 through 100, for plotting trials on 1 through 100 on x-axis
        self.trialList = []
        for i in range(0, self.numTrials):
            self.trialList.append(i)

        self.randomAction = False

        self.negativeRewardCount = 0;

        self.initializeQValues()

        self.successfulTripsCount = 0
        self.averageSuccessfulTrips = []


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
        #print("total actions: " );
        #print(self.totalActions);

        #print("total negative rewards: " );
        #print(self.negativeRewardCount);
        self.deterministicNegativeActionList.append(self.deterministicNegativeAction)
        self.deterministicNegativeAction = 0


        self.averageList.append(self.totalReward/self.totalActions)

        self.totalDeterministicRewardList.append(self.totalReward/self.totalActions)

        # print(self.totalReward/self.totalActions);
        self.planner.route_to(destination)
        self.totalReward = 0;
        self.totalActions = 1;
        self.negativeRewardCount = 0;

        self.trialCount += 1

        if (self.env.successfulTrip):
            self.SuccessfulTripsList.append(0)
            self.successfulTripsCount += 1;
        else:
            self.SuccessfulTripsList.append(1)

        self.averageSuccessfulTrips.append(self.successfulTripsCount / self.trialCount)

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

        epsilon = 0.1

        # random.randint(a, b) returns random int N such that a <= N <= b
        # 0.2 chance of selecting random action
        # 0.8 chance of selecting action with highest Q-value
        if random.randint(1,10) > 10 * epsilon:
            self.randomAction = False
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
            self.randomAction = True
            # select any random action
            nextAction = random.choice(self.valid_actions)

        return nextAction

    def getQValueForStateActionPair(self, state, action):
        # if (state,action) not in self.qValues :
        #             self.qValues[(state,action)] = self.DEFAULT_Q_VALUE

        return self.qValues[(state,action)]

    def act(self):

        state = self.getState()

        action = self.getAction(state)

        #self.next_waypoint = action

        currentQValue = self.getQValueForStateActionPair(state, action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if self.state in self.previousStateList and not self.randomAction:
            self.totalActions += 1;
            self.totalReward += reward;
            if (reward < 0): 
                self.deterministicNegativeAction += reward
        else:
            self.previousStateList.append(self.state)



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
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.numTrials)  # press Esc or close pygame window to quit

    #Testing size
    print(len(a.trialList))
    print(len(a.averageSuccessfulTrips))

    #-----------Average Reward Per Deterministic Action-----------#
    # x = np.array (a.trialList )
    # y = np.array (a.totalDeterministicRewardList)

    # plt.xlabel("Trial Number")
    # plt.ylabel("Average Reward Per Deterministic Action")

    #-----------Average Successful Full Trips-----------#
    x = np.array (a.trialList )
    y = np.array (a.averageSuccessfulTrips)

    plt.xlabel("Trial Number")
    plt.ylabel("Average Successful Trips")

    #creating the scatter plot
    plt.scatter(x, y, s=30, alpha=0.15, marker='o')

    #create the best fit line
    par = np.polyfit(x, y, 1, full=True)

    #graph the best fit line
    slope=par[0][0]
    intercept=par[0][1]
    xl = [min(x), max(x)]
    yl = [slope*xx + intercept  for xx in xl]




    # error bounds
    yerr = [abs(slope*xx + intercept - yy)  for xx,yy in zip(x,y)]
    par = np.polyfit(x, yerr, 2, full=True)

    yerrUpper = [(xx*slope+intercept)+(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(x,y)]
    yerrLower = [(xx*slope+intercept)-(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(x,y)]

    #ploting the best fit line
    plt.plot(xl, yl, '-r')

    #uncomment to plot the error bounds
    #plt.plot(x, yerrLower, '--r')
    #plt.plot(x, yerrUpper, '--r')

    #show the graph
    plt.show()


if __name__ == '__main__':
    run()