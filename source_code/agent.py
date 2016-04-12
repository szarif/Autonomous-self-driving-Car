import random
import sys

from collections import OrderedDict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class Agent():
    def __init__(self, env):
        self.env = env
        self.state = None
        self.next_waypoint = None




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



    def get_next_waypoint(self):
        return self.next_waypoint

    def get_state(self):
        return self.state

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # # TODO: update current state
        # self.updateState(t)
        #
        # # TODO: Select action according to your policy
        # action = random.choice(Environment.valid_actions[1:])
        #
        # # Execute action and get reward
        # reward = self.env.act(self, action)
        #
        # state = None;
        #
        # # TODO: Learn policy based on state, action, reward
        # self.updatePolicy(state, action, reward)

        self.doAction()

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

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

        state = light + oncoming + left + right
        return state

    def updateQValueForStateActionPair(self, state, action, reward, currentQValue, maxNextStateActionQValue):
        # alpha is the learning rate
        alpha = .2

        # gamma
        gamma = .2

        updatedQValue = (currentQValue + alpha * (reward + (gamma * maxNextStateActionQValue) - currentQValue))

        self.qValues[(state, action)] = updatedQValue


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
