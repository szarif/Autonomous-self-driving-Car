import random
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

    def __init__(self, env):
        super(LearningAgent).__init__()  # sets self.env = env, state = None, next_waypoint = None, and a default color

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

        # TODO: update current state
        self.updateState(t)

        # TODO: Select action according to your policy
        action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        reward = self.env.act(self, action)

        state = None;

        # TODO: Learn policy based on state, action, reward
        self.updatePolicy(state, action, reward)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def updateState(self, t):
        # TODO: update current state
        pass


    def updatePolicy(self, state, action, reward):
        # TODO: Learn policy based on state, action, reward
        pass



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
