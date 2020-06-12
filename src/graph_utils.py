import numpy as np


class MyCustomNode:
    count_nodes = 0  # static variable for this class that is reset when a new graph object is created

    def __init__(self, num_actions, num_observations, num_nodes):
        self.index = MyCustomNode.count_nodes
        self.action_distrib = (1.0 / num_actions) * np.ones(num_actions)
        self.next_node_distrib = (1.0 / num_nodes) * np.ones(shape=(num_observations, num_nodes))
        MyCustomNode.count_nodes += 1


class MyCustomGraph(list):

    def __init__(self, agent_index, N_n, num_actions, num_observations):
        MyCustomNode.count_nodes = 0  # reset the number of nodes
        self.index = agent_index
        self.num_nodes = N_n
        self.num_actions = num_actions
        self.num_observations = num_observations

    def add_node(self):
        node = MyCustomNode(self.num_actions, self.num_observations, self.num_nodes)
        self.append(node)
