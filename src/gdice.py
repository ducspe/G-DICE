# third party libraries
import numpy as np
import sys


# my libraries
import utils
from routines import *
from graph_utils import MyCustomNode, MyCustomGraph
from test import draw_graph

# Flags
TEST_FLAG = False

# Define the parameters of the algorithm:
N_n = 3  # number of nodes in a graph
N_k = 10  # number of iterations
N_s = 100  # total number of samples
N_b = 10  # number of best samples
alpha = 0.1  # learning rate
gamma = 1  # discount factor for each step. Make it 1 if you actually don't need diminishing rewards over time
horizon = 3
N_simulations = 100  # from a statistical point of view, 30 is a minimal number. More is better. See also Bessel's correction factor

# Read and parse the problem file:
ut = utils.Utils("../problems/dectiger.dpomdp")

# Convenience variables to avoid too many calls to utils
decpomdp = ut.decpomdp
num_agents = decpomdp.num_agents()
num_states = decpomdp.num_states()
num_joint_observations = decpomdp.num_joint_observations()
reward_matrix = ut.make_reward_matrix()


transition_model = ut.make_transition_matrix()  # S x S x A
observation_model = ut.make_observation_matrix()  # O x S x A


# Construct a graph for each agent (this is a list implementation actually)
policy_graphs = []  # we will construct a graph for each agent and append it to this list
for agent_index in range(num_agents):
    print("Initializing FSM for agent %s" % agent_index)
    num_actions = decpomdp.num_actions(agent_index)
    num_observations = decpomdp.num_observations(agent_index)

    G = MyCustomGraph(agent_index, N_n, num_actions, num_observations)

    for q in range(N_n):
        G.add_node()

    policy_graphs.append(G)

# End graph initialization process


v_b = -sys.float_info.max  # initial best value is -1.7976931348623157e+308
v_w = -sys.float_info.max  # initial worst value is -1.7976931348623157e+308


# for npgi -w 2:
# test_actions = [[0, 0, 0, 0, 2], [0, 0, 0, 2, 0]]
# test_next_nodes = [[[2, 1], [3, 3], [4, 3], [0, 0], [0, 0]], [[1, 2], [3, 4], [4, 4], [0, 0], [0, 0]]]

# for npgi "0 2" tree with value=-17:
# test_actions = [[0, 2], [0, 2]]
# test_next_nodes = [[[1, 1], [0, 0]], [[1, 1], [0, 0]]]

# for npgi "1 2 0" tree with value -32:
# test_actions = [[1, 2, 0], [1, 2, 0]]
# test_next_nodes = [[[1, 1], [2, 2], [0, 0]], [[1, 1], [2, 2], [0, 0]]]

# for npgi "2 1 1 2" tree with value=-60
test_actions = [[2, 1, 1], [2, 1, 1]]
test_next_nodes = [[[1, 1], [2, 2], [0, 0]], [[1, 1], [2, 2], [0, 0]]]

best_policy = None

initial_belief = ut.make_initial_belief()
# This comment marks the working branch
for k in range(N_k):  # N_k is the number of iterations (how many times we will attempt to improve the policy)
    print("Iteration: ", k)

    storage_list = []  # this is a list of tuples tracking all useful information: action taken in a node, next_nodes etc.

    for sample in range(N_s):  # N_s is the total number of samples
        # Draw a sample from the policy distribution to get a concrete policy
        # Create a deterministic next node transition as well (for each agent in a current node q, what is the next node given an observation)

        deterministic_action_for_node = []  # container for a deterministic action to be taken in a particular node
        deterministic_next_node = []  # list of numpy arrays; each array corresponds to an agent
                                      # and stores deterministic next nodes given a current node and an observation
        for i in range(num_agents):
            num_observations = decpomdp.num_observations(i)
            deterministic_action_for_node.append(-1 * np.ones(N_n, dtype=int))
            deterministic_next_node.append(-1 * np.ones(shape=(N_n, num_observations), dtype=int))
            for q in range(N_n):
                if TEST_FLAG:
                    deterministic_action_for_node[i][q] = test_actions[i][q]
                else:
                    deterministic_action_for_node[i][q] = np.random.choice(policy_graphs[i].num_actions, p=policy_graphs[i][q].action_distrib)
                for obs in range(num_observations):
                    if TEST_FLAG:
                        deterministic_next_node[i][q][obs] = test_next_nodes[i][q][obs]
                    else:
                        deterministic_next_node[i][q][obs] = np.random.choice(N_n, p=policy_graphs[i][q].next_node_distrib[obs])


        values_list = []  # Container for the rewards after each simulation/evaluation process. We compute the average reward using it
        ################################################################################################################
        # Start evaluation part:
        for simulation_counter in range(N_simulations):
            state = np.random.choice(num_states, p=initial_belief)
            cumulated_reward = 0.0
            current_nodes = np.zeros(num_agents, dtype=int)  # keeps track of the current nodes of each FSM.
            for t in range(horizon):
                actions_taken = -1 * np.ones(num_agents, dtype=int)  # stores individual actions for each agent at one time step
                for i in range(num_agents):
                    actions_taken[i] = deterministic_action_for_node[i][current_nodes[i]]

                joint_action_index = ut.get_joint_action_index(actions_taken)

                cumulated_reward += gamma**t * reward_matrix[state][joint_action_index]

                state = np.random.choice(num_states, p=transition_model[:, state, joint_action_index])

                joint_observation = np.random.choice(num_joint_observations, p=observation_model[:, state, joint_action_index])

                individual_observations_list = ut.get_individual_observations_list(joint_observation)

                for i in range(num_agents):
                    obs = individual_observations_list[i]
                    current_nodes[i] = deterministic_next_node[i][current_nodes[i]][obs]

            values_list.append(cumulated_reward)

        average_reward = np.mean(values_list)

        # End evaluation part
        ################################################################################################################
        if average_reward > v_w:
            storage_list.append((deterministic_action_for_node, average_reward, deterministic_next_node))

        if average_reward > v_b:
            best_policy = (deterministic_action_for_node, deterministic_next_node)
            v_b = average_reward
            print("Found a new v_b: ", v_b)

    best_policy_storage_list = sorted(storage_list, key=lambda tup: tup[1])[-N_b:]  # sort by the average reward element of each tuple
    best_values_list = [tuple_item[1] for tuple_item in best_policy_storage_list]

    if best_values_list:
        v_w = min(best_values_list)
        print("Updated v_w: ", v_w)

        # Maximum Likelihood Estimation to improve the probability distributions:
        for i in range(num_agents):
            for q in range(N_n):
                updated_action_distrib = compute_updated_action_distrib(policy_graphs[i], q, best_policy_storage_list)
                policy_graphs[i][q].action_distrib = alpha * updated_action_distrib + (1 - alpha) * policy_graphs[i][q].action_distrib
                print("Updated action distribution of node %s for agent %s is %s" % (q, i, policy_graphs[i][q].action_distrib))

                updated_next_node_distribution = compute_updated_next_node_distrib(policy_graphs[i], q, best_policy_storage_list)
                policy_graphs[i][q].next_node_distrib = alpha * updated_next_node_distribution + (1 - alpha) * policy_graphs[i][q].next_node_distrib
                print("Updated next node distribution of node %s for agent %s is %s" % (q, i, policy_graphs[i][q].next_node_distrib))

print("Best policy is %s with value %s" % (best_policy[0], v_b))

draw_graph(best_policy)
