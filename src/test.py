import networkx as nx
import utils
from networkx.drawing.nx_agraph import write_dot

ut = utils.Utils("../problems/dectiger.dpomdp")
num_agents = ut.decpomdp.num_agents()


def draw_graph(best_policy):
    deterministic_action_for_node = best_policy[0]
    deterministic_next_node = best_policy[1]

    for agent in range(num_agents):
        G = nx.MultiDiGraph()  # Directed graph with self loops and parallel edges
        num_observations = ut.decpomdp.num_observations(agent)

        for node, action in enumerate(deterministic_action_for_node[agent]):
            action_name = ut.decpomdp.action_name(agent, action)
            node_label = "N " + str(node) + " : " + "A " + str(action) + " [" + action_name + "]"
            G.add_node(node, label=node_label)

            observation_table = deterministic_next_node[agent][node]

            for obs in range(num_observations):
                next_node = observation_table[obs]
                obs_name = ut.decpomdp.observation_name(agent, obs)
                edge_label = str(obs) + " [" + obs_name + "]"
                G.add_edge(node, next_node, label=edge_label)

        file_name = "gdice_best_policy_agent" + str(agent) + ".dot"
        write_dot(G, file_name)
