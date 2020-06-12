# third party libraries
import numpy as np


def compute_updated_action_distrib(G, q, best_policy_storage_list):
    policy_vectors = [tuple_item[0][G.index] for tuple_item in best_policy_storage_list]
    updated_distribution = np.zeros(G.num_actions)

    actions_taken_in_q = []
    for vec in policy_vectors:
        actions_taken_in_q.append(vec[q])

    for action in range(G.num_actions):
        count = actions_taken_in_q.count(action)
        prob = count/len(actions_taken_in_q)

        updated_distribution[action] = prob

    return updated_distribution


def compute_updated_next_node_distrib(G, q, best_policy_storage_list):
    next_node_vectors = [tuple_item[2][G.index][q] for tuple_item in best_policy_storage_list]

    count_matrix = np.zeros(shape=(G.num_observations, G.num_nodes))
    for vec in next_node_vectors:
        for obs in range(G.num_observations):
            count_matrix[obs][vec[obs]] += 1

    updated_distribution = np.zeros(shape=(G.num_observations, G.num_nodes))
    for obs in range(G.num_observations):
        row_vector = count_matrix[obs]
        normalization_factor = sum(row_vector)
        updated_distribution[obs] = (1.0 / normalization_factor) * row_vector

    return updated_distribution
