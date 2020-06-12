from madp_python3_wrapper import MADPDecPOMDPDiscrete as madp
import numpy as np
import itertools
from collections import OrderedDict


class Utils:

    def __init__(self, problem_path):
        self.decpomdp = madp(problem_path)
        self.local_to_joint_action_dict = OrderedDict()
        
        num_agents = self.decpomdp.num_agents()
        agent_num_actions = []
        agent_num_observations = []
        for agent in range(num_agents):
            agent_num_actions.append(self.decpomdp.num_actions(agent))
            agent_num_observations.append(self.decpomdp.num_observations(agent))

        i = len(agent_num_actions) - 1

        list1 = range(agent_num_actions[i])
        while i != 0:
            list2 = range(agent_num_actions[i-1])
            temp = list(itertools.product(list2, list1))
            list1 = temp
            i -= 1

        count_joint_actions = 0
        for item in list1:
            key = str(item)
            key = key.replace("(", "").replace(")", "")
            self.local_to_joint_action_dict[key] = count_joint_actions
            count_joint_actions += 1

        print("Built action index: ", self.local_to_joint_action_dict)

        self.joint_to_local_action_dict = OrderedDict()

        for key, value in self.local_to_joint_action_dict.items():
            self.joint_to_local_action_dict[value] = key

        print("Inverse action index: ", self.joint_to_local_action_dict)

        ################################################################################################################
        # Build observation mappings:
        self.local_to_joint_observation_dict = OrderedDict()
        i = len(agent_num_observations) - 1

        list1 = range(agent_num_observations[i])
        while i != 0:
            list2 = range(agent_num_observations[i-1])
            temp = list(itertools.product(list2, list1))
            list1 = temp
            i -= 1

        count_joint_observations = 0
        for item in list1:
            key = str(item)
            key = key.replace("(", "").replace(")", "")
            self.local_to_joint_observation_dict[key] = count_joint_observations
            count_joint_observations += 1

        print("Built observation index: ", self.local_to_joint_observation_dict)

        self.joint_to_local_observation_dict = OrderedDict()

        for key, value in self.local_to_joint_observation_dict.items():
            self.joint_to_local_observation_dict[value] = key

        print("Inverse observation index: ", self.joint_to_local_observation_dict)

    def get_joint_action_index(self, joint_action_list):
        key = ""
        for action in joint_action_list:
            key += str(action) + ", "

        key = key.strip(", ")
        joint_action_index = self.local_to_joint_action_dict[key]
        return joint_action_index

    def get_joint_observation_index(self, joint_observation_list):
        key = ""
        for observation in joint_observation_list:
            key += str(observation) + ", "

        key = key.strip(", ")
        joint_observation_index = self.local_to_joint_observation_dict[key]
        return joint_observation_index

    def get_individual_observation_index(self, joint_observation_index, agent_index):
        observations_list = self.joint_to_local_observation_dict[joint_observation_index].split(",")
        return int(observations_list[agent_index])

    def get_individual_observations_list(self, joint_observation_index):
        observations_list = [int(item) for item in self.joint_to_local_observation_dict[joint_observation_index].split(",")]
        return observations_list

    def make_transition_matrix(self):
        state_transition_matrix = np.zeros(shape=(self.decpomdp.num_states(), self.decpomdp.num_states(), self.decpomdp.num_joint_actions()))
        for next_state in range(self.decpomdp.num_states()):
            for old_state in range(self.decpomdp.num_states()):
                for joint_action in range(self.decpomdp.num_joint_actions()):
                    state_transition_matrix[next_state][old_state][joint_action] = self.decpomdp.transition_probability(next_state, old_state, joint_action)
                
        return state_transition_matrix

    def make_observation_matrix(self):
        observation_matrix = np.zeros(shape=(self.decpomdp.num_joint_observations(), self.decpomdp.num_states(), self.decpomdp.num_joint_actions()))
        for joint_observation in range(self.decpomdp.num_joint_observations()):   
            for state in range(self.decpomdp.num_states()):
                for joint_action in range(self.decpomdp.num_joint_actions()):
                    observation_matrix[joint_observation][state][joint_action] = self.decpomdp.observation_probability(joint_observation, state, joint_action) 

        return observation_matrix

    def make_reward_matrix(self):
        reward_matrix = np.zeros(shape=(self.decpomdp.num_states(), self.decpomdp.num_joint_actions()))
        for state in range(self.decpomdp.num_states()):
            for joint_action in range(self.decpomdp.num_joint_actions()):
                reward_matrix[state][joint_action] = self.decpomdp.reward(state, joint_action) 

        return reward_matrix

    def make_initial_belief(self):
        belief = np.zeros(self.decpomdp.num_states())
        for state in range(self.decpomdp.num_states()):
            belief[state] = self.decpomdp.initial_belief_at(state)
        return belief

