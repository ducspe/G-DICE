#include <pybind11/pybind11.h>
#include "MADPWrapper.h"


PYBIND11_MODULE(madp_python3_wrapper, m) {
    
    pybind11::class_<pgi::madpwrapper::MADPDecPOMDPDiscrete>(m, "MADPDecPOMDPDiscrete")
        .def(pybind11::init<const std::string&>() )
        .def("num_agents", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_agents)
        .def("num_states", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_states)
        .def("num_joint_actions", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_joint_actions)
        .def("num_joint_observations", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_joint_observations)
        .def("num_actions", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_actions)
        .def("num_observations", & pgi::madpwrapper::MADPDecPOMDPDiscrete::num_observations)
        .def("state_name", & pgi::madpwrapper::MADPDecPOMDPDiscrete::state_name)
        .def("action_name", & pgi::madpwrapper::MADPDecPOMDPDiscrete::action_name)
        .def("observation_name", & pgi::madpwrapper::MADPDecPOMDPDiscrete::observation_name)
        .def("initial_belief_at", & pgi::madpwrapper::MADPDecPOMDPDiscrete::initial_belief_at)
        .def("reward", & pgi::madpwrapper::MADPDecPOMDPDiscrete::reward)
        .def("observation_probability", & pgi::madpwrapper::MADPDecPOMDPDiscrete::observation_probability)
        .def("transition_probability", & pgi::madpwrapper::MADPDecPOMDPDiscrete::transition_probability)
        ;

}