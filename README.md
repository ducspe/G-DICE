# This is a re-implementation of the G-DICE algorithm based on the paper: "The Cross-Entropy Method for Policy Search in Decentralized POMDPs" by FA Oliehoek et al.

# The algorithm is run on the dectiger problem from the "problems/" folder

# Compile using: 
# sudo python3 setup.py develop
# Then execute : 
# python3 gdice.py

# The policy graphs are viewable with the installation of "xdot". Once the tool is installed, execute "xdot gdice_best_policy_agent{x}.dot" on the ".dot" files available in the "src/" folder.