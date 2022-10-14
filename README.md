# Graph-Bandit

This repository contains the implementation of  graph bandit algorithms and the corresponding numerical experiments. The code are written in Python. 

**Python packages** required for running the experiments: 

* For running the Python notebooks: jupyter notebook or jupyter lab. 
* Graph-related utilities: networkx.
* Plotting utilities: matplotlib, seaborn.
* For showing the progress bar during the experiments: tqdm.
* For saving and loading experiment data: pickle.

**Quick Start:** Directly run the **'Robotic Application.ipynb'**  notebook to see the network used in our robotic application and the regret for our proposed algorithm.

## Contents of the Python files

**graph_bandit.py**: the class definition of graph bandit environment, which includes a class method that trains a Q-learning agent.

**agents.py**: contains the agent implementing our propose algorithm(under the name doubling_agent), as well as the local Thompson Sampling and UCB agents.

**core.py**: contains a function that visits all nodes at least once(used in initialization), and the train_agent() function.

**known_rewards_helper_functions.py**: the shortest path algorithm for off-line planning.

**graph_bandit_helper_tools.py**: contains a graph generator, a graph drawing utility, and a wrapper for training a Q-learning agent.

## Contents of the Python notebooks

**Main.ipynb**: contains the experiments comparing our proposed algorithm with various benchmarks on various graphs.

**Main Plotting.ipynb**: plotting utilities for the results obtained from **Main.ipynb**

**Sensitivity Analysis.ipynb:** experiments showing how the performance of our algorithm depends on graph parameters $|S|,D,$ and $\Delta$. 

**Robotic Application.ipynb**: contains the synthetic robotic application of providing Internet access to rural/suburban areas using an UAV.
