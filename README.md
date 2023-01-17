# Graph-Bandit

This repository contains the implementation of  graph bandit algorithms and the corresponding numerical experiments. The code are written in Python. 

The link to our paper: [Multi-armed Bandit Learning on a Graph](https://arxiv.org/abs/2209.09419)

**Python packages** required for running the experiments: 

* For running the Python notebooks: jupyter notebook or jupyter lab. 
* Parallel Programming: joblib.
* Graph-related utilities: networkx.
* Plotting utilities: matplotlib, seaborn.
* For showing the progress bar in non-parallel experiments: tqdm.
* For saving and loading experiment data: pickle.

**Quick Start:** Directly run the **'Robotic Application.ipynb'**  notebook to see the network used in our robotic application and the regret for our proposed algorithm.

## Contents of the Python files

**graph_bandit.py**: the class definition of graph bandit environment. Implements the step() method.

**agents.py**: contains the agent implementing our propose algorithm(under the name GUCB_agent), as well as the local Thompson Sampling, local UCB, QL $\epsilon$-greedy, QL-UCB-H, and UCRL2 agents.

**core.py**: contains a function that visits all nodes at least once(used in initialization), and the train_agent() function.

**planning.py**: Contains two offline planning algorithm. The shortest path algorithm for off-line planning in G-UCB and the value iteration planning in UCRL2.

**utils.py**: contains a graph generator, a graph drawing utility, and a wrapper for training a Q-learning agent.

## Contents of the Python notebooks

**Main.ipynb**: contains the experiments comparing our proposed algorithm with various benchmarks on various graphs.

**Main Plotting.ipynb**: plotting utilities for the results obtained from **Main.ipynb**

**Sensitivity Analysis.ipynb:** experiments showing how the performance of our algorithm depends on graph parameters $|S|,D,$ and $\Delta$. 

**Robotic Application.ipynb**: contains the synthetic robotic application of providing Internet access to rural/suburban areas using an UAV.

**Direct SP.ipynb**: comparing the learning regret between following the path with the shortest weighted distance and with the shortest length to the source.

**Direct SP Plotting.ipynb**: the plotting notebook for the above.

**Our Algorithm vs UCRL2.ipynb**: detailed experiments investigating the effect of UCB and doubling scheme on learning performance.

**Our Algorithm vs UCRL2-plotting.ipynb**: the plotting notebook for the above.

**Simulation Efficiency**: experiments comparing the simulation efficiency of G-UCB and UCRL2.
