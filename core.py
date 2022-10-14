import networkx as nx
import numpy as np
from tqdm import trange

import graph_bandit



def visit_all_nodes(gb):
    n_nodes  = len(gb.G)
    while True:
        unvisited = [i for i in range(n_nodes) if gb.nodes[i]['n_visits']==0]
        
        if len(unvisited)==0:
            break

        dest = unvisited[0]
        
        next_path = nx.shortest_path(gb.G,gb.state,dest)
        if len(next_path)==1:
            gb.step(dest)
        else:
            for s in next_path[1:]:
                gb.step(s)



def train_agent(n_samples,T,G,means, init_node,execute_agent):
    regrets = np.zeros((n_samples,T))
    for i in trange(n_samples):

        env = graph_bandit.GraphBandit(means[i],  G)

        ## Visit all nodes
        visit_all_nodes(env)

        H0 = len(env.visitedStates)

        # Start learning

        env.state = init_node

        while len(env.visitedStates)-H0<T:
            execute_agent(env)
            
        # print(env.visitedStates.shape,regrets.shape)
            
        regrets[i,:]= env.expectedRegret()[:T]
        
    return regrets