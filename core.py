import networkx as nx
import numpy as np
from tqdm import trange

import graph_bandit
import inspect
from functools import partial


def visit_all_nodes(env):
    n_nodes  = len(env.G)
    while True:
        unvisited = [i for i in range(n_nodes) if env.nodes[i]['n_visits']==0]
        
        if len(unvisited)==0:
            break

        dest = unvisited[0]
        
        next_path = nx.shortest_path(env.G,env.state,dest)
        if len(next_path)==1:
            env.step(dest)
        else:
            for s in next_path[1:]:
                env.step(s)



# def train_agent(n_samples,T,G,means, init_node,agent):
#     regrets = np.zeros((n_samples,T))
#     for i in trange(n_samples):

#         env = graph_bandit.GraphBandit(means[i],  G)
#         if inspect.isfunction(agent) or type(agent)==partial:
#             execute_agent = agent
#         elif inspect.isclass(agent):
#             execute_agent = agent()



#         ## Visit all nodes
#         visit_all_nodes(env)

#         H0 = len(env.visitedStates)

#         # Start learning

#         env.state = init_node

#         while len(env.visitedStates)-H0<T:
#             execute_agent(env)
            
#         # print(env.visitedStates.shape,regrets.shape)
            
#         # regrets[i,:]= env.expectedRegret()[:T]
        
#         regrets[i,:]= env.expectedRegret()[-T:]
        
        
#     return regrets


from joblib import Parallel, delayed,cpu_count
def train_agent(n_samples,T,G,means, init_node,agent,parallelized=False):
    def main_loop(i):

        env = graph_bandit.GraphBandit(means[i],  G)

        if inspect.isfunction(agent) or type(agent)==partial:
            execute_agent = agent
        elif inspect.isclass(agent):
            execute_agent = agent()

        ## Visit all nodes
        visit_all_nodes(env)

        H0 = len(env.visitedStates)

        # Start learning

        env.state = init_node

        while len(env.visitedStates)-H0<T:
            execute_agent(env)
            
        # print(env.visitedStates.shape,regrets.shape)
            
        # regrets[i,:]= env.expectedRegret()[:T]
        
        return env.expectedRegret()[-T:]
    
    
    if parallelized:
        regrets = Parallel(n_jobs = cpu_count())(delayed(main_loop)(i) for i in range(n_samples))
    else:
        regrets = np.zeros((n_samples,T))
        for i in trange(n_samples):
            regrets[i,:] = main_loop(i)

        
    return np.array(regrets)