import numpy as np
import networkx as nx

import warnings

def offline_SP_planning(G_cyc,means):
    G = G_cyc.copy()
    G.remove_edges_from(nx.selfloop_edges(G_cyc))

    mu_star = np.max(means)
    s_star = np.argmax(means)

    c = mu_star - means

    n_nodes = G.number_of_nodes()

    distance = np.ones(n_nodes)*np.inf

    distance[s_star] = 0

    policy = {s_star:s_star}

    # Value iteration for acyclic all-to-all weighted shortest path.
    n_calls = 0
    n_iter = 0
    for _ in range(n_nodes):
        n_iter+=1
        updated = False
        
        # Bellman-Ford
        for s in G:
            for w in G.neighbors(s):
                n_calls +=1
                if distance[s]>distance[w]+c[w]: 
                    distance[s]=distance[w]+c[w]
                    policy[s] = w
                    updated = True
                   
        # Terminate early if no update is made.
        if not updated:
            break
    # print('n_iter',n_iter)
    return policy,n_calls,n_iter
             
def EVI_known_transition_planning(G,means,epsilon = 0.0001):
    '''
    epsilon: The stopping condition.
    '''
    assert(epsilon>0)
    u = np.zeros(G.number_of_nodes())

    iter_count = 0
    
    max_iter = np.min([1e4,1/epsilon]) 

    #  max_iter: Hard upper-bound of the number of iterations, to ensure the algorithm does not fall into infinite loop

    while iter_count<=max_iter:
        iter_count+=1 

        u_old = np.array(u)

        for s in G:
            u[s] = means[s]+np.max(u_old[G[s]])
        # print(iter_count, u)

        if np.max(u-u_old) - np.min(u-u_old)<epsilon:
            # print('Gap',np.max(u-u_old) - np.min(u-u_old))
            break

    # print('iter_count',iter_count)

    policy = {s:list(G[s])[np.argmax(u[G[s]])] for s in G}
    if iter_count> max_iter:
        warnings.warn('Value iteration terminated before reaching the stopping condition, the resulting policy may be suboptimal.')
    return policy, u
    
        
        