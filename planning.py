import numpy as np
import networkx as nx

import warnings

def offline_SP_planning(G,means):
    # G = G_cyc.copy()
    # G.remove_edges_from(nx.selfloop_edges(G_cyc))
    # The above was removed because it causes unnecessary edge removal operation in each planning call.

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
            best_nb_d = distance[s]
            for w in G.neighbors(s):
                if w == s:
                    continue
                if best_nb_d>distance[w]+c[w]: 
                    best_nb_d=distance[w]+c[w]
                    policy[s] = w
                    updated = True

            distance[s] = best_nb_d
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
    policy = {}

    iter_count = 0
    
    max_iter = np.min([1e4,1/epsilon]) 

    #  max_iter: Hard upper-bound of the number of iterations, to ensure the algorithm does not fall into infinite loop

    while iter_count<=max_iter:
        iter_count+=1 

        u_old = np.array(u)

        for s in G:
            best_nb_u = 0
            for w in G[s]:
                if u_old[w]>best_nb_u:
                    best_nb_u = u_old[w]
                    policy[s] = w

            u[s] = means[s]+best_nb_u
            # print(iter_count, u)

        if np.max(u-u_old) - np.min(u-u_old)<epsilon:
            # print('Gap',np.max(u-u_old) - np.min(u-u_old))
            break

    # print('iter_count',iter_count)

    # The following operation is slow. Don't use it!
    # policy = {s:list(G[s])[np.argmax(u[G[s]])] for s in G}
    return policy, u
    
        
        