import numpy as np
import networkx as nx

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
    return policy,n_calls,n_iter
             
        
    
        
        