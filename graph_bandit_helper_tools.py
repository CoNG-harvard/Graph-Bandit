import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import graph_bandit
from tqdm import trange



           
        

def return_graph(graph_type='fully_connected', n_nodes=6, n_children=None):
    """
    Returns specified graph type.
    
    param graph_type: string. fully_connected, line, circle, star, or tree
    param n_nodes: Number of nodes in graph.
    param n_children: Number of children per node in the tree graph (only applicable for tree graph)
    """
    G = nx.Graph()

    if graph_type=='fully_connected':
        for i in range(n_nodes):
            for j in range(n_nodes):
                G.add_edge(i,j)
    elif graph_type=='line' or graph_type=='circle':
        for i in range(n_nodes):
            G.add_edge(i,i)
            if i<n_nodes-1:
                G.add_edge(i,i+1)
        if graph_type=='circle':
            G.add_edge(0,n_nodes-1)
    elif graph_type=='star':
        G.add_edge(0,0)
        for i in range(1,n_nodes):
            G.add_edge(i,i)
            G.add_edge(0,i)
    elif graph_type=='tree':
        assert n_children is not None
        G.add_edge(0,0)
        children = {0:0}
        for i in range(1,n_nodes):
            G.add_edge(i,i)
            available_nodes = np.sort(list(G.nodes))
            for node in available_nodes:
                if children[node] < n_children:
                    G.add_edge(node,i)
                    children[node] += 1
                    children[i] = 0
                    break
    elif graph_type=='maze':
        G=nx.Graph()
        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(2,3)
        G.add_edge(0,4)
        G.add_edge(1,5)
        G.add_edge(2,6)
        G.add_edge(3,7)
        G.add_edge(4,5)
        G.add_edge(5,6)
        G.add_edge(6,7)
        G.add_edge(4,8)
        G.add_edge(5,9)
        G.add_edge(6,10)
        G.add_edge(7,11)
        G.add_edge(8,9)
        G.add_edge(9,10)
        G.add_edge(10,11)
        G.add_edge(8,12)
        G.add_edge(9,13)
        G.add_edge(10,14)
        G.add_edge(11,15)
        G.add_edge(12,13)
        G.add_edge(13,14)
        G.add_edge(14,15)
        for i in range(16):
            G.add_edge(i,i)
    elif graph_type =='grid':
        ldim = int(np.ceil(np.sqrt(n_nodes)))
        udim = int(np.floor(np.sqrt(n_nodes)))
        G = nx.grid_2d_graph(ldim,udim)
    else:
        raise ValueError("Invalid graph type. Must be fully_connected, line, circle, star, or tree.")
    return G
        

def draw_graph(G, zero_indexed=True):
    """
    Draws graph.
    
    param G: networkx graph.
    param zero_indexed: if True, nodes are zero-indexed, else indexing starts at one.
    """
    if zero_indexed:
        labels = {n:n for n in G.nodes}
    else:
        labels = {n:n+1 for n in G.nodes}
    nx.draw(G,labels=labels)
    plt.show()
    
    

    

def testQLearning(T, n_samples, epsilon, G,\
                 means=None, mean_magnitude=1, stdev_magnitude=1,\
                 init_nodes=None,prior_uncertainty=10
                ):
    """
    param T: time horizon (per epoch)
    param n_samples: number of randomized samples
    param epsilon: exploration parameter (epsilon greedy)
    param G: networkx graph
    param mean_magnitude: magnitude of mean reward; e.g. mean_magnitude=a-> means=np.random.normal(size=(n_samples,6))*a
    param stdev_magnitude: magnitude of mean reward; e.g. stdev_magnitude=a-> means=np.ones((n_samples,6))*10*a
    param initNodes: Dictionary of initial nodes for each algorithm
    """
    nNodes = len(G.nodes)
    regrets = np.zeros((n_samples, T)) 
    rewards = np.zeros((n_samples, T)) 

    if means is None:
        means = np.random.uniform(low=0.5, high=5.5,size=(n_samples,nNodes))
        
    if init_nodes is None:
        init_nodes = {alg: None for alg in algorithms}

    for i in trange(n_samples):      
        # Q-learning
        QL = graph_bandit.GraphBandit(means[i], G, belief_update='Bayesian_full_update',\
                                                    bayesian_params=[0, prior_uncertainty*mean_magnitude**2,\
                                                                    stdev_magnitude**2])
        QL.train_QL_agent(H=T, epsilon=epsilon, init_node=init_nodes)
        
        regrets[i,:] = QL.expectedRegret()[-T:]
        rewards[i,:] = QL.collected_rewards

            
    return regrets, rewards

