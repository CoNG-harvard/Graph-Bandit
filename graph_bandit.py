import numpy as np
import random

class GraphBandit:
    """
    The graph bandit implemented as a gym environment
    """
    
    def __init__(self, mean, G, init_state=0):
        """
        param mean: Vector of means for bandits (one mean for each node in graph G).
        param G: networkx Graph.
        param init_state: the initial state (node) of the agent.      
        """
        # Initialization parameters
        self.mean = mean.copy()
        self.G = G.copy()
        
        self.state = init_state
        self.nodes = self.G.nodes
        self.edges = self.G.edges
        self.visited_expected_rewards = []
        
        # Number of nodes
        self.num_nodes = self.mean.shape[0]
               
                    
        # Store history of rewards and number of vistits for each node, as well as number of (state -> action) encounters
        for n in self.nodes:
            self.nodes[n]['r_sum'] = 0
            self.nodes[n]['n_visits'] = 0 
        for e in self.edges:
            self.edges[e]['n_visits'] = 0 

        # List of all visited states
        self.visitedStates = [] 
        
    def step(self, action):
        # Take a step in the graph with 
        # Returns: observation, reward, done
        assert action <= self.num_nodes
        if (action, self.state) in self.G.edges:
            reward = np.random.uniform(low = self.mean[action]-0.5, high=self.mean[action]+0.5)
            # reward = np.random.binomial(1, self.mean[action])

            state_old = self.state
            self.state = action
                        
            self.nodes[action]['r_sum'] += reward 
            
            
            self.nodes[self.state]['n_visits'] += 1
            self.edges[(state_old,action)]['n_visits'] += 1
            
            self.visited_expected_rewards.append(self.mean[self.state])

        else:
            print(self.state)
            print(action)
            reward = -100
            done = True
            print("Something is wrong. Illegal action")
            
        self.visitedStates.append(self.state)
        return self.state, reward, False
   
  
    def expectedRegret(self):
        # Returns vector of expected regret
        mu_max = np.max(self.mean)
        mu_visited = self.mean[self.visitedStates]
        
        return mu_max - np.array(self.visited_expected_rewards)


