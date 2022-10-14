import numpy as np
import random

class GraphBandit:
    """
    The graph bandit implemented as a gym environment
    """
    
    def __init__(self, mean, G, belief_update=None, bayesian_params=[0, 1, 1], init_state=0):
        """
        param mean: Vector of means for bandits (one mean for each node in graph G).
        param G: networkx Graph.
        param belief_update: How to update beliefs (when applicable (in Q-learning algorithms)). None=use sample rewards in QL-algorithms, average=use average, Bayesian=use Bayesian update. 
        param bayesian_params: [mu0, sigma0^2, sigma^2]; a list of the belief parameters. mu0 is the initial mean estimate, sigma0 is the initial standard deviation in the initial belief, and sigma is the known variance of the reward distributions. All parameters are the same for all nodes.
        param init_state: the initial state (node) of the agent.
        param uncertainty: Constant in UCB estimate; e.g. mu_UCB = mu_est + uncertainty/sqrt(t)
        """
        # Initialization parameters
        self.mean = mean.copy()
        self.G = G.copy()
        self.belief_update = belief_update
        self.bayesian_params = bayesian_params.copy()
        self.state = init_state
        self.nodes = self.G.nodes
        self.visited_expected_rewards = []
        
        # Number of nodes
        self.num_nodes = self.mean.shape[0]
        
        # Rewards collected during training
        self.collected_rewards = []
               
        
        # Initialize Q-table
        self.q_table = np.zeros((self.num_nodes,self.num_nodes))      
                 
        # Eliminate actions that are 'illegal' (agent can only go via edges)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if (i,j) not in G.edges:
                    self.q_table[i,j] = -np.inf
                    
        # Store history of rewards and number of vistits for each node, as well as number of (state -> action) encounters
        for n in self.nodes:
            self.nodes[n]['r_hist']=[] 
            self.nodes[n]['n_visits']=0 
          
        # List of all visited states
        self.visitedStates = [] 
        
    def step(self, action):
        # Take a step in the graph with 
        # Returns: observation, QL_reward, done
        assert action <= self.num_nodes
        if (action, self.state) in self.G.edges:
            reward = np.random.uniform(low = self.mean[action]-0.5, high=self.mean[action]+0.5)

            
            state_old = self.state
            self.state = action
            
            
                        
            self.collected_rewards.append(reward)
            self.nodes[action]['r_hist'].append(reward)
            
            self.nodes[self.state]['n_visits'] += 1
            self.visited_expected_rewards.append(self.mean[self.state])
                   
                
        else:
            print(self.state)
            print(action)
            reward = -100
            done = True
            print("Something is wrong. Illegal action")
            
        self.visitedStates.append(self.state)
        return self.state, reward, False
        
    def reset(self, node):
        """
        Resets agent at node.
        If node=None, random node is picked.
        """
        if node is None:
            self.state = np.random.randint(self.num_nodes)
        else:
            self.state = node
        return self.state
    
    def explore(self, node):
        """
        Returns least explored available action.
        """
        
        neighbors = list(self.G[node])
        N_explorations = [self.nodes[nb]['n_visits'] for nb in neighbors]
        return neighbors[np.argmin(N_explorations)]


    def train_QL_agent(self, H=100, init_node= None, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Training the agent
        
        param H: total number of steps
        param init_node: initial node 
        param alpha: Q-learning parameter (if applicable)
        param gamma: RL discount factor (if applicable)
        param epsilon: exploration parameter (if applicable)
        """
        epsilon0 = epsilon
       
        self.reset(init_node)

        for h in range(0, H):
            
            state = self.state
            
            # Reduce epsilon

            epsilon = 0.7*(3*self.num_nodes + 1) / (3*self.num_nodes + h)

            # 'Greedy efficient' exploration
            if random.uniform(0, 1) < epsilon:
                action = self.explore(state)
                assert((self.state,action) in self.G.edges)
            else:
                action = np.argmax(self.q_table[state]) # Exploit learned values
                assert((self.state,action) in self.G.edges)

            # Next state, reward, and store reward
            if action is not None:

                next_state, QL_reward, done = self.step(action)

                self.update_q_table(state, action, gamma)
    
    def update_q_table(self, state, action, gamma):
        next_state = action
        old_value = self.q_table[state, action]

        assert(np.isfinite(old_value))
        next_max = np.max(self.q_table[next_state])

        for node in self.G[action]:
            old_value = self.q_table[node, action]
            if self.belief_update!='Bayesian_full_update' and self.belief_update!='average_full_update':
                new_value = (1 - alpha) * old_value + alpha * (QL_reward + gamma * next_max)
            else: 
                new_value = np.mean(self.nodes[action]['r_hist']) + gamma * next_max
            self.q_table[node, action] = new_value
    
    def expectedRegret(self):
        # Returns vector of expected regret
        mu_max = np.max(self.mean)
        mu_visited = self.mean[self.visitedStates]
        
        return mu_max - np.array(self.visited_expected_rewards)


