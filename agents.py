import numpy as np

from known_rewards_helper_functions import offline_SP_planning,EVI_known_transition_planning


def get_ucb(gb,nodes=None):
    
    if nodes is None:
        nodes = gb.nodes
    ave_reward = [np.mean(gb.nodes[i]['r_hist']) for i in nodes] 
    nm = [gb.nodes[i]['n_visits'] for i in nodes]
    
    tm = len(gb.visitedStates)
    ucb = ave_reward + np.sqrt(2*np.log(tm)/nm)
    
    return ucb

def doubling_agent(env):
    ucb = get_ucb(env)
    # Compute optimal policy.
    policy,_,_ = offline_SP_planning(env.G,ucb)

    # Travel to the node with the highest UCB
    while ucb[env.state] < np.max(ucb):
        next_s = policy[env.state]
        env.step(next_s)

    target_count = 0+env.nodes[env.state]['n_visits']
    # Keep sampling the best UCB node until its number of samples doubles
    for _ in range(target_count):
        env.step(env.state)

def local_ucb_agent(env):
    neighbors = [_ for _ in env.G[env.state]]

    neighbor_ucb = get_ucb(env,neighbors)

    best_nb = neighbors[np.argmax(neighbor_ucb)]

    env.step(best_nb)

def get_ts_parameters(env,neighbors,
                    var_0 = 1.0,
                    mu_0 = 0.0,
                    var = 1.0):
    xsum = np.array([np.sum(env.nodes[i]['r_hist']) for i in neighbors])
    n = np.array([env.nodes[i]['n_visits'] for i in neighbors])

    var_1 = 1/(var_0 + n/var) 
    mu_1 = var_1 * (mu_0/var_0 + xsum/var)

    return mu_1,var_1

def local_ts_agent(env,
                    var_0 = 1,
                    mu_0 = 0.0,
                    var = 1):

    neighbors = [_ for _ in env.G[env.state]]
    

    # Bayesian estimation of mu and var estimation with Gaussian Prior

    mu_1,var_1 = get_ts_parameters(env,neighbors,var_0,mu_0,var)

    # Posterior sampling
    mu_sample = np.random.normal(mu_1,np.sqrt(var_1))

    
    # Take a step in the environment
    best_nb = neighbors[np.argmax(mu_sample)]
    env.step(best_nb)

def UCRL2_ucb(env,nodes=None,delta = 0.01):
    '''
        delta: a number in (0, 1], characterizing the probability that the true mean falls into the confidence interval [LCB, UCB]
    '''
    if nodes is None:
        nodes = env.nodes
        
    ave_reward = np.array([np.mean(env.nodes[i]['r_hist']) for i in nodes] )
    nm = np.array([env.nodes[i]['n_visits'] for i in nodes])
    
    tm = len(env.visitedStates)
    
    S = env.G.number_of_nodes()
    A = 2 * env.G.number_of_edges() # Since our graph is undirected, A = 2|E|

    ucb = ave_reward + np.sqrt(7*np.log(2*S*A*tm/delta)/(2*nm))
    return ucb

def UCRL2_agent(env,delta = 0.01):
    
    ucb = UCRL2_ucb(env,delta = 0.01)
    
    tm = len(env.visitedStates)
    epsilon = 1/np.sqrt(tm) # The stopping condition of value iteration as specified in Jacksch(2008).
    
    # Compute the policy.
    policy,_ = EVI_known_transition_planning(env.G,ucb,epsilon = epsilon)
    
    prev_visits = {s:env.nodes[s]['n_visits'] for s in env.G}
    visits_this_episode = {s:0 for s in env.G}
    
    # Keep executing the policy until the doubling terminal condition specified in Jacksh(2008) is meet.
    while visits_this_episode[env.state]< np.max([1,prev_visits[env.state]]):
        visits_this_episode[env.state]+=1
        next_s = policy[env.state]
        env.step(next_s)


# def PSRL_agent(env,var_0 = 1,
#                     mu_0 = 0.0,
#                     var = 1):


#     # Sample distributions.
#     mu_1,var_1 = get_ts_parameters(env,env.nodes, var_0,mu_0,var)
#     sampled_means = np.random.normal(mu_1,np.sqrt(var_1))

#     # Compute the epsilon-optimal policy based on the sampled means
#     tm = len(env.visitedStates)
#     epsilon = 1/np.max([1,np.sqrt(tm)])
#     policy,_ = EVI_known_transition_planning(env.G,sampled_means,epsilon = epsilon)

#     prev_visits = {s:env.nodes[s]['n_visits'] for s in env.G}
#     visits_this_episode = {s:0 for s in env.G}

#     # Keep executing the policy until the doubling terminal condition specified in Agrawal(2017), which is the same as in Jacksh(2008), is meet.
#     while visits_this_episode[env.state]< np.max([1,prev_visits[env.state]]):
#     # for i in range(10):
#         visits_this_episode[env.state]+=1
#         next_s = policy[env.state]
#         env.step(next_s)
#             