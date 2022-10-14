import numpy as np

from known_rewards_helper_functions import offline_SP_planning


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

def local_ts_agent(env,
                    var_0 = 0.5,
                    mu_0 = 5,
                    var = 0.5):

    neighbors = [_ for _ in env.G[env.state]]
    

    # Bayesian estimation of mu and var estimation with Gaussian Prior

    xsum = np.array([np.sum(env.nodes[i]['r_hist']) for i in neighbors])
    n = np.array([env.nodes[i]['n_visits'] for i in neighbors])

    var_1 = 1/(var_0 + n/var) 
    mu_1 = var_1 * (mu_0/var_0 + xsum/var)

    # Posterior sampling
    mu_sample = np.random.normal(mu_1,np.sqrt(var_1))

    
    # Take a step in the environment
    best_nb = neighbors[np.argmax(mu_sample)]
    env.step(best_nb)