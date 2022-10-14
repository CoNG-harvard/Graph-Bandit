import numpy as np
class NormalBayesianEstimator:
    #def __init__(self,mu_lim=(-1,1),m=50,epsilon=0.01):
    def __init__(self,mu_0, var_0, var):
        """
            mu_lim = The upper and lower bounds of the unknown mean parameter mu to be estimated.
            m,epsilon: exploration strength(var_1) drops below epsilon after m updates.
        """
        
#         self.mu_0 = np.mean(mu_lim)
#         self.var_0= (mu_lim[0]-mu_lim[1])**2/4
#         if 1-epsilon*self.var_0 == 0:
#             self.var = m*epsilon/(1-epsilon*self.var_0+np.abs(np.random.normal(0,0.001)))
#         else:
#             self.var = m*epsilon/(1-epsilon*self.var_0)

#         self.mu_0 = mu_0
#         self.var_0 = var_0
#         self.var = var
#         self.xsum = 0
#         self.n = 0
        
#         if self.var_0 == 0:
#             self.var_0 = 1e-5
#         if self.var == 0:
#             self.var = 1e-5
        
#         self.mu_1 = self.mu_0 # Current mean
#         self.var_1 = self.var_0 # Currenc variance
        self.S = 0
        self.F = 0
        
    def get_param(self):
        return self.S, self.F
#         return self.mu_1,self.var_1
    
    def update(self,x):
        r = (x-0.5)/5
        result = np.random.binomial(1, r)
        if result == 0:
            self.F += 1
        elif result == 1:
            self.S += 1
#         self.xsum += x
#         self.n +=1
        

#         self.var_1 = 1/(1/self.var_0 + self.n/self.var) 
#         self.mu_1 = self.var_1 * (self.mu_0/self.var_0 + self.xsum/self.var)
        
class AverageEstimator:
    def __init__(self):
        """
            mu_lim = The upper and lower bounds of the unknown mean parameter mu to be estimated.
            m,epsilon: exploration strength(var_1) drops below epsilon after m updates.
        """
        self.mu = None
        self.xsum = 0
        self.n = 0
        
    def get_param(self):
        return self.mu, None
    
    def update(self,x):
        self.xsum += x
        self.n +=1
        self.mu = self.xsum / self.n

        

        