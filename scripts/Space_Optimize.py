from bayes_opt import BayesianOptimization
from collections import defaultdict
import json 
class Optimization: 
    def __init__(self,
                raw_model,  
                fit_params,
                seed,
                init_points, 
                opt_round,
                save_output):
        self.raw_model = raw_model
        self.fit_params = fit_params 
        self.seed = seed
        self.init_points = init_points
        self.opt_round = opt_round
        self.save_output = save_output
    
    def Bayesian_opt(self):
        Bayes = BayesianOptimization(self.raw_model, 
                                    pbounds=self.fit_params,
                                    random_state=self.seed,
                                    verbose=2)
        Bayes.maximize(init_points=self.init_points,n_iter=self.opt_round)
        if self.save_output: 
            Bayes.points_to_csv('bayes_params.csv')
        
        return Bayes.res['max']['max_params']
    
