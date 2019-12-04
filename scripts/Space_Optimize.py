import skopt
from collections import defaultdict
import json 

class Optimization: 
    def __init__(self,
                raw_model, 
                dimensions,
                search_params, 
                fit_params,
                seed):
        self.raw_model = raw_model
        self.dimensions = dimensions
        self.search_params = search_params 
        self.fit_params = fit_params
        self.seed = seed

    def BayesSearch(self,X_train,y_train,X_test,y_test,
                    n_jobs,
                    cv,
                    optimizer_kwargs,
                    jsonize): 
        evaluation = skopt.BayesSearchCV(raw_model,search_spaces=self.search_params, 
                                        optimizer_kwargs=optimizer_kwargs,
                                        fit_params=self.fit_params,cv=cv,
                                        random_state = self.seed)
        evaluation.fit(X_train,y=y_train)
        model_parameters = defaultdict(list) 
        print(f"Best_params:{evaluation.best_params_}")
        print(f"Best_score:{evaluation.best_score_}")
        print(f"cv_results:{evaluation.cv_results_}")
        model_parameters['best_params'].append(evaluation.best_params_)
        model_parameters['best_score'].append(evaluation.best_score_)
        model_parameters['cv_results'].append(evaluation.cv_results_)
        if 'base_estimator' in optimizer_kwargs.keys():
            model_parameters['best_estimator'].append(evaluation.best_estimator_)

        if jsonize: 
            with open(f'{self.raw_model.__name__} parameters','w+') as model:
                json.dump(model_parameters,model)



