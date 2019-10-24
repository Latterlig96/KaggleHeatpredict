import lightgbm as lgb 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class GBM:
    def __init__(self,
                train_gbm : bool, 
                test_predict : bool, 
                save_model : bool,
                save_history : bool,
                src_dir : str,
                plot_history : bool,
                model_dir : str,
                seed : int,
                verbose : int,
                shuffle : bool,
                name : str): 
                self.train_gbm = train_gbm 
                self.test_predict = test_predict 
                self.save_model = save_model
                self.save_history = save_history
                self.src_dir = src_dir,
                self.verbose = verbose
                self.seed = seed
                self.name = name
                self.shuffle = shuffle
                self.best_iter = [] 
                self.history = {}
                self.train_loss = [] 
                self.val_loss = [] 
                self.losses = [] 
    
    def run(self, 
            X_train, y_train,
            X_test,y_test, 
            tt_split = None,
            test_size = None,
            parameters = None,
            folds_split = None, 
            n_split = None,
            categorical_feature = None):

            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values 
            if isinstance(y_test, pd.DataFrame): 
                y_test = y_test.values
            
            if self.src_dir: 
                os.makedirs('{}_{}'.format(self.src_dir,self.name))
            
            try: 
                print("X_train_shape",X_train.shape)
                print("X_test_shape",X_test.shape)
            except ValueError: 
                print("Shape does not fit")
            
            if tt_split:
                pass
            elif folds_split: 
                kf = StratifiedKFold(n_split = n_split,random_state = self.seed)
            
            for i,train_index, val_index in enumerate(kf.split(X_train,y_train)):
                if self.train_gbm: 
                    train_X = X_train.iloc[train_index]
                    val_X = X_train.iloc[val_index]
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = y_train.iloc[train_index]
                        val_y = y_train.iloc[val_index]
                    else: 
                        train_y = y_train[train_index]
                        val_y  = y_train[val_index]
                    
                    lgb_train = lgb.Dataset(train_X,train_y,categorical_feature = categorical_feature)
                    lgb_val = lgb.Dataset(val_X,val_y,categorical_feature = categorical_feature,reference = lgb_train)

                    gbm = lgb.train(train_set=lgb_train,
                                    num_boost_round=parameters['num_boost_round'],
                                    valid_sets=lgb_val,
                                    early_stopping_rounds=parameters['early_stopping_rounds'],
                                    evals_result = self.history,
                                    verbose_eval = parameters['verbose_eval'])
                    gbm = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                    self.best_iter.append(gbm.best_iteartion)
                if self.save_model: 
                    print("Saving model")
                    gbm.save_model('{0}/fold_{1}_{2}_eval_history.txt'.format(self.src_dir,i,self.name))
                if self.save_history: 
                    print("Saving Hisotry")
                    pd.to_pickle(self.history,'{0}/fold_{1}_{2}_pickle_eval_history.pkl'.format(self.src_dir,i,self.name))
                






                    

                










            

            

            


            

            


            
            

            
        
            

