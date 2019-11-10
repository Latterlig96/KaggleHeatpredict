import lightgbm as lgb 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,f1_score


class GBM:
    def __init__(self,
                train_gbm : bool, 
                test_predict : bool, 
                save_model : bool,
                save_history : bool,
                src_dir : str,
                seed : int,
                name : str,
                categorical_features : list,
                importance : bool,
                stratify : bool,
                eval_metric,
                time_series : bool,
                prepare_submission : bool): 
                self.train_gbm = train_gbm 
                self.test_predict = test_predict 
                self.save_model = save_model
                self.save_history = save_history
                self.src_dir = src_dir,
                self.seed = seed
                self.name = name
                self.prepare_submission = prepare_submission
                self.history = {}
                self.categorical_features = categorical_features
                self.importance = importance
                self.stratify = stratify
                self.eval_metric = eval_metric
                self.time_series = time_series
    
    def fold_run(self, 
            X_train : pd.DataFrame, 
            y_train,
            X_test : pd.DataFrame,
            n_folds : int,
            parameters = None,
            categorical_feature = None):

            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values 

            if self.src_dir: 
                print("Making dir {}".format(self.src_dir))
                os.makedirs('{}'.format(self.src_dir))
            
            try: 
                print("X_train_shape",X_train.shape)
                print("X_test_shape",X_test.shape)
            except ValueError: 
                print("Shape does not fit")

            if self.stratify: 
                print("Make {} stratify folds".format(n_folds))
                kf = StratifiedKFold(n_splits=n_folds,random_state = self.seed)
            elif self.time_series:
                kf = TimeSeriesSplit(n_splits=n_folds,random_state = self.seed)
            else: 
                kf = KFold(n_splits=n_folds,random_state = self.seed)

            valid_predictions = np.zeros((X_train.shape[0],n_folds))
            test_predictions = np.zeros((X_train.shape[0],n_folds))
            print("vaild_predict",valid_predictions.shape[0])
            print("test_predictions",test_predictions.shape[0])
            
            i = 0
            for train_index, val_index in kf.split(X_train,y_train):
                print("Train LightGBM")
                if self.train_gbm: 
                    train_X = X_train.iloc[train_index]
                    val_X = X_train.iloc[val_index]
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = y_train.iloc[train_index]
                        val_y = y_train.iloc[val_index]
                    else: 
                        train_y = y_train[train_index]
                        val_y  = y_train[val_index]
                    
                    lgb_train = lgb.Dataset(train_X,train_y,categorical_feature = self.categorical_features)
                    lgb_val = lgb.Dataset(val_X,val_y,categorical_feature = self.categorical_features,reference = lgb_train)

                    gbm = lgb.train(params = parameters,
                                    train_set=lgb_train,
                                    num_boost_round=parameters['num_boost_round'],
                                    valid_sets=lgb_val,
                                    early_stopping_rounds=parameters['early_stopping_rounds'],
                                    evals_result = self.history,
                                    verbose_eval = parameters['verbose_eval'])
                    valid_predictions[val_index,i] = gbm.predict(val_X,num_iteration=gbm.best_iteration)
                    r2 = r2_score(val_y[val_index],valid_predictions[val_index,i])
                    print(f"R squared for current validation set {r2}")
                    
                if self.save_model: 
                    print("Saving model")
                    gbm.save_model('{0}/fold_{1}_{2}_eval_history.txt'.format(self.src_dir,i,self.name))
                if self.save_history: 
                    print("Saving Hisotry")
                    pd.to_pickle(self.history,'{0}/fold_{1}_{2}_pickle_eval_history.pkl'.format(self.src_dir,i,self.name))
                if self.importance: 
                    self.visualize_importance(i)
                if self.test_predict: 
                    test_predictions[:,i] = self.predict_test(X_test,i)
                if self.prepare_submission: 
                    self.output_submission(self,test_predictions,i)
                i += 1
            return valid_predictions,test_predictions

    
    def predict_test(self,X_test,index): 
        gbm = lgb.Booster(model_file ="{}/fold_{}_{}_eval_history.txt".format(self.src_dir,index,self.name))
        test_predict = gbm.predict(X_test,num_iteration = gbm.best_iteration)
        return test_predict

    def visualize_importance(self,index): 
        print("Visualize plot importance")
        if self.train_gbm:
            gbm = lgb.Booster(model_file ="{}/fold_{}_{}_eval_history.txt".format(self.src_dir,index,self.name))
            importance = gbm.feature_importance() 
            names = gbm.feature_name() 
            df_importance = pd.DataFrame() 
            df_importance['importance'] = importance 
            df_importance['names'] = names
            df_importance.sort_values('importance',ascending=False,inplace=True)
            importance_plot = df_importance.plot(x='names',y='importance',kind='barh',figsize=(10,10))
            plt.show()
    
    def output_submission(self,predictions,index,save = True): 
        print("Preparing submission")
        os.makedirs('submissions')
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['predictions'] = predictions
        if save: 
            submission.to_csv('./submissions/{}_{}_.csv'.format(self.name,index))
        return submission 








                
                

                






                    

                










            

            

            


            

            


            
            

            
        
            

