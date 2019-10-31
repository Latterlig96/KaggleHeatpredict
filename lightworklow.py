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
                plot_history : bool,
                model_dir : str,
                seed : int,
                name : str,
                categorical_features : list,
                importance : bool,
                stratify : bool,
                eval_metric,
                target: int,
                time_series : bool): 
                self.train_gbm = train_gbm 
                self.test_predict = test_predict 
                self.save_model = save_model
                self.save_history = save_history
                self.src_dir = src_dir,
                self.seed = seed
                self.name = name
                self.best_iter = [] 
                self.history = {}
                self.losses = []
                self.categorical_features = categorical_features
                self.importance = importance
                self.stratify = stratify
                self.eval_metric = eval_metric
                self.target = target
                self.time_series = time_series
    
    def fold_run(self, 
            X_train : pd.DataFrame, 
            y_train,
            X_test : pd.DataFrame,
            test_size : int,
            n_split : int,
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
                print("Make {} stratify folds".format(n_split))
                kf = StratifiedKFold(n_split = n_split,random_state = self.seed)
            elif self.time_series:
                kf = TimeSeriesSplit(n_split = n_split,random_state = self.seed)
            else: 
                kf = KFold(n_split = n_split,random_state = self.seed)

            valid_predictions = np.zeros((X_train.shape[0],len(target)))
            test_predictions = np.zeros((X_train.shape[0],len(target)))
            print("vaild_predict",valid_predictions.shape[0])
            print("test_predictions",test_predictions.shape[0])

            for i,train_index, val_index in enumerate(kf.split(X_train,y_train)):
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

                    gbm = lgb.train(train_set=lgb_train,
                                    num_boost_round=parameters['num_boost_round'],
                                    valid_sets=lgb_val,
                                    early_stopping_rounds=parameters['early_stopping_rounds'],
                                    evals_result = self.history,
                                    verbose_eval = parameters['verbose_eval'],
                                    feval = self.eval_metric)
                    valid_predictions[val_index,i] = gbm.predict(val_X,num_iteration=gbm.best_iteration)
                    self.best_iter.append(gbm.best_iteartion)
                    r2 = r2_score(y_train,valid_predictions[val_index,i])
                    print("R Squared for best validation iteration:{}".format(r2))
                if self.save_model: 
                    print("Saving model")
                    gbm.save_model('{0}/fold_{1}_{2}_eval_history.txt'.format(self.src_dir,i,self.name))
                if self.save_history: 
                    print("Saving Hisotry")
                    pd.to_pickle(self.history,'{0}/fold_{1}_{2}_pickle_eval_history.pkl'.format(self.src_dir,i,self.name))
                if self.importance: 
                    self.visualize_importance(gbm,i)
                if self.test_predict: 
                    test_predictions[:,i] = self.predict_test(gbm,i)
                    r2 = r2_score(y_train,test_predictions[:,i])
                    print("R Squared for test iteration:{}".format(r2))
                if self.prepare_submission: 
                    self.output_submission(self,test_predictions,i)
    
    def predict_test(self,X_test,gbm,index): 
        gbm = lgb.Booster(model_file ="{}/fold_{}_{}_eval_history.txt".format(self.src_dit,index,self.name))
        test_predict = gbm.predict(X_test,num_iteration = gbm.best_iteration)
        return test_predict

    def visualize_importance(self,gbm,index): 
        print("Visualize plot importance")
        if self.train_gbm:
            gbm = gbm.Booster(model_file ="{}/fold_{}_{}_eval_history.txt".format(self.src_dir,index,self.name))
            importance = gbm.feature_importance() 
            names = gbm.feature_names() 
            df_importance = pd.DataFrame() 
            df_importance['importance'] = importance 
            df['names'] = names
            df_importance.sort_values('importance',ascending=False,inplace=True)
            importance_plot = df_importance.plot(x='names',y='importance',kind='barh',figsize=(10,10))
            plt.show()
    
    def output_submission(self,predictions,index,save = True): 
        print("Preparing submission")
        submission = pd.read_csv('./data/sample_submission.csv')
        submid = pd.DataFrame({"Id":submission['id']})
        submission = pd.concat([submid,pd.DataFrame(predictions,columns=['Target'])],axis=1)
        if save: 
            submission.to_csv('./submissions/{}_{}_.csv'.format(self.name,index))
        return submission 








                
                

                






                    

                










            

            

            


            

            


            
            

            
        
            

