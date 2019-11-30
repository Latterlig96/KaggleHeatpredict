import lightgbm as lgb 
import xgboost as xgb
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import catboost
import json 
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,f1_score


class GBM:
    def __init__(self,
                train_gbm : bool,
                train_xg : bool,
                train_cat : bool,
                test_predict : bool, 
                save_model : bool,
                save_history : bool,
                seed : int,
                name : str,
                importance : bool,
                stratify : bool,
                eval_metric,
                time_series : bool,
                prepare_submission : bool,
                jsonize : bool): 
                self.train_gbm = train_gbm
                self.train_xg = train_xg
                self.train_cat = train_cat
                self.test_predict = test_predict 
                self.save_model = save_model
                self.save_history = save_history
                self.seed = seed
                self.name = name
                self.prepare_submission = prepare_submission
                self.history = {}
                self.importance = importance
                self.stratify = stratify
                self.eval_metric = eval_metric
                self.time_series = time_series
                self.jsonize = jsonize
    
    def fold_run(self,
            src_dir : str,
            X_train : pd.DataFrame, 
            y_train,
            X_test : pd.DataFrame,
            n_folds : int,
            parameters = None,
            categorical_features = None):

            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values 

            if src_dir:
                if os.path.isdir(src_dir):
                    pass
                else:
                    print(f"Making dir:{src_dir}")
                    os.makedirs(src_dir)
            try: 
                print("X_train_shape",X_train.shape)
                print("X_test_shape",X_test.shape)
            except ValueError: 
                print("Shape does not fit")

            if self.stratify:
                # Bugs...
                #print("Make {} stratify folds".format(n_folds))
                #kf = StratifiedKFold(n_splits=n_folds,random_state = self.seed)
                raise Exception("Bugs... Could not proceed")

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
                if self.train_gbm:
                    print("Train LightGBM")
                    train_X = X_train.iloc[train_index]
                    val_X = X_train.iloc[val_index]
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = y_train.iloc[train_index]
                        val_y = y_train.iloc[val_index]
                    else: 
                        train_y = y_train[train_index]
                        val_y  = y_train[val_index]
                    
                    lgb_train = lgb.Dataset(train_X,train_y,categorical_feature = categorical_features)
                    lgb_val = lgb.Dataset(val_X,val_y,categorical_feature = categorical_features,reference = lgb_train)

                    gbm = lgb.train(params = parameters,
                                    train_set=lgb_train,
                                    num_boost_round=parameters['num_boost_round'],
                                    valid_sets=[lgb_train,lgb_val],
                                    early_stopping_rounds=parameters['early_stopping_rounds'],
                                    evals_result = self.history,
                                    verbose_eval = parameters['verbose_eval'],
                                    feval = self.eval_metric)
                    valid_predictions[val_index,i] = gbm.predict(val_X,num_iteration=gbm.best_iteration)

                    r2= r2_score(val_y[val_index],valid_predictions[val_index,i])
                    print(f"R2 Score for current validation set:{r2}")
                
                    if self.save_model:
                        print("Saving model")
                        gbm.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history.txt')
                    
                    if self.save_history: 
                        print("Saving Hisotry")
                        pd.to_pickle(self.history,f'{src_dir}/fold_{i}_{self.name}_pickle_eval_history.pkl')
                    
                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir)
                    
                    if self.importance: 
                        self.visualize_importance(i,src_dir)
                        
                elif self.train_xg: 
                    print("Train XGBooost")
                    train_X = X_train.iloc[train_index]
                    val_X = X_train.iloc[val_index]
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = y_train.iloc[train_index]
                        val_y = y_train.iloc[val_index]
                    else: 
                        train_y = y_train[train_index]
                        val_y = y_train[val_index]
                    
                    xg_train = xgb.DMatrix(train_X,label=train_y)
                    xg_val = xgb.DMatrix(val_X,label=val_y)
                    eval_list = [(xg_train,'train'),(xg_val,'val')]

                    xgboost_train = xgb.train(parameters,xg_train,evals=eval_list,
                                              evals_result=self.history, 
                                              num_boost_round=parameters['boost_round'],
                                              early_stopping_rounds=parameters['early_stopping'],
                                              verbose_eval = parameters['verbose_eval'])

                    valid_predictions[val_index,i] = xgboost_train.predict(xg_val,
                                                                           ntree_limit = xgboost_train.best_ntree_limit)

                    r2 = r2_score(val_y[val_index],valid_predictions[val_index,i])
                    print(f"R2 Score for current validation set:{r2}")
                    
                    if self.save_model:
                        print("Saving model")
                        xgboost_train.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history.txt')

                    if self.save_history: 
                        print("Saving Hisotry")
                        pd.to_pickle(self.history,f'{src_dir}/fold_{i}_{self.name}_pickle_eval_history.pkl')
           
                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir,xgboost_train)

                    if self.importance: 
                        self.visualize_importance(i,src_dir) 
                
                elif self.train_cat: 
                    print("Training CatBoost")
                    train_X = np.array(X_train.iloc[train_index],dtype=np.float32)
                    val_X = np.array(X_train.iloc[val_index],dtype=np.float32)
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = np.array(y_train.iloc[train_index],dtype=np.float32)
                        val_y = np.array(y_train.iloc[val_index],dtype=np.float32)
                    else: 
                        train_y = np.array(y_train[train_index],dtype=np.float32)
                        val_y = np.array(y_train[val_index],dtype=np.float32)
                    
                    cat_train = catboost.Pool(train_X,label=train_y)
                    cat_test = catboost.Pool(val_X,label=val_y)
                    self.cat = catboost.CatBoostRegressor(**parameters).fit(cat_train,use_best_model=True,
                                                                       eval_set=cat_test,verbose_eval=True) 
                    #Index Error after first epoch, need to fix it
                    valid_predictions[val_index,i] = self.cat.predict(cat_test)
                    r2 = r2_score(val_y[val_index],valid_predictions[val_index,i])
                    print(f"R2 Score for current validation set:{r2}")
                    if self.save_model:
                        print("Saving model")
                        self.cat.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history',format='json')

                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir)
                i += 1
            if self.jsonize:
                print("Saving model parameters to json")
                model_dict = {"model":f"{src_dir}_{i}_folds",
                              "parameters":parameters}
                with open(f"./parameters/{src_dir}_{i}_fold.json",'w+') as model_param: 
                    json.dump(model_dict,model_param)
            
            if self.prepare_submission: 
                self.output_submission(self,test_predictions,i)

            return valid_predictions,test_predictions
            
    def predict_test(self,X_test,index,source_dir,xgb_model = None):
        if self.train_gbm:
            gbm = lgb.Booster(model_file =f"{source_dir}/fold_{index}_{self.name}_eval_history.txt")
            test_predict = gbm.predict(X_test,num_iteration = gbm.best_iteration)
            return test_predict
        elif self.train_xg and xgb_model is not None:
            xg_mod = xgb.Booster(model_file=f"{source_dir}/fold_{index}_{self.name}_eval_history.txt")
            test_predict = xg_mod.predict(xgb.DMatrix(X_test),ntree_limit=xgb_model.best_ntree_limit)
            return test_predict
        elif self.train_cat:
            self.cat.load_model(f'{source_dir}/fold_{index}_{self.name}_eval_history',format='json')
            test_predict = self.cat.predict(catboost.Pool(X_test))
            return test_predict

    def visualize_importance(self,index,source_dir): 
        print("Visualize plot importance")
        if self.train_gbm:
            gbm = lgb.Booster(model_file =f"{source_dir}/fold_{index}_{self.name}_eval_history.txt")
            importance = gbm.feature_importance() 
            names = gbm.feature_name()
        elif self.train_xg: 
            xg_mod = xgb.Booster(model_file=f"{source_dir}/fold_{index}_{self.name}_eval_history.txt")
            importance = list(xg_mod.get_fscore().values())
            names = list(xg_mod.get_fscore().keys())
        elif self.train_cat:
            cat_model = self.cat.load_model(f'{source_dir}/fold_{index}_{self.name}_eval_history',format='json')
            importance = cat_model.feature_importances_
            names = cat_model.feature_names_

        df_importance = pd.DataFrame() 
        df_importance['importance'] = importance
        df_importance['names'] = names 
        df_importance.sort_values('importance',ascending=False,inplace=True)
        importance_plot = df_importance.plot(x='names',y='importance',kind='barh',figsize=(10,10))
 
        if self.train_gbm: 
            plt.title("LightGBM current validation set importance")
        elif self.train_xg: 
            plt.title("XGBoost current validation set importance")
        elif self.train_cat: 
            plt.title("CatBoost current validation set importance")
    
    def output_submission(self,predictions,index,save = True): 
        print("Preparing submission")
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['predictions'] = predictions
        if save: 
            submission.to_csv('./submissions/{}_{}_.csv'.format(self.name,index))
        return submission 