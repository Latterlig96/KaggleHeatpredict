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
from sklearn.metrics import r2_score,mean_squared_error,auc,mean_squared_log_error
from ngboost.ngboost import NGBoost 
from ngboost.distns import Normal 
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE 


class GBM:
    """
        The main purpose of this function is to provide necesarry

        functionality for users that would like to use Base-Tree models 

        such as LightGBM,XGBoost,Catboost or the most recent model NGBoost.

        LightGBM and XGBoost and NGBoost can be performed on classification 

        and regression tasks when Catboost can be used just for regression problems. 

        Fuction can be also used to provide base functionality for ensemble models. 

        Arguments:
        ============================================================================
        : train_gbm - bool - whether to train LightGBM 

        : train_xg  - bool - whether to train XGBoost 

        : train_cat - bool - whether to train Catboost 

        : train_ng - bool - whether to train NGBoost

        : test_predict - bool - bool to choose whether to test model on unseen test data 

        : save_model - bool - whether to save_model, specific source dir for model 

        is specified in fold_run function 

        : save_history - bool - whether to save model loss history 

        : seed - int - random seed 
        
        : name - str - this provides some easy to use name in source dir to find 
        specified model 

        : importance - bool - whether to plot feature importance for model 
        (this function does not cover the CatBoost and NGBoost, so can be turned to False
        when training one of mentioned models)

        : stratify - bool - whether to use StratifiedKFold (sklearn implementation)

        : eval_metric - function - custom metric made by the user that can be passed to model 
        as a loss metric 

        : time_series - bool - whether to use TimeSeriesSplit (sklearn implementation) 

        : prepare_submission - bool - whether to save predictions of models into pandas DataFrame 

        : jsonize - bool - this arguments provides saving the model run parameters that are saved 
        into json format 

        # Mentions 
        In XGBoost,CatBoost and NGBoost model was used function nan_to_num offered by numpy API.
        
        This is because there were a lot of problems with Sklearn API that could not handle this 

        kind of prepared dataset and was throwing error that dataset consist of NaN values, but in reality 
        
        there were 0 NaN Values.
        """
    def __init__(self,
                train_gbm : bool,
                train_xg : bool,
                train_cat : bool,
                train_ng : bool,
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
                jsonize : bool,
                show_metric_results : bool): 
                self.train_gbm = train_gbm
                self.train_xg = train_xg
                self.train_cat = train_cat
                self.train_ng = train_ng
                self.test_predict = test_predict 
                self.save_model = save_model
                self.save_history = save_history
                self.seed = seed
                self.name = name
                self.prepare_submission = prepare_submission
                self.show_metric_results = show_metric_results
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

            """
            # Arguments: 
            : src_dir - str - main dir for saving model,history and fold runs 
            
            : X_train,y_train - training dataset with labels 

            : X_test,y_test - test dataset with labels

            : n_folds - number of folds to split dataset 

            : parameters - run_parameters that are necessary to run the Tree-Based models,

            for better understading of these parameters, you should go and read LightGBM,

            XGBoost,CatBoost API 

            : categorical_features - list - list of categorical features in dataset, 

            necessary for LightGBM and CatBoost

            # Returns: 
            valid_predictions,test_predictions - predictions made by model
            """

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
            test_predictions = np.zeros((X_test.shape[0],n_folds))
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
                    valid_predictions[val_index,i] = np.clip(np.nan_to_num(valid_predictions[val_index,i]),a_min=0,a_max=None)

                    r2= r2_score(val_y[val_index],valid_predictions[val_index,i])
                    log_error = np.sqrt(mean_squared_log_error(val_y[val_index],valid_predictions[val_index,i]))
                    print(f"R2 Score for current validation set:{r2}")
                    print(f"RMSLE for current val set:{log_error}")
                
                    if self.save_model:
                        print("Saving model")
                        gbm.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history.txt')
                    
                    if self.save_history: 
                        print("Saving Hisotry")
                        pd.to_pickle(self.history,f'{src_dir}/fold_{i}_{self.name}_pickle_eval_history.pkl')
                    
                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir)
                        test_predictions[:,i] = np.clip(np.nan_to_num(test_predictions[:,i]),a_min=0,a_max=None)

                    if self.importance: 
                        self.visualize_importance(i,src_dir)
                    
                    if self.show_metric_results:
                        self.show_results(i)
                        
                elif self.train_xg: 
                    print("Train XGBooost")
                    train_X = np.nan_to_num(X_train.iloc[train_index])
                    val_X = np.nan_to_num(X_train.iloc[val_index])
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = np.nan_to_num(y_train.iloc[train_index])
                        val_y = np.nan_to_num(y_train.iloc[val_index])
                    else: 
                        train_y = np.nan_to_num(y_train[train_index])
                        val_y = np.nan_to_num(y_train[val_index])
                    
                    xg_train = xgb.DMatrix(train_X,label=train_y,feature_names=X_train.columns)
                    xg_val = xgb.DMatrix(val_X,label=val_y,feature_names=X_train.columns)
                    eval_list = [(xg_train,'train'),(xg_val,'val')]

                    xgboost_train = xgb.train(parameters,xg_train,evals=eval_list,
                                              evals_result=self.history, 
                                              num_boost_round=parameters['boost_round'],
                                              early_stopping_rounds=parameters['early_stopping'],
                                              verbose_eval = parameters['verbose_eval'])

                    valid_predictions[val_index,i] = xgboost_train.predict(xg_val,
                                                                           ntree_limit = xgboost_train.best_ntree_limit)
                    valid_predictions[val_index,i] = np.clip(np.nan_to_num(valid_predictions[val_index,i]),a_min=0,a_max=None)

                    r2 = r2_score(val_y[val_index],valid_predictions[val_index,i])
                    log_error = np.sqrt(mean_squared_log_error(val_y[val_index],valid_predictions[val_index,i]))
                    print(f"R2 Score for current validation set:{r2}")
                    print(f"RMSLE for current val set:{log_error}")
                
                    if self.save_model:
                        print("Saving model")
                        xgboost_train.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history.txt')

                    if self.save_history: 
                        print("Saving Hisotry")
                        pd.to_pickle(self.history,f'{src_dir}/fold_{i}_{self.name}_pickle_eval_history.pkl')
           
                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir,xgboost_train)
                        test_predictions[:,i] = np.clip(np.nan_to_num(test_predictions[:,i]),a_min=0,a_max=None)

                    if self.importance: 
                        self.visualize_importance(i,src_dir)
                    
                    if self.show_metric_results: 
                        self.show_results(i)
                
                elif self.train_cat: 
                    print("Training CatBoost")
                    train_X = np.nan_to_num(np.array(X_train.iloc[train_index],dtype=np.float32))
                    val_X = np.nan_to_num(np.array(X_train.iloc[val_index],dtype=np.float32))
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = np.nan_to_num(np.array(y_train.iloc[train_index],dtype=np.float32))
                        val_y = np.nan_to_num(np.array(y_train.iloc[val_index],dtype=np.float32))
                    else: 
                        train_y = np.nan_to_num(np.array(y_train[train_index],dtype=np.float32))
                        val_y = np.nan_to_num(np.array(y_train[val_index],dtype=np.float32))

                    cat_train = catboost.Pool(train_X,label=train_y)
                    cat_test = catboost.Pool(val_X,label=val_y)
                    self.cat = catboost.CatBoostRegressor(**parameters).fit(cat_train,use_best_model=True,
                                                                       eval_set=cat_test,verbose_eval=True)
                    self.history = self.cat.get_evals_result()
                    #Index Error after first epoch, need to fix it
                    valid_predictions[val_index,i] = self.cat.predict(cat_test)
                    valid_predictions[val_index,i] = np.clip(np.nan_to_num(valid_predictions[val_index,i]),a_min=0,a_max=None)
                    r2 = r2_score(val_y[val_index],valid_predictions[val_index,i])
                    log_error = np.sqrt(mean_squared_log_error(val_y[val_index],valid_predictions[val_index,i]))
                    print(f"R2 Score for current validation set:{r2}")
                    print(f"RMSLE for current val set:{log_error}")
                    if self.save_model:
                        print("Saving model")
                        self.cat.save_model(f'{src_dir}/fold_{i}_{self.name}_eval_history',format='json')

                    if self.test_predict: 
                        test_predictions[:,i] = self.predict_test(X_test,i,src_dir)
                        test_predictions[:,i] = np.clip(np.nan_to_num(test_predictions[:,i]),a_min=0,a_max=None)

                elif self.train_ng: 
                    print("Train NGBooost")
                    train_X = np.nan_to_num(X_train.iloc[train_index])
                    val_X = np.nan_to_num(X_train.iloc[val_index])
                    if isinstance(y_train,pd.DataFrame): 
                        train_y = np.nan_to_num(y_train.iloc[train_index])
                        val_y = np.nan_to_num(y_train.iloc[val_index])
                    else: 
                        train_y = np.nan_to_num(y_train[train_index])
                        val_y = np.nan_to_num(y_train[val_index])

                        ng = NGBoost(Dist=Normal,Score=MLE,
                                     Base=default_tree_learner,natural_gradient=True,
                                     n_estimators = 150,learning_rate = 0.01,verbose=True,
                                     verbose_eval=50).fit(train_X,train_y)
                        valid_predictions[val_index,i] = ng.predict(val_X)
                        valid_predictions[val_index,i] = np.clip(np.nan_to_num(valid_predictions[val_index,i]),a_min=0,a_max=None)
                        rmse = np.sqrt(mean_squared_error(val_y,valid_predictions[val_index,i]))
                        r2 = r2_score(val_y,valid_predictions[val_index,i])
                        log_error = np.sqrt(mean_squared_log_error(val_y[val_index],valid_predictions[val_index,i]))
                        print(f"RMSE for current fold:{rmse}")
                        print(f"R2 Score for current fold:{r2}")
                        print(f"RMSLE for current val set:{log_error}")
                        test_predictions[:,i] = np.clip(np.nan_to_num(ng.predict(X_test)),a_min=0,a_max=None)
                i += 1
            if self.jsonize:
                print("Saving model parameters to json")
                if os.path.isdir('parameters'):
                    pass
                else:
                    print("Making Dir: parameters")
                    os.makedirs('parameters')
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
    
    def show_results(self,index):
        if self.train_gbm: 
            rmse_score = np.min(self.history['valid_1']['rmse'])
            print("Best score for current validation set:\n")
            print(f"RMSE:{rmse_score}")
        elif self.train_xg: 
            rmse_score = np.max(self.history['val']['rmse'])
            print("Best score for current validation set:\n")
            print(f"RMSE:{rmse_score}")
        
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