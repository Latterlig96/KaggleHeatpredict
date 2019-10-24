import pandas as pd 
import numpy as np
from utils import reduce_mem_usage

def load_data(build_meta : str,
            train : str ,
            test : str,
            weather_test : str ,
            weather_train : str ,
            train_mode : bool , 
            test_mode : bool ,
            to_datetime : bool ,
            merge : bool ,
            return_unmerged : bool) -> pd.DataFrame:

            if train_mode: 
                build_data = pd.read_csv(build_meta) 
                train_data = pd.read_csv(train) 
                weather_train_data = pd.read_csv(weather_train)
            else:
                build_data = pd.read_csv(build_meta) 
                test_data = pd.read_csv(test) 
                weather_test_data = pd.read_csv(weather_test)
            if train_mode: 
                train_data, NAlist = reduce_mem_usage(train_data)
            elif test_mode: 
                test_data, NAlist = reduce_mem_usage(test_data)
            if merge and train_mode: 
                train = train_data.merge(build_data,on='building_id',how='left')
                train = train.merge(weather_train_data,on=['site_id','timestamp'],how='left')
            elif merge and test_mode: 
                test = test_data.merge(build_data,on='building_id',how='left')
                test = test.merge(weather_test_data,on=['site_id','timestamp'],how='left')
            if to_datetime:
                if train_mode:
                    train['timestamp'] = pd.to_datetime(train['timestamp'])
                    train['year'] = train['timestamp'].dt.year.astype(np.uint16)
                    train['month'] = train['timestamp'].dt.month.astype(np.uint8)
                    train['day'] = train['timestamp'].dt.month.astype(np.uint8)
                    train['weekday'] = train['timestamp'].dt.weekday.astype(np.uint8)
                    train['hour'] = train['timestamp'].dt.hour.astype(np.uint8)
                else: 
                    test['timestamp'] = pd.to_datetime(test['timestamp'])
                    test['year'] = test['timestamp'].dt.year.astype(np.uint16)
                    test['month'] = test['timestamp'].dt.month.astype(np.uint8)
                    test['day'] = test['timestamp'].dt.month.astype(np.uint8)
                    test['weekday'] = test['timestamp'].dt.weekday.astype(np.uint8)
                    test['hour'] = test['timestamp'].dt.hour.astype(np.uint8)                       
            if return_unmerged and train_mode:
                if any(build_data.columns == 'timestamp') and any(train_data.columns == 'timestamp'):
                    build_data['timestamp'] = pd.to_datetime(build_data['timestamp'])
                    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
                return build_data,train_data,weather_train_data
            elif test_mode: 
                return test
            else:
                return train

            
            

            

            

