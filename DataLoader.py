import pandas as pd 
import numpy as np
from utils import reduce_mem_usage
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import gc

def train_df(build_meta_csv: str, 
        train_csv : str, 
        weather_train_csv : str,
        merge : bool,
        datetime : bool,
        unmerged : bool,
        drop : bool,
        col_drop = None,
        axis = None) -> pd.DataFrame:
        # Reading data
        build = pd.read_csv(build_meta_csv)
        train_data = pd.read_csv(train_csv)
        weather = pd.read_csv(weather_train_csv)
        # Reducing memory on the train data
        train = reduce_mem_usage(train_data)
        if merge:
            train = train_data.merge(build,on='building_id',how='left')
            train = train.merge(weather,on=['site_id','timestamp'],how='left')
            del build,weather 
            gc.collect()
        if datetime:
            train['timestamp'] = pd.to_datetime(train['timestamp'])
            train['year'] = train['timestamp'].dt.year.astype(np.uint16)
            train['month'] = train['timestamp'].dt.month.astype(np.uint8)
            train['day'] = train['timestamp'].dt.month.astype(np.uint8)
            train['weekday'] = train['timestamp'].dt.weekday.astype(np.uint8)
            train['hour'] = train['timestamp'].dt.hour.astype(np.uint8)
        if drop:
            train = train.drop(col_drop,axis=axis)
            gc.collect()
        if unmerged: 
            return build,train,weather
        else: 
            return train

def test_df(test_csv : str, 
        weather_test_csv : str, 
        merge : bool, 
        datetime : bool,
        unmerged : bool,
        drop : bool,
        col_drop = None,
        axis = None) -> pd.DataFrame:
        # Reading data 
        test_data = pd.read_csv(test_csv)
        weather_test = pd.read_csv(weather_test_csv)
        # Reducing memory on the test data 
        test = reduce_mem_usage(test_data)
        if merge:
            test = test_data.merge(weather_test,on=['timestamp'],how='left')
            del weather_test
            gc.collect()
        if datetime:
            test['timestamp'] = pd.to_datetime(test['timestamp'])
            test['year'] = test['timestamp'].dt.year.astype(np.uint16)
            test['month'] = test['timestamp'].dt.month.astype(np.uint8)
            test['day'] = test['timestamp'].dt.month.astype(np.uint8)
            test['weekday'] = test['timestamp'].dt.weekday.astype(np.uint8)
            test['hour'] = test['timestamp'].dt.hour.astype(np.uint8)
        if drop:
            test = test.drop(col_drop,axis=axis)
            gc.collect()
        if unmerged: 
            return build_meta,test,weather_test, 
        else: 
            return test
        
def filling_nan_values(df: pd.DataFrame) -> pd.DataFrame: 
    ratio = df.count()/len(df) 
    cols = ratio[ratio < 1].index
    for col in cols: 
        print(f"Filling Column:{col}")
        df[col] = df[col].fillna(df[col].mean())
    return df










        


        

        




            
            

            

            

