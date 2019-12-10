import pandas as pd
import numpy as np
import scipy.stats as stats

# Based on  https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


def percentile_condition(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing outliers based on percentile conditioning")
    print(f"Input data shape: {df.shape}")
    # ========================================================
    labels = df['meter_reading']
    IQR = np.percentile(labels,75) - np.percentile(labels,25)
    const = IQR * 1.5 
    lower_treshold = np.percentile(labels,25) - const 
    upper_treshold = np.percentile(labels,75) + const 
    # Non outliers data 
    cleaned_labels = df['meter_reading'].between(lower_treshold,upper_treshold)
    print("False/True values while removing the outliers:\n")
    print(cleaned_labels.value_counts())
    # Removing indexes of outliers data
    indexes = df[~cleaned_labels].index 
    cleaned_df = df.drop(indexes,axis=0)
    print(f"Final shape after removing outliers: {cleaned_labels.shape[0]}")
    return cleaned_df 

#Very nice and gentle code implementation for finding bad rows implemented by Robert Stackton
def make_is_bad_zero(Xy_subset, min_interval=48,summer_month_start = 5,summer_month_end = 8):
    """Helper routine for 'find_bad_zeros'.
    
    This operates upon a single dataframe produced by 'groupby'. We expect an 
    additional column 'meter_id' which is a duplicate of 'meter' because groupby 
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero

    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()
    ids = all_sequence_ids[is_zero].rename("ids")
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        keep = set(ids[(Xy_subset.month < summer_month_start) |
                       (Xy_subset.month > summer_month_end)].unique())
        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)
    elif meter == 1:
        time_ids = ids.to_frame().join(Xy_subset.timestamp).set_index("timestamp").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result

def find_bad_zeros(X : pd.DataFrame, y):
    """Returns an Index object containing only the rows which should be deleted."""
    Xy = X.assign(meter_reading=y, meter_id=X.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])

def find_bad_building1099(X : pd.DataFrame, y):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index

def trim_site(X : pd.DataFrame) -> pd.DataFrame: 
    """ Trim some outliers readings from specified sites""" 
    X.loc[X['site_id'] == 6,'meter_reading'] = np.clip(X.loc[X['site_id'] == 6,'meter_reading'],a_min=0,a_max=1.9e5)
    X.loc[(X['primary_use'] == 'Education') &(X['site_id'] == 13),'meter_reading'] = np.clip(X.loc[(X['primary_use'] == 'Education') &(X['site_id'] == 13),'meter_reading'],a_min=0,a_max=1.5e5)
    X.loc[X['site_id'] == 9,'meter_reading'] = np.clip(X.loc[X['site_id'] == 9 ,'meter_reading'],a_min =0,a_max = 1.9e5)
    return X
