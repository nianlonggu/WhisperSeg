import pandas as pd
import numpy as np

"""
Marmoset, for model whisperseg-large-marmoset-v2.0, dataset from Melissa
"""

def detect_continuous_e_ts( df ):
    continous_ts_list = []
    for idx in range( len(df) ):
        if df.iloc[idx]["cluster"] == "e_ts":
            if len(continous_ts_list) == 0 or len(continous_ts_list[-1]) == 2:
                continous_ts_list.append( [ idx ] )
            else:
                if idx > 0 and df.iloc[idx]["onset"] - df.iloc[idx-1]["offset"] > 0.01:
                    if idx - continous_ts_list[-1][0] <= 5:
                        continous_ts_list.pop(-1)
                    else:
                        continous_ts_list[-1].append(idx)
                    continous_ts_list.append( [ idx ] )
        else:
            if idx > 0 and idx < len(df) - 1 and df.iloc[idx-1]["cluster"] == "e_ts" and df.iloc[idx+1]["cluster"] == "e_ts":
                if df.iloc[idx+1]["onset"] - df.iloc[idx-1]["offset"] < 0.01:
                    continue
            elif len(continous_ts_list) > 0 and len(continous_ts_list[-1]) == 1:
                if idx - continous_ts_list[-1][0] <= 5:
                    continous_ts_list.pop(-1)
                else:
                    continous_ts_list[-1].append(idx)
    if len(continous_ts_list)>0 and len(continous_ts_list[-1]) == 1:
        continous_ts_list.pop(-1)
    return continous_ts_list

def convert_continuous_e_ts_to_e_tw( df ):
    continous_ts_list = detect_continuous_e_ts( df )
    indices_to_skip = []
    for item in continous_ts_list:
        indices_to_skip += np.arange( item[0], item[1] ).tolist()
    new_df = df[~df.index.isin(indices_to_skip)]
    for start, end in continous_ts_list:
        try:
            assert df.iloc[end-1]["offset"] > df.iloc[start]["onset"]
        except:
            continue
        new_df = pd.concat([ new_df, 
                             pd.DataFrame({ "onset": [df.iloc[start]["onset"]],
                               "offset": [df.iloc[end-1]["offset"]],
                               "cluster": ["e_tw"]
                             })
                           ], ignore_index=True)
    new_df = new_df.sort_values( "onset" )
    new_df = new_df.reset_index( drop = True )
    
    return new_df

def clean_e_tw_follows( df ):
    indices_to_remove = []
    is_checking = 3
    current_tw_idx = None
    for idx in range(len(df)):
        if df.iloc[idx]["cluster"] == "e_tw":
            is_checking = 3
            current_tw_idx = idx
        else:
            if is_checking > 0:
                if df.iloc[idx]["cluster"].startswith("e_p") and idx > 0 and df.iloc[idx]["onset"] - df.iloc[idx-1]["offset"] < 0.1:
                    indices_to_remove.append( idx )
                    if is_checking > 1:
                        df.loc[current_tw_idx, "offset"] = df.iloc[idx]["offset"]
                    is_checking -= 1
                elif idx > 0 and df.iloc[idx]["onset"] - df.iloc[idx-1]["offset"] < 0.01:
                    indices_to_remove.append( idx )
                    if is_checking > 1:
                        df.loc[current_tw_idx, "offset"] = df.iloc[idx]["offset"]
                    is_checking -= 1
                else:
                    is_checking = 0
                
    new_df = df[~df.index.isin(indices_to_remove)]
    new_df = new_df.sort_values("onset").reset_index(drop=True)
    return new_df

def post_process_marmoset( df ):
    try:
        new_df = clean_e_tw_follows(convert_continuous_e_ts_to_e_tw(df))
    except:
        new_df = df
    return new_df


PROCESS_TOOLBOX= {
    "whisperseg-large-marmoset-v2.0":post_process_marmoset
}
