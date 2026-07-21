import pandas as pd
import numpy as np
df = pd.read_csv('data/data_clean.csv')
# print(df['sleep_duration'].describe())
#? In this stage, additional features are engineered from the cleaned dataset
#? by combining existing variables. Both continuous and categorical interaction
#? features were explored to capture relationships that were not directly
#? represented by the original features.
def sleepDU_cate(df):
    df['sleep_cate'] = None
    df.loc[df['sleep_duration'] < 6 , 'sleep_cate' ] =0
    df.loc[(df['sleep_duration']>=6) & (df['sleep_duration'] <=8), 'sleep_cate' ] = 1
    df.loc[df['sleep_duration'] > 8 ,'sleep_cate']=2
    return df
df = sleepDU_cate(df)
def sleepL_stress_flag(df):
    df["sleepLOW_stressH_flag"] = ((df["sleep_cate"] == 0) &(df["stress_level"]=='high')).astype(int)
    # df["sleepLOW_stressMIS_flag"] = ((df["sleep_cate"] == 0) &(df["stress_level"]=='Missing')).astype(int)
    # df["sleepN_stressMIS_flag"] = ((df["sleep_cate"] == 1) &(df["stress_level"]=='Missing')).astype(int)
    # df["sleepN_stressL_flag"] = ((df["sleep_cate"] == 1) &(df["stress_level"]=='low')).astype(int)
    # df["sleepH_stressL_flag"] = ((df["sleep_cate"] == 2) &(df["stress_level"]=='low')).astype(int)
    # df["sleepH_stressMIS_flag"] = ((df["sleep_cate"] == 2) &(df["stress_level"]=='Missing')).astype(int)
    return df
df = sleepL_stress_flag(df)

def sleep_active_flag(df):
    # df["sleepN_activityL_flag"] = ((df["sleep_cate"] == 1) &(df["physical_activity_level"]==0)).astype(int)
    # df["sleepH_activityL_flag"] = ((df["sleep_cate"] == 2) &(df["physical_activity_level"]==0)).astype(int)
    # df["sleepH_activityH_flag"] = ((df["sleep_cate"] == 2) &(df["physical_activity_level"]==2)).astype(int)
    # df["sleepL_activityH_flag"] = ((df["sleep_cate"] == 0) &(df["physical_activity_level"]==2)).astype(int)

    return df
df = sleep_active_flag(df)

# def lifestyle(df):
#     df['stressH_Lactive_sleepL'] = ((df['stress_level'] =='high')& (df['physical_activity_level']==0)&(df['sleep_cate']==0)).astype(int)
#     df['stressL_Hactive_sleepH'] = ((df['stress_level'] =='low')& (df['physical_activity_level']==2)&(df['sleep_cate']==2)).astype(int)
    
  
#     return df
# df = lifestyle(df)
#? After evaluating their impact on model performance, only the interaction
#? features that consistently improved the validation score were retained.
#? Among them, the `stress_exercise` feature showed the greatest positive
#? contribution to the model.
def Stress_Sleep(df):
    df['stress_sleep'] = (df['stress_level'] + 1) / df['sleep_duration']
    return df
df = Stress_Sleep(df)

def Stress_physic(df):
    df['stress_physic'] = (df['stress_level'] + 1) / (df['physical_activity_level']+1)
    return df
df = Stress_physic(df)

def Stress_exercise(df):
    df['stress_exercise'] = (df['stress_level'] + 1) / (df['exercise_duration'] + 1)
    return df
df = Stress_exercise(df)