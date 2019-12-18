import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
from math import sin, cos, sqrt, atan2, radians

def create_tables(model):
    importances_all_ARX = np.load('ARX_feature_importances.npy')
    importances_all_RRF = np.load('RRF_feature_importances.npy')
    df_features = pd.DataFrame(np.zeros((36, 18)))
    labels = ['importance_0.5h','importance_1.0h','importance_1.5h','importance_2h','importance_3h','importance_4h','top feature_0.5h','top feature_1h','top feature_1.5h','top feature_2h','top feature_3h','top feature_4h','value_top_0.5h','value_top_1h','value_top_1.5h','value_top_2h','value_top_3h','value_top_4h']
    df_features.columns = labels
    df_features['top feature_0.5h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    df_features['top feature_1h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    df_features['top feature_1.5h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    df_features['top feature_2h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    df_features['top feature_3h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    df_features['top feature_4h'] = pd.to_numeric(df_features['top feature_0.5h'],downcast='signed')
    for h in range(6):        
        if(model == 'ran_for'):  
            weights = importances_all_RRF[:,h]
        else:
            weights = importances_all_ARX[:,h]
        i = 0
        j = 4
        for n in range(36):
            df_features.loc[n,labels[h]] = sum(abs(weights[i:j]))
            if((abs(weights[i:j]).argmax())==0):
                df_features.loc[n,labels[h+6]] = 3  #aziT3  
            elif (abs(weights[i:j]).argmax()==1):
                df_features.loc[n,labels[h+6]] = 2  #T3
            elif (abs(weights[i:j]).argmax()==2):
                df_features.loc[n,labels[h+6]] = 1  #T2            
            else: 
                df_features.loc[n,labels[h+6]] = 0  #T1
            df_features.loc[n,labels[h+12]] = abs(weights[i:j]).max()
            i = j
            j = j+4
    df_features.loc[25,labels[h]]  = df_features.loc[25,labels[h]] + (sum(abs(weights[144:149])))
    df_stations = pd.read_csv('df_stations.csv')
    def man_dist_2_target(lat,lon):
        R = 6373.0
        lat_station = radians(lat)
        lon_station = radians(lon)        
        lat_target = radians(42.166240)
        lon_target = radians(-5.264526)
        dlon = lon_target - lon_station
        dlat = lat_target - lat_station     

        a1 = sin(0 / 2)**2 + cos(lat_target) * cos(lat_target) * sin(dlon / 2)**2
        a2 = sin(dlat / 2)**2 + cos(lat_station) * cos(lat_target) * sin(0 / 2)**2
        c1 = 2 * atan2(sqrt(a1), sqrt(1 - a1))
        c2 = 2 * atan2(sqrt(a2), sqrt(1 - a2))
        WO_distance = R * c1*np.sign(lon_station - lon_target)
        NS_distance = R * c2*np.sign(lat_station - lat_target)
        return [WO_distance,NS_distance]
    df_stations['WO_dist'],df_stations['NS_dist'] = df_stations.apply(lambda x: man_dist_2_target(x['latitude'],x['longitude'])[0], axis=1),df_stations.apply(lambda x: man_dist_2_target(x['latitude'],x['longitude'])[1], axis=1)
    df_importances = pd.concat((df_stations,df_features), axis=1)
    if(model == 'ran_for'):
        df_importances.to_csv('df_importances_RRF.csv',index=0)
    else:
        df_importances.to_csv('df_importances_ARX.csv',index=0)
    Print("Tables for plots updated with newest data for feature importances")
create_tables('ran_for')
create_tables(' ')
