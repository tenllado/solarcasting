# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import os
import json
import datetime
import argparse
import pandas as pd
import numpy as np
import math
from functools import reduce
pd.set_option('display.max_columns', 200)

class TrainTestSetGenerator(object):
    orig_folder = None
    dest_folder = None    
    X_y_files = None
    start_date_train = None
    end_date_train = None
    start_date_test = None
    end_date_test = None
    n_samples = None
    radiation_differences = False
    include_azimuth_last_sample = True
    forecast_horizon = None
    config_file = None
    verbose = True

    def __init__(self, orig_folder=None, dest_folder=None, X_y_files = None, start_date_train = None, end_date_train = None, start_date_test = None, end_date_test = None, n_samples = None, radiation_differences = False, include_azimuth_last_sample = True, forecast_horizon = None, config_file=None, verbose=True):
        
        self.orig_folder = orig_folder
        self.dest_folder = dest_folder
        self.X_y_files = X_y_files
        self.start_date_train = start_date_train
        self.end_date_train = end_date_train
        self.start_date_test = start_date_test
        self.end_date_test = end_date_test
        self.n_samples = n_samples
        self.radiation_differences = radiation_differences
        self.include_azimuth_last_sample = include_azimuth_last_sample
        self.forecast_horizon = forecast_horizon 
        self.config_file = config_file
        self.verbose = verbose
        self.parse_config_file()
        self.check_errors()
        
    def concatenate_data(self):

        file_name = self._get_file_names()        
        if os.path.isfile(os.path.join(self.dest_folder, file_name[0] + ".csv")):
            print ("Train data X already exists: " + file_name[0])
            return None
        elif os.path.isfile(os.path.join(self.dest_folder, file_name[1] + ".csv")):
            print ("Train data y already exists: " + file_name[1])
            return None
        
        elif os.path.isfile(os.path.join(self.dest_folder, file_name[2] + ".csv")):
            print ("Test data X already exists: " + file_name[2])
            return None
        
        elif os.path.isfile(os.path.join(self.dest_folder, file_name[3] + ".csv")):
            print ("Test data y already exists: " + file_name[3])
            return None

        csv_files = self._get_csv_files()

        if len(csv_files) == 0:
            print ("No csv file found.")
            return None
        df_X, feature_list = self._engineer_features(csv_files[0])
        n_codigos = df_X['codigo'].nunique()
        print('Number of stations: '+str(n_codigos))
        print("Selected features : "+str(feature_list))        
        df_X = self._build_vectors(df_X)
        df_X = self._rename_df(df_X, feature_list)
        df_Xy = self._add_target(df_X, csv_files[1], csv_files[2], feature_list, n_codigos)
        df_X, df_y, df_X_train, df_y_train, df_X_test, df_y_test = self._split_data(df_Xy)
        self._save_csvs(df_X, df_y, df_X_train, df_y_train, df_X_test, df_y_test)        
      
    def _save_csvs(self, df_X, df_y, df_X_train, df_y_train, df_X_test, df_y_test):
        """
        This function saves X and y data for the whole dataset, train and test set
        """

        # Name of the file 
        file_name = self._get_file_names()

        # if base folder doesn't exist create it
        print ('folder: '+self.dest_folder)
        if not os.path.isdir(self.dest_folder):
            os.makedirs(self.dest_folder)

        # destination folders for the file
        destination_folder_dataset = os.path.join(self.dest_folder, "complete_dataset")
        destination_folder_train_test = os.path.join(self.dest_folder, "train_test_data")

        # if destination folders doesn't exists we create them
        if not os.path.isdir(destination_folder_dataset):
            os.makedirs(destination_folder_dataset)
        if not os.path.isdir(destination_folder_train_test):
            os.makedirs(destination_folder_train_test)
            
        # saving whole dataset
        X_path = os.path.join(destination_folder_dataset, file_name[0] + ".csv")
        y_path = os.path.join(destination_folder_dataset, file_name[1] + ".csv")
        df_X.to_csv(X_path, index=False)            
        df_y.to_csv(y_path, index=False)
        print(X_path)
        print(y_path)
        # saving train data
        X_train_path = os.path.join(destination_folder_train_test, file_name[2] + ".csv")
        y_train_path = os.path.join(destination_folder_train_test, file_name[3] + ".csv")
        df_X_train.to_csv(X_train_path, index=False)            
        df_y_train.to_csv(y_train_path, index=False)
        print(X_train_path)
        print(y_train_path)
        print("Saving data...")
        # saving train data
        X_test_path = os.path.join(destination_folder_train_test, file_name[4] + ".csv")
        y_test_path = os.path.join(destination_folder_train_test, file_name[5] + ".csv")
        df_X_test.to_csv(X_test_path, index=False)            
        df_y_test.to_csv(y_test_path, index=False)
        print(X_test_path)
        print(y_test_path)
        
    def _get_csv_files(self):
        X_path = os.path.join(self.orig_folder, self.X_y_files[0])
        df_X = pd.read_csv(X_path)
        y_path = os.path.join(self.orig_folder, self.X_y_files[1])
        df_y = pd.read_csv(y_path)
        y_persist_path = os.path.join(self.orig_folder, self.X_y_files[2])
        df_y_persist = pd.read_csv(y_persist_path)
        return [df_X, df_y, df_y_persist]
                
    def _get_file_names(self):        
        return ["df_X","df_y","df_X_train","df_y_train","df_X_test","df_y_test"]

    def parse_config_file(self):
        """This method parses the self.config file and assigns the values to attributes"""

        if self.config_file is not None:

            if self.verbose:
                print ('Parsing config file')

            with open(self.config_file) as data_file:
                config_data = json.load(data_file)
            
            if "original_folder" in config_data:
                self.orig_folder = config_data["original_folder"]

            if "destination_folder" in config_data:
                self.dest_folder = config_data["destination_folder"]

            if "X_y_files" in config_data:
                self.X_y_files = config_data["X_y_files"]

            if "start_date_train" in config_data:
                self.start_date_train = config_data["start_date_train"]

            if "end_date_train" in config_data:
                self.end_date_train = config_data["end_date_train"]

            if "start_date_test" in config_data:
                self.start_date_test = config_data["start_date_test"]

            if "end_date_test" in config_data:
                self.end_date_test = config_data["end_date_test"]

            if "n_samples" in config_data:
                self.n_samples = config_data["n_samples"]

            if "radiation_differences" in config_data:
                self.radiation_differences = config_data["radiation_differences"]

            if "include_azimuth_last_sample" in config_data:
                self.include_azimuth_last_sample = config_data["include_azimuth_last_sample"]
                
            if "forecast_horizon" in config_data:
                self.forecast_horizon = config_data["forecast_horizon"]
                
    def check_errors(self):
        """ This method checks if all the variables have the correct values """
        if self.start_date_train is None:
            raise Exception(1, "No start_date for training period specified.")
        
        elif self.end_date_train is None:
            raise Exception(2, "No end_date for training period specified.")
        
        elif self.start_date_test is None:
            raise Exception(3, "No start_date for test period specified.")
        
        elif self.end_date_test is None:
            raise Exception(4, "No end_date for test period specified.")

        elif self.start_date_train > self.end_date_train:
            raise Exception(5, "start_date for training period greater than end_date.")

        elif self.start_date_test > self.end_date_test:
            raise Exception(5, "start_date for test period greater than end_date.")

        elif (self.n_samples is None) or (self.n_samples < 1) or (self.n_samples > 3):
            raise Exception(6, "Specify number of samples between 1 and 3.")

        elif (self.n_samples == 1) and (self.radiation_differences == True):
            raise Exception(6, "For radiation_differences at least n_samples = 2 necessary.")
        
        elif (set(self.forecast_horizon) - set([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])):
            raise Exception(8, "Possible forecast horizons are 0.5, 1.0, 1.5, 2.0, 3.0 or 4.0")
        
        elif self.dest_folder is None:
            raise Exception(7, "No destination_folder specified.")

        elif self.orig_folder is None:
            raise Exception(7, "No original_folder specified.")

        elif self.X_y_files is None:
            raise Exception(7, "No X and y files specified.")
      
    def _engineer_features(self,df_X):
        
        df_X.drop(["latitude","longitude"],axis=1,inplace=True)        
        # change format of hora into float
        df_X['hora'] = df_X[['hora']]//100+(df_X[['hora']]%100/60)
        # reduce memory usage by downcasting numeric datatypes
        df_X['fecha']=pd.to_numeric(df_X['fecha'], downcast='signed')
        df_X['year']=pd.to_numeric(df_X['year'], downcast='signed')
        df_X['hora']=pd.to_numeric(df_X['hora'], downcast='float')        
        df_X['rel_rad']=pd.to_numeric(df_X['rel_rad'], downcast='float')
        
        if (self.n_samples == 1):
            df_t1 = df_X[['codigo','fecha','hora','year','azimuth','rel_rad']] 
            if not(self.include_azimuth_last_sample):
                df_t1.drop("azimuth",axis=1,inplace=True)
                feature_list = ['_T1']
            else:
                df_t1['azimuth']=pd.to_numeric(df_t1['azimuth'], downcast='float')
                feature_list = ['_aziT1','_T1']
            return df_t1, feature_list

        elif (self.n_samples == 2):
            df_t1 = df_X[['codigo','fecha','hora','year','azimuth','rel_rad']]             
            df_t2 = df_t1[['codigo','fecha','hora','year','rel_rad']]            
            df_t2['hora'] = df_t2[['hora']]-0.5               
            df_t21 = pd.merge(df_t1,df_t2,on=['codigo','fecha','hora','year'],how='inner')
            df_t21['hora'] = df_t21[['hora']]+0.5
            df_t21 = df_t21[['codigo','fecha','hora','year','azimuth','rel_rad_x','rel_rad_y']]
            if (self.radiation_differences):
                df_t21['dif_T21'] = df_t21['rel_rad_y']-df_t21['rel_rad_x']      
                df_t21 = df_t21[['codigo','fecha','hora','year','azimuth','dif_T21','rel_rad_y']]
                feature_list = ['_aziT2','_dif-T21','_T1']
                if (not(self.include_azimuth_last_sample)):
                    df_t21.drop("azimuth",axis=1,inplace=True)
                    feature_list = ['_dif-T21','_T1']
            elif not(self.radiation_differences):
                    df_t21['azimuth']=pd.to_numeric(df_t21['azimuth'], downcast='float')
                    feature_list = ['_aziT2','_T2','_T1']
                    if (not(self.include_azimuth_last_sample)):
                        df_t21.drop("azimuth",axis=1,inplace=True)
                        feature_list = ['_T2','_T1']   
            return df_t21, feature_list
        elif (self.n_samples == 3):
            df_t3 = df_X[['codigo','fecha','hora','year','azimuth','rel_rad']]            
            df_t3['hora'] = df_t3[['hora']]+1.0            
            df_t2 = df_t3[['codigo','fecha','hora','year','rel_rad']]            
            df_t2['hora'] = df_t2[['hora']]-0.5
            df_t1 = df_t2.copy()            
            df_t1['hora'] = df_t1[['hora']]-0.5            
            data_frames =[df_t3,df_t2,df_t1]
            df_t321 = reduce(lambda left,right: pd.merge(left,right,on=['codigo','fecha','hora','year'],how='inner'), data_frames)            
            df_t321 = df_t321[['codigo','fecha','hora','year','azimuth','rel_rad_x','rel_rad_y','rel_rad']]
            if (self.radiation_differences):
                df_t321['dif_T32'] = df_t321['rel_rad_y']-df_t321['rel_rad_x']
                df_t321['dif_T21'] = df_t321['rel_rad']-df_t321['rel_rad_y']                
                df_t321 = df_t321[['codigo','fecha','hora','year','azimuth','dif_T32','dif_T21','rel_rad']]
                feature_list = ['_aziT3','_dif-T32','_dif-T21','_T1']
                if not(self.include_azimuth_last_sample):
                    df_t321.drop("azimuth",axis=1,inplace=True)
                    feature_list = ['_dif-T32','_dif-T21','_T1']
            elif not(self.radiation_differences):
                    df_t321['azimuth']=pd.to_numeric(df_t321['azimuth'], downcast='float')
                    feature_list = ['_aziT3','_T3','_T2','_T1']
                    if (not(self.include_azimuth_last_sample)):
                        df_t321.drop("azimuth",axis=1,inplace=True)
                        feature_list = ['_T3','_T2','_T1']   
            return df_t321, feature_list
        
    def _build_vectors(self,df_X):
        start_date = min (self.start_date_test,self.start_date_train)        
        start_date_str = str (start_date)[:4]+"-"+str(start_date)[4:6]+"-"+str(start_date)[6:8]
        end_date = max (self.end_date_test,self.end_date_train)
        end_date_str = str (end_date)[:4]+"-"+str(end_date)[4:6]+"-"+str(end_date)[6:8]
        start_time = df_X['hora'].min()
        start_time_str = str(datetime.datetime.strptime(str(int(math.floor(start_time)*100 + start_time%1*60)), '%H%M').time())
        end_time = df_X['hora'].max()
        end_time_str = str(datetime.datetime.strptime(str(int(math.floor(end_time)*100 + end_time%1*60)), '%H%M').time())
        group_by_codigos = df_X.groupby('codigo')        
        groups_df = [g for k,g in group_by_codigos]
                
        def indexedDataframe(startdate,enddate,starttime,endtime):
            periode_parameters = pd.date_range(startdate,enddate,freq="30Min")
            periode_parameters = periode_parameters[periode_parameters.indexer_between_time(starttime,endtime)].to_pydatetime()
            df_time_index = pd.DataFrame(index=periode_parameters,columns=['fecha','hora'])
            df_time_index['fecha'] = df_time_index.index.date
            df_time_index['hora'] = df_time_index.index.time
            df_time_index['year'] = df_time_index.index.year
            df_time_index['fecha'] = df_time_index['fecha'].apply(lambda x:int("{:%Y%m%d}".format(x)))
            df_time_index['hora'] = df_time_index['hora'].apply(lambda x:int("{:%H%M}".format(x)))
            df_time_index['hora'] = df_time_index[['hora']]//100+(df_time_index[['hora']]%100/60)
            df_time_index['fecha'] = pd.to_numeric(df_time_index['fecha'],downcast='signed')
            df_time_index['hora'] = pd.to_numeric(df_time_index['hora'],downcast='float')
            df_time_index['year'] = pd.to_numeric(df_time_index['year'],downcast='signed')
            return df_time_index.reset_index(drop=True)
        df_index = indexedDataframe(start_date_str,end_date_str,start_time_str,end_time_str)        
        for i in range(0,len(groups_df)):
            df_con = pd.merge(df_index,groups_df[i],on=['fecha','hora','year'],how='inner')
            df_index = df_con
            df_con.drop_duplicates(subset=['fecha','hora','year'], keep='first', inplace=True)
        return df_con
    
    def _rename_df(self,df_X,feature_list):
        column_labels = ['fecha','hora','year']
        n_features = len(feature_list)
        for i in range(0,df_X.shape[1]-3,(n_features+1)):
            codigo = df_X.iloc[0,i+3]
            for f in feature_list:
                column_labels.append(codigo+f)              

        df_X.drop(['codigo_x','codigo_y'],axis=1,inplace=True)        
        df_X.columns = column_labels        
        
        return df_X

    def _add_target(self, df_X, df_y, df_y_persist, feature_list, n_codigos):
        df_X['year']=pd.to_numeric(df_X['year'], downcast='signed')
        df_X['fecha']=pd.to_numeric(df_X['fecha'], downcast='signed')
        df_y['year']=pd.to_numeric(df_y['year'], downcast='signed')
        df_y['fecha']=pd.to_numeric(df_y['fecha'], downcast='signed')
        df_y['hora'] = df_y[['hora']]//100+(df_y[['hora']]%100/60)
        df_y['hora']=pd.to_numeric(df_y['hora'], downcast='float')
        df_y['elevation']=pd.to_numeric(df_y['elevation'], downcast='float')
        df_y_persist['year']=pd.to_numeric(df_y_persist['year'], downcast='signed')
        df_y_persist['fecha']=pd.to_numeric(df_y_persist['fecha'], downcast='signed')
        df_y_persist['hora']=pd.to_numeric(df_y_persist['hora'], downcast='float')
        p_cols = ['year','fecha','hora']
        for fc in self.forecast_horizon:
            p_cols.append("p_"+str(fc))
        df_y_persist = df_y_persist[p_cols]                
        elevation = df_y[['year','fecha','hora','elevation']] 
        Y = df_y[['year','fecha','hora','rel_rad']]        
        for h in [0.5 if x<3 else 1.0 for x in self.forecast_horizon]:
            elevation['hora'] = elevation[['hora']]-h
            df_X = pd.merge(df_X, elevation, on=["year","fecha","hora"], how='inner')
        for h in [0.5 if x<3 else 1.0 for x in self.forecast_horizon]:
            Y['hora'] = Y[['hora']]-h
            df_X = pd.merge(df_X, Y, on=["year","fecha","hora"], how='inner')
        new_labels = list(df_X.columns[:-2*len(self.forecast_horizon)])        
        for fc in self.forecast_horizon:
            new_labels.append(("elevation_"+str(fc)+"h"))
        for fc in self.forecast_horizon:
            new_labels.append((str(fc)+"h"))
        df_X.columns = new_labels
        df_X = pd.merge(df_X, df_y_persist, on=["year","fecha","hora"], how='inner')
        return df_X

    def _split_data(self, df_Xy):
        if (self.start_date_train < self.start_date_test):
            df_Xy_train = df_Xy[df_Xy['fecha']<self.start_date_test]
            df_Xy_test = df_Xy[df_Xy['fecha']>self.end_date_train]
        else:
            df_Xy_train = df_Xy[df_Xy['fecha']>self.end_date_test]
            df_Xy_test = df_Xy[df_Xy['fecha']<self.start_date_train]
        df_Xy.drop(["fecha","hora","year"], axis=1, inplace=True)
        df_Xy = df_Xy.sample(frac=1).reset_index(drop=True)
        df_Xy_train.drop(["fecha","hora","year"], axis=1, inplace=True)
        df_Xy_train = df_Xy_train.sample(frac=1).reset_index(drop=True)
        df_Xy_test.drop(["fecha","hora","year"], axis=1, inplace=True)
        df_Xy_test = df_Xy_test.sample(frac=1).reset_index(drop=True)
        print("Number of training samples: "+str(df_Xy_train.shape[0]))
        print("Number of test samples: "+str(df_Xy_test.shape[0]))        
        df_X = df_Xy.drop(df_Xy.columns[-2*len(self.forecast_horizon):],axis=1)      
        df_y = df_Xy.drop(df_Xy.columns[:-2*len(self.forecast_horizon)],axis=1)        
        df_X_train = df_Xy_train.drop(df_Xy_train.columns[-2*len(self.forecast_horizon):],axis=1)
        df_y_train = df_Xy_train.drop(df_Xy_train.columns[:-2*len(self.forecast_horizon)],axis=1)
        df_y_train = df_y_train.fillna(0)
        df_X_test = df_Xy_test.drop(df_Xy_test.columns[-2*len(self.forecast_horizon):],axis=1) 
        df_y_test = df_Xy_test.drop(df_Xy_test.columns[:-2*len(self.forecast_horizon)],axis=1)
        df_y_test = df_y_test.fillna(0)
        return df_X, df_y, df_X_train, df_y_train, df_X_test, df_y_test
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', nargs='?', action="store", dest="config_file", help="File with the parameters for generating training and test set.")
    parser.add_argument('--original-folder', dest="orig_folder", action="store", help="The folder where the original data is.")
    parser.add_argument('--destination-folder', dest="dest_folder", action="store", help="Folder to save the data.")
    parser.add_argument('--X-y-files', dest="X_y_files", action="store", help="List with names of original X and y csv-files.", type=list)
    parser.add_argument('--start-date-train', dest="start_date_train", action="store", help="The start date for the training set. (YYYYMMDD).", type=int)
    parser.add_argument('--end-date-train', dest="end_date_train", action="store", help="The end date for the training set. (YYYYMMDD).", type=int)
    parser.add_argument('--start-date-test', dest="start_date_test", action="store", help="The start date for the test set. (YYYYMMDD).", type=int)
    parser.add_argument('--end-date-test', dest="end_date_test", action="store", help="The end date for the test set. (YYYYMMDD).", type=int)
    parser.add_argument('--n-samples', dest="n_samples", action="store", help="N_samples is 1,2 or 3 - number of radiation samples per station", type=int)
    parser.add_argument('--radiation-differences', dest="radiation_differences", action="store", help="give differences between two radiation samples instead of absolute value", type=bool)
    parser.add_argument('--include-azimuth-last-sample', dest="include_azimuth_last_sample", action="store", help="include azimuth angle for sample farthest from prediction time", type=bool)
    parser.add_argument('--forecast-horizon', dest="forecast_horizon", action="store", help="Choose between one and 6 forecast horizon(s) out of 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 hours", type=list)


    arguments = parser.parse_args()

    config_file = arguments.config_file    
    orig_folder = arguments.orig_folder
    dest_folder = arguments.dest_folder
    X_y_files = arguments.X_y_files
    start_date_train = arguments.start_date_train
    end_date_train = arguments.end_date_train
    start_date_test = arguments.start_date_test
    end_date_test = arguments.end_date_test
    n_samples = arguments.n_samples
    radiation_differences = arguments.radiation_differences
    include_azimuth_last_sample = arguments.include_azimuth_last_sample
    forecast_horizon = arguments.forecast_horizon
    

    traintestsetgenerator = TrainTestSetGenerator(orig_folder, dest_folder, X_y_files, start_date_train, end_date_train, start_date_test, end_date_test, n_samples, radiation_differences, include_azimuth_last_sample, forecast_horizon, config_file)
    traintestsetgenerator.concatenate_data()
