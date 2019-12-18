"""This module searches the best hyperparameters in a machine learning model"""

import json
import os
import time
import argparse
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.externals import joblib

class Train(object):
    """this class trains a machine learning model with given hyperparameters"""
       
    train_test_data_path = None
    train_files = None    
    estimator_model = None
    model_destination_folder = None    
    config_file = None
    verbose = True
    
    def __init__(self, train_test_data_path=None, train_files=None, estimator_model = None, model_destination_folder=None, config_file=None, verbose=True):

        self.train_test_data_path = train_test_data_path
        self.train_files = train_files        
        self.estimator_model = estimator_model
        self.model_destination_folder = model_destination_folder              
        self.config_file = config_file
        self.verbose = verbose
        self.parse_config_file()
        
    def fit(self):
        """This method fits the training data to the model"""        
        df_X_train, df_y_train = self._get_csv_files()        
        self._train_model(df_X_train, df_y_train)                 
        
    def _get_csv_files(self):
        X_train_path = os.path.join(self.train_test_data_path, self.train_files[0])
        df_X_train = pd.read_csv(X_train_path)
        y_train_path = os.path.join(self.train_test_data_path, self.train_files[1])
        df_y_train = pd.read_csv(y_train_path)
        df_y_train = df_y_train[df_y_train.columns[:df_y_train.shape[1]//2]]
        return df_X_train, df_y_train

    def _train_model(self, df_X_train, df_y_train):
        """This method trains two models: one with features from all selected stations the other one only with features from the target station (local model)
        For linear (ARX) and Random Forest (RRF) regression the feature importances are extracted and stored."""
        
        model_destination_folder = self.model_destination_folder
        if not os.path.isdir(model_destination_folder):
            os.makedirs(model_destination_folder)
        
        n_horizons = df_y_train.shape[1]
        n_features = df_X_train.shape[1]
        forecast_horizon = df_y_train.columns        
        X = np.array (df_X_train)
        y = np.array(df_y_train)                                            
                                                    
        # local model

        target_cols_ind = [col for col in df_X_train if col.startswith('VA01')]
        X_local = np.array(df_X_train[target_cols_ind])
        target_el_ind = [col for col in df_X_train if col.startswith('elevation')]
        target_elevation = np.array((df_X_train[target_el_ind]))                                      
        X_local = np.append(X_local, target_elevation,axis=1)
                
        if (self.estimator_model == 'ARX'):
            print("training: ARX")
            feature_importances = np.zeros([n_features,n_horizons])            
            for i in range (n_horizons):                                              
                lasso = linear_model.Lasso(alpha=0.0001, fit_intercept=True, random_state=7, max_iter=10000000)    
                lasso.fit(X,y[:,i])
                scores_ls = cross_val_score(lasso, X, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_ = (np.sqrt(-scores_ls))    
                feature_importances[:,i] = lasso.coef_
                
                print("\n")
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]))
                
                lasso_local = linear_model.Lasso(alpha=0.0001, fit_intercept=True, random_state=7, max_iter=10000000)
                lasso_local.fit(X_local,y[:,i])
                scores_ls_local = cross_val_score(lasso_local, X_local, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_local = (np.sqrt(-scores_ls_local))
                                
                print("mean RMSE on train set: ",scores_.mean())
                print("mean RMSE for local model on train set: ",scores_local.mean())            
                print("std error: ",scores_.std())
                print("std error for local model: ",scores_local.std())          
                print("----------------------------------------------------------------------------------")
                
                filename = os.path.join(self.model_destination_folder, self.estimator_model + "_"+str(forecast_horizon[i])+ ".pkl")
                filename_local = os.path.join(self.model_destination_folder, self.estimator_model + "_local_"+str(forecast_horizon[i])+ ".pkl")
                joblib.dump(lasso, filename)
                joblib.dump(lasso_local, filename_local)            
            np.save("./plots/ARX_feature_importances", np.array(feature_importances))

        if (self.estimator_model == 'RRF'):
            print("training: RRF")
            feature_importances = np.zeros([n_features,n_horizons])            
            for i in range (n_horizons):                                              
                forest_reg = RandomForestRegressor(max_features= df_X_train.shape[1]//3, min_samples_split= 2,n_estimators =100,n_jobs = -1)   
                forest_reg.fit(X,y[:,i])
                scores_fr = cross_val_score(forest_reg, X, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_ = (np.sqrt(-scores_fr))    
                feature_importances[:,i] = forest_reg.feature_importances_
                print("\n")
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]))               
            
                forest_reg_local = RandomForestRegressor( n_estimators =100,n_jobs = -1)
                forest_reg_local.fit(X_local,y[:,i])
                scores_fr_local = cross_val_score(forest_reg_local, X_local, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_local = (np.sqrt(-scores_fr_local))           
                              
                print("mean RMSE on train set: ",scores_.mean())
                print("mean RMSE for local model on train set: ",scores_local.mean())            
                print("std error: ",scores_.std())
                print("std error for local model: ",scores_local.std())              
                print("-----------------------------------------------------------------------------------")
                
                filename = os.path.join(self.model_destination_folder, self.estimator_model + "_"+str(forecast_horizon[i])+ ".pkl")
                filename_local = os.path.join(self.model_destination_folder, self.estimator_model + "_local_"+str(forecast_horizon[i])+ ".pkl")
                joblib.dump(forest_reg, filename)
                joblib.dump(forest_reg_local, filename_local)
            
            np.save("./plots/RRF_feature_importances", np.array(feature_importances))

        if (self.estimator_model == 'RT'):
            print("training: RT")                      
            for i in range (n_horizons):
                dtr = DecisionTreeRegressor(max_features= df_X_train.shape[1]//2)                  
                dtr.fit(X,y[:,i])
                scores_dtr = cross_val_score(dtr, X, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_ = (np.sqrt(-scores_dtr)) 
                print("\n")
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]))               
            
                dtr_local = DecisionTreeRegressor() 
                dtr_local.fit(X_local,y[:,i])
                scores_dtr_local = cross_val_score(dtr_local, X_local, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_local = (np.sqrt(-scores_dtr_local))           
                              
                print("mean RMSE on train set: ",scores_.mean())
                print("mean RMSE for local model on train set: ",scores_local.mean())            
                print("std error: ",scores_.std())
                print("std error for local model: ",scores_local.std())              
                print("--------------------------------------------------------------------------------")
                
                filename = os.path.join(self.model_destination_folder, self.estimator_model + "_"+str(forecast_horizon[i])+ ".pkl")
                filename_local = os.path.join(self.model_destination_folder, self.estimator_model + "_local_"+str(forecast_horizon[i])+ ".pkl")
                joblib.dump(dtr, filename)
                joblib.dump(dtr_local, filename_local)

        if (self.estimator_model == 'ANN'):
            print("training: ANN")                      
            for i in range (n_horizons):
                mlp = MLPRegressor(alpha=0.1, hidden_layer_sizes = (300,300,300,), max_iter = 10000, verbose = 'False', learning_rate = 'adaptive')                  
                mlp.fit(X,y[:,i])
                scores_mlp = cross_val_score(mlp, X, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_ = (np.sqrt(-scores_mlp)) 
                print("\n")
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]))               
            
                mlp_local = MLPRegressor(alpha=0.1, hidden_layer_sizes = (300,300,300,), max_iter = 10000,verbose = 'False', learning_rate = 'adaptive') 
                mlp_local.fit(X_local,y[:,i])
                scores_mlp_local = cross_val_score(mlp_local, X_local, y[:,i], scoring='neg_mean_squared_error',cv=10)                       
                scores_local = (np.sqrt(-scores_mlp_local))           
                              
                print("mean RMSE on train set: ",scores_.mean())
                print("mean RMSE for local model on train set: ",scores_local.mean())            
                print("std error: ",scores_.std())
                print("std error for local model: ",scores_local.std())              
                print("--------------------------------------------------------------------------------")
                
                filename = os.path.join(self.model_destination_folder, self.estimator_model + "_"+str(forecast_horizon[i])+ ".pkl")
                filename_local = os.path.join(self.model_destination_folder, self.estimator_model + "_local_"+str(forecast_horizon[i])+ ".pkl")
                joblib.dump(mlp, filename)
                joblib.dump(mlp_local, filename_local)

    def parse_config_file(self):
        """This method parses the self.config file and assigns the values to attributes"""

        if self.config_file is not None:

            if self.verbose:
                print ('Parsing config file')

            with open(self.config_file) as data_file:
                config_data = json.load(data_file)
            
            if "train_test_data_path" in config_data:
                self.train_test_data_path = config_data["train_test_data_path"]

            if "train_files" in config_data:
                self.train_files = config_data["train_files"]            

            if "estimator_model" in config_data:
                self.estimator_model = config_data["estimator_model"]

            if "model_destination_folder" in config_data:
                self.model_destination_folder = config_data["model_destination_folder"]

                      
        
def main():
    """main function of module"""

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config-file', nargs='?', action="store", dest="config_file", help="File with the parameters to fit a model on the training data.")
    parser.add_argument('--train-test_data_path', dest="train_test_data_path", action="store", help="The folder where the train and test data is located.")      
    parser.add_argument('--train-files', dest="train_files", action="store", help="List with names of training data and target.", type=list)
    parser.add_argument('--estimator-model', dest="estimator_model", action="store", help="Enter 'ARX' for linear regression model/ 'ANN' for multilayer perceptron/ 'RT' for binary tree/ 'RRF' for random forest regressor", type=string)  
    parser.add_argument('--model-destination-folder', dest="model_destination_folder", action="store", help="The folder where the trained model is located.")      
        
    arguments = parser.parse_args()

    config_file = arguments.config_file    
    train_test_data_path = arguments.train_test_data_path
    train_files = arguments.train_files    
    estimator_model = arguments.estimator_model
    model_destination_folder = arguments.model_destination_folder    
       

    train = Train(train_test_data_path, train_files, estimator_model, model_destination_folder)
    train.fit()
