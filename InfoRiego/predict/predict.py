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

class Predict(object):
    """this class trains a machine learning model with given hyperparameters"""
       
    train_test_data_path = None
    test_files = None    
    estimator_model = None
    trained_models_folder = None
    text_output_folder = None
    predictions_data_path = None
    config_file = None
    verbose = True
    
    def __init__(self, train_test_data_path=None, test_files=None, estimator_model=None, trained_models_folder=None, text_output_folder=None, predictions_data_path=None, config_file=None, verbose=True):

        self.train_test_data_path = train_test_data_path
        self.test_files = test_files        
        self.estimator_model = estimator_model
        self.trained_models_folder = trained_models_folder
        self.text_output_folder = text_output_folder
        self.predictions_data_path = predictions_data_path
        self.config_file = config_file
        self.verbose = verbose
        self.parse_config_file()
        
    def apply_model(self):
        """This method predicts the target with a pretrained model"""        
        df_X_test, df_y_test = self._get_csv_files()
        model_filenames = self._get_models(df_y_test)        
        self._make_predictions(df_X_test, df_y_test, model_filenames)                 
        
    def _get_csv_files(self):
        X_test_path = os.path.join(self.train_test_data_path, self.test_files[0])
        df_X_test = pd.read_csv(X_test_path)
        y_test_path = os.path.join(self.train_test_data_path, self.test_files[1])
        df_y_test = pd.read_csv(y_test_path)        
        return df_X_test, df_y_test
    
    def _get_models(self, df_y_test):
        n_horizons = df_y_test.shape[1]//2        
        forecast_horizon = df_y_test.columns[:df_y_test.shape[1]//2]       
        model_filenames = []
        for h in forecast_horizon:
            filename = os.path.join(self.trained_models_folder, self.estimator_model + "_"+str(h)+ ".pkl")
            model_filenames.append(filename)
        for h in forecast_horizon:
            filename_local = os.path.join(self.trained_models_folder, self.estimator_model + "_local_"+str(h)+ ".pkl")
            model_filenames.append(filename_local)
        return model_filenames
        

    def _make_predictions(self, df_X_test, df_y_test, model_filenames):
        text_output_folder = self.text_output_folder
        if not os.path.isdir(text_output_folder):
            os.makedirs(text_output_folder)
        predictions_data_path = self. predictions_data_path
        if not os.path.isdir( predictions_data_path):
            os.makedirs( predictions_data_path)
        out_file = os.path.join(self.text_output_folder, self.estimator_model + "_results.txt")
        out = open(out_file, "a")
        n_horizons = df_y_test.shape[1]//2        
        forecast_horizon = df_y_test.columns[:df_y_test.shape[1]//2]        
                                                    
        X_test = np.array(df_X_test)
        y_test = np.array(df_y_test)
                                              
        # local model

        target_cols_ind = [col for col in df_X_test if col.startswith('VA01')]
        X_local_test = np.array(df_X_test[target_cols_ind])
        target_el_ind_test = [col for col in df_X_test if col.startswith('elevation')]
        target_elevation_test = np.array((df_X_test[target_el_ind_test]))                                      
        X_local_test = np.append(X_local_test, target_elevation_test,axis=1)   
        
        for i in range (n_horizons):                                              
                model = joblib.load(model_filenames[i])
                local_model = joblib.load(model_filenames[i+n_horizons])             
                print("\n")
                print("\n", file=out)
                if((self.estimator_model)=='ANN'):
                    print("MODEL: Multilayer Perceptron")
                    print("MODEL: Multilayer Perceptron", file=out)
                if((self.estimator_model)=='ARX'):
                    print("MODEL: Linear Regression")
                    print("MODEL: Linear Regression", file=out)
                if((self.estimator_model)=='RT'):
                    print("MODEL: Decision Tree")
                    print("MODEL: Decision Tree", file=out)
                if((self.estimator_model)=='RRF'):
                    print("MODEL: Random Forest Regression")
                    print("MODEL: Random Forest Regression", file=out)
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]))
                print("FORECASTHORIZON: "+ str(forecast_horizon[i]),file=out)
                predictions = model.predict(X_test)
                filename = os.path.join(self.predictions_data_path, self.estimator_model + "_"+str(forecast_horizon[i]))
                np.save(filename, np.array(predictions))
                RMSE = np.sqrt(((np.square(y_test[:,i]-predictions)).sum())/y_test.shape[0])
                RMSE_per = np.sqrt(((np.square(y_test[:,i]-y_test[:,(i+n_horizons)])).sum())/y_test.shape[0])
                Skill = (1-(RMSE/RMSE_per))*100           
                                                     
                predictions_local = local_model.predict(X_local_test)
                filename_local = os.path.join(self.predictions_data_path, self.estimator_model + "_local_"+str(forecast_horizon[i]))
                np.save(filename_local, np.array(predictions_local))
                RMSE_local = np.sqrt(((np.square(y_test[:,i]-predictions_local)).sum())/y_test.shape[0])    
                Skill_local = (1-(RMSE_local/RMSE_per))*100                                        
                
                print("\n")
                print("\n",file=out)
                print("ALL FEATURES:")
                print("ALL FEATURES:", file=out)                 
                print("RMSE on test set: ",RMSE)
                print("RMSE on test set: ",RMSE, file=out)
                print('RMSE persistence model: ',RMSE_per)
                print('RMSE persistence model: ',RMSE_per, file=out)
                print("Skill in %: ",Skill)
                print("Skill in %: ",Skill, file=out)                
                print("\n")
                print("\n", file=out)
                print("LOCAL MODEL:")              
                print("LOCAL MODEL:", file=out) 
                print("RMSE_local on test set: ",RMSE_local)
                print("RMSE_local on test set: ",RMSE_local, file=out) 
                print("Skill in % for local model: ",Skill_local)
                print("Skill in % for local model: ",Skill_local, file=out)                                
                print("----------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------", file=out)
                                         
        out.close()
                
        
    def parse_config_file(self):
        """This method parses the self.config file and assigns the values to attributes"""

        if self.config_file is not None:

            if self.verbose:
                print ('Parsing config file')

            with open(self.config_file) as data_file:
                config_data = json.load(data_file)
            
            if "train_test_data_path" in config_data:
                self.train_test_data_path = config_data["train_test_data_path"]

            if "test_files" in config_data:
                self.test_files = config_data["test_files"]            

            if "estimator_model" in config_data:
                self.estimator_model = config_data["estimator_model"]                

            if "trained_models_folder" in config_data:
                self.trained_models_folder = config_data["trained_models_folder"]

            if "text_output_folder" in config_data:
                self.text_output_folder = config_data["text_output_folder"]

            if "predictions_data_path" in config_data:
                self.predictions_data_path = config_data["predictions_data_path"]
                      
        
def main():
    """main function of module"""

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config-file', nargs='?', action="store", dest="config_file", help="File with the parameters to predict on the test data.")
    parser.add_argument('--train-test_data_path', dest="train_test_data_path", action="store", help="The folder where the train and test data is located.")      
    parser.add_argument('--test-files', dest="test_files", action="store", help="List with names of test data and target.", type=list)
    parser.add_argument('--estimator-model', dest="estimator_model", action="store", help="Enter 'ARX' for linear regression model/ 'ANN' for multilayer perceptron/ 'RT' for binary tree/ 'RRF' for random forest regressor", type=string)  
    parser.add_argument('--trained-models-folder', dest="trained_models_folder", action="store", help="The folder where the pretrained models are located.")
    parser.add_argument('--text-output-folder', dest="text_output_folder", action="store", help="The folder where the text file with the results of the validation wil be placed.")
    parser.add_argument('--predictions-data-path', dest="predictions_data_path", action="store", help="The folder where the csv files with the predictions wil be placed.")
        
    arguments = parser.parse_args()

    config_file = arguments.config_file    
    train_test_data_path = arguments.train_test_data_path
    test_files = arguments.test_files    
    estimator_model = arguments.estimator_model
    trained_models_folder = arguments.trained_models_folder
    text_output_folder = arguments.text_output_folder
    predictions_data_path = arguments.predictions_data_path
    

    predict = Predict(train_test_data_path, test_files, estimator_model, trained_models_folder, text_output_folder, predictions_data_path)
    predict.apply_model()
