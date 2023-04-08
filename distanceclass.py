
import joblib
#from tensorflow.keras.models import model_from_json

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import sys
sys.path.append('tools')
from configmanager import ConfigurationManager

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DistanceClass:

    def __init__(self, configurationManager):

        self.configurationManager = configurationManager

        self.dnn_distance_model_folder=self.configurationManager.config_readable['dnn_distance_model_folder']
        self.dnn_distance_model=self.configurationManager.config_readable['dnn_distance_model']


        # load weights into new model
        model_path = self.dnn_distance_model

        self.loaded_model = load_model(model_path)
        print("Loaded Dnn model from disk")

        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        self.scalar_X = joblib.load(self.dnn_distance_model_folder+"/xtrain_scaler.save")
        
        self.scalar_y = joblib.load(self.dnn_distance_model_folder+"/ytrain_scaler.save")

        
    def distance_single_prediction(self,xmin, ymin, xmax, ymax, width,height,class_type):
        # get data
        
        X_test = [[width, height, class_type]]
        #print("X_test",X_test)
        # standardized data
        #scalar = StandardScaler()
   
        X_test = self.scalar_X.transform(X_test)
        #print("scalar.transform(X_test)",X_test)

        # evaluate loaded model on test data
        #self.loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        
        '''    temp = [[xmin]]
        print("temp",temp)
        temp = scalar.transform(temp)
        temp = scalar.transform(temp)
        print("scalar.transform(temp)",temp)
        temp = scalar.inverse_transform(temp)
        print("scalar.inverse_transform(temp)",temp)'''
        

        y_pred = self.loaded_model.predict(X_test)
        #print("loaded_model.predict(X_test)",y_pred)
        
        

        # scale up predictions to original values
        y_pred = self.scalar_y.inverse_transform(y_pred)
        #print("scalar.inverse_transform(y_pred)",y_pred)
        
        return y_pred[0][0]