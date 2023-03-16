
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import model_from_json,Model, load_model
from keras.layers import Dense, Input, Activation, BatchNormalization,Add,Dropout
import matplotlib.pyplot as plt

class DistanceClass:

    def __init__(self, dnn_distance_model=None, models_path_folder=None):

        self.path = models_path_folder
        self.model = dnn_distance_model

        # load weights into new model
        json_file = open(self.path+'dnn/{}.json'.format(self.model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json( loaded_model_json )
        model_path = self.path+"dnn/{}.h5".format(self.model)
        self.loaded_model.load_weights(model_path)
        print("Loaded model from disk")

        self.scalar_X = joblib.load(self.path+"dnn/xtrain_scaler.save")
        
        self.scalar_y = joblib.load(self.path+"dnn/ytrain_scaler.save")

        
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