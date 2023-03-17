import tensorflow as tf
from tensorflow import keras
import joblib
#from sklearn.externals import joblib
import numpy as np
import pandas as pd

class RNNClass:

    def __init__(self, configurationManager):

        #model=None, config_path=None

        self.configurationManager = configurationManager

        self.path = self.configurationManager.config_readable['rnn_trajectory_model_folder']
        self.model= keras.models.load_model(self.path+"lstm_x_center.h5")
        print("Loaded keras rnn model from disk")
        self.scalar = joblib.load(self.path+"x_center_transform.save")
        self.scalar.clip = False

        '''dataset_train = pd.read_csv(self.path+'Google_Stock_Price_Train.csv')
        dataset_train = dataset_train['Open']
        self.inputs = dataset_train.values
        self.inputs = self.inputs.reshape(-1,1)'''

  
    def pred(self, df_rnn, pred_str):

        #df_rnn = df_rnn[pred_str]
        df_rnn = list(df_rnn)
        elon_array = np.array(df_rnn)

        
        #inputs = elon_array.values()
        inputs = elon_array.reshape(-1,1)
    
        test_set_scaled = self.scalar.transform(inputs)
        
        X_Test = []

        X_Test.append(test_set_scaled[0:len(test_set_scaled), 0])

        X_Test = np.array(X_Test)

        
        X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], 1))

        y_pred = self.model.predict(X_Test)

        y_pred = self.scalar.inverse_transform(y_pred)
    

        return y_pred