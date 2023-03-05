import cv2
import time
import sys
from datetime import datetime
import os
import sys
import pandas as pd
import copy


class DNNCompare:

    def __init__(self, configurationManager):
        self.configurationManager = configurationManager

        video_path_file= self.configurationManager .config_readable['video_path_file']
        flight_id, ext = os.path.splitext(os.path.basename(video_path_file))

        #ucusun kendı dnn dataframe
        path_flight_dnn = self.configurationManager .config_readable['dnn_dataset']+flight_id+".csv"
        self.flight_dnn =pd.read_csv(path_flight_dnn)
        '''print(len(self.flight_dnn))
        print(self.flight_dnn.head())'''

        #pred dnn hazılrık
        temp_output_folder = os.path.join(self.configurationManager.config_readable['output_path_folder']+flight_id+"/")
        self.path_dnn_pred_dataframe = temp_output_folder+'/dnn_pred_dataframe.csv'

        df_cols = ['object_type', 'frame_id','left','top','width','height','area','range_distance', 'confidence']
        self.dnn_pred_dataframe = pd.DataFrame(columns=df_cols)

        self.dnn_pred_dataframe.to_csv(self.path_dnn_pred_dataframe , index=False)

        self.dnn_pred_dataframe =pd.read_csv(self.path_dnn_pred_dataframe )


    def save_dnn_pred(self, right_detection, frame_id, scale):

        width = (right_detection.risk_factor_x_max*scale)-(right_detection.risk_factor_x_min*scale)
        height = (right_detection.risk_factor_y_max*scale)-(right_detection.risk_factor_y_min*scale)
        area= width * height
        self.dnn_pred_dataframe = self.dnn_pred_dataframe.append({'object_type': str(right_detection.object_type),
                                                                        'frame_id':str(frame_id),
                                                                        'left': str(right_detection.risk_factor_x_min*scale),
                                                                        'top':str(right_detection.risk_factor_y_min*scale),
                                                                        'width': str(width), 
                                                                        'height': str(height),
                                                                        'area': str(area),
                                                                        'range_distance': str(right_detection.range_distance),
                                                                        'confidence': str(right_detection.confidence)}, 
                                                                        ignore_index=True, verify_integrity=False,
                                                                                sort=False)


    def save_dnn_pred_path(self,):
        self.dnn_pred_dataframe.to_csv(self.path_dnn_pred_dataframe, index=False)

        
