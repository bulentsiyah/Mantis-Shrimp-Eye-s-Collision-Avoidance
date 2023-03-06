import cv2
import time
import sys
from datetime import datetime
import os
import sys
import pandas as pd
import copy

from collisioncalculationcontext import RightDetection

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

        self.df_cols = ['object_type', 'frame_id','left','top','width','height','area','range_distance', 'confidence']
        self.dnn_pred_dataframe = pd.DataFrame(columns=self.df_cols)

        self.dnn_pred_dataframe.to_csv(self.path_dnn_pred_dataframe , index=False)

        self.dnn_pred_dataframe =pd.read_csv(self.path_dnn_pred_dataframe )

        self.list_before_dataframe = []

        self.right_detection = RightDetection()
        


    def save_dnn_pred(self, right_detection, frame_id, scale):

        right_detection = copy.deepcopy(right_detection)

        self.right_detection = RightDetection(risk_situation=right_detection.risk_situation,
                                              confidence=right_detection.confidence,
                                              object_type=right_detection.object_type,
                                              risk_factor_x_min=right_detection.risk_factor_x_min,
                                              risk_factor_y_min=right_detection.risk_factor_y_min,
                                              risk_factor_x_max=right_detection.risk_factor_x_max,
                                              risk_factor_y_max=right_detection.risk_factor_y_max,
                                              range_distance=right_detection.range_distance)

        risk_factor_x_min = self.right_detection.risk_factor_x_min*scale
        risk_factor_y_min = self.right_detection.risk_factor_y_min*scale

        risk_factor_x_max = self.right_detection.risk_factor_x_max*scale
        risk_factor_y_max = self.right_detection.risk_factor_y_max*scale

        object_type = str(self.right_detection.object_type)
        frame_id = str(frame_id)
        left = str(risk_factor_x_min)
        top = str(risk_factor_y_min)

        width = (risk_factor_x_max)-(risk_factor_x_min)
        height = (risk_factor_y_max)-(risk_factor_y_min)
        area= str(width * height)

        width = str(width)
        height = str(height)

        range_distance = str(self.right_detection.range_distance)
        confidence= str(self.right_detection.confidence)
        list = [object_type, frame_id, left,top, width,height,area,range_distance,confidence]
        self.list_before_dataframe.append(list)
        


    def save_dnn_pred_path(self,):
        '''self.dnn_pred_dataframe = self.dnn_pred_dataframe.append({'object_type': object_type,
                                                                        'frame_id':frame_id,
                                                                        'left': left,
                                                                        'top':top,
                                                                        'width': width, 
                                                                        'height': height,
                                                                        'area': area,
                                                                        'range_distance': range_distance,
                                                                        'confidence':confidence}, 
                                                                        ignore_index=True, verify_integrity=False,
                                                                                sort=False)'''
        
        self.dnn_pred_dataframe = pd.DataFrame( self.list_before_dataframe,columns=self.df_cols)
        self.dnn_pred_dataframe.to_csv(self.path_dnn_pred_dataframe, index=False)

        
