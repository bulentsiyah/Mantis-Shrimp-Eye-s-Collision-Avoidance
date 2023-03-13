import cv2
import time
import sys
from datetime import datetime
import os
import sys
import pandas as pd
import copy

from collisioncalculationcontext import RightDetection

import sys
sys.path.append('tools')
from tools import ResultTypes

class DNNCompare:

    def __init__(self, camera_parameters, configurationManager):
        self.configurationManager = configurationManager
        self.camera_parameters = camera_parameters

        video_path_file= self.configurationManager .config_changeable['video_path_file']
        self.flight_id, ext = os.path.splitext(os.path.basename(video_path_file))

        #ucusun kendı dnn dataframe
        path_flight_dnn = self.configurationManager .config_readable['dnn_dataset']+self.flight_id+".csv"
        self.flight_dnn_dataframe =pd.read_csv(path_flight_dnn)

        #pred dnn hazılrık
        self.output_folder = self.configurationManager.config_readable['output_path_folder']
        self.output_folder_flight = os.path.join(self.output_folder+self.flight_id+"/")
        self.path_dnn_pred_dataframe = self.output_folder_flight+'/dnn_pred_dataframe.csv'

        self.dnn_pred_dataframe =None

        self.list_pred_before_dataframe = []

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
        self.list_pred_before_dataframe.append(list)
        


    def save_dnn_pred_path(self,context_return):
        '''self.dnn_pred_dataframe = self.dnn_pred_dataframe.append({'object_type': object_type,
                                                                    'frame_id':frame_id,
                                                                    'left': left,
                                                                    'top':top,
                                                                    'width': width, 
                                                                    'height': height,
                                                                    'area': area,
                                                                    'range_distance': range_distance,
                                                                    'confidence':confidence
                                                                    }, 
                                                                    ignore_index=True, verify_integrity=False,
                                                                    sort=False)'''
        
        df_cols = ['object_type', 'frame_id','left','top','width','height','area','range_distance', 'confidence']
        
        self.dnn_pred_dataframe = pd.DataFrame(self.list_pred_before_dataframe,columns=df_cols)
        self.dnn_pred_dataframe.to_csv(self.path_dnn_pred_dataframe, index=False)

        self.save_dnn_result_path(context_return=context_return)


    def save_dnn_result_path(self,context_return):
        df_result_cols = ['frame_id', 'result_type','dnn_object_type', 'dnn_width','dnn_height','dnn_range_distance', 'pred_object_type', 'pred_width','pred_height','pred_range_distance','pred_confidence']
        list_before_result_dataframe = []

        flight_dnn_dataframe_frame_id = self.flight_dnn_dataframe['frame_id'].tolist() 
        flight_dnn_dataframe_frame_id = [ int(x) for x in flight_dnn_dataframe_frame_id ]

        dnn_pred_dataframe_frame_id= self.dnn_pred_dataframe['frame_id'].tolist() 
        dnn_pred_dataframe_frame_id = [ int(x) for x in dnn_pred_dataframe_frame_id ]

        main_list = copy.deepcopy(flight_dnn_dataframe_frame_id)

        for element in dnn_pred_dataframe_frame_id:
            element = int(element)
            if element not in main_list:
                main_list.append(element)


        main_list = sorted(main_list)

        kac_karsilasma = 0
        ucus_var_tahmin_yok = 0
        ucus_yok_tahmin_var = 0
        max_range = 0
        for element in main_list:
            element = int(element)
            frame_id=""
            result_type=""

            dnn_object_type=""
            dnn_width=""
            dnn_height=""
            dnn_range_distance=""

            pred_object_type=""
            pred_width=""
            pred_height=""
            pred_range_distance=""
            pred_confidence = ""

            kac_karsilasma = kac_karsilasma+1
            if element not in flight_dnn_dataframe_frame_id: # flight yok-preddte var--helıkopter yok sen var demıssın 
                ucus_yok_tahmin_var = ucus_yok_tahmin_var + 1
                row_dnn_pred_dataframe= self.dnn_pred_dataframe[self.dnn_pred_dataframe["frame_id"]==str(element)] #self.dnn_pred_dataframe.iloc[element]
                frame_id=row_dnn_pred_dataframe["frame_id"].values[0]
                result_type = ResultTypes(2).name #FalsePositive
                pred_object_type=row_dnn_pred_dataframe["object_type"].values[0]
                pred_width=row_dnn_pred_dataframe["width"].values[0]
                pred_height=row_dnn_pred_dataframe["height"].values[0]
                pred_range_distance=row_dnn_pred_dataframe["range_distance"].values[0]
                pred_confidence = row_dnn_pred_dataframe["confidence"].values[0]
            elif element not in dnn_pred_dataframe_frame_id: # flight var--predte yok--helıkopter var sen yok demıssın
                ucus_var_tahmin_yok = ucus_var_tahmin_yok + 1
                row_flight_dnn_dataframe= self.flight_dnn_dataframe[self.flight_dnn_dataframe["frame_id"]==element]  #self.flight_dnn_dataframe.iloc[element]
                frame_id=row_flight_dnn_dataframe["frame_id"].values[0]
                result_type = ResultTypes(1).name #FalseNegative
                dnn_object_type=row_flight_dnn_dataframe["object_type"].values[0]
                dnn_width=row_flight_dnn_dataframe["width"].values[0]
                dnn_height=row_flight_dnn_dataframe["height"].values[0]
                dnn_range_distance=row_flight_dnn_dataframe["range_distance"].values[0]
            else:
                row_flight_dnn_dataframe= self.flight_dnn_dataframe[self.flight_dnn_dataframe["frame_id"]==element]
                row_dnn_pred_dataframe= self.dnn_pred_dataframe[self.dnn_pred_dataframe["frame_id"]==str(element)]
                frame_id=row_flight_dnn_dataframe["frame_id"].values[0]
                result_type = ResultTypes(0).name #TruePositive

                #dnn
                dnn_object_type=row_flight_dnn_dataframe["object_type"].values[0]
                dnn_width=row_flight_dnn_dataframe["width"].values[0]
                dnn_height=row_flight_dnn_dataframe["height"].values[0]
                dnn_range_distance=row_flight_dnn_dataframe["range_distance"].values[0]

                #pred
                pred_object_type=row_dnn_pred_dataframe["object_type"].values[0]
                pred_width=row_dnn_pred_dataframe["width"].values[0]
                pred_height=row_dnn_pred_dataframe["height"].values[0]
                pred_range_distance=row_dnn_pred_dataframe["range_distance"].values[0]
                pred_confidence = row_dnn_pred_dataframe["confidence"].values[0]

                if max_range < float(dnn_range_distance):
                    max_range = dnn_range_distance


            list = [frame_id, result_type, dnn_object_type, dnn_width,dnn_height,dnn_range_distance,pred_object_type,pred_width,pred_height,pred_range_distance,pred_confidence]
            list_before_result_dataframe.append(list)


        dnn_result_dataframe = pd.DataFrame(list_before_result_dataframe,columns=df_result_cols)
        dnn_result_dataframe.to_csv(self.output_folder_flight+'/dnn_result_dataframe.csv', index=False)

        #tüm train operasyonu ana dataframei
        path_dnn_ana_result_dataframe = self.output_folder+'/dnn_ana_result_dataframe.csv'
        try:
            dnn_ana_result_dataframe =pd.read_csv(path_dnn_ana_result_dataframe)
        except:
            df_ana_result_cols = ['olusturulma_zamani', 'flight_id','toplam_frame','kac_karsilasma', 'ucus_var_tahmin_yok','ucus_yok_tahmin_var','max_range', 'fps', 'camera_size','model']
            dnn_ana_result_dataframe = pd.DataFrame(columns=df_ana_result_cols)


        olusturulma_zamani = datetime.now()
        flight_id = self.flight_id
        toplam_frame=context_return.method_call_number
        kac_karsilasma=kac_karsilasma
        ucus_var_tahmin_yok=ucus_var_tahmin_yok
        ucus_yok_tahmin_var=ucus_yok_tahmin_var
        max_range=max_range
        fps=context_return.method_fps
        camera_size="camera_type:",self.camera_parameters.camera_type , " width:" , self.camera_parameters.width , " height:" ,self.camera_parameters.height

        right_detection_model= self.configurationManager.config_readable['right_detection_model']
        right_detection_model= os.path.basename(right_detection_model)
        model=right_detection_model

        dnn_ana_result_dataframe = dnn_ana_result_dataframe.append({'olusturulma_zamani': str(olusturulma_zamani),
                                                                    'flight_id': str(flight_id),
                                                                        'toplam_frame': str(toplam_frame),
                                                                        'kac_karsilasma': str(kac_karsilasma),
                                                                        'ucus_var_tahmin_yok': str(ucus_var_tahmin_yok),
                                                                        'ucus_yok_tahmin_var': str(ucus_yok_tahmin_var),
                                                                        'max_range': str(max_range),
                                                                        'fps': str(fps),
                                                                        'camera_size': str(camera_size),
                                                                        'model': str(model)
                                                                        }, 
                                                                        ignore_index=True, verify_integrity=False,
                                                                                sort=False)

        dnn_ana_result_dataframe.to_csv(path_dnn_ana_result_dataframe, index=False)

        
