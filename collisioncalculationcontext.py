
import time
from ultralytics import YOLO
import cv2

import sys
sys.path.append('tools')
from configmanager import ConfigurationManager

from tools import DrawingOpencv, Utils, ObjectTypes

class RightDetection:
    def __init__(self,risk_situation=False,confidence="0",object_type="None",risk_factor_x_min=0,risk_factor_y_min=0,risk_factor_x_max=0,risk_factor_y_max=0,range_distance=0):
        self.object_type = object_type
        self.confidence= confidence
        self.risk_situation = risk_situation
        self.risk_factor_x_min = risk_factor_x_min
        self.risk_factor_y_min = risk_factor_y_min
        self.risk_factor_x_max = risk_factor_x_max
        self.risk_factor_y_max = risk_factor_y_max
        self.range_distance = range_distance

class ContextReturn:
        
    def __init__(self):
        self.frame = None
        self.return_call_time = None
        self.right_detection = RightDetection()
        self.method_fps = 0
        self.method_call_number = 0
        

class CollisionCalculationContext:

    def __init__(self, configurationManager, camera_parameters, visual_drawing=False):
        
        self.configurationManager = configurationManager
        self.camera_parameters = camera_parameters 

        self.context_return = ContextReturn()
        self.call_time_for_context_return = time.time() 
        self.frame = None

        self.visual_drawing = visual_drawing

        self.right_detection_model = YOLO(self.configurationManager.config_readable['right_detection_model']) 

        self.__start_time= time.time()
        self.__method_call_number = 0
        self.__method_fps = 0

        self.__run_rightDetectionAnalysis= True
        
        
    def context(self, frame):

        self.frame = frame

        self.__method_call_number = self.__method_call_number+1
        
        # Region sana belirli sıklıkta cevap verebilirim
        '''temp_diff_time = time.time() - self.__method_last_call_time
        if temp_diff_time< self.context_treshold_fps_frq:
            if self.context_return.frame is None:
                self.context_return.frame = frame
            self.context_return.return_call_time= 0
            print("----1 call_frq_time < self.treshold_fps_frq------:",temp_diff_time)
            return self.context_return'''

        
        #self.__method_last_call_time= time.time()

        if self.frame is None:
            print("black pixel")
            return self.context_return

        # işlem başlıyor
        
        # bunu ilerde async task donustur
        if self.__run_rightDetectionAnalysis:
            frame, right_detection=self.rightDetectionAnalysis(frame=self.frame)
            self.context_return.frame = frame
            self.context_return.right_detection = right_detection

        end_time = time.time()
        duration = end_time - self.__start_time
        if int(duration) % Utils.frame_fps_kac_saniye  == 0:
            self.__method_fps  = int(round(self.__method_call_number / duration) ) 
            self.__method_call_number =0
            self.__start_time = time.time()


        #return hazırlığı
        self.context_return.frame = self.frame
        self.context_return.return_call_time= self.call_time_for_context_return
        self.context_return.method_fps = self.__method_fps 
        self.context_return.method_call_number=self.context_return.method_call_number + 1

                
        return self.context_return
    

    def rightDetectionAnalysis(self, frame):

        try:
            right_detection= RightDetection()
            results = self.right_detection_model.predict(frame, device="0", show=False ) # class=[0,2,3]--- hide_labels ---hide_conf
            howmany_haveyougot=results[0].boxes.boxes

            for i in howmany_haveyougot:
                xmin,ymin,xmax,ymax,confidence,class_id=i
                x1=int(xmin) 
                y1=int(ymin)
                x2=int(xmax)
                y2=int(ymax)
                confidence=float(confidence)
                class_id=int(class_id)

                enum_value = ObjectTypes(class_id).name

                if confidence >= Utils.yolo_confidence:
                    right_detection=RightDetection(risk_situation=True,confidence=str(confidence),object_type=enum_value,risk_factor_x_min=float(x1),risk_factor_y_min=float(y1),risk_factor_x_max=float(x2),risk_factor_y_max=float(y2), range_distance=0)
                    if self.visual_drawing:
                        DrawingOpencv.drawing_rectangle(frame, enum_value, (x1,y1), (x2,y2))
                        # cv2.imwrite("test.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        

            return frame, right_detection
        except:
            print("exception: def rightDetectionAnalysis")
            pass
