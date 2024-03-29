
import time
from ultralytics import YOLO
import copy

from distanceclass import DistanceClass
from rnnclass import RNNClass
from collections import deque

import sys
sys.path.append('tools')
from configmanager import ConfigurationManager

from tools import DrawingOpencv, Utils, ObjectTypes

import cv2

class ContextReturn:
        
    def __init__(self):
        self.right_frame = None
        self.left_frame = None
        self.return_call_time = None
        self.right_detection = RightDetection()
        self.left_detection = LeftDetection()
        self.method_fps = 0
        self.method_call_number = 0

class LeftDetection:
    def __init__(self,motion_detection=None, risk_situation=False,confidence="0",object_type="None",risk_factor_x_min=0,risk_factor_y_min=0,risk_factor_x_max=0,risk_factor_y_max=0):
        self.motion_detection = motion_detection
        self.object_type = object_type
        self.confidence= confidence
        self.risk_situation = risk_situation
        self.risk_factor_x_min = risk_factor_x_min
        self.risk_factor_y_min = risk_factor_y_min
        self.risk_factor_x_max = risk_factor_x_max
        self.risk_factor_y_max = risk_factor_y_max
                     
class RightDetection:
    def __init__(self,risk_situation=False,confidence="0",object_type="None",risk_factor_x_min=0,risk_factor_y_min=0,risk_factor_x_max=0,risk_factor_y_max=0,range_distance=0,trajectory_pred_x_center=[],trajectory_pred_y_center=[]):
        self.object_type = object_type
        self.confidence= confidence
        self.risk_situation = risk_situation
        self.risk_factor_x_min = risk_factor_x_min
        self.risk_factor_y_min = risk_factor_y_min
        self.risk_factor_x_max = risk_factor_x_max
        self.risk_factor_y_max = risk_factor_y_max
        self.range_distance = range_distance
        self.trajectory_pred_x_center = trajectory_pred_x_center
        self.trajectory_pred_y_center = trajectory_pred_y_center


        

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

        self.__dnn_calculation = True
        if self.__dnn_calculation:
            self.distance_class = DistanceClass(configurationManager=configurationManager)


        self.__rnn_calculation = False
        self.__rnn_req_input_size = 30
        if  self.__rnn_calculation:
            self.rnn_class = RNNClass(configurationManager=configurationManager)
            self.deque_xcenter = deque(maxlen = self.__rnn_req_input_size)
            self.deque_ycenter = deque(maxlen = self.__rnn_req_input_size)


        self.__run_leftDetectionAnalysis= True
        self.__motion_calculation= True

        self.__motion_frame_1= None
        self.__motion_frame_2= None
        
        
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
            temp_frame, right_detection=self.rightDetectionAnalysis(frame=copy.copy(self.frame))
            self.context_return.right_frame = temp_frame
            self.context_return.right_detection = right_detection


        if self.__run_leftDetectionAnalysis:
            temp_frame, left_detection=self.leftDetectionAnalysis(frame=self.frame)
            self.context_return.left_frame = temp_frame
            self.context_return.left_detection = left_detection

            

        end_time = time.time()
        duration = end_time - self.__start_time
        if int(duration) % Utils.frame_fps_kac_saniye  == 0:
            self.__method_fps  = int(round(self.__method_call_number / duration) ) 
            self.__method_call_number =0
            self.__start_time = time.time()


        #return hazırlığı
        self.context_return.return_call_time= self.call_time_for_context_return
        self.context_return.method_fps = self.__method_fps 
        self.context_return.method_call_number=self.context_return.method_call_number + 1

                
        return self.context_return
    

    def leftDetectionAnalysis(self, frame):

        try: 

            left_detection= LeftDetection()

            if self.__motion_calculation:
                left_detection.motion_detection = self.leftMotionDetection(frame=frame)


            return frame, left_detection
        
        except Exception as e:
            print(f'caught {type(e)}: e')
            print("exception: def leftDetectionAnalysis")
            


    def leftMotionDetection(self, frame):
        try:

            if self.__motion_frame_1 is None:  
                self.__motion_frame_1 = frame

            if self.__motion_frame_2 is None:
                self.__motion_frame_2 = frame

            self.__motion_frame_2 = frame

            diff = cv2.absdiff(self.__motion_frame_1 , self.__motion_frame_2)
            #cv2.imshow("Diffrence",diff)
            
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            #blur = cv2.GaussianBlur(gray, (5,5), 0)
            #cv2.imshow("Blurred",blur)
            
            _,thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations = 1)
            
            contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #cv2.drawContours(frame1, contours, -1, color, 3)

            start_point_min_y = frame.shape[0]
            start_point_max_y = 0 #frame_height
            end_point_min_x = frame.shape[1]
            end_point_max_x = 0 #frame_width

            for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if start_point_min_y > y:
                        start_point_min_y = y

                    if start_point_max_y < (y+h):
                        start_point_max_y = (y+h)


                    if end_point_min_x > x:
                        end_point_min_x = x

                    if end_point_max_x < (x+w):
                        end_point_max_x = (x+w)

                    cv2.rectangle(self.__motion_frame_1 , (x, y), (x+w, y+h), (0, 255, 0), 2)

            #cv2.imshow("Feed",self.__motion_frame_1 )

            frame = copy.copy(self.__motion_frame_1)

            padding= 0
            frame = frame[start_point_min_y-padding:start_point_max_y+padding,end_point_min_x-padding:end_point_max_x+padding ] 
            
            self.__motion_frame_1 = copy.copy(self.__motion_frame_2)
  
            
            return frame
        
        except Exception as e:
            print(f'caught {type(e)}: e')
            print("exception: def leftMotionDetection")
    

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

                range_distance = 0
                trajectory_pred_x_center = []
                trajectory_pred_y_center = []

                enum_value = ObjectTypes(class_id).name

                if confidence >= Utils.yolo_confidence:
                    if self.__dnn_calculation:
                        range_distance  = self.distance_class.distance_single_prediction(xmin=x1,ymin=y1,xmax=x2,ymax=y2,width=x2-x1,height=y2-y1,class_type=class_id)
                        range_distance = range_distance 


                    if self.__rnn_calculation:
                        x_center = int((x1+x2)/2)
                        y_center = int((y1+y2)/2)

                        self.deque_xcenter.append(x_center)
                        self.deque_ycenter.append(y_center)

                        count_i = len(self.deque_xcenter)
                        if count_i >= self.__rnn_req_input_size:
                            trajectory_pred_x_center = self.rnn_class.pred(list_x_center=self.deque_xcenter,list_y_center= self.deque_xcenter,list_distance=self.deque_xcenter,pred_str='x_center')
                            trajectory_pred_y_center =  list(self.deque_ycenter)[0:10] #self.rnn_pred_x #self.rnn_class.pred(self.deque_ycenter, 'y_center')

                    right_detection=RightDetection(risk_situation=True,
                                                   confidence=str(confidence),
                                                   object_type=enum_value,
                                                   risk_factor_x_min=int(x1),
                                                   risk_factor_y_min=int(y1),
                                                   risk_factor_x_max=int(x2),
                                                   risk_factor_y_max=int(y2), 
                                                   range_distance=range_distance,
                                                   trajectory_pred_x_center=trajectory_pred_x_center,
                                                   trajectory_pred_y_center=trajectory_pred_y_center)
                    if self.visual_drawing:
                        DrawingOpencv.drawing_rectangle(frame, enum_value, (x1,y1), (x2,y2))
                        # cv2.imwrite("test.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        

            return frame, right_detection
        except Exception as e:
            print(f'caught {type(e)}: e')
            print("exception: def rightDetectionAnalysis")

