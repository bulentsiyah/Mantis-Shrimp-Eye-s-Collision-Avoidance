from datetime import datetime, timedelta
import cv2 
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
import math
from math import atan2, pi, cos, sin

import matplotlib.pyplot as plt
import numpy as np

from enum import Enum



@dataclass
class ResultTypes(Enum): # ilk kelime afferım bıldın ıkıncı neyi bıldın?
    TruePositive = 0 # gercekte var ve sende var demıssın
    FalseNagative = 1 # gercekte havada helıkopter var o sen yok demıssın--yanıldın
    FalsePositive = 2  # gercekte havada helıkopter yok o sen var demıssın--yanıldın
    TrueNegative = 3 # gercekte yok ve sende yok demıssın


    '''
    True Positive: Gerçekte var olan bir nesne doğru olarak sınıflandırılır.
    False Positive: Gerçekte yok olan bir nesne yanlış olarak var olarak sınıflandırılır.
    True Negative: Gerçekte yok olan bir nesne doğru olarak yok olarak sınıflandırılır.
    False Negative: Gerçekte var olan bir nesne yanlış olarak yok olarak sınıflandırılır.
        
    '''


@dataclass
class ObjectTypes(Enum):
    Airplane = 0
    Helicopter = 1
    Bird = 2
    Drone = 3
    Flock = 4
    Airborne = 5

class Utils:
    frame_fps_kac_saniye = 1
    yolo_confidence = 0.5
    camera_height_max = 2081
    camera_height_min = 480

class DrawingOpencv:
    """
    Tum ekrana cizim islemlerinin yapildigi siniftir.
    
    """

    visual_beacon_min_distance = 30
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
               'cv2.TM_SQDIFF_NORMED']
    methods_str = cv2.TM_CCOEFF_NORMED
    threshold_visual_vs_beacon = 3.09 # meter
    threshold_matching = 0.40 # %70
    max_match_template_corr_score = 1500111222
    min_match_template_corr_score =  200111222

    color_beacon = (255,0,0)
    color_text_beacon = (255,0,0)


    line_thick = 0.65
    color_blue = (0, 0, 255) #blue
    color_red = (255, 0, 0) #red
    color_pink = (240, 131, 244) #pink
    color_green = (7, 216, 17) #green
    color_red_bgr = (0, 0, 255)  # BGR formatında renk seçimi
    color_blue_bgr = (255, 0, 0)  # BGR formatında renk seçimi


    left_section = ["CV: Motion Detection", "Siamese: Similarity", "CV:Kalman"]
    right_section = ["CNN: Object Detection", "DNN: Distance", "RNN: Direction"]

    '''left_section_color = [ (0, 0, 255), (65, 105, 225), (30, 144, 255)]
    right_section_color = [(255, 0, 0), (220, 20, 60), (255, 69, 0)]'''

    left_section_color = [ (255, 165, 0),(128, 0, 128), (0, 128, 128)]
    right_section_color = [ (255, 0, 0),  (0, 255, 0), (0, 0, 255)]


    #right_detection_image = np.zeros((right_part_height, int(her_blogun_genisligi_w), 3), dtype=np.uint8) * 255

    def __init__(self):
        
        self.right_detection_image = None #np.zeros((right_part_height, int(her_blogun_genisligi_w), 3), dtype=np.uint8) * 255
        

    def main_print_show(self,frame, deepcopy_frame,camera_parameters, right_detection):
        imshow = True
        if imshow:

            frame_w = frame.shape[1]
            frame_h= frame.shape[0]
            # Beyaz bir resim oluştur (%40 daha geniş)
            white_image = np.zeros((frame_h, int(frame_w * 1.4), 3), dtype=np.uint8) * 255

            white_image_h = white_image.shape[0]
            white_image_w = white_image.shape[1]

            # Siyah resmi beyaz resmin tam ortasına kopyala
            x_offset = int((white_image_w - frame_w) / 2)
            y_offset = int((white_image_h - frame_h) / 2)
            white_image[y_offset:y_offset+frame_h, x_offset:x_offset+frame_w] = frame

            her_blogun_genisligi_w = int((white_image_w-frame_w)/2)
            left_parts = 3
            left_part_height = int(white_image_h / left_parts)

            right_parts = 3
            right_part_height = int(white_image_h / right_parts)

            if self.right_detection_image is None:
                self.right_detection_image = np.zeros((right_part_height, int(her_blogun_genisligi_w), 3), dtype=np.uint8) * 255


            try:
                if right_detection.object_type != "None":
                    start_point_min_y = int(right_detection.risk_factor_y_min)
                    start_point_max_y = int(right_detection.risk_factor_y_max)
                    end_point_min_x = int(right_detection.risk_factor_x_min)
                    end_point_max_x =int(right_detection.risk_factor_x_max)
                    self.right_detection_image = deepcopy_frame[start_point_min_y:start_point_max_y,end_point_min_x:end_point_max_x ] 


                    diff_start_point_w = start_point_max_y - start_point_min_y

                    try:
                        
                        start_point_center_y = int((start_point_min_y+start_point_max_y)/2)
                        start_point_min_y = int(start_point_center_y - (right_part_height/2))
                        start_point_max_y =  int(start_point_min_y + right_part_height)

                        diff_end_point_w = end_point_min_x - end_point_max_x
                        end_point_center_y = int((end_point_min_x+end_point_max_x)/2)
                        end_point_min_x = int(end_point_center_y - (her_blogun_genisligi_w/2))
                        end_point_max_x =  int(end_point_min_x +her_blogun_genisligi_w)
                        
                        self.right_detection_image = deepcopy_frame[start_point_min_y:start_point_max_y,end_point_min_x:end_point_max_x ] 
                    except:
                        pass


                    try:
                        zoom = 3
                        if diff_start_point_w <=10:
                            zoom = 5
                        elif diff_start_point_w <=15:
                            zoom = 4.5
                        elif diff_start_point_w <=20:
                            zoom = 4
                        elif diff_start_point_w <=25:
                            zoom = 3.5

                        # resmin yüksekliği ve genişliği
                        h, w = self.right_detection_image.shape[:2]
                        # merkez noktasını hesapla
                        center = (w//2, h//2)
                        # yeni boyutları hesapla
                        new_h, new_w = int(h * zoom), int(w * zoom)

                        # yeniden boyutlandır
                        resized = cv2.resize(self.right_detection_image, (new_w, new_h))

                        # yeni boyutlu resmin merkezini hesapla
                        x, y = (new_w//2, new_h//2)

                        # resmi kes
                        x1, y1 = x - center[0], y - center[1]
                        x2, y2 = x1 + w, y1 + h
                        resized = resized[y1:y2, x1:x2]

                        # orijinal boyutlara döndür
                        self.right_detection_image = cv2.resize(resized, (w, h))

                    except:
                        pass
            except:
                pass
            
            # Sol tarafı 3 eşit parçaya ayırıyoruz
            for i in range(left_parts):
                start_point = (0, i * left_part_height)
                end_point = (her_blogun_genisligi_w, (i+1) * left_part_height)
                
                thickness = 2
                color = DrawingOpencv.left_section_color[i] #DrawingOpencv.color_blue
                white_image = cv2.rectangle(white_image, start_point, end_point, color, thickness)

                center_x = 5+int(start_point[0])#int((start_point[0]+end_point[0])/2)
                center_y = 20+int(start_point[1])
                cv2.putText(white_image, DrawingOpencv.left_section[i], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0/2, color, thickness)
                

            # Sağ tarafı da 3 eşit parçaya ayırıyoruz
            for i in range(right_parts):

                start_point = (white_image_w-her_blogun_genisligi_w, i * right_part_height)
                end_point = (white_image_w, (i+1) * right_part_height)


                color = DrawingOpencv.right_section_color[i]  #color_red
                thickness = 2
                white_image = cv2.rectangle(white_image, start_point, end_point, color, thickness)

                center_x = 5+int(start_point[0])#int((start_point[0]+end_point[0])/2)
                center_y = 20+int(start_point[1])
                cv2.putText(white_image, DrawingOpencv.right_section[i], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0/2, color, thickness)

                if i ==0:
                    try:
                        h, w = self.right_detection_image.shape[:2]
                        white_image[start_point[1]:start_point[1]+h, start_point[0]:start_point[0]+w] = self.right_detection_image
                    except:
                        print("for i in range(right_parts): i 0")
                        pass

                if i==1:
                    try:
                        center_x = center_x
                        center_y = center_y + 50
                        text = str(right_detection.range_distance)
                        cv2.putText(white_image, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0/2, color, thickness)
                    except:
                        print("for i in range(right_parts): i 1")
                        pass


                


            scale = white_image_w / camera_parameters.width
            new_width = camera_parameters.width
            new_height = int(white_image_h / scale)

            white_image = cv2.resize(white_image, (new_width, new_height))
            
            

            return white_image




    @staticmethod
    def angle_between(last_cage_left_top_corner, new_cage_left_top_corner, frame = None, visual_debug=False, camera_parameters=None):
        delta_x_image = new_cage_left_top_corner[0] - last_cage_left_top_corner[0]
        delta_y_image = new_cage_left_top_corner[1] - last_cage_left_top_corner[1]
        angle_rad = atan2(delta_x_image, delta_y_image)
        angle_deg = angle_rad * 180.0 / pi
        angle_deg = math.atan2(last_cage_left_top_corner[1] - new_cage_left_top_corner[1], last_cage_left_top_corner[0] - new_cage_left_top_corner[0]) * 180/math.pi
        delta_x = (delta_y_image * camera_parameters.pixel_in_centimeters)/100
        delta_y = -(delta_x_image * camera_parameters.pixel_in_centimeters)/100
       

        # # angle_rad = atan2(delta_x_image, delta_y_image)
        # # angle_deg = angle_rad * 180.0 / pi
        # # angle_deg = math.atan2(last_cage_left_top_corner[1] - new_cage_left_top_corner[1], last_cage_left_top_corner[0] - new_cage_left_top_corner[0]) * 180/math.pi
        # # angle_rad = math.atan2(new_cage_left_top_corner[1]- last_cage_left_top_corner[1] , (new_cage_left_top_corner[0]- last_cage_left_top_corner[0]) )
        # # angle_deg = math.degrees(angle_rad)
        
        # delta_x = (delta_y_image * camera_parameters.pixel_in_centimeters)/100
        # delta_y = -(delta_x_image * camera_parameters.pixel_in_centimeters)/100
        # angle_rad = math.atan2(delta_x, delta_y )
        # angle_deg = math.degrees(angle_rad)

        return angle_deg, angle_rad, delta_x, delta_y


    @staticmethod
    def drawing_beacon_rectangle(frame, top_left_corner=None, right_bottom_corner=None):
        try:
            cv2.rectangle(frame, top_left_corner, right_bottom_corner, DrawingOpencv.color_beacon, thickness=0)

            # First we crop the sub-rect from the image

            sub_img = frame[top_left_corner[1]:right_bottom_corner[1], top_left_corner[0]:right_bottom_corner[0]]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

            # Putting the image back to its position
            frame[top_left_corner[1]:right_bottom_corner[1], top_left_corner[0]:right_bottom_corner[0]] = res

            cv2.rectangle(frame, top_left_corner, right_bottom_corner, DrawingOpencv.color_beacon, thickness=0)

            center_rectangle =  (int(top_left_corner[0]+10), int((top_left_corner[1]+right_bottom_corner[1])/2))

            line_thick=0.30
            cv2.putText(frame, "Visual Beacon", center_rectangle, cv2.FONT_HERSHEY_DUPLEX, line_thick, DrawingOpencv.color_beacon, 1)
        except:
            pass

    @staticmethod
    def drawing_time_stamp_text(frame, selected_class=None):

        """
        Parameters
        -----------
        frame: image - o anki frame
        selected_class: integer - secili sinifin labeli

        Returns
        -----------
        None: geri donuse gerek yok gelen frame uzerine yaziliyor
        """

        ttemp_y = 3
        ttemp_y_end = 60
        sub_img = frame[ttemp_y:ttemp_y_end, 5:320]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.5, 1.0)
        frame[ttemp_y:ttemp_y_end, 5:320] = res

        if selected_class==None:
            selected_class = ""
        else:
            selected_class ="Selected Class Id: "+ str(selected_class)


        time_stamp = str(datetime.now())[:-3]
        textsize = cv2.getTextSize(time_stamp, cv2.FONT_HERSHEY_COMPLEX_SMALL, DrawingOpencv.line_thick, 1)[0]
        textX = int(10)
        textY = 30 #int(textsize[1]) + int((ttemp_y_end+ttemp_y)/2)
        
        cv2.putText(frame, "Selected Class Id: "+ str(selected_class), (textX,textY), cv2.FONT_HERSHEY_COMPLEX_SMALL, DrawingOpencv.line_thick, DrawingOpencv.color_blue, 1, cv2.LINE_AA)

    @staticmethod
    def drawing_frame_number_text(frame, frame_number, frame_fps):
        """
        Parameters
        -----------
        frame:image - o anki frame
        frame_number: integer - o anki bulunan frame sayisi
        frame_fps: integer -

        Returns
        -----------
        None: geri donuse gerek yok gelen frame üzerine yaziliyor. 
        """

        try:
            time_stamp = "frame number: "+str(frame_number)
            textsize = cv2.getTextSize(time_stamp, cv2.FONT_HERSHEY_COMPLEX_SMALL, DrawingOpencv.line_thick, 1)[0]
            textX = int(10)
            textY = 5 + int(textsize[1])

            '''total_second = int(frame_number / frame_fps)
            microseconds = (total_second * 1000) + int(frame_number % frame_fps)
            conversion = timedelta(seconds=total_second, microseconds=microseconds)
            converted_time = str(conversion)'''

            time_stamp = time_stamp + " fps:" + str(frame_fps)

            cv2.putText(frame, time_stamp,(textX,textY), cv2.FONT_HERSHEY_COMPLEX_SMALL, DrawingOpencv.line_thick, DrawingOpencv.color_green, 1, cv2.LINE_AA)
        except:
            pass


    @staticmethod
    def drawing_rectangle(frame, class_id, x1_y1, x2_y2):
        """
        Parameters
        -----------
        frame:image - o anki frame
        class_id: integer -
        x1_y1: tuple - sol ust koordinat degerleri
        x2_y2: tuple - sag alt koordinat degerleri

        Returns
        -----------
        None: geri donuse gerek yok gelen frame üzerine yaziliyor.
        
        """

        id_t = 0
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        color = colors[int(id_t) % len(colors)]
        color = [i * 255 for i in color]

        x1_y1_text_padding = (x1_y1[0], x1_y1[1]-5)
        cv2.putText(frame, str(class_id), x1_y1_text_padding, cv2.FONT_HERSHEY_COMPLEX_SMALL, DrawingOpencv.line_thick, DrawingOpencv.color_red, 1, cv2.LINE_AA)
        cv2.rectangle(frame, x1_y1, x2_y2, DrawingOpencv.color_red, 1)

    @staticmethod
    def opencv_put_text(frame, string_text):
        """
        Parameters
        -----------
        frame:image - o anki frame
        string_text: string - yazilmasi istenilen text

        Returns
        -----------
        None: geri donuse gerek yok gelen frame uzerine yaziliyor.
        """
        
        cv2.putText(frame, string_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, DrawingOpencv.color_red, 2)
        