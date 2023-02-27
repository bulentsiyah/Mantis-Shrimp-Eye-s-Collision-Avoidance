import cv2
import time
import sys
from datetime import datetime
import os
import sys

from tools import Utils

class VideoCapture:
    """
    Video üzerindeki tüm işlemlerin yapıldığı sınıftır.
    """

    def __init__(self, configurationManager, camera_parameters, pure_frame_save=True, vision_frame_save=True):
        """
        Parameters
        -----------
        print_showing: Boolean - sınıf içerisindeki çıktıların terminalde görünmesini/gizlenmesini sağlar
        pure_frame_save: Boolean - videonun orjinalinin/üzerine birşey yazılmamış halini seçili kamera parametrelerine göre kaydedilmesini sağlar.
        vision_frame_save: Boolean - videonun üzerine hesaplamalar yapılmış son halinin seçili kamere parametrelerine göre kaydedilmesini sağlar.

        Returns
        ----------
        return: None
        """
        self.configurationManager = configurationManager

        self.camera_parameters = camera_parameters


        video_path = self.configurationManager .config_readable['video_path_file']


        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_width = int(self.video_capture.get(3))
        self.frame_height = int(self.video_capture.get(4))

        self.frame_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

        self.frame_number = 0
        self.id = os.path.splitext(os.path.basename(video_path))[0]

        self.resizing = True
        self.new_width = int(self.video_capture.get(3))
        self.new_height = int(self.video_capture.get(4))

        self.selecetROI = True
        self.box = None
        self.ret = True

        if self.frame_width != self.camera_parameters.width:
            if self.resizing:
                self.scale = self.frame_width / self.camera_parameters.width
                self.new_width = self.camera_parameters.width
                self.new_height = int(self.frame_height / self.scale)
            else:
                self.new_width = self.frame_width
                self.new_height = self.frame_height


        test_id, ext_main_image_filepath= os.path.splitext(os.path.basename(video_path))
        ustuneKayit_True_ayriAyri_False = eval(self.configurationManager.config_readable['ustuneKayit_True_ayriAyri_False'])
        temp_video_on_ek = ""
        if ustuneKayit_True_ayriAyri_False == False:
            temp_video_on_ek = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y, %H:%M:%S")

        self.pure_frame_save = pure_frame_save
        if self.pure_frame_save:
            temp_save_file =os.path.dirname(sys.argv[0]) +self.configurationManager.config_readable['video_save_path_folder']+temp_video_on_ek+test_id+'_output_saa_pure.avi'
            self.pure_frame_save_out = cv2.VideoWriter(temp_save_file,cv2.VideoWriter_fourcc(*'XVID'), self.frame_fps, (self.new_width,self.new_height)) 
        
        self.vision_frame_save = vision_frame_save
        if self.vision_frame_save:
            temp_save_file = os.path.dirname(sys.argv[0]) +self.configurationManager.config_readable['video_save_path_folder']+temp_video_on_ek+test_id+'_output_saa_vision.avi'
            self.vision_frame_save_out = cv2.VideoWriter(temp_save_file,cv2.VideoWriter_fourcc(*'XVID'), self.frame_fps, (self.new_width,self.new_height)) 

        self.__method_init_time= time.time()
        self.__method_call_number = 0
        self.method_fps = 0
        

    def save_pure_frame_save(self, frame):
        if self.pure_frame_save:
            self.pure_frame_save_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def save_vision_frame_save(self, frame):
        if self.vision_frame_save:
            try:
                self.vision_frame_save_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except:
                pass


    def all_video_release(self):
        self.video_capture.release()


    
    def get_image(self):
        """
        video dan sırası gelen frame i alan kısımdır, aynı zaman da bölge seçmek, ekran görüntüsü almak gibi kısımlar da burada yapılır.
        
        Parameters
        -----------
        None:
        Returns
        -----------
        None:
        """
        self.ret, self.img = self.video_capture.read()

        self.frame_number = self.frame_number + 1
        self.__method_call_number = self.__method_call_number + 1
        
        if self.ret == False:
            cv2.destroyAllWindows()
            self.img = None
            return
        
        if self.frame_width != self.camera_parameters.width:
            if self.resizing:
                self.img = cv2.resize(self.img, (self.new_width, self.new_height))
        
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord("q")):
            self.ret = False
            self.img = None
            return
            
        #if self.frame_number % (self.frame_fps*Utils.frame_fps_kac_saniye)  == 0:
        diff_time = int(time.time()-self.__method_init_time)
        if diff_time !=0:
            if diff_time % Utils.frame_fps_kac_saniye  == 0:
                #self.method_fps  = int((self.frame_fps*Utils.frame_fps_kac_saniye) / (time.time() -self.method_start_time))
                self.method_fps  = int(self.__method_call_number / diff_time ) 
                self.__method_call_number = 0
                self.__method_init_time= time.time()

        return self.img