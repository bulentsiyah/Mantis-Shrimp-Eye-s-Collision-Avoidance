import warnings
warnings.filterwarnings("ignore")
import argparse
import sys, os
import cv2
import time
from datetime import datetime
import copy
import glob

from collisioncalculationcontext import CollisionCalculationContext

from threading import Thread
from dnncompare import DNNCompare

sys.path.append('tools')
from configmanager import ConfigurationManager
from videocapture import VideoCapture
from tools import DrawingOpencv
from cameraparameters import CameraParameters


class TestOnVideo:

    def __init__(self,pure_frame_save=False,vision_frame_save=True, test_id=""):

        self.configurationManager = ConfigurationManager()
        #agr fonksıyonları kullanmak ıcın
        threshold = self.configurationManager.config_changeable['threshold']
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--threshold', help="Select Threshold", default=threshold)
        args = parser.parse_args()
        self.configurationManager.set_threshold(threshold=str(args.threshold))

        self.cv_imshow_title = self.configurationManager.config_readable['cv_imshow_title']

        self.drawopencv = DrawingOpencv()

        self.camera_parameters = CameraParameters("amazon_prime_air_half")

        self.videocapture = VideoCapture(camera_parameters=self.camera_parameters, configurationManager=self.configurationManager, pure_frame_save=pure_frame_save,vision_frame_save=vision_frame_save)

        self.dnn_compare = DNNCompare(camera_parameters=self.camera_parameters,configurationManager=self.configurationManager)

        self.collisioncalculation = CollisionCalculationContext(camera_parameters=self.camera_parameters, configurationManager=self.configurationManager, visual_drawing=True)

        self.context_return = None

        self.threadTrueNormalFalse = False


    def myfunc(self, frame,frame_number):
        self.context_return = self.collisioncalculation.context(frame= frame, frame_number=frame_number)


    def run(self):
        """
        İslemin basladigi ana siniftir.                  
        """

        while True:
            frame = self.videocapture.get_image()

            if (self.videocapture.ret == False):
                break

            # saf halini kaydetmek istersen kaydet
            self.videocapture.save_pure_frame_save(frame)

            #collision işimiz başladı
            if self.threadTrueNormalFalse:
                t = Thread(target=self.myfunc, args=(frame, self.videocapture.frame_number,))
                t.start()
            else:
                self.context_return = self.collisioncalculation.context(frame=copy.deepcopy(frame))
                
            
            #coolison işimiz bitti
            frame= self.context_return.frame
            #self.drawopencv.drawing_frame_number_text(frame, self.videocapture.frame_number, context_return.method_fps)
            self.drawopencv.drawing_frame_number_text(frame, self.videocapture.frame_number, self.videocapture.method_fps)
            # işlenmiş halini kaydetmek istersen kaydet
            self.videocapture.save_vision_frame_save(frame=frame)

            #DNN için işlemler
            if self.context_return.right_detection.object_type != "None":
                self.dnn_compare.save_dnn_pred(right_detection=self.context_return.right_detection,frame_id=self.videocapture.frame_number,scale=self.videocapture.scale)

            #genel gosterım 
            try:
                '''scale = 2.0
                new_width = int(self.camera_parameters.width / scale)
                new_height = int( self.camera_parameters.height / scale)
                frame_show = cv2.resize(frame, (new_width, new_height))'''

                imshow = False
                if imshow:
                    cv2.imshow(self.cv_imshow_title,  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except:
                print("main exception")
                pass


            main_print_show = False
            if main_print_show:
                print("Video Capture fps rate:","FPS: "+str(self.videocapture.work_time_fps))

        self.dnn_compare.save_dnn_pred_path(context_return=self.context_return)
        self.videocapture.all_video_release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # coklu oldugu anlasılısa video path setlenebılır
    configurationManager = ConfigurationManager()
    video_path = configurationManager.config_changeable['video_path_file']
    flight_id, ext = os.path.splitext(os.path.basename(video_path))
    if ext !=".mp4":
        for mp4_file in glob.iglob(os.path.join(video_path, '*.mp4')):
            flight_id= os.path.basename(mp4_file)
            configurationManager.set_video_path_file(video_path_file=str(mp4_file))
            test_on_video = TestOnVideo(pure_frame_save=False,vision_frame_save=True)
            test_on_video.run()

    else:
        test_on_video = TestOnVideo(pure_frame_save=False,vision_frame_save=True)
        test_on_video.run()
