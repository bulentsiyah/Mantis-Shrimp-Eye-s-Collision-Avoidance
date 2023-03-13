from ultralytics import YOLO
import torch

import sys
sys.path.append('tools')
from configmanager import ConfigurationManager


if __name__ == '__main__':
    print('cuda_avail:', torch.cuda.is_available())
    print('cuda_device:', torch.cuda.device_count())

    configurationManager =  ConfigurationManager()

    data = configurationManager.config_readable['yolo_dataset']+"/custom.yaml"
    last_weights = configurationManager.config_readable['right_detection_model']

    epoch = 10
    imgsz = 1024
    last_folder = "best_"+str(350+epoch)+"_" + str(imgsz)

    # Load a model
    model = YOLO(last_weights)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
    data=data,
    imgsz=imgsz,
    epochs=epoch,
    batch=8,
    name=last_folder,
    device=0
    )
