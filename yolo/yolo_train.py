from ultralytics import YOLO
import torch
import os

os.chdir("../")
os.getcwd()

try:
    import sys
    sys.path.append('tools')
    from configmanager import ConfigurationManager
except:
    pass


if __name__ == '__main__':
    print('cuda_avail:', torch.cuda.is_available())
    print('cuda_device:', torch.cuda.device_count())


    configurationManager =  ConfigurationManager()
    data = configurationManager.config_readable['yolo_dataset']+"/custom.yaml"
    last_weights = configurationManager.config_readable['right_detection_model']

    os.chdir("yolo/")
    os.getcwd()

    epoch = 50
    imgsz = 1024
    last_folder = "best_"+str(420+epoch)+"_" + str(imgsz)

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
