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
    data = configurationManager.config_readable['right_detection_dataset']+"custom_1b3a52c0f1764533b32e36992b17b6e1.yaml"
    last_weights = configurationManager.config_readable['right_detection_model']

    last_weights_name = last_weights.split("/")[-1]
    last_weights_name = last_weights_name.split("_")[1]


    os.chdir("./yolo/")
    print(os.getcwd())

    last_weights = "../"+last_weights
    data = "../"+data


    epoch = 50
    imgsz = 1024
    last_folder = "best_"+str(int(last_weights_name)+epoch)+"_" + str(imgsz)

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
