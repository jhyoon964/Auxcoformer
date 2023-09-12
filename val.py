from ultralytics import YOLO
from multiprocessing import freeze_support

    
    
    
def run():    
    freeze_support()
    model = YOLO("pretrain/Auxcoformer.yaml", task='detect')

    model.load("root")    
    metrics = model.val(data='ultralytics/cfg/datasets/crack600.yaml')

    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
if __name__ == '__main__':
    run()


