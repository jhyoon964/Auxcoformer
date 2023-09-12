from ultralytics import YOLO
import cv2
import os
import urllib

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':
    
    model = YOLO("pretrain/yolo_low_low.yaml")

    model.load("root")

    model.train(cfg="ultralytics/cfg/custom.yaml")

