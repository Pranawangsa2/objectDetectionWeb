import os
from yolo.yolov5 import detect
import cv2
import pandas as pd
from django.core.files.storage import FileSystemStorage

weight = 'data/yolov5m6.pt'
fss = FileSystemStorage()

def imageProcess(path, conf, iou):
    imgProcessed, imgLabel = detect.run(weights=weight, source=path, conf_thres=conf, iou_thres=iou)
    imgProcessedPath = os.path.join(fss.base_location, 'images', 'processed', 'processed.png')
    cv2.imwrite(imgProcessedPath, imgProcessed)
    df = pd.DataFrame(imgLabel, columns=['Label', 'Confidence'])
    print(df)
    return df
