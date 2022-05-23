#!/usr/bin/env python3


CLEAN=True
DATASET_FOLDER = 'dataset'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

from utils import *
import cv2
import numpy


if __name__ == '__main__':
    data = ImageData(preload=True)
    print('pls')
    smote = SMOTE_balancing().compute(data)
    print('work')
    
