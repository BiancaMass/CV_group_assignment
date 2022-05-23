#!/usr/bin/env python3


CLEAN=True
DATASET_FOLDER = 'dataset'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

from utils import *
import cv2
import numpy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = ImageData(preload=True)
    eq_imgs = {}
    for em in EMOTIONS:
        eq_imgs[em] = []
        for i in range(len(data.data_struct['train'][em])):
            img = data.data_struct['train'][em][i]
            eq_imgs[em].append(cv2.equalizeHist(img))

    f, ax = plt.subplots(5,7)
    fn, axn = plt.subplots(5,7)

    for em in EMOTIONS:
        for i in range(5):
            old_img = data.data_struct['train'][em][i]
            new_img = eq_imgs[em][i]
            ax[i,EMOTIONS.index(em)].imshow(old_img,cmap='gray')
            axn[i,EMOTIONS.index(em)].imshow(new_img,cmap='gray')
            #cv2.imwrite(f'old_{em}_{i}.png',old_img)
            #cv2.imwrite(f'new_{em}_{i}.png',new_img)
    f.savefig('old.png')
    fn.savefig('new.png')
