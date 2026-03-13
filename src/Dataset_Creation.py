import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

input_dir_path = "data\\raw-890\\raw-890\\" 
ref_dir_path = "data\\reference-890\\reference-890\\"

raw_paths = []
ref_paths = []
for f in os.listdir(input_dir_path):
    full_path = input_dir_path + f
    raw_paths.append(full_path)

for f in os.listdir(ref_dir_path):
    full_path = ref_dir_path + f
    ref_paths.append(full_path)

def get_mini_dataset():

    X_train_paths,X_test_paths,y_train_paths,y_test_paths = train_test_split(raw_paths,ref_paths,test_size=0.01,random_state=3)

    # For the time-being I will use the test set containing 9 images just to see whether my algo is working or not

    raw_imgs = []
    ref_imgs = []
    
    for img_path in X_test_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
        raw_imgs.append(img)

    for img_path in y_test_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
        ref_imgs.append(img)

    raw_imgs = np.array(raw_imgs)
    ref_imgs = np.array(ref_imgs)

    return raw_imgs,ref_imgs

def get_whole_dataset():
    raw_imgs = []
    ref_imgs = []
    
    for img_path in raw_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
        raw_imgs.append(img)

    for img_path in ref_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
        ref_imgs.append(img)

    raw_imgs = np.array(raw_imgs)
    ref_imgs = np.array(ref_imgs)

    return raw_imgs,ref_imgs



