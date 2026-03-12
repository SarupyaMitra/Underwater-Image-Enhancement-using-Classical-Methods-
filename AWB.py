import numpy as np
from Dataset_Creation import get_mini_dataset
import matplotlib.pyplot as plt 

raw_imgs,ref_imgs = get_mini_dataset()
#print(raw_imgs.shape)

def perform_gray_world(img):
    op_img = np.zeros_like(img)
    R_channel = img[:,:,0]
    G_channel = img[:,:,1]
    B_channel = img[:,:,2]

    meanR = np.mean(R_channel)
    meanG = np.mean(G_channel)
    meanB = np.mean(B_channel)

    # Practically green channel is kept unchanged, only blue and red channels are modified

    alpha = meanG/meanR
    beta = meanG/meanB

    new_R_channel = (alpha*R_channel)
    new_B_channel = (beta*B_channel)

    op_img[:,:,0] = new_R_channel
    op_img[:,:,1] = G_channel
    op_img[:,:,2] = new_B_channel

    return op_img



def gray_world_AWB(raw_imgs):
    outputs = []
    for i in range(len(raw_imgs)):
        img = raw_imgs[i]
        op_img = perform_gray_world(img)
        outputs.append(op_img)
    return np.array(outputs)

def perform_max_RGB(img):
    op_img = np.zeros_like(img)
    R_channel = img[:,:,0]
    G_channel = img[:,:,1]
    B_channel = img[:,:,2]

    maxR = np.max(R_channel)
    maxG = np.max(G_channel)
    maxB = np.max(B_channel)

    alpha = maxG/maxR
    beta = maxG/maxB

    new_R_channel = (alpha*R_channel)
    new_B_channel = (beta*B_channel)

    op_img[:,:,0] = new_R_channel
    op_img[:,:,1] = G_channel
    op_img[:,:,2] = new_B_channel

    return op_img

def max_RGB_AWB(raw_imgs):
    outputs = []
    for i in range(len(raw_imgs)):
        img = raw_imgs[i]
        op_img = perform_max_RGB(img)
        outputs.append(op_img)
    return np.array(outputs)
     




if __name__ == '__main__':
    outputs = max_RGB_AWB(raw_imgs)
    figcounter = 0
    column_headers = ['Raw','Reference','Output']
    fig,ax = plt.subplots(nrows=3,ncols=3)
    for i in range(3):
        ax[0,i].set_title(column_headers[i])
        ax[i,figcounter].imshow(raw_imgs[i+3])
        ax[i,figcounter].axis('off')
        figcounter += 1

        ax[i,figcounter].imshow(ref_imgs[i+3])
        ax[i,figcounter].axis('off')
        figcounter += 1

        ax[i,figcounter].imshow(outputs[i+3])
        ax[i,figcounter].axis('off')
        figcounter += 1

        if figcounter==3:
            figcounter = 0 
    plt.tight_layout()
    plt.show()
        
