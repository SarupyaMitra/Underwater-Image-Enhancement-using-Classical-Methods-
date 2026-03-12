import numpy as np
from DCP import get_scene_radiance
from Dataset_Creation import get_mini_dataset
import matplotlib.pyplot as plt


def perform_red_compensation(img,mean_chrom_r):
    neutral_mean_chrom_r = 1/3
    deficit = neutral_mean_chrom_r - mean_chrom_r

    gain = deficit / neutral_mean_chrom_r   
    compensated_img = np.zeros_like(img)
    compensated_img[:,:,1:] = img[:,:,1:]  # Since only red channel will be affected but not green and blue.
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         chrom_r = img[i,j,0] / (np.sum(img[i,j])+ 1e8)
    #         compensated_chrom_r = chrom_r*gain*(1-chrom_r) + chrom_r
    #         R_compensated = np.sum(img[i,j]) * compensated_chrom_r
    #         compensated_img[i,j,0] = np.clip(R_compensated,0,1)

    chrom_r = img[:,:,0] / (np.sum(img,axis=2) + 1e-8)
    compensated_chrom_r = (chrom_r*gain*(1-chrom_r)) + chrom_r
    R_compensated_channel = np.sum(img,axis=2) * compensated_chrom_r
    compensated_img[:,:,0] = np.clip(R_compensated_channel,0,1)

    return compensated_img

def preprocess(img):
    chromatic_img = np.zeros_like(img).astype(np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            chromatic_img[i,j,:] = img[i,j,:]/(np.sum(img[i,j]) + 1e-8)

    mean_chrom_r = np.mean(chromatic_img[:,:,0])
    mean_chrom_g = np.mean(chromatic_img[:,:,1])
    mean_chrom_b = np.mean(chromatic_img[:,:,2])

    ratio_b = mean_chrom_b/mean_chrom_r
    ratio_g = mean_chrom_g/mean_chrom_r

    if(ratio_b > 1.3 or ratio_g > 1.3):
        #print("Mean chromatic red is less than 0.25, hence red compensation performed")
        compensated_img = perform_red_compensation(img,mean_chrom_r)
    else:
        #print("Mean chromatic red is not less than 0.25, no red compensation performed")
        return img

    return compensated_img

if __name__=="__main__":
    raw_imgs,ref_imgs = get_mini_dataset()

    for i in range(len(raw_imgs)):
        input_img = raw_imgs[i]
        compensated_img = preprocess(input_img)
        output = get_scene_radiance(compensated_img)


        fig,ax = plt.subplots(nrows=1,ncols=4)
        ax[0].imshow(input_img)
        ax[0].set_title("Input")
        ax[0].axis("off")
        ax[1].imshow(compensated_img)
        ax[1].set_title("Red Channel compensated image")
        ax[1].axis("off")
        ax[2].imshow(output)
        ax[2].set_title("Compensated DCP Output")
        ax[2].axis("off")
        ax[3].imshow(ref_imgs[i])
        ax[3].set_title("Reference Output")
        ax[3].axis("off")
        plt.show()

    # input_img = raw_imgs[1]
    # compensated_img = preprocess(input_img)
    # compensated_img = compensated_img.astype(np.float64)/255.0
    # output = get_scene_radiance(compensated_img)


    # fig,ax = plt.subplots(nrows=1,ncols=4)
    # ax[0].imshow(input_img)
    # ax[0].set_title("Input")
    # ax[0].axis("off")
    # ax[1].imshow(compensated_img)
    # ax[1].set_title("Red Channel compensated image")
    # ax[1].axis("off")
    # ax[2].imshow(output)
    # ax[2].set_title("Compensated DCP Output")
    # ax[2].axis("off")
    # ax[3].imshow(ref_imgs[1])
    # ax[3].set_title("Reference Output")
    # ax[3].axis("off")
    # plt.show()
