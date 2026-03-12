import numpy as np
import cv2
from Dataset_Creation import get_mini_dataset
import matplotlib.pyplot as plt
from AWB import max_RGB_AWB,perform_max_RGB


def get_dark_channel(img,patch_size=(15,15)):
    dark_channel = np.zeros((img.shape[0],img.shape[1]))
    min_channel = np.zeros((img.shape[0],img.shape[1]))

    # img_R = img[:,:,0]
    # img_G = img[:,:,1]
    # img_B = img[:,:,2]
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         min_channel[i,j] = min(img_R[i,j] , img_G[i,j] , img_B[i,j])
    min_channel = np.min(img,axis=2) # Efficient way to write the above code segment
    
    # We need the patch to be centered around each pixel hence padding is necessary
    amt_to_pad = patch_size[0]//2
    min_channel_padded = np.pad(min_channel,pad_width=((amt_to_pad,amt_to_pad),(amt_to_pad,amt_to_pad)),mode='reflect')
                                                    # ((top,bottom),(left,right))
    #print(min_channel_padded.shape)  # ---> (270,270)

    for i in range(amt_to_pad,min_channel_padded.shape[0] - amt_to_pad):
        for j in range(amt_to_pad,min_channel_padded.shape[1]-amt_to_pad):
            neighborhood = min_channel_padded[i - amt_to_pad : i +amt_to_pad + 1 ,j - amt_to_pad : j + amt_to_pad + 1]
            dark_channel[i-amt_to_pad , j - amt_to_pad] = np.min(neighborhood)

    return min_channel,dark_channel

def estimate_A(dark_channel,DCP_input_img):
    no_of_pixels = dark_channel.shape[0]*dark_channel.shape[1]
    dark_channel_flattened = dark_channel.ravel()
    no_of_pixel_to_be_considered = max(int((0.1/100)*no_of_pixels),1)  # if 0.1% of total pixels is less than 1 then no pixel will be considered hence we use max()
    top_indices = np.argsort(dark_channel_flattened)[-no_of_pixel_to_be_considered:]
    # argsort will give me indices of the pixel intensities in ascending order. We only want to get 0.1% of pixels hence the slicing

    # Now we need the spatial positions of the brightest pixels. 
    row_indices = top_indices//dark_channel.shape[1]
    col_indices = top_indices%dark_channel.shape[1]

    max_intensity = -1
    for r,c in zip(row_indices,col_indices):
        intensity = np.max(DCP_input_img[r,c,:])  # We defined intensity as Max{r,g,b}
        if intensity>max_intensity:
            A = DCP_input_img[r,c,:]
            max_intensity = intensity

    return A

def estimate_transmission_map(A,DCP_input_img):
    normalized_img = np.zeros_like(DCP_input_img)
    normalized_img[:,:,0] = DCP_input_img[:,:,0]/(A[0] + 1e-8)
    normalized_img[:,:,1] = DCP_input_img[:,:,1]/(A[1] + 1e-8)
    normalized_img[:,:,2] = DCP_input_img[:,:,2]/(A[2] + 1e-8)
    _,dark_channel = get_dark_channel(normalized_img)
    w = 0.95

    return np.ones(dark_channel.shape) - w*dark_channel

def get_scene_radiance(DCP_input_img):
    DCP_input_img_normalized = DCP_input_img.astype(np.float64)/255.0  # dcp assumes that t(x) and A are in [0,1]. So its better to normalize img
    _,dark_channel = get_dark_channel(DCP_input_img_normalized)
    A = estimate_A(dark_channel=dark_channel,DCP_input_img=DCP_input_img_normalized)
    t_map = estimate_transmission_map(A,DCP_input_img_normalized)
    t_refined = cv2.ximgproc.guidedFilter(
        guide = DCP_input_img,
        src = t_map.astype(np.float32),
        radius = 40,
        eps = 1e-3
    )
    final_t = np.maximum(t_refined,0.1)[:,:,np.newaxis]
    output = ((DCP_input_img_normalized-A)/final_t) + A
    output = np.clip(output,0,1)
    return output

if __name__ == "__main__":
    raw_imgs,ref_imgs = get_mini_dataset()
    AWB_outputs = max_RGB_AWB(raw_imgs)
    for i in range(len(AWB_outputs)):
        DCP_input_img = AWB_outputs[i]
        min_channel,dark_channel = get_dark_channel(DCP_input_img)

        output = get_scene_radiance(DCP_input_img)
        
        
        fig,ax = plt.subplots(nrows=1,ncols=6)

        ax[0].imshow(raw_imgs[i])
        ax[0].set_title("Input")
        ax[0].axis("off")

        ax[1].imshow(DCP_input_img)
        ax[1].set_title("AWB(MAX RGB)")
        ax[1].axis("off")

        ax[2].imshow(min_channel,cmap='gray')
        ax[2].set_title("Min Channel Output")
        ax[2].axis("off")

        ax[3].imshow(dark_channel,cmap='gray')
        ax[3].set_title("Dark Channel Output")
        ax[3].axis("off")

        ax[4].imshow(output)
        ax[4].set_title("Final Output")
        ax[4].axis("off")  

        ax[5].imshow(ref_imgs[i])
        ax[5].set_title("Reference Output")
        ax[5].axis("off")    
        plt.show()