import numpy as np
from AWB import max_RGB_AWB,raw_imgs,ref_imgs
import cv2
import matplotlib.pyplot as plt

block_size= (64,64)
def get_freq_of_block(block):
    freq = np.zeros(256)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            freq[block[i,j]] += 1
    return freq

def get_blocks(L_channel):
    # We have 256*256 image and we will break it up into 4*4 blocks and each block having 64*64 pixels
    blocks = []
    freqs = []
    for i in range(0,L_channel.shape[0],block_size[0]):
        for j in range(0,L_channel.shape[1],block_size[1]):
            block =  L_channel[i:i+block_size[0] , j:j+block_size[1]]
            blocks.append(block)
            freq = get_freq_of_block(block)
            freqs.append(freq)
    return blocks,freqs

def const_lim(freq):
    clip_limit = 512
    extrasum = 0

    # Initial Clipping
    for i in range(len(freq)):
        if(freq[i]>clip_limit):
            extrasum += freq[i] - clip_limit
            freq[i] = clip_limit

    # Redistribution
    while(extrasum>0):
        possible_no_to_add = int(extrasum/256) 
        # There can be cases where extrasum < 256, then  possible_no_to_add=0
        # We handle this case in the residual case (written below)
        if (possible_no_to_add==0) :  
            break
        else:
            for i in range(len(freq)):
                if(freq[i]<clip_limit):
                    amt_bin_can_accept = clip_limit-freq[i]
                    freq[i] += min(amt_bin_can_accept,possible_no_to_add)
                    extrasum -= min(amt_bin_can_accept,possible_no_to_add)
                    if(extrasum <= 0):
                        break
    
    # In case any residual is still left
    while(extrasum>0):
        for i in range(len(freq)):
            if freq[i]<clip_limit:
                freq[i] += 1
                extrasum -= 1
            if extrasum <= 0:
                break
    return freq

def get_lookup(new_freq):
    pdf = new_freq/np.sum(new_freq)
    cdf = np.cumsum(pdf)
    new_pixel_lookup = np.zeros(256)
    for i in range(len(new_freq)):
        new_pixel_lookup[i] = np.round(255*cdf[i])

    return new_pixel_lookup

def perform_hist_eq(block,new_lookup):
    new_block = np.zeros_like(block)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            new_block[i,j] = new_lookup[block[i,j]]
    return new_block

def perform_interpolation(interpolation_lookups,L_channel):
    h,w = L_channel.shape
    bh,bw = block_size
    op_L_channel = np.zeros_like(L_channel)
    for i in range(h):
        for j in range(w):
            old_pixel_value  = L_channel[i,j]
            block_i = i//bh
            block_j = j//bw

            # Now we need to find the block's centres as we need to compute the pixel's distance from the centre
            # to use as weights during interpolation

            block_centre_i = block_i*bh + bh//2
            block_centre_j = block_j*bw + bw//2


            # Biggest problem in this interpolation is to see 
            # whether the pixel is above/below and left/right of 
            # its block's center to pick the correct four neighbors

            # Checking if pixel is above or below block's centre
            if i < block_centre_i: # pixel is above block centre
                i_top = max(block_i - 1,0)
                i_bottom = block_i
            elif i == block_centre_i:  # pixel is at the center
                i_top = block_i
                i_bottom = block_i
            else:   # pixel is below block centre
                i_top = block_i
                i_bottom = min(block_i+1,(h//bh) - 1)

            # Checking if pixel is left or right of block's centre
            if j < block_centre_j:  # pixel is left of block center
                j_left  = max(block_j-1,0)
                j_right = block_j
            elif j == block_centre_j:   # pixel is at the centre
                j_left = block_j
                j_right = block_j
            else:                # pixel is right of block centre
                j_left = block_j
                j_right = min(block_j + 1,(w//bw) - 1)

            top_centre = i_top * bh + bh//2
            left_centre = j_left * bw + bw//2

            if i_top == i_bottom:
                fy = 0
            else:
                fy = (i - top_centre)/bh

            if j_left == j_right:
                fx = 0
            else:
                fx = (j - left_centre)/bw

            output_pixel_value = (
                interpolation_lookups[i_top,j_left][old_pixel_value]*(1-fx)*(1-fy) +  # Top-left block
                interpolation_lookups[i_top,j_right][old_pixel_value]*(fx)*(1-fy) + # Top right block
                interpolation_lookups[i_bottom,j_left][old_pixel_value]*(1-fx)*fy + # Bottom-left block
                interpolation_lookups[i_bottom,j_right][old_pixel_value]*(fx)*fy  # Bottom right block
            )

            op_L_channel[i,j] = output_pixel_value
    return op_L_channel

def CLAHE_pipeline(img):
    img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    L_channel = img_lab[:,:,0]
    interpolation_lookups = []
    blocks,freqs = get_blocks(L_channel)
    for i,block in enumerate(blocks):
        new_freq_of_block = const_lim(freqs[i])
        lookup = get_lookup(new_freq_of_block)
        interpolation_lookups.append(lookup)
       # new_block = perform_hist_eq(block,lookup) --->  I don't actually need the newblock, because final output image will 
       # be found after interpolation.   
    
    interpolation_lookups = np.array(interpolation_lookups)   # Shape is (64,256)
    interpolation_lookups = np.reshape(interpolation_lookups,(4,4,256))
    new_L_channel = perform_interpolation(interpolation_lookups,L_channel)
    op_lab_img = np.zeros_like(img_lab)
    op_lab_img[:,:,0] = np.clip(new_L_channel, 0, 255).astype(np.uint8)
    op_lab_img[:,:,1] = img_lab[:,:,1]
    op_lab_img[:,:,2] = img_lab[:,:,2]   

    final_op_img = cv2.cvtColor(op_lab_img,cv2.COLOR_LAB2RGB)
    return final_op_img  


if __name__ == '__main__':
    #print(block_size[0])
    outputs = max_RGB_AWB(raw_imgs)
    output = CLAHE_pipeline(outputs[0])
    output_row = output[:,182:202,:]


    # Inbuilt CLAHE function:
    lab = cv2.cvtColor(outputs[0],cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(tileGridSize=(4,4)) 
    l_eq = clahe.apply(l)
    output_lab = cv2.merge((l_eq,a,b))
    output1 = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output1_row = output1[:,182:202,:]



    #output =  outputs[7]
    fig,ax = plt.subplots(nrows=1,ncols=3)
    ax[0].imshow(raw_imgs[0])
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(output)
    ax[1].set_title("My CLAHE Output")
    ax[1].axis("off")
    ax[2].imshow(output1)
    ax[2].set_title("CV2's CLAHE Output")
    ax[2].axis("off")
    plt.show()
    