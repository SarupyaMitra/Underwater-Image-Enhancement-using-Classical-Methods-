import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim
from AWB import max_RGB_AWB,perform_max_RGB
from CLAHE import CLAHE_pipeline
from Dataset_Creation import get_whole_dataset,get_mini_dataset
from Min_GB_DCP import get_scene_radiance_GB
from Chromaticity_guided_DCP import preprocess
from DCP import get_scene_radiance


def uicm(img):

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    rg = R - G
    yb = 0.5*(R + G) - B

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    var_rg = np.var(rg)
    var_yb = np.var(yb)

    std_rg = np.sqrt(var_rg)
    std_yb = np.sqrt(var_yb)

    return -0.0268*np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586*np.sqrt(std_rg**2 + std_yb**2)


def uism(img):
    # img in [0,1]
    R = img[:,:,0].astype(np.float32)
    G = img[:,:,1].astype(np.float32)
    B = img[:,:,2].astype(np.float32)

    def sobel_magnitude(channel):
        sx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(channel, cv2.CV_64F, 0, 1)
        return np.sqrt(sx**2 + sy**2)

    edge_r = np.mean(sobel_magnitude(R))
    edge_g = np.mean(sobel_magnitude(G))
    edge_b = np.mean(sobel_magnitude(B))

    # perceptual weights
    return 0.299*edge_r + 0.587*edge_g + 0.114*edge_b

def uiconm(img, block_size=8):
    # handle both uint8 and float input
    if img.max() > 1.0:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)

    h, w, _ = img.shape

    # compute on luminance
    luminance = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    contrast_vals = []
    for i in range(0, h-block_size, block_size):
        for j in range(0, w-block_size, block_size):
            block = luminance[i:i+block_size, j:j+block_size]
            Imax  = np.max(block)
            Imin  = np.min(block)
            if (Imax + Imin) > 0:
                contrast = (Imax - Imin) / (Imax + Imin + 1e-8)
                contrast_vals.append(contrast)

    return np.mean(contrast_vals) if contrast_vals else 0.0


def uiqm(img):
    if img.max() > 1.0:
        img = img.astype(np.float64) / 255.0

    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    return c1*uicm(img) + c2*uism(img) + c3*uiconm(img)


def show_ops(model_no,raw_imgs,ref_imgs):
    psnr_values = []
    ssim_values = []
    uiqm_values = []
    raw_img_uiqm_values = []

    for i in range(len(raw_imgs)):
        raw_img_uiqm_val = uiqm(raw_imgs[i])
        raw_img_uiqm_values.append(raw_img_uiqm_val)
        if model_no==1:                    # AWB+CLAHE
            AWB_output = perform_max_RGB(raw_imgs[i])
            op = CLAHE_pipeline(AWB_output)
            final_op = op
            if(i == len(raw_imgs)-1):   # Just show the name of the model at the end
                print("\n For AWB+CLAHE Model:")

        elif model_no == 2:              # Basic DCP
            DCP_OP = get_scene_radiance(raw_imgs[i])
            final_op = DCP_OP
            if (i == len(raw_imgs)-1):
                print("\n For basic DCP:")

        elif model_no==3:               # Red Compensated DCP
            processed_ip = preprocess(raw_imgs[i])
            red_comp_DCP = get_scene_radiance(processed_ip)
            final_op = red_comp_DCP
            if (i == len(raw_imgs)-1):
                print("\n For Red compensated DCP:")

        elif model_no==4:              # MIN GB DCP
            DCP_GB = get_scene_radiance_GB(raw_imgs[i])
            final_op = DCP_GB
            if (i == len(raw_imgs)-1):
                print("\n For only Green-Blue DCP:")

        psnr_val = psnr(ref_imgs[i],final_op,data_range=255)
        psnr_values.append(psnr_val)
        ssim_val = ssim(ref_imgs[i],final_op,data_range=255,channel_axis=-1)
        ssim_values.append(ssim_val)
        uiqm_val = uiqm(final_op)
        uiqm_values.append(uiqm_val)

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    mean_uiqm = np.mean(uiqm_values)

    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    std_uiqm = np.std(uiqm_values)

    raw_img_uiqm_mean = np.mean(raw_img_uiqm_values)

    print("Mean PSNR:", mean_psnr)
    print("Std PSNR:", std_psnr)
    print("Mean SSIM:", mean_ssim)
    print("Std SSIM:", std_ssim)

    print("Mean UIQM of raw image:", raw_img_uiqm_mean)
    print("Mean UIQM of output:", mean_uiqm)
    print("Std UIQM of output:", std_uiqm)



if __name__=="__main__":
    raw_imgs,ref_imgs = get_whole_dataset() 
    # print(f"Raw Images shape: {raw_imgs.shape}")
    # print(f"Reference Images shape: {ref_imgs.shape}")
    show_ops(1,raw_imgs,ref_imgs)
    show_ops(2,raw_imgs,ref_imgs)
    show_ops(3,raw_imgs,ref_imgs)
    show_ops(4,raw_imgs,ref_imgs)








