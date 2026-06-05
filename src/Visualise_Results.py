'This file is for visualising the model outputs for a few inputs'

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim

from AWB import perform_max_RGB
from CLAHE import CLAHE_pipeline
from Dataset_Creation import get_whole_dataset,get_mini_dataset
from Min_GB_DCP import get_scene_radiance_GB
from Chromaticity_guided_DCP import preprocess
from DCP import get_scene_radiance
from Compare import uiqm
from functools import partial



def update_grid(class_idx,fig,axes,final_imgs):
    """Clears the grid and draws the images for the selected index."""

    images = final_imgs[class_idx]
    for col_idx, ax in enumerate(axes):
        ax.clear()                  # Clear previous image and titles
        ax.imshow(images[col_idx])  # Plot new image
        ax.axis('off')              # Keep axes clean
        names_of_images = ["Raw Image","AWB+CLAHE output","DCP output","Red-compensated DCP output","Min Green-Blue DCP output","Reference Image"]
        ax.set_title(f"{names_of_images[col_idx]}")
        
    fig.canvas.draw_idle()          # Redraw the figure window safely

def on_key_press(event,fig,axes,state,final_imgs):
    """Waits for the Enter key to cycle to the index of images."""
    if event.key == 'enter':
        # Increment index and wrap around to 0 when reaching the end
        state["current_idx"] = (state["current_idx"] + 1) % len(final_imgs)  
        # Due to the % len(final_imgs), we will cyclically loop through the images in the display window.
        update_grid(state["current_idx"],fig,axes,final_imgs)


def visualise_outputs():
    raw_imgs,ref_imgs = get_whole_dataset()
    rng = np.random.default_rng(seed=3)
    random_indexes = rng.choice(len(raw_imgs),size=5,replace=False)

    # print(random_indexes)
    sampled_raw_imgs = raw_imgs[random_indexes]
    sampled_ref_imgs = ref_imgs[random_indexes]

    final_imgs = []

    for i in range(len(sampled_raw_imgs)):
        AWB_CLAHE_op = CLAHE_pipeline(perform_max_RGB(sampled_raw_imgs[i]))
        DCP_op = get_scene_radiance(sampled_raw_imgs[i])
        red_comp_DCP_op = get_scene_radiance(preprocess(sampled_raw_imgs[i]))
        min_GB_DCP_op = get_scene_radiance_GB(sampled_raw_imgs[i])

        final_imgs.append([sampled_raw_imgs[i],AWB_CLAHE_op,DCP_op,red_comp_DCP_op,min_GB_DCP_op,sampled_ref_imgs[i]])
    # final_imgs will be: [[6 imgs],[6 imgs],[6 imgs],[6 imgs],[6 imgs]]

    
    fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(15,5))
    state = {"current_idx":0}
    # Use partial to attach our local variables to the callback function
    callback_with_args = partial(on_key_press, fig=fig, axes=axs, state=state, final_imgs=final_imgs)
    fig.canvas.mpl_connect('key_press_event', callback_with_args)

    # Display the first class initially
    names_of_images = ["Raw Image","AWB+CLAHE output","DCP output","Red-compensated DCP output","Min Green-Blue DCP output","Reference Image"]
    for col_idx, ax in enumerate(axs):
        ax.imshow(final_imgs[0][col_idx])
        ax.axis('off')
        ax.set_title(f"{names_of_images[col_idx]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualise_outputs()




