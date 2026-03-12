To evaluate the performance of the implemented underwater image enhancement models, three metrics are used:
1) PSNR (Peak Signal-to-Noise Ratio)
2) SSIM (Structural Similarity Index Measure)
3) UIQM (Underwater Image Quality Measure)
The first two metrics require reference images, while the third is a no-reference underwater image quality index.


NOTE: The reference images in the dataset were created using Deep Learning based enhancement methods and hence classical methods cannot exactly reproduce the reference images. So SSIM and PSNR values are not always accurately reflecting the perceptual improvement produced by the classical methods. Therefore, while PSNR and SSIM are reported for completeness, UIQM is a more meaningful metric for under water image enhancement algorithms.



The Mean UIQM of raw images in the dataset is 0.9856293082993665

1. ##### **Automatic White Balancing(max RGB) + Contrast Limited Adaptive Histogram Equalization:**

    For CLAHE, the image was broken into 4\*4 non-overlapping blocks each blocks having 64\*64 pixels. The results:



&nbsp;		Mean PSNR: 14.491193873872154

&nbsp;		Std PSNR: 2.0849822476830884

&nbsp;		Mean SSIM: 0.6596929965375062

&nbsp;		Std SSIM: 0.13256120415080697

&nbsp;		

&nbsp;		Mean UIQM of output: 2.1898920166317897

&nbsp;		Std UIQM of output: 0.4448216635136627





##### 2\. Basic Dark Channel Prior(DCP):

The results are:



&nbsp;	Mean PSNR: 6.0638328780235975

&nbsp;	Std PSNR: 1.1356170919497688

&nbsp;	Mean SSIM: 0.011151461206919226

&nbsp;	Std SSIM: 0.02295188788138808

&nbsp;	

&nbsp;	Mean UIQM of output: 1.2913712992695061

&nbsp;	Std UIQM of output: 0.5419831039570466

##### 

##### 3\. Red channel compensated DCP:

The results are:



&nbsp;	Mean PSNR: 6.061178886068516

&nbsp;	Std PSNR: 1.135729736348809

&nbsp;	Mean SSIM: 0.010828370775700634

&nbsp;	Std SSIM: 0.022863174303680886



&nbsp;	Mean UIQM of output: 1.393124910810237

&nbsp;	Std UIQM of output: 0.5463559640663779



##### 4\. Min Green-Blue DCP: 

The results are:



&nbsp;	Mean PSNR: 6.059335138146466

&nbsp;	Std PSNR: 1.1351206865766215

&nbsp;	Mean SSIM: 0.010397026153530846

&nbsp;	Std SSIM: 0.022695533819141165

&nbsp;	

&nbsp;	Mean UIQM of output: 1.6995861964285832

&nbsp;	Std UIQM of output: 0.5171365096593712


#### Interpretation:

In terms of UIQM, all enhancement methods improve the UIQM score compared to the raw images.

AWB+CLAHE model achieved highest UIQM, which indicates strong improvements in contrast and color balance.

Among the Dark Channel Prior based methods, the Min Green-Blue DCP produces a higher UIQM score compared to basic DCP variants, suggesting that removing the red channel bias imporves perceptual quality.
