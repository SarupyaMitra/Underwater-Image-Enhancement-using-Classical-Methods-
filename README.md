##### Dataset Used:

The experiments were conducted on the UIEB(Underwater Image Enhancement Benchmark) Dataset. Dataset Link :- https://li-chongyi.github.io/proj\_benchmark.html

Each image was reshaped to 256\*256 pixels before processing.



###### Important Observation:

Red Colour is absorbed rapidly underwater and typically within 5-10 meters, most of the red wavelength components are significantly attenuated.



##### Implemented Models:

I have implemented 4 different classical models from scratch:

1. Max RGB Automatic White Balancing + Contrast Limited Adaptive Histogram Equalization Model
2. Basic Dark Channel Prior Model
3. Red Channel Compensated Dark Channel Prior
4. Min Green-Blue Dark Channel Prior





##### Model Descriptions:

###### A) Max RGB AWB + CLAHE Model:

 	In this model, Max-RGB Automatic White Balancing was applied to the input raw image and then applied CLAHE. For CLAHE processing, the RGB image was converted to LAB space and CLAHE was applied to the L-channel only. The L-channel was divided into 4\*4 non-overlapping blocks, each block having 64\*64 pixels.



###### B) Basic DCP:

 	This model applies the Dark Channel Prior(DCP) based dehazing algorithm described in \[1]. The reason for this model is that underwater environments behave similarly to hazy or foggy atmospheres, as water acts as a scattering medium.



###### C) Red Channel Compensated DCP:

 	Since red wavelengths are first absorbed, underwater images usually suffer from significant red channel attenuation.


To compensate for this effect, red channel is enhanced by:

a) Computing the mean chromatic values of red, green and blue channels. Let them be r', g', b'.

b) To determine whether red channel compensation is needed, the following ratios are evaluated: g'/r' and b'/r'. If either of these ratio exceeds 1.3, red channel compensation is applied.

c) In perfectly balanced image, the expected chromatic mean of each channel is approximately 1/3. The red deficit is computed as:
        deficit = 1/3 - r'
The corresponding gain factor is then defined as:
        gain = deficit / (1/3)
d) For each pixel, the chromatic red component is adjusted using the following transformation:
        r_new(x,y) = r(x,y) + gain\*r(x,y)\*(1-r(x,y))
The final compensated red channel is reconstructed as:
        R_new(x,y) = (R(x,y) + G(x,y) +  B(x,y))\*r_new(x,y)



###### D) Min Green-Blue DCP:

 	This model also tries to solve the Red channel problem, but in a different way. While computation of the standard dark channel we basically go to each spatial location(say, (x,y)) and set its value to minimum(R(x,y),G(x,y),B(x,y)) where C(x,y) denotes the channel value at location (x,y).

But since the Red channel values are typically low in underwater images, the dark channel is mostly dominated by the red channel values introducing a bias. 

To mitigate the issue, the dark channel is computed only using Green and Blue channels: minimum(G(x,y),B(x,y)).

This modification reduces the bias creted by the red channel and leads to improved transmission estimation.





##### References:

\[1] : K. He, J. Sun and X. Tang, "Single Image Haze Removal Using Dark Channel Prior," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 12, pp. 2341-2353, Dec. 2011.

