# Thermal Texture and Suerpixel Analysis

One of the ways to categorize and analyse the features of an image is to look at the texture of an image. As tactility cannot be conveyed numerically, texture is instead conveyed by the gradient across different directions. Texture is typically described by the response to a bank of filters. An example is the Leung-Malik filter bank which is a set of 48 filters each with a different purpose including edge filters for different orientations and low pass filters of different sigmas.

The thermal camera footage can be converted to an image by normalizing it with respect to individual frame's limits or the datasets limits. Texture might be a way of characterising the behaviour of the thermal response as the distribution of energy might reveal sinks related to cracks or delamination.

## Software Requirements
  - OpenCV 4
  - lm (supplied from [here](https://github.com/CVDLBOT/LM_filter_bank_python_code))
  - scipy
  - numpy

## Software

## Leung-Malik Filter Bank

Leung-Malik (LM) filter bank is a set of 48 patches of size 49 x 49 pixels. Each patch is a filter designed to reveal a specific aspect of the texture. The filter bank is arranged as follows:
  - Filters 1-4 : Gaussian Function (low-pass filter) at four different sigmas
  - Filters 5-12 : Blob filter at eight different scales
  - Filters 13-30 : Edge filter for six different orientations at three different scales
  - Filters 31-48 : Bar filter

Filter banks such as this are used in mixture models such as the Gaussian Mixture Model (GMM) where the texture can be modelled as the combined effect of the responses. In GMM, each response is modelled as a Gaussian distribution described by its mean and standard deviation and each set of parameters has an associated weight describing its importance in the mixture.

## Superpixels SLICO

## Results

The results described are the result of applying the LM filter bank to each image in the pi-camera-data-127001-2019-10-14T12-41-20 dataset. The images have been locally normalized and saved as PNGs separately. The results are stored [here](pi-data-lm-response.7z). The folder is arranged as a series of folders each containing the responses of one image to the bank of filters. Each folder is named after the input image file and the responses are named loosely after the type of filter applied. Only images 91398 to 150000 were used as the more interesting data occurs across this period. The results are in a zipped file as it is a lot of images.

The results were combined together to form a video by arranging each source image and its responses in a 7x7 grid (49 images) with the source image in the top left corner and the filters arranged in the order described earler. Below is a GIF showing the first 60 seconds of the video sped up to 300 FPS. The complete video recorded at 20 FPS can be found [here](pi-texture-response.mp4).

![First 60 secs of LM](pi-texture-response-300.gif) 

![First 60 secs of SLICO](pi-thermal-slico-300.gif)
