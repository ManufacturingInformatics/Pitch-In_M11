
## Results

As the complete results are over 20GB in size, a single frame has been selected to represent the impact of different models and scaling factors. Frame 1048 was selected as it shows the author displaying several fingers which can be viewed as an image with several distinct features.

Below is the reference image in the different data types colormapped to make it a bit easier to see.

| **uint8** | **uint16** | **float32** | **float64** |
|:---------:|:---------:|:---------:|:---------:|
|![](./ContoursBase/pi-frame-1048-uint8-contours.png)|![](./ContoursBase/pi-frame-1048-uint16-contours.png)|![](./ContoursBase/pi-frame-1048-float32-contours.png)|![](./ContoursBase/pi-frame-1048-float64-contours.png)|

A few differences can already be seen with the different data types. Under this colormap, the darker blue indicates a lower temperature and black the hottest. When the 8-bit reference image is compared to the 16-bit, we can see that the 16-bit has a greater range of lower temperatures indicated with the blue colors. As expected, the 16-bit image has a richer range of colors. The floating point images are showing similar behaviour with the range of range of blue colors it displays. Interestingly, it's displaying higher temperatures in the palm area compared to the other data types.

The author notes that these values are of course affected by the number of levels used in the colormap and any data truncation caused when the image was read in.

When the program was executed, a large number of error messages were displayed indicating that the denoised versions of the 32-bit floating point images could not be read back in citing an inability to decode the bytes. This suggests that when Waifu2x denoises the images and attempts to "fill in" the missing areas, it somehow generates or corrupts the data. To handle this without stopping the program, the metrics are all left as their default values of 0. This behaviour did not apply to 64-bit floating point images.

### Minimum Values
#### uint8
##### Minimum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/diff-min-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/diff-min-temp-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/diff-min-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/diff-min-temp-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/diff-min-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/diff-min-temp-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/diff-min-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/diff-min-temp-all-denoise-uint8.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/var-diff-min-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/var-diff-min-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/var-diff-min-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/var-diff-min-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/var-diff-min-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/var-diff-min-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/var-diff-min-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/var-diff-min-all-denoise-uint8.png)|

##### Maximum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/diff-max-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/diff-max-temp-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/diff-max-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/diff-max-temp-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/diff-max-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/diff-max-temp-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/diff-max-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/diff-max-temp-all-denoise-uint8.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/var-diff-max-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/var-diff-max-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/var-diff-max-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/var-diff-max-all-denoise-uint8.png)|

##### Variance
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/diff-var-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/diff-var-temp-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/diff-var-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/diff-var-temp-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/diff-var-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/diff-var-temp-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/diff-var-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/diff-var-temp-all-denoise-uint8.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/var-diff-max-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/var-diff-max-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/var-diff-max-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/var-diff-max-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/var-diff-max-all-denoise-uint8.png)|

##### Temperature Range
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/diff-range-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/diff-range-temp-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/diff-range-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/diff-range-temp-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/diff-range-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/diff-range-temp-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/diff-range-temp-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/diff-range-temp-all-denoise-uint8.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint8/var-diff-range-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint8/var-diff-range-all-denoise-uint8.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint8/var-diff-range-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint8/var-diff-range-all-denoise-uint8.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint8/var-diff-range-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint8/var-diff-range-all-denoise-uint8.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint8/var-diff-range-all-denoise-uint8.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint8/var-diff-range-all-denoise-uint8.png)|

#### uint16
##### Minimum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/diff-min-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/diff-min-temp-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/diff-min-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/diff-min-temp-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/diff-min-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/diff-min-temp-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/diff-min-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/diff-min-temp-all-denoise-uint16.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/var-diff-min-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/var-diff-min-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/var-diff-min-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/var-diff-min-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/var-diff-min-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/var-diff-min-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/var-diff-min-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/var-diff-min-all-denoise-uint16.png)|

##### Maximum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/diff-max-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/diff-max-temp-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/diff-max-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/diff-max-temp-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/diff-max-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/diff-max-temp-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/diff-max-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/diff-max-temp-all-denoise-uint16.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/var-diff-max-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/var-diff-max-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/var-diff-max-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/var-diff-max-all-denoise-uint16.png)|

##### Variance
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/diff-var-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/diff-var-temp-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/diff-var-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/diff-var-temp-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/diff-var-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/diff-var-temp-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/diff-var-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/diff-var-temp-all-denoise-uint16.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/var-diff-max-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/var-diff-max-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/var-diff-max-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/var-diff-max-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/var-diff-max-all-denoise-uint16.png)|

##### Temperature Range
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/diff-range-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/diff-range-temp-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/diff-range-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/diff-range-temp-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/diff-range-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/diff-range-temp-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/diff-range-temp-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/diff-range-temp-all-denoise-uint16.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/uint16/var-diff-range-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/uint16/var-diff-range-all-denoise-uint16.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/uint16/var-diff-range-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/uint16/var-diff-range-all-denoise-uint16.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/uint16/var-diff-range-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/uint16/var-diff-range-all-denoise-uint16.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/uint16/var-diff-range-all-denoise-uint16.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/uint16/var-diff-range-all-denoise-uint16.png)|

#### float64
##### Minimum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/diff-min-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/diff-min-temp-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/diff-min-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/diff-min-temp-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/diff-min-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/diff-min-temp-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/diff-min-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/diff-min-temp-all-denoise-float64.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/var-diff-min-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/var-diff-min-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/var-diff-min-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/var-diff-min-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/var-diff-min-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/var-diff-min-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/var-diff-min-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/var-diff-min-all-denoise-float64.png)|


##### Maximum Value
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/diff-max-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/diff-max-temp-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/diff-max-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/diff-max-temp-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/diff-max-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/diff-max-temp-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/diff-max-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/diff-max-temp-all-denoise-float64.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/var-diff-max-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/var-diff-max-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/var-diff-max-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/var-diff-max-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/var-diff-max-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/var-diff-max-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/var-diff-max-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/var-diff-max-all-denoise-float64.png)|

##### Variance
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/diff-var-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/diff-var-temp-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/diff-var-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/diff-var-temp-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/diff-var-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/diff-var-temp-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/diff-var-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/diff-var-temp-all-denoise-float64.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/var-diff-var-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/var-diff-var-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/var-diff-var-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/var-diff-var-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/var-diff-var-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/var-diff-var-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/var-diff-var-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/var-diff-var-all-denoise-float64.png)|

##### Temperature Range
###### Difference
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/diff-range-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/diff-range-temp-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/diff-range-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/diff-range-temp-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/diff-range-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/diff-range-temp-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/diff-range-temp-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/diff-range-temp-all-denoise-float64.png)|

###### Variance
|**anime style art**|**anime style art RGB**|
|:-:|:-:|
|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art/float64/var-diff-range-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/anime_style_art_rgb/float64/var-diff-range-all-denoise-float64.png)|
|**CUNet**|**Photo**|
|![](arrowshape-temperature-HDF5-denoise-Plots/cunet/float64/var-diff-range-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/photo/float64/var-diff-range-all-denoise-float64.png)|
|**UKBench**|**Upconv 7 anime style art RGB**|
|![](arrowshape-temperature-HDF5-denoise-Plots/ukbench/float64/var-diff-range-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_anime_style_art_rgb/float64/var-diff-range-all-denoise-float64.png)|
|**UpConv 7 Photo**|**Up ResNet 10**|
![](arrowshape-temperature-HDF5-denoise-Plots/upconv_7_photo/float64/var-diff-range-all-denoise-float64.png)|![](arrowshape-temperature-HDF5-denoise-Plots/upresnet10/float64/var-diff-range-all-denoise-float64.png)|

### denoised Images

Under the folder [Contours](./Contours) are a series of colormapped images showing the results of scaling frame 1048 using different models and scaling factors. As there are a lot of images, only the results using scaling factor 10 are shown in this documents. It is up to to the user if they want to compare the whole set.

<table>
    <thead>
        <tr>
            <th>uint8</th>
            <th>uint16</th>
            <th>float32</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center" colspan=3>Original</td>
        </tr>
        <tr>
            <td align="center"><img src="./ContoursBase/pi-frame-1048-uint8-contours.png" width="240" height="180">
            <td align="center"><img src="./ContoursBase/pi-frame-1048-uint16-contours.png" width="240" height="180">
            <td align="center"><img src="./ContoursBase/pi-frame-1048-float64-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>Anime Style</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/anime-style-art-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/anime-style-art-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/anime-style-art-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>Anime Style RGB</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/anime-style-art-rgb-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/anime-style-art-rgb-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/anime-style-art-rgb-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>CUNet</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/cunet-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/cunet-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/cunet-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>Photo</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/photo-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/photo-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/photo-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>UKBench</td>
        </tr>
        <tr>
            <td align="center" colspan=3>UpConv7 Anime Style Art RGB</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/upconv-7-anime-style-art-rgb-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upconv-7-anime-style-art-rgb-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upconv-7-anime-style-art-rgb-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>UpConv7 Photo</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/upconv-7-photo-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upconv-7-photo-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upconv-7-photo-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr>
        <tr>
            <td align="center" colspan=3>UpResNet10</td>
        </tr>
        <tr>
            <td align="center"><img src="./Contours/upresnet10-uint8-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upresnet10-uint16-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
            <td align="center"><img src="./Contours/upresnet10-float64-denoise-3-pi-frame-1048-contours.png" width="240" height="180">
        </tr> 
    </tbody>
</table>
