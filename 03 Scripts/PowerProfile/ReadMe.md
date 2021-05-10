# Power Profile Estimation
This folder is the collection of methods used to estimate the power profile from the estimated power density matricies

## Requirements
 + [Numpy](https://www.numpy.org/)
 + [SciPy](https://www.scipy.org/)
 + [skimage](https://scikit-image.org/)
 + [OpenCV v4.x.x](https://opencv.org/)
 
## Power Estimation
When a laser is emitted from a source it's power is distributed over a certain area. Power density is in terms of W/m2
(Watts per Metre Squared) meaning you can calculate the power of the laser by summing the power density values within the laser's
area of effect (LAOE) and multiplying it by the AOE's area. The difficulty then comes from detecting that area of effect. The 
challenge is choosing an algorithm that can repeatedly pickout the LAOE across the entire data set with a high accuracy.

The main challenge for whichever algorithm is chosen is distinguishing the data we're interested in from the noise that persists
across each frame. The noise is present as two "bands" across the top and bottom of the image as shown below.

As the LAOE tends to overlap with these noise "bands", the boundary is destroyed or lost within the noise very easily.

As we're searching for a distinctive spatial artifact in the data, image processing techniques are employed as they are designed to
pick out objects within 2D data. So far, the best method is a mixture of Canny Edge detection and contour finding functions.

The Canny Edge detection method can be viewed as a low pass filter as it searches for pixels whose gradient relative to nearby
pixels is high enough to be identified as an edge. So if a pixel doesn't vary much relative to its neighbours, like say when it
is in the noise "bands", it is not regarded as distinct enough. The edge results are passed to a contour finding algorithm
that searches for enclosed shapes such as the LAOE. The results from this algorithm are shown below:

As you can see, it picks up the periods of activity but the calculated values vary greatly from one period to the next. When 
compared with the target power value of 500W (green line) it repeatedly over and under estimates the values. To improve these
results, a number of pre-processing techniques and their impacts where investigated. Only a brief description is given here, please
visit the respective folders for more information.

## Methods Investigated
### Canny Parameter Selection
The Canny function has two parameters; lower threshold and upper threshold. If the processed gradient is equal to or higher than
the upper threshold, then it is marked down as an edge. If it is less than the lower threshold, then it is not an edge. Anything
inbetween is disregarded as the algorithm is not sure whether it is an edge or not. As there's no sure way of choosing the ideal
values, this script takes a brute force approach and searches a given parameter space for ideal values.

For more information on the Canny algorithm, look [here](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html)

### Fourier Transform
The Fourier transform is an algorithm that searches for the magnitude and frequency of the components that collectively makeup
the data. It's typically applied to time-based data where the frequency indicates how often an object occurs across a given
period of time. When applied to 2D data, the frequency indicates how often a given artifact occurs across the 2D parameter
space. In the case of a camera monitoring a physical area, it is how often a component across the physical space. The idea with
this is to filter for just the components the laser introduces and apply the algorithm to the results.

For a better explanation of this can be found [here](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html)

### Distance Metric
When the laser is turned on, the recorded data changes dramatically as the laser introduces more heat to a target area.
Distance metrics are a way of measuring and weighting this change from one pixel to the next. The idea is that the change
brought about by the laser will indicate the LAOE making the boundary easier to detect. The problem then becomes choosing
which metrics can be used with the data and whether the results are of any use.

### Hough Circle Transform
The Hough Circle Transform is an algorithm that searches for circles in a given image. It can process the image directly or on representative information such as the gradient of edges in the image. The algorithm searches for and ranks circles based on how likely the circle actually in the image. For example, a circle whose radius is the width of the image will contains alot of the values in the image but without a distinct border will be ranked low. Visually inspecting the LAOE can show that the boundary is mostly circular with a distinct border that is likely detectable.

### Notes
Some of these programs can take several hours to run so please make sure your computer is up to the task before running the
scripts for yourself.
