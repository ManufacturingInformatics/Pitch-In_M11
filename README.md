# Hopper Data Processing

A variety of data can be captured and processed from the powder hoppers. This data can be used to infer information that isn't or can't be measured from the hopper directly. This project relates to two problems; estimating the angular velocity (RPM) of the powder scraper and analysing the behaviour of the powder artifact that occurs on the output.

## Software
 - Python 3 (tested on v3.7.5)
 - OpenCV 4 with contrib
 - scikit-image
 - Numpy
 - networkx
 
## Hardware Used

**NO HARDWARE YET**

## Notes On Folder Structure

The scripts and results relating to estimating the scraper roating speed are in the [ScraperRPM](03%20Scripts/ScraperRPM) folder.

The scripts and results relating to identifying and processing the scraper powder artifact are in the [PowderSegmentation](03%20Scripts/PowderSegmentation) folder.

Each folder may have its own README or similar documentation explaining specific scripts and providing background on specific techniques and concepts used.

## Analysing Powder Artifact

The purpose of the scraper in the hopper is to pull the powder around from the source and deposit it at the output where it is combined with a carrier gas and sent to the BEAM machine. However due to the nature of the powder and the angle of its linear velocity, the powder rotates back on itself to form a spiral or vortex. This artifact affects the feed rate of the powder to the build site. The goal is to use an imaging system to segment the powder from the background and identify some parameters or information about it. The specific parameters of interest are not known at this stage so the goal is to segment the powder from the background.

### Segmentation

Separating a powder from the background is a difficult task to its poorly defined edges, dynamic nature and in this case the fast moving nature of it. It is likely a machine learning method will need to be employed to properly segment it. However, as we can't get a significant amount to data to train a new system, a pre-existing, trained system would be preferrable.

**NO EXAMPLE FOOTAGE IS AVAILABLE AT PRESENT SO THE PAPERS BELOW SHOULD BE TREATED AS A LITERATURE REVIEW**

#### Useful links

[blog post explaining the basics of graph cutting in image segmentation](https://sandipanweb.wordpress.com/2018/02/11/interactive-image-segmentation-with-graph-cut/)

#### Notable papers
##### "GrabCut" Interactive Foreground Extraction using Iterated Graph Cuts ([here](01%20Inputs/fromUoS/Papers/ImageSegmentation/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts.pdf))

Rother et al propose an extension of the popular graph cutting technique for images by arranging the probelm of separating the foreground and background as an optimization problem that operates in minimal labelling by the user. The result is a fast and stable algorithm. It has been incorporated into the popular image processing library [OpenCV](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html). The improvements made by the authors are primarily come from the way they arrange the color modelling problem as an optimization problem.

A popular way to model the features of an image is to use a Gaussian Mixture Model (GMM) which models a set of features as a combination of Gaussian distributions. Examples of features include the different color channels and the response to a filter representing filter. Each distribution is described by a mean and standard deviation parameter. The energy required to segment the foreground and background is described using the Gibbs Energy form combining a unary term and a smoothing term. The unary term is composed of the different Gaussian parameters describing the foreground and background, the data and which class the data points belong to. The smoothing parameter is based off Euclidean distance in color space and contrast between data points. By arranging it as an iterative minimization problem, the solution has a better "understanding" of the data compared to previous single step learning algorithms and requires significantly less input from the user. For this algorithm the user only needs to supply a broad and incomplete labelling. In the OpenCV implementation, this is in the form of a filled bounding box representing the starting region for the foreground. The optimization is also arranged such that it is guaranteed to reach a stable solution in a detectable manner.

The authors note that whilst the guaranteed stability is a bonus, it doesn't ensure it will reach a good solution. The incomplete labelling supplied by the user can also lead the optimization to a poor conclusion. Additional runs on the algorithm with more precise labelling or a different segmentation algorithm may be required to clean up the results.

##### Multispectral Imaging for Fine-Grained Recognition of Powders on Complex Backgrounds ([here](01%20Input/fromUoS/Papers/ImageSegmentation/multispectral-imaging-for-fine-grained-recognition-of-powders-on-complex-backgrounds.pdf))

Zhi et al. investigates using different frequencies of light to identify materials in an image. Through this they have created a database of materials and example images. The examples are a mixture of backgrounds with and without powders. As it's difficult to obtain a representative set of examples showing a powder's appearance across different backgrouds, the authors also propose a method generating artificial powders. The materials are identified based on their response with a multispectral camera looking at the visible spectrum and the infrared spectrum. One of the conclusions found was that a wavelength of 1000-1700nm can identify powders with little colour information in the other spectrums. 

The camera being used could record up to 951 bands of light. As this volume and dimensionality would drastically reduce the performance of the ML application, the authors identified an optimal set of 34 bands using greedy Nearest Neighbour Cross Validation score.

The blending model used to generate synthetic data is called Beer-Lambert Blending Model. The model generates the intensity of a thin powder as a function of the intensity of an infinitely thick powder and thhe background. The parameters controlling the generator are powder scattering coefficient, thin powder intensity, mean thick powder intensity and the mean background intensity. The model is based around manipulating the transparency of the generated data to give the illusion of a blurred edge. This falls under a class of blending called Alpha Blending.

The authors record an accuracy of 60-70% with known powder location and over 40% mean Insersection of Union without a known location.

The Github for the paper and database can be found [here](https://github.com/tiancheng-zhi/ms-powder)

##### Tracing liquid level and material boundaries in transparent vessels using the graph cut computer vision approach ([here](01%20Input/fromUoS/Papers/ImageSegmentation/Tracing-liquid-level-and-material-boundaries-in-transparent-vessels-using-the-graph-cut-computer-vision-approach.pdf))

Eppal proposed an improved method for identifying the bounding of different materials and phases inside a transparent container. Identifying the boundaries of materials at different stages is essential for automating chemistry research. The method is based on a methodology called Graph Cut which treats pixels as verticies and similarity between neighbouring pixels as edges of a graph. The goal is to take this "graph" and cut it into two separate graphs by removing the edges between dissimilar sections. In the case of images it's identifying the boundary between an object and the background. The sequence of edges to remove is based on the inverse of the intensity difference between neighbouring pixels. An extension of this method is the max-flow/min-cut method which treats the pixels as a flow of something from a source to a sink. The source would be the object of interest and the sink is the other side oft he boundary. The goal becomes finding the minimum cut that separates the source from the sink.

As it is assumed that the vessel is upright with the top of the vessel at the top of the image, the author makes the assumption that the bottom 10% of the vessel is covered by the target material while the top 10% is air. This defines the material's pixels as the source and the air's pixels as the sink. The main issue with this approach is the requirement to define the image region corresponding to each phase before hand.

This method performs well even with powdered materials and materials with an irregular surface.

The author has made the C++ source code for the method available [here](https://github.com/sagieppel/Tracing-liquid-level-and-material-boundaries-in-transparent-vessels-using-the-graph-cut--maxflow-mod).

##### Supervised and unsupervised segmentation using superpixels, model estimation, and graph cut ([here](01%20Input/fromUoS/Papers/ImageSegmentation/Supervised-and-unsupervised-segmentation-using-superpixels-model-estimation-and-graph-cut.pdf))

Superpixels has proved to be viable way of reducing and classifying images. Each of the superpixels can be classified in a number of ways including color and texture. By associating the features with different segmentation classes (e.g. background, midground, foreground etc.), an algorithm can learn how to segment images into different levels. It is up to the designer how to weight the differences between feature vectors of superpixels. This paper extends previous works by applying spatial regularization to the superpixels to maintain continuity between them and presents a new weighting algorithm.

Superpixels in this paper are described by their color and texture. Texture is described by a Leung-Malik filterbank that applies a series of different filters and combines it with the intensity of an image's channel to creatre the texture response. The response becomes rotation invariant by choosing the maximum response over all orientations. Color is described by the mean, standard deviation, energy and median of each color channel in the superpixel.

For an unsupervised learning problem, the classes are derived using a Gaussian Mixture Model. Each class is described as a mixture of probabilities across each feature. The mixing parameters are estimated using the expectation maximization (EM) algorithm. In a basic situation, the EM is applied to each image independently assuming each image is different from the others but can be applied to a set of images if some aspect leaks them. Both cases are explored in the paper.

The weighting algorithm proposed by the authors is based on the comparison of posteriori class probabilities. These distributions provide an idea as what's important for the particular application weighting certain classes over others.

The code developed as part of the paper can be found [here](https://github.com/Borda/pyImSegm).

**UNABLE TO INSTALL AND RUN GITHUB CODE. UNABLE TO COMPILE gco-wrapper WHICH IS REQUIRED TO PERFORM THE GRAPH CUTTING IN THEIR EXAMPLE**

In the current case of separating the powder from the background, both supervised and unsupervised cases are viable depending on the amount of footage obtained.

##### Hierarchical semantic segmentation using modular convolutional neural networks ([here](01%20Input/fromUoS/Papers/ImageSegmentation/hierarchical-semantic-segmentation-using-modular-convolutional-NN.pdf))

Sagi and Eppel developed a modular Convolutional Neural Network (CNN) to segment an image at increasing levels of detail. The modular aspect comes from the ability to use each network separately once trained and that each network was trained independently allowing a "plug and play" mechanic. The problem context they were using to test their methodology was identifying a glass laboratory-grade vessel and segmenting their contents into different components. They connected in series a network to segment the vessel from the background and another to separate the components into different groups. The range of possible groups are empty or filles, liquid or solid and granular types of materials including emulsion, vapor and suspension. The second network was trained using a database of material images collected from a range of YouTube channels with the owner's permission. The database is available [here](https://github.com/sagieppel/Materials-in-Vessels-data-set). It was shown by the authors that their modular system outperforms a single Fully Convolutional Neural Network (FCNN) trained to perform both tasks.

One of the key features of this network is the use of valve filters to limit which feature detectors are applied to which regions of the image. A valve filter is convolved with the generated segmentation mask is convolved to form a relevance mas. This controls where in the image a specific filter is likely to be more relevant/useful. This new mask is multiplied with the feature map to form a Normalized Feature Map (NFM) which is passed as an input (after RELU) to the next layer. The map controols the activation of features whether they appear in the region of interest or the background. In other words, there's no point in detecting features in something outside of what we're interested in.

The code for the second network that segments the contents can be found [here](https://github.com/sagieppel/Focusing-attention-of-Fully-convolutional-neural-networks-on-Region-of-interest-ROI-input-map-).

The code for the combined FCN that performs both background segmentation and contents segmentation can be found [here](https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation).

One possible use of this paper is to use the content segmentation algorithm to separate the powder from the background where the initial region of interest mask passed to it is perhaps just a cropped version of the image.

## Estimating Scraper RPM

The hopper is part of the BEAM setup that takes the powdered metal and attempts to feeds it into the machine at a target feed rate. However, the target feed rate is rarely stable and often has an oscillation accompanying it. Part of the hopper is a device called a scraper which when the powder drops down literally scrapes and pulls the powder around to the chute (?) where the powder travels to the machine. It is believed that the RPM of the scraper is related to the oscillations in the feed rate. However, there is no direct way at present to read the RPM of the scraper. The motor controlling the scraper is inaccessible so recording devices cannot be connected to read directly from it. The purpose of this project is to develop an image processing system that can record the scraper and estimate the RPM from its motion.
 
### Image Processing
 
Tracking an object in an image is a well researched area with a wide scope of techniques including machine learning available. Each method is based around a specific way in which an object is tracked and how the results are returned. For example it could be identifying and following the bounding box surrounding an object, the velocity of a group of pixels or uniquely identifying each object in a frame and tracking them. Below are the techniques investigated as part of this project and the results

#### Evaluation 

As footage of the scraper wasn't available early on in the project, artificial data was generated to simulate the motions of the tracked dot. The file ![RotatingDotGen](03%20Scripts/ScraperRPM/RotatingDotGen.py) contains a set of classes that generates a black dot on a white background rotating about the centre of the image. It also includes a threaded animator class that rotate the dot to simulate the process. A white background was chosen as it is known that the scraper object is white/off-white and the dot is black as it's a distinct color that can be easily identified relative to the background.

![Rotating Animation](03%20Scripts/ScraperRPM/rotating_dot_animation.gif)

Each technique is evaluated by starting an animation and feeding the frames into the detector. The animation classes keep track of the dot's current angular position so the accuracy of the estimated position can be compared. The rate at which the animation updates is also controllable which in turn gives us control over the RPM of the rotated dot.

Below is a couple of plots showing the recorded angular position and estimated angular velocity of a rotating dot moving at approximately 5.33 rad/s. The velocity is set based on the set period between updates and how much the dot is moved by. The velocity is estimated as the recorded change in angular position divided by time between measurements. The angular position is calculated from the position of the dot in the image relative to the centre of the image.

![Recorded angular position from animator](03%20Scripts/ScraperRPM/dot_anim_debug_angle_position.png)

![Estimated angular velocity of animator](03%20Scripts/ScraperRPM/dot_anim_debug_angle_velocity.png)

The initial jump in the beginning of the position plot is due to the time between the animation being started and the data being recorded. The first value is set as 0.0 radians but by the time the first measurement occured the animator has moved the dot.

The periodic jumps in the velocity come from how the angle updates. When the dot is perpendicular to the right vertical side the angle flips from positive pi radians to negative pi radians. When estimating velocity this is read as a large value hence the spike.

When we zoom in the majority of the behaviour outside of the spikes we see the following.

![Zoomed in plot of the estimation of the angular velocity](03%20Scripts/ScraperRPM/dot_anim_debug_angle_velocity_zoom.png)

The red dotted line denotes the target velocity based off the update rate. As we can see, the animation isn't smooth with updates and introduces a sinusoidal behaviour into the response. The sudden drops to zero are from the data being sampled too quickly and the animator not moving the dot yet. This can be avoided with a correctly chosen sampling rate.

#### Ball Tracker

The name Ball Tracker refers to the example code from [here](https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/) which the program is based off. The goal of this method is to segment the rotating dot from the background and estimate it's position with respect to the centre of the image (the assumed point of rotation). From there the angular velocity can be estimated based on the change in the angular position over time.

This method has three stages of processing the first of which is identifying pixels within a certain target color range. As we're tracking a non-white blob on a white background, this method is suitable as we control what color the blob is and therefore can confidently define the target range. While this stage does most of the work it can pick up small artifacts or individual pixels that sneak inside the range, which is why the second stage is denoising and artifact removal. 

This stage is simply applying erosion followed by dialation transforms to the masked data. The erosion removes small artifacts while the dialation expands the borders of larger artifacts making them easier to detect whilst undoing any unintentional damage caused by the erosion. 

The final stage is contour edge detection to find the border of the blob and from there it's centre. The contours defining the edge is a series of coordinates marking the edge. The centre of the blob is found by finding the moments of these points and calculating the coordinates of the centroid. In image processing, the moments are weighted averages of the pixel intensities that can be used to find properties of a shape such as the centroid which can be thought of as the "centre" of the shape. The coordinates are defined from the moments as follows.

|||
|:-:|:-:|
|![Centroid X position](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-a69df753d9fa4da6805fbc1c7a312207_l3.png)|![Centroid Y Position](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-4d46e9d273a00cb5d12fc74e052a6e91_l3.png)|

The distance from the centroid the centre of the image is then found and then the angle between the distance components is calculated. The angular velocity is then calculated as the change in this angle between two known time points divided by the time period. The angular velocity can be estimated from the linear velocity but this has proved to be unreliable so far.

Below is a GIF showing the different stages of this process. The yellow circle drawn at the end is the smallest circle that closes the contour points and the red trail is the history of centroids founds. As you can see tracks the artificial data pretty well.

![Steps of the BallTracker class](03%20Scripts/ScraperRPM/rotating_dot_balltracker_debug.gif)

When the estimated angular velocity over a period of time is plotted we get the following history. The varying nature of the peaks is similar the velocity behaviour extracted from the animation class directly. The red dotted line represents the target velocity of the animator calculated from the change in angle each update and the time period between updates. The target velocity in this case is around 5.33 radians/s.

![Estimated velocity from BallTracker class](03%20Scripts/ScraperRPM/rotating_dot_balltracker_est_vel.png)

#### Optical Flow

When a moving object is captured by a camera, it (usually) appears smeared or blurred creating a trail behind it. Optical flow refers to the blur pattern that is generated. The faster the object is moving, the greater the blue or distortion between frames.

![Moving tennis ball](https://thumbs.dreamstime.com/b/tennis-ball-moving-motion-blur-background-black-white-73706007.jpg)

By studying how the object pixels are distorted by motion, we can estimate how fast it is moving within the pixel space. Optical flow works under the assumption that the pixel intensities do not vary between consecutive frames and the pixels have similar motion. The first assumption is to provide an environment where we can track a pixel. If a pixel changes color between frames then we cannot realistically track it. The second assumption simply means that the object of interest is collectively performing a single motion (e.g. moving from right to left) rather than different sections performing different movement. Optical flow is about looking at how pixels of the same or very similar intensity have changed one from one frame to the next. This movement is expressed as movement vectors in the **x** and **y** directions. OpenCV supports two method of optical flow analysis; Lucas-Kanade method (LK) and Gunnar-Farneback method (GF).

LK (paper [here](01%20Input/fromUoS/Papers/MotionDetection/an-iterative-image-registration-technique-with-an-application-to-stereo-vision.pdf)) is about tracking a limited set of points defined by a set of features. The points are identified in one frame and tracked across future frames. In the program developed based off the OpenCV [documentation](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html), the points are based off Shi-Tomasi corner points. The results are returned as the location of points in the frame that contain the identified features.

For the artificial rotating dot, LK typically finds four data points to track as shown by this GIF below.

![Tracked points from Lucas-Kanade method](03%20Scripts/ScraperRPM/rotating_dot_opticflow_LK.gif)

The angular velocity of the blob at present is represented as an aggregate of the tracked points. The angular velocity of each point relative to the centre of the image is calculated based off the change between the locations found in two separate frames. It is assumed that the order of the found pixels is the same between frames and the reference frame was updated every five frames. The velocities are then combined by looking at either the minimum, maximum or mean. The best function out of these three has yet to be determined.

In comparison GF (paper [here](01%20Input/fromUoS/Papers/MotionDetection/two-frame-motion-estimation-based-on-poly-expansion.pdf)) is about tracking all the pixels in a frame and calculating the individual component velocities of every pixel. The motion is calculated based on the change between two consecutive frames. The results are returned as a two channel array the same size as the input image. The GIF below shows the velocities as red with different brightness levels.

![Estimated centroid (blue) using Gunnar-Farneback method](03%20Scripts/ScraperRPM/rotating_dot_opticflow_centroid_GF.gif)

#### Object Trackers

![Output of All Default Trackers](03%20Scripts/ScraperRPM/rotating_dot_trackers.gif)

##### Angular Position

|CSRT|KCF|MedianFlow|MIL|
|:---:|:---:|:---:|:---:|
|![CSRT](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerCSRT.png)|![KCF](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerKCF.png)|![Median Flow](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerMedianFlow.png)|![MIL](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerMIL.png)|

|MOSSE|Boosting|TLD|
|:---:|:---:|:---:|
|![MOSSE](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerMOSSE.png)|![Boosting](03%20Scripts/ScraperRPM/rotating-dot-tracker-ang-TrackerBoosting.png)|![TLD](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-ang-TrackerTLD.png)||

##### Angular Velocity

|CSRT|KCF|MedianFlow|MIL|
|:---:|:---:|:---:|:---:|
|![CSRT](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerCSRT.png)|![KCF](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerKCF.png)|![Median Flow](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerMedianFlow.png)|![MIL](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerMIL.png)|

|MOSSE|Boosting|TLD|
|:---:|:---:|:---:|
|![MOSSE](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerMOSSE.png)|![Boosting](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerBoosting.png)|![TLD](03%20Scripts/ScraperRPM/Plots/rotating-dot-tracker-vel-TrackerTLD.png)|

#### Kalman Filter

![Kalman Velocity Filter Results](03%20Scripts/ScraperRPM/Animations/rotating_dot_kalman.gif)
