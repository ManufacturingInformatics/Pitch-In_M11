# Estimating Laser Parameters and Surface Temperature
## Parameters and Methodologies
This program estimates the following parameters by the following methods:
  * Surface Temperature and Peak Temperature
      * Takes the image data as 16-bit float values representing radiative heat loss and converts it to estimated surface
      temperature for each point
      * Requires a known emissivity value
      * Requires the previous temperature value. Takes room temperature (23 C) for previous temperature of first frame.
  * Laser Power Density
      * Converts surface temperature to estimate of power density (W/m^2), wattage over each pixel area
      * Assumes the laser power is uniform over each area (1D approximation)
        * Reasonable given small enough area
  * Laser Power
      * Estimates the area of effect of laser and multiplies the estimated area by sum of the laser density values within the area
      * Area is estimated by the following methods:
        * KD-Tree to find the non-zero value furthest from the peak power density value. This distance is used to estimate the area
          of effect as a circle
          * The max distance is capped at a certain value
          * Max value estimated from visual inspection of data
          * Massively underestimates the laser power
          * Laser power is estimated instead by re-arranging a formula for describing the behaviour of a He-Ne laser
           * From this [doc](https://web.pa.msu.edu/courses/2010fall/phy431/PostNotes/PHY431-Notes-GaussianBeamOptics.pdf)
        * Area of effect is estimated as the area of the largest contour according to OpenCV (v4.0.1) function findContours with 
          Simple Chain Approximation
          * Over estimates slightly
          * Power density image is converted to an 8-bit image and thresholded using Otsu algorithm
          * Filters out the contour that follows the edge of the image
        * Area of effect is estimated as the area of the largest contour according to OpenCV (v4.0.1) function findContours with 
          no approximation of contours
          * Power density image is converted to an 8-bit image and thresholded using Otsu algorithm
          * Over estimates slightly
          * Very similar behaviour and response to Simple Chain Approximation
          * Filters out the contour that follows the edge of the image
        * Canny Edge detection algorithm in OpenCV (v4.0.1) is used to threshold the image and the findContours algorithm is used
          to estimate the area of effect
          * Largest contour is used for estimate of effect
          * Power estimate is aroud half that of other contour estimates
  * Laser radius
      * KD-Tree to find the non-zero value furthest from the peak power density value.
          * This value is the radius of the total area of effect
          * Assuming a gaussian beam, this radius can be said to be x4 the standard deviation of the beam
          * The standard deviation is the laser beam radius
          * Value is plausible and well within limits
      * Second moment of the laser power density matrix
          * Value is very erratic
          * Suspected that the calculation is not implemented correctly
          
## Files and results included
* Main program file
* Peak and furthest point identified by KD-Tree drawn on thresholded image
* Canny Edge detection thresholded images
* Otsu Thresholded images
  * Only one set as the same algorithm is used for two of the contour algorithms so it is taken to be same for both
* Labelled plots of the results so far
  * Red line represents limit of the parameters
* Plots of the Splines fitted to the material parameter (solid and powder) estimates
  * Thermal Conductivity, K
    * Powder thermal conductivity taken to be 1% of solid conductivity when temperature is below melting point
  * Specific Heat Capacity, C
    * Taken to be the same for powder and solid material
  * Volume Expansion Factor, Ev
    * Estimated as 3x linear expansion factor
  * Solid Material density, p
    * Based off Ev data and a known density value
   * Thermal Diffusivity, D
    * Based off formal formula D = K/Cp
    * Calculated for powder and solid material using the different K splines
    
## Notes
* Splines used in calculation
  * Thermal Conductivity: Extrapolated Spline for powder
  * Thermal Diffusivity: Extrapolated Spline for solid
  * Splines chosen from trial and error of different combinations
* Unknowns
  * Actual emissivity value of the material is unknown
    * Taken to be e=0.55, emissivity for stainless steel
  * Known data for material parameters is over a narrow range (See plots)
    * As splines are cubic, extrapolations over a long temperature range tends to grow rapidly
    * More data or better understanding of the material parameter behaviour is required to choose a better fitting
    * Cubic was chosen for ease of use
    * Functions are provided that generate the data but does not fit to splines within the respective function so 
      the choice of fitting can be changed.
  * The actual values of laser parameters is unknown
    * The limit of the laser power are known from the machine specifications
      * [500-2500] Watts
    * Visual inspection shows that the laser area of effect is well within the limits of the image so it can be inferred
      that the laser radius cannot be greater than the limits of the image
      * 128x128 pixels
      * Pixel pitch is 20*10^-6 m
        * From received document about thermal camera, **NOT** datasheet
      * So, the radius has to be less than:
        * sqrt(128^2 + 128^2) = 181.019 pixels
        * sqrt((128*20*10^-6)^2 + (128*20*10^-6)^2) = 0.00362 m
  
