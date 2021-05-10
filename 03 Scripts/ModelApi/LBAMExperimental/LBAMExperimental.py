import numpy as np
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.signal import find_peaks
import cv2
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import color

def FFT2PlotData(data_fft2,hz=True):
    """ Generates basic plot data from 2D fft data

        fft2_data : 2D fft image
        hz : Flag to convert freq data to Hertz. Default True.

        Returns ordered magnitude and ordered angle data ready for plotting
    """
    # get flattened magnitude data
    F_mag = np.abs(data_fft2).ravel()
    # get flattened angle data
    F_angle = np.angle(data_fft2).ravel()
    # if hz flag is set, convert angle data to hertz
    if hz:
        F_angle *= 2.0*np.pi
    # sort angle data
    angle_sort = np.argsort(F_angle)
    # return sorted magnitude and freq data so it can be plotted straight away
    return F_mag[angle_sort],F_angle[angle_sort]

def filterFreq(I,freq_list,freq_tol=0.5,is_hz=True):
    """ Filter the given power density matrix for the frequencies listed given the tolerance

        I : Power density matrix
        freq_list : List of frequencies to mask for
        freq_tol : Tolerance on searching for frequencies
        is_hz    : Flag indicating if the frequency list is in hz or rads
        Returns a masked copy of the power density matrix

        Performs a 2D FFT on the given power density matrix and extracts just the
        values that are within range of the given frequencies

        freq_list[i]-freq_tol <= np.angle(fftn(I)) <= freq_list[i]+freq_tol
    """
    # perform 2D fourier transform of the given power matrix
    I_fft = fftn(I)
    # extract the magnitude and angle
    mag,freq = np.abs(I_fft),np.angle(I_fft)
    # if the target list is in hz, convert freq matrix to hx
    if is_hz:
        freq *= 2.0*np.pi
    # create matrix for boolean search results
    freq_search = np.full(I_fft.shape,False,dtype=np.bool)
    # searches for freqencies that are within the frequency range defined by target freq +/- freq_tol
    for f in freq_list:
        freq_search = np.logical_or(freq_search,np.logical_and(freq>=(f-freq_tol),freq<=(f+freq_tol)))
    # create copy to hold masked values
    I_copy = np.zeros(I_fft.shape,dtype=I_fft.dtype)
    # update values
    I_copy[freq_search]=I_fft[freq_search]
    # return inverse fft of masked results
    return ifftn(I_copy)

def findFilterFreq(Qr,mag_filt=0.05,hz=True,first_inact=0):
    """ Search the power density matrix for the freq to be used in the filter

        Qr : Data as processed by readH264
        mag_filt : Factor of maximum magnitude to filter the peaks by
        hz : Flag to return frequencies in terms of Hertz. Default True
        first_in : Index of the first frame where the laser is NOT on. Default 0 (i.e. first frame)

        Searches the data to identify the laser spatial frequencies to search for.
        As the peak frequency algorithm picks up lots of little peaks, only the
        frequencies whose magnitudes are above mag_filt times the maximum magnitude
        are returned. This gives helps return only the most dominant components.

        Returns a list of the frequencies of the most prominent components
    """
    # find the frame with the highest values
    # use it as an example of an active frame where the laser is turned on
    d_max = np.unravel_index(np.argmax(Qr),Qr.shape)[2]
    # find the frequencies introduced between first inactive frame and the found active frame
    I_diff = fftn(Qr[:,:,d_max])-fftn(Qr[:,:,first_inact])
    # get the magnitude and frequency data for the fft difference
    mm,ff = FFT2PlotData(I_diff,hz=hz)
    # search for peaks, reutrns the indicies
    mag_peaks = find_peaks(mm)[0]
    # filter peaks for just the most prominent components
    mag_peaks = mag_peaks[mm[mag_peaks]>(mag_filt*np.max(mm[mag_peaks]))]
    # return the frequencies associated with these peaks
    return ff[mag_peaks]

def findFilterFreqDiff(Qr,mag_filt=0.05,hz=True,first_inact=0):
    """ Search the power density matrix for the freq to be used in the filter. Looks for the frequencies
        introduced between an active and inactive frame

        Qr : Data as processed by readH264
        mag_filt : Factor of maximum magnitude to filter the peaks by
        hz : Flag to return frequencies in terms of Hertz. Default True
        first_in : Index of the first frame where the laser is NOT on. Default 0 (i.e. first frame)

        Searches the data to identify the laser spatial frequencies to search for.
        As the peak frequency algorithm picks up lots of little peaks, only the
        frequencies whose magnitudes are above mag_filt times the maximum magnitude
        are returned. This gives helps return only the most dominant components.

        Returns a list of the frequencies of the most prominent components
    """
    # find the frame with the highest values
    # use it as an example of an active frame where the laser is turned on
    d_max = np.unravel_index(np.argmax(Qr),Qr.shape)[2]
    # find the frequencies introduced between first inactive frame and the found active frame
    I_diff = fftn(Qr[:,:,d_max])-fftn(Qr[:,:,first_inact])
    # get the magnitude and frequency data for the fft difference
    mm,ff = FFT2PlotData(I_diff,hz=hz)
    # search for peaks, reutrns the indicies
    mag_peaks = find_peaks(mm)[0]
    # filter peaks for just the most prominent components
    mag_peaks = mag_peaks[mm[mag_peaks]>(mag_filt*np.max(mm[mag_peaks]))]
    # return the frequencies associated with these peaks
    return ff[mag_peaks]

def findPeakFilterPerc(peaks_list,hist_bins=7):
   """ Calculate by what percentage of the maximum the peaks should be filtered by

       peaks_list : Collection of unfiltered peak values
       hist_bins : Number of bins to use in histogram

       This function returns the ideal percentage of the maximum the peaks can be filtered by to get only
       the most dominant peaks

             peaks_list[peaks_list >= max(peaks_list)*perc]

       A histogram is used to identify which percentage bin the majority of values belongs to. The maximum value
       in that bin is used to calculate the returned percentage value.

       Returns percentage value to be used in boolean indexing
   """
   # calculate the histogram of the values based on percentage of maximum
   # returns populations and edges of those bins
   max_peak = np.max(peaks_list)
   n,edges = np.histogram(peaks_list/max_peak,bins=hist_bins)
   # find which bin has the highest population
   # sorted ascending
   nn = n.argsort()
   # Find which values are in the target bin
   peaks_bin = np.where(np.logical_and(peaks_list/max_peak>=edges[nn[-1]],peaks_list/max_peak<=edges[nn[-2]]))
   # Calculate the percentage as the maximum value in the bin divided by the max result
   # the list is treated as an array to allow slicing and save space
   return np.asarray(peaks_list)[peaks_bin].max()/max_peak

def maskFreq(I,freq_list,freq_tol=0.5,is_hz=True):
    """ Remove the target frequencies from the data I

        I : Power density matrix
        freq_list : List of frequencies to remove
        freq_tol : Tolerance on searching for frequencies
        is_hz    : Flag indicating if the frequency list is in hz or rads
        Returns a masked copy of the power density matrix

        Performs a 2D FFT on the given power density matrix and searches for the values whose
        frequencies are within range.

        freq_list[i]-freq_tol <= np.angle(fftn(I)) <= freq_list[i]+freq_tol

        The indicies that are not found are used to copy the fft values to a new matrix. The
        ifft of that new matrix is returned.

        Returns ifft of matrix whose frequency components are not in the target ranges
        
    """
    # perform 2D fourier transform of the given power matrix
    I_fft = fftn(I)
    # extract the magnitude and angle
    mag,freq = np.abs(I_fft),np.angle(I_fft)
    # if the target list is in hz, convert freq matrix to hx
    if is_hz:
        freq *= 2.0*np.pi
    # create matrix for boolean search results
    freq_search = np.full(I_fft.shape,False,dtype=np.bool)
    # searches for freqencies that are within the frequency range defined by target freq +/- freq_tol
    for f in freq_list:
        freq_search = np.logical_or(freq_search,np.logical_and(freq>=(f-freq_tol),freq<=(f+freq_tol)))
    # create copy of the fft matrix
    I_copy = np.zeros(I_fft.shape,dtype=I_fft.dtype)
    # update values
    I_copy[~freq_search]=I_fft[~freq_search]
    # return inverse fft of masked results
    return ifftn(I_copy)

def diffCol(I):
    """ Calculate difference between columns in the matrix

        I : Matrix to operate one

        Finds the absolute element-wise difference between one column and the next.

        diff[:,0]= ||I[:,0]-I[:,1]||

        Returns the difference matrix
    """
    cols = I.shape[1]
    diff = np.zeros(I.shape,dtype=I.dtype)
    for c in range(1,cols):
       diff[:,c-1]=np.abs(I[:,c]-I[:,c-1])
    return diff

def diffRow(I):
    """ Calculate difference between rows in the matrix

        I : Matrix to operate one

        Finds the absolute element-wise difference between one row and the next.

        diff[0,:]= ||I[0,:]-I[1,:]||

        Returns the difference matrix
    """
    rows = I.shape[1]
    diff = np.zeros(I.shape,dtype=I.dtype)
    for r in range(1,rows):
       diff[r-1,:]=np.abs(I[r,:]-I[r-1,:])
    return diff

def drawHoughCircles(circles,shape,**kwargs):
    """ Draw the givenn list of circles from the HoughCircles function onto a blank matrix

        circles : Array of circles data returned by OpenCVs HoughCircles function
        shape : Shape of the mask to draw results on. Mask size is (shape,3)
        **kwargs : Additional arguments to customise drawing
            centre_col : color to draw circle centres with, tuple of uint8 values
            bounday_col : color to draw circle boundaries with, tuple of uint8 values
            iter_cols : Each circle is drawn with a unique color and centre color decided by it's order in the matrix.
                        (255/ci,255/ci,255/ci). Centre and boundary color ARE THE SAME
            num_c : Number of circles to draw. By default it draws all of circles. This allows user to select a certain number

        HoughCircles searches for circular objects in a matrix. The function returns an array
        of circles found within the data given the used function parameters. The array is organised as
        a row of tuples describing centre location (x,y) and radius.

        The centres are drawn as 1 radius circles in one color and the bounday in another color. Draws
        boundaries and then centres so the centres can be more clearly seen for large collections of circles

        Return the matrix that has the circles drawn on.
    """
    # set colors
    if 'centre_col' in kwargs.keys():
        ccol = kwargs["centre_col"]
    else:
        ccol = (0, 100, 100)
        
    if 'boundary_col' in kwargs.keys():
        bcol = kwargs["bounday_col"]
    else:
        bcol = (255, 0, 255)

    if 'num_c' in kwargs.keys():
        len_c = kwargs['num_c']
    else:
        len_c = circles.shape[1]
        
    # blank matrix
    mask = np.zeros((*shape,3),dtype='uint8')
    # iterate through circles array
    # array is [1xCx3]
    for c in range(len_c):
        # draw circle boundary
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), circles[0,c,2], bcol, 1)
    for c in range(len_c):
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), 1, ccol, 1)
    # return circle result
    return mask

def powerEstHoughCircle(circle,I,PP=20e-6):
   """ Estimating power using circle found in power density matrix by a Hough Circle algorithm

       circle : Portion of the Hough Circle results matrix representing one circle. It's an array
                of a list of three values [x,y,radius]
       I : Power density matrix, W/m2
       PP : Pixel Pitch, m. Default: 20e-6 m

       This is designed to be used with Numpy's apply along matrix command as applied to results

       Return sums the values within circle and multiplies by the area
   """
   # create empty mask to draw results on 
   mask = np.zeros(I.shape[:2],dtype='uint8')
   # draw filled circle using given parameters
   cv2.circle(mask,(*circle[:2],),circle[2],(255),cv2.FILLED)
   # find where it was drawn
   i,j = np.where(mask==255)
   # sum the power density values in that area and multiply by area
   return np.sum(I[i,j])*(np.pi*(circle[2]*PP)**2.0)

def remCirclesOutBounds(circles,shape):
   """ Remove the circles form the set of circles whose area goes out of bounds in the image

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       shape : Shape of the data they were found in

       This function checks if given the postion and radius of the circle whether any portion of it
       goes outside the bounds of shape. It performs a logical check of

       Returns filtered list of circles
   """
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.logical_or(
      np.logical_or(circles[0,:,0]-circles[0,:,2]>128,circles[0,:,0]+circles[0,:,2]>128), # checking if any x coordinates are out of bounds
      np.logical_or(circles[0,:,1]-circles[0,:,2]>128,circles[0,:,1]+circles[0,:,2]>128)))[0] # checking if any y coordinates are out of bounds
   # if any circles were found, delete them and return new array
   # Numpy's delete command is performance costly hence the precheck
   if outofbounds_idx.shape[0]>0:
      # if circles is given straight from Hough circles, then its shape is 1xCx3
      # if the user unwraps it, then it is Cx3
      # this if statement simply makes sure that the correct axis is set
      if len(circles.shape)==3:
         return np.delete(circles,outofbounds_idx,axis=1)
      else:
         return np.delete(circles,outofbounds_idx,axis=0)
   # if no circles are to be removed, return original array
   else:
      return circles

def powerEstBestCircle(I,radii_range,PP=20e-6):
    """ Estimate power using power density matrix by selecting the highest
        scoring circle found using skimage hough_circle function

        I : Single power density frame to search, W/m2
        radii_range: 1D Numpy array of radii to search for in pixels.
                     E.g. numpy.arange(30,70,1.0)
        PP : Pixel pitch of the thermal camera, metres

        Returns the estimated power in Watts
    """
    # normalize image so it can be used by skimage
    I_norm = I/I.max(axis=(0,1))
    # search for circles
    res = hough_circle(I_norm,radii_range)
    # get the top three circles
    accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=1)
    # choose the highest rated circle to estimate power with
    return powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I,PP)

def powerEstAccumCircle(I,radii_range,PP=20e-6,min_accum=0.6,num_peaks=10):
   """ Estimated power profile using the power density matrix by selecting the
       highest scoring circle after removing the circles whose scores are less
       then min_accum.

       I : Single power density frame to search, W/m2
       radii_range: 1D Numpy array of radii to search for in pixels.
                    E.g. numpy.arange(30,70,1.0)
       PP : Pixel pitch of the thermal camera, metres. Default: 20e-6 m.
    """
   # normalize image so it can be used by skimage
   I_norm = I/I.max(axis=(0,1))
   # search for circles
   res = hough_circle(I_norm,radii_range)
   # get the top three circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=num_peaks)
   # search for any circles with the min score
   ai = np.where(accums>=min_accum)[0]
   # if there are none, then return zero
   if ai.shape[0]==0:
      return 0.0
   # else choose the highest scoring one
   else:
      # as accums are already sorted by value, the highest score in the filtered
      # list is the first one.
      return powerEstHoughCircle([cx[ai][0],cy[ai][0],radii[ai][0]],I,PP)
   
def distAllMetrics(I0,I1,mstr):
    """ Apply the specified distance metric to the difference between the frames
        I0,I1

        I0: First power density frame.
        I1 : Second power density frame
        mstr: List of names of distance metrics to try. Has to be supported by
              SciPy's cdist function.

        This generator iterates through the list of specified metrics using the
        cdist function to calculate distance.

        The originbal purpose of this is to test which distance metrics could be
        mathematically applied to the power density matricies. If a blank matrix is
        returned, then it couldn't be applied to the distance between the matricies
        for some reason.
        
        Yields the name of the metric used and the distance results. If it fails,
        for whatever reason then the name of the metric is returned along with
        an empty matrix.
    """
    from scipy.spatial.distance import cdist
    for m in mstr:
        try:
            # attempt cdist function for given metric
            yield (m,cdist(I0,I1,metric=m))
        except:
            # if exception occurs, return zero
            yield (m,np.zeros(I0.shape,dtype=I0.dtype))

def sigmoid(x,lower,upper,growth,v,Q):
    """ Generalized sigmoid function

        lower : lower asymptote
        upper : upper asymptote
        growth: growth factor
        v     : affects which asymptote the max growth occurs near
        Q     : related to y(0)

        Returns the value of the sigmoid function in response to input x
        and the parameters.
    """
    return lower + (upper-lower)/((1 + Q*np.exp(-growth*x))**(1.0/v))

def exponential(t,A,B,C,D):
    """ Generic Exponential Function

        y(t) = A + B*exp(C*t)

        Returns result
    """
    from numpy import exp
    return A + B*exp((C*t)+D)

def sumOfSquaredError(parameterTuple,xData,yData):
    """ Calculated sum squared error between sigmoid values for x-data and yData.
    
        Arguments:
            - parameterTuple : Collection of paramters values for sigmoid function
            - xData: Data for x values
            - yData: Predicted yData to match against sigmoid function
            
        Designed to be used in Scipy curve fitting functions as a cost function.
        
        Returns the sum squared error between the yData and the sigmoid values using xData
    """
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    # generate values
    val = sigmoid(xData, *parameterTuple)
    # returns sum sq error
    return np.sum((yData - val) ** 2.0)

def gen_p0(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution
        for data x,y
        
        x : x-data
        y : data

        Used in script AbsorbtivityData.py
        
        Parameters are:
            - lower asymptote
            - upper asymptote
            - growth factor
            - v value
            - Q value
            
        Returns list of parameters to be used with sigmoid function in this
        module
    """
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.1,200.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=3000)
    return result.x

def gen_p1(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution
       for data x,y

        x : X data
        y : Y data

        Uses different data ranges for the sigmoid growth factor than function
        gen_pl.

        Used in script AbsorbtivityData.py

        Parameters are:
            - lower asymptote
            - upper asymptote
            - growth factor
            - v value
            - Q value
            
        Returns list of parameters to be used with sigmoid function in this
        module.
    """
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.001,10.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=5000)
    return result.x

def gen_p2(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution
        for data x,y

        x : X data
        y : y data

        Uses different data ranges for the sigmoid growth factor than function
        gen_pl and gen_p0.

        Used in script AbsorbtivityData.py

        Parameters are:
            - lower asymptote
            - upper asymptote
            - growth factor
            - v value
            - Q value
            
        Returns list of parameters to be used with sigmoid function in this
        module.
    """
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.0001,1.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=3000)
    return result.x
            
def readAbsorbParams(folderpath):
    """ Read absorbtivity parameters written to file into a dictionary

        folderpath : Path to the folder where the documents are

        Originally used in script EstimateThermalConductivity.py

        Returns a dictionary of the numpy.polynomial.polynomial.Polynomial objects
        for the parameters

        Dictionary keys are:
            "growth" : growth rate of sigmoid for range of velocities
            "lower"  : lower asymptote of sigmoid
            "upper"  : upper asymptote of sigmoid
            "Q"      : Q-factor of sigmoid
            "v"      : v-factor of sigmoid
    """
    from numpy import genfromtxt
    import numpy.polynomial.polynomial as poly

    # create dictionary object
    params_dict = {}
    # read in parameters from file as array and convert to list
    params = genfromtxt(folderpath+'/absorb-poly-growth-params.txt',delimiter=',').tolist()
    # construct polynomial object, only using 3 values as 4th is nan
    params_dict["growth"] = poly.Polynomial(params[:3])

    params= genfromtxt(folderpath+'/absorb-poly-lower-params.txt',delimiter=',').tolist()
    params_dict["lower"] = poly.Polynomial(params[:3])

    params= genfromtxt(folderpath+'/absorb-poly-upper-params.txt',delimiter=',').tolist()
    params_dict["upper"] = poly.Polynomial(params[:3])

    params = genfromtxt(folderpath+'/absorb-poly-Q-params.txt',delimiter=',').tolist()
    params_dict["Q"] = poly.Polynomial(params[:3])

    params= genfromtxt(folderpath+'/absorb-poly-v-params.txt',delimiter=',').tolist()
    params_dict["v"] = poly.Polynomial(params[:3])
    # return poly dictionary
    return params_dict

def twoD_Gaussian(x_y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """ 2D non-rotated gaussian function

        x_y : Data to process, Just x and y data combined together like x,y

        Processes the x and y data according to a non-rotated 2D gaussian
        function.

        Used in LaserParamtersTimelapse.py

        Returns the results raveled.
    """
    x,y = x_y
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def remCirclesROI(circles,xlim,ylim):
   """ Remove the circles from the target region of interest in the image

       circles: Array containing [centre_x,centre_y,radius] for each circle found
       xlim : Two element array describing the lower and upper x limits for the region of intererst [lower,upper]
       ylim : Two element array describing the lower and upper y limits for the region of interest [lower,upper]

       This function checks to see if any portion of the specified circles goes outside the limits given. If it does,
       it is removed from the array.

       If xlim or ylim are single element arrays then it is assumed to be the upper limits. e.g. ylim = 128 => [0,128]
       If any of the elements are negative, then -1 is returned.

       Returned filtered list of circles whose areas are within the limits specified by the user
   """
   # if the limits are only one element in length
   if len(xlim)==1:
      xup = xlim
      xdwn = 0
   else:
      xup = xlim[1]
      xdwn = xlim[0]

   if len(ylim)==1:
      yup = ylim
      ydwn = 0
   else:
      yup = ylim[1]
      ydwn = ylim[0]

   # if any of the limits passed are negative, return -1
   if (np.asarray(xlim)<0).any():
      return -1

   if (np.asarray(ylim)<0).any():
      return -1
         
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.logical_or(
      np.logical_or(circles[0,:,0]-circles[0,:,2]<xdwn,circles[0,:,0]+circles[0,:,2]>xup), # checking if any x coordinates are out of bounds
      np.logical_or(circles[0,:,1]-circles[0,:,2]<ydwn,circles[0,:,1]+circles[0,:,2]>yup)))[0] # checking if any y coordinates are out of bounds
   # if any circles were found, delete them and return new array
   # Numpy's delete command is performance costly hence the precheck
   if outofbounds_idx.shape[0]>0:
      # if circles is given straight from Hough circles, then its shape is 1xCx3
      # if the user unwraps it, then it is Cx3
      # this if statement simply makes sure that the correct axis is set
      if len(circles.shape)==3:
         return np.delete(circles,outofbounds_idx,axis=1)
      else:
         return np.delete(circles,outofbounds_idx,axis=0)
   # if no circles are to be removed, return original array
   else:
      return circles

def maskCirclesRadius(circles,rmin,rmax):
   """ Remove circles whose radii are outside the limits

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       rmin : Min. radius in terms of pixels
       rmax : Max. radius in terms of pixels

       Searches for circles whose radii are too small or too big as per the limits set by
       arguments rmin and rmax. The returns list has circles are within the limits.

       rmin <= r <= rmax

       Returns the filtered list of circles
   """
   # searches for circles that are too small or too big
   outofbounds_idx = np.where(np.logical_or(circles[0,:,2]<rmin,circles[0,:,2]>rmax))[0]
   # if any circles were found, delete them and return new array
   # Numpy's delete command is performance costly hence the precheck
   if outofbounds_idx.shape[0]>0:
      # if circles is given straight from Hough circles, then its shape is 1xCx3
      # if the user unwraps it, then it is Cx3
      # this if statement simply makes sure that the correct axis is set
      if len(circles.shape)==3:
         return np.delete(circles,outofbounds_idx,axis=1)
      else:
         return np.delete(circles,outofbounds_idx,axis=0)
   # if no circles are to be removed, return original array
   else:
      return circles

def customOpenAction(src,kernel,**kwargs):
   """ Custom opening operation applied to source image

      src : Source 8-bit image
      kernel : Kernel applied to the source image during BOTH the erode and dilate operations
      **kwargs : Customization parameters
         erode_it : number of times the eroding operation is applied
         dilate_it : number of times the dilate operation is applied
         erode_k : Kernel for erode operation, Overrides the kernel arg
         dilate_k : Kernal for dilate operation. Overrides the kernel arg


      Applies a customized open morphological transform to the source image. The listed supported arguments for the kwargs
      kwargs show how it can be customised. If no additional options are given then, the kernel is applied once in the open
      operation and then once in the closed operation
      Returns the image after the specified open operation
   """
   # check kwargs
   if 'erode_it' in kwargs.keys():
      erode_it = kwargs['erode_it']
   else:
      erode_it = 1

   if 'dilate_it' in kwargs.keys():
      dilate_it = kwargs['dilate_it']
   else:
      dilate_it = 1

   if 'erode_k' in kwargs.keys():
      erode_k = kwargs['erode_k']
   else:
      erode_k = kernel

   if 'dilate_k' in kwargs.keys():
      dilate_k = kwargs['dilate_k']
   else:
      dilate_k = kernel

   # apply erode operation
   ero = cv2.erode(src,erode_k,iterations=erode_it)
   # apply dialate operation
   dia = cv2.dilate(ero,dilate_k,iterations=dilate_it)

   return dia

def findLaserBorderBest(I,PP,r0,rr=3.5):
   """ Find the ideal laser border given the investigation into the best parameters

       I : image to search
       PP : pixel pitch
       r0 : laser radius
       rr : Ratio between laser radius and the radius of the laser boundary. Dependent on height
   """
   if I.dtype != np.uint8:
      I_img = I.astype('uint8')
   else:
      I_img = I
   # find the difference matrix
   diff_all = diffRow(I_img)+diffCol(I_img)
   # mask the right third of the image
   diff_all[:,int(I.shape[0]*(2/3))]=0
   # perform open operation
   res = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
   # search for circles across a wide range
   circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,int(I.shape[0]/8),param1=100,param2=30,minRadius=0,maxRadius=4*r0)
   # if circles were found
   if circles is not None:
      # remove out of bounds circles
      circles = remCirclesOutBounds(circles,I.shape[:2])
      # if there are still circles left
      if circles.shape[1]>0:
         # calculate power using that circle
         return powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I,PP)
      else:
      # if there are no circles after filtering
         return 0.0
   # if there are no circles, return 0.0
   else:
      return 0.0
         
def estimateCircleEntropy(circle,I):
   """ Estimate the Shannon Entropy within the circle

       circle : [x,y,r]
       I : power density matrix the circles were detected in

       Finds the pixels within the area of the circle and calculate the entropy of this area
       using the Shannon Entropy approach. The values within the circle are copied over to a square
       where the values outside of the circle are 0

       Returns the entropy within the circle roi
   """
   from skimage.filter.rank import entropy
   # find the shape of the target image
   # use it to create a meshgrid of coordinates
   xx,yy = np.meshgrid(np.arange(0,I.shape[0]),np.arange(0,I.shape[1]))
   # find which pixels are within range
   xr,yr = np.where((((xx-circle[0])**2.0)+((yy-circles[1])**2.0))**0.5 <= circles[2])
   # create blank matrix for results
   vals = np.zeros(I.shape[:2],np.uint8)
   # copy values over
   vals[xr,yr] = I[xr,yr]
   # calculate entropy using the values
   return entropy(vals)

def CannyMedian(I,sigma=0.33):
    """ Calculate the Canny Parameters using the Median of the frame

        I : Data frame
        sigma : Tolerance on parameters

        Parameters are calculated as:
            lt : int(max(0,(1.0-sigma)*median))
            ut : int(min(255,(1.0+sigma)*median))

        Source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        
        Returns lower threshold and upper threshold
    """
    v = np.median(I.astype('uint8'))
    return int(np.maximum(0.0,(1.0-sigma)*v)),int(np.minimum(255,(1.0+sigma)*v))

def CannyMean(I,sigma=0.33):
    """ Calculate the Canny Parameters using the Mean of the frame

        I : Data frame
        sigma : Tolerance on parameters

        Parameters are calculated as:
            lt : int(max(0,(1.0-sigma)*mean))
            ut : int(min(255,(1.0+sigma)*mean))

        Source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        
        Returns lower threshold and upper threshold
    """
    v = np.mean(I.astype('uint8'))
    return int(np.maximum(0.0,(1.0-sigma)*v)),int(np.minimum(255,(1.0+sigma)*v))

# anistroptically enhanced thermal conductivity
def anisKFactor(T,Tm,P,v,a1,b1,a2,b2,va):
    """ Anistropicaly enhanced thermal conductivity factors

        T : Temperature (K)
        Tm : Melting Point (K)
        *params : parameters for the function
            a1,b1 : parameters for factor for z direction
            a2,b2,va : parameters for factor for y direction

        Returns the estimated anistropically enhanced thermal conductivity
        factors based off the given parameters.
    """
    # if temperature is less than melting temperature
    if T<Tm:
        return 1.0,1.0,1.0
    else:
        lambda_x = 1.0
        lambda_z = a1*(P/np.sqrt(v)) + b1
        if v<=va:
            lambda_y = a2*v + b2
        else:
            lambda_y = 1.0
        return lambda_x,lamda_y,lambda_z

def absorbEnergyDensity(beta,D,r,P,v):
    """ Absorbed Energy density

        beta : Absorbtivity matrix
        D : Thermal diffusivity matrix
        r : laser radius
        P : laser power: scanning velocity

        Returns the absorbed energy density matrix
    """
    return (beta/(np.pi*np.sqrt(2.0*D*(r**3.0))))*(P/np.sqrt(v))

def heatingDepth(D,r,v,l):
    """ Heating Depth

        D : Thermal diffusivity
        r : laser radius (m)
        v : Scanning velocity (m/s)
        l : layer thickness (m)

        Returns the heating depth (m)
    """
    tau = (2.0*r)/v
    return np.sqrt(4.0*D*tau)/l

def format_bytes(size):
   """ Convert the given number of bytes into a readable form

       size : size of something in bytes

       Constantly divides size by 1024 until the result becomes less than
       1024. At which point, the current result and the appropriate label
       (stored in an internal dictionary) is returned. Labels are currently
       up to tera.

       E.g. format_bytes(1024) => 1,kilo

       Returns the size reduced down to its largest components and the
       appropriate label.
"""
   # number of bytes in each term
   power = 1024
   # counter for number of divisions
   n = 0
   # labels for each stage of divisiion
   power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
   # see how many times the number of bytes can be divided
   while size > power:
      size /= power
      n+=1
   # return the result and the respective label
   return size,power_labels[n]+'bytes'

