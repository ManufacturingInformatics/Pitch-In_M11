from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from TSLoggerPy.TSLoggerPy import TSLoggerPy as logger
import cv2
import os

# get current working directory
curr_cwd = os.getcwd()

def readH264(path,flag='mask'):
   """ Read in image data from target H264 video file as float16
	
	 path : target file path
	 flag : how to process the data, default='mask' means the data is masked
   """
   # known size of the images
   rows = 128
   cols = 128

   # read in raw bytes as a 1D array
   arr = np.fromfile(path,dtype='uint16')

   if flag=='mask':
      ## update values based on code
      # get code
      code_array = np.bitwise_and(arr,0xF000)
      # CODE_VAL_SEUIL2
      arr[code_array==0xD000] = 0xF800
      # CODE_VAL_CONTOUR
      arr[code_array==0xB000] = 0xF81F
      # CODE_VAL_MAX
      arr[code_array==0xC000] = 0x0000
      # CODE_VAL_SEUIL1
      arr[code_array==0xE000] = 0x001F

   ## just lower 12-bits
   arr = np.bitwise_and(arr,0x0FFF)

   ## convert data to frames
   # break the data into chunks that are 1d frames
   frames_set = np.split(arr,int(arr.shape[0]/(rows*cols)))
   # combined frames together into a 3d array and interpret as float16 data type
   return np.dstack([np.reshape(f,(rows,cols)) for f in frames_set]).astype('float16')

# functions for converting temperature between celcius and kelvin
def CelciusToK(C):
    return float(C)+273.15

def KelvinToC(K):
    return float(K)-273.15

# starter temperature, room temperature
T0 = CelciusToK(23.0)

# melting temperature of 316L
Tm = 1390 # C
Tm_K = CelciusToK(Tm) # K

### ALL TEMPS IN C
#temperature resolution for [0-20] C or [0 - 293.15] K 
starter_res = 200

## temperature conductivity
print("Creating thermal conductivity data")
def genKdata():
    # temp range [0-20] avg
    K_range_1 = np.full(starter_res,16.3)
    # temp range [20-100]
    K_range_2 = np.full(int(starter_res*(80/20)),(16.3+14.6)/2)
    # temp range [100-500]
    K_range_3 = np.full(int(starter_res*(400/20)),21.5)
    # temp data, complete
    K_data = np.concatenate([K_range_1,K_range_2,K_range_3],axis=0)
    # temperature range for plotting data
    K_temp_range = np.linspace(0.0,CelciusToK(500.0),K_data.shape[0])
    return K_data,K_temp_range

res_scale = 2
K_data,K_temp_range = genKdata()
# temperature range to test fitting on
T_test_data = np.linspace(0,CelciusToK(1500),res_scale*(1500/20))


### interpolate data
## thermal conductivity
# returns a function that can be used later
# s parameter is the min square distance between interpolation and data
K_spline = UnivariateSpline(K_temp_range,K_data)
K_ispline = InterpolatedUnivariateSpline(K_temp_range,K_data)

def Kp(Ks,T,Tm,scale=0.01):
    """ Thermal conductivity of powder
        
        Ks : Thermal condutivity of solid material, numpy array
        T  : Temperature data to genereate Ks, K
        Tm : Melting point of material, K

        Taken to be fraction of solid thermal conductivity when
        temperature is less than the melting point, Tm.

        Source : 3-Dimensional heat transfer modeling for laser 
        powder-bed fusion additive manufacturing with volumetric 
        heat sources based on varied thermal conductivity and 
        absorptivity (2019), Z. Zhang et. al.

        Returns thermal conductivity for powder as approximated
    """

    from numpy import where,asarray,copy
    # make a copy of solid thermal conductivity matrix
    Kp = np.copy(Ks)
    # for all values where the temperature is less than the melting point
    # scale the thermal conductivity values
    Kp[where(T<Tm)] *= scale
    # return modified matrix
    return Kp

Kp_data = Kp(K_data,K_temp_range,Tm_K)
Kp_spline = UnivariateSpline(K_temp_range,Kp_data)
Kp_ispline = InterpolatedUnivariateSpline(K_temp_range,Kp_data)

## specific heat conductivity
print("Creating specific heat capacity data")
def genCdata():
    # temp range [0-20]
    C_range_1 = np.full(starter_res,450.0)
    # temp range [20-93]
    C_range_2 = np.full(int(starter_res*(73/20)),(485.0+500.0)/2.0)
    # temp range [93-100]
    C_range_3 = np.full(int(starter_res*(7/20)),500.0)
    # specific heat capacity, complete
    C_data = np.concatenate([C_range_1,C_range_2,C_range_3],axis=0)
    # temperature data for plotting data
    C_temp_range = np.linspace(0,CelciusToK(100),C_data.shape[0])
    return C_data,C_temp_range

C_data,C_temp_range = genCdata()

C_spline = UnivariateSpline(C_temp_range,C_data)
C_ispline = InterpolatedUnivariateSpline(C_temp_range,C_data)

# volume expansion factor, mm^3/mm^3.C
# same as m^3/m^3.K
# as both C and K are absolute units of measurement, a change is C is the same in K
# converting between mm to m cancels each other out
print("Generating volume expansion factor data")
def genEvData():
    """ Generate data for volume expansion factor based off linear expansion factors """
    # for T = [20-100]
    Ev_range_1 = np.full(int(starter_res*(80/20)),3.0*(16.6+18.2+19.4)/3.0)
    # for T = [100-500]
    Ev_range_2 = np.full(int(starter_res*(400/20)),3.0*(18.2+19.4)/2.0)
    # for T = [500-1000]
    Ev_range_3 = np.full(int(starter_res*(500/20)),3.0*19.4)
    # combine data together
    Ev_data = np.concatenate([Ev_range_1,Ev_range_2,Ev_range_3],axis=0)
    # temperature data for temperature range so it can be plotted
    Ev_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),Ev_data.shape[0])
    # return data and temp data
    return Ev_data,Ev_temp_range

Ev_data, Ev_temp_range = genEvData()

Ev_spline = UnivariateSpline(Ev_temp_range,Ev_data)
Ev_ispline = InterpolatedUnivariateSpline(Ev_temp_range,Ev_data)

def genDensityData(Ev):
    """ Generate density data based of thermal volumetric expansion 
        
        Ev : spline for thermal volumetric expansion
    """
    # for dT = [20 - 100] C
    p_range_1 = np.full(int(starter_res*(80/20)),(7900.0+8000.0+8027.0)/3.0)
    ## calculate density data for dT =[100 - 1000] C using Ev spline and prev range value as starter point
    p_range_2 = []
    for dT in np.linspace(0.0,900.0,int(starter_res*(20/900))):
        p_range_2.append(p_range_1[0]/(1+Ev(dT)*dT))
    # convert to array
    p_range_2 = np.array(p_range_2)
    # combine data ranges together
    p_data = np.concatenate([p_range_1,p_range_2],axis=0)
    # create temperature data
    p_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),p_data.shape[0])
    return p_data, p_temp_range

p_data,p_temp_range = genDensityData(Ev_ispline)

p_spline = UnivariateSpline(p_temp_range,p_data)
p_ispline = InterpolatedUnivariateSpline(p_temp_range,p_data)

def genThermalDiff(K,p,C,T):
    """ Generate thermal diffusivity data for solid material using previous splines

        K : thermal conductivity spline for solid material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix

        returns thermal diffusivity data for the temperature range
    """

    return K(T)/(p(T)*C(T))

def genThermalDiff_powder(K,p,C,T):
    """ Generate thermal diffusivity data for powder using previous splines

        K : thermal conductivity spline for powder material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix
        Tm : metling temperature of the solid material

        returns thermal diffusivity data for the temperature range
    """
    # generate thermal conductivity data
    # then modifiy it according to approximation function
    return Kp(K(T),T,Tm)/(p(T)*C(T))

# thermal diffusivity of the solid material
Ds_data = genThermalDiff(K_ispline,p_ispline,C_ispline,T_test_data)

Ds_spline = UnivariateSpline(T_test_data,Ds_data)
Ds_ispline = InterpolatedUnivariateSpline(T_test_data,Ds_data)

# thermal diffusivity using thermal conductivity scaling approximation
Dp_data = genThermalDiff_powder(Kp_ispline,p_ispline,C_ispline,T_test_data)

Dp_spline = UnivariateSpline(T_test_data,Dp_data)
Dp_ispline = InterpolatedUnivariateSpline(T_test_data,Dp_data)

## emissivity
# emissivity of stainless steel
e_ss = 0.53
# emissivity of steel galvanised
e_p = 0.28

# function for converting radiative heat to temperature
def Qr2Temp(qr,e,T0):
    """ Function for converting radiative heat to surface temperature

        qr : radiative heat W/m2
        e  : emissivity of the material [0.0 - 1.0]
        T0 : starting temperature, K

        Returns numpy array surface temperature in Kelvin using Stefan-Boltzman theory in W/m2

        qr = e*sigma*(T^4 - T0^4)
        => T = ((qr/(e*sigma))**4.0 + T0**4.0)**0.25
    """
    from scipy.constants import Stefan_Boltzmann as sigma
    from numpy import asarray, nan_to_num,full
    # if emissivity is 0.0, return T0
    # result of setting qr/(e*sigma)
    # avoids attempt to divide by 0
    if e==0.0:
        #print("emissivity is 0")
        # if T0 is a numpy array and then return T0 as is
        if type(T0)==numpy.ndarray:
            return T0
        # if T0 is a single value float
        elif type(T0)==float:
            # construct and return a matrix of that value
            return full(qr.shape,T0,qr.dtype)
    else:
        # calculate left hand portion of addition
        # due to e_ss*sigma being so small (~10^-8)
        div = (e*sigma)**-1.0
        #qr/(e*sigma)
        a = (qr*div)
        # T0^4
        #print("T0:",T0)
        b = asarray(T0,dtype=np.float32)**4.0
        #print("To^4:",b)
        # (qr/(e*sigma)) + T0^4
        c = a+b
        return (c)**0.25

def Temp2IApprox(T,T0,K,D,t):
    """ Function for converting temperature to power density

        K : thermal conductivity, spline fn
        D : thermal diffusivity, spline fn
        t : interaction time between T0 and T
        T : temperature K, numpy matrix
        T0: starter temperature K , constant

        Returns power density approximation ndarray and laser power estimate

        Approximation based on the assumption of uniform energy for the given area
    """
    # get numpy fn to interpret temperature as matrix and get pi constant
    from numpy import asarray, pi, sqrt, abs
    # temperature difference
    Tdiff = (T-T0)
    # thermal conductivity matrix
    K = K(Tdiff)
    # thermal diffusivity matrix
    D = D(Tdiff)
    # 2*sqrt(Dt/pi)
    a = ((D*t)/np.pi)
    # result of sqrt can be +/-
    # power density cannot be negative 
    b = (2.0*np.sqrt(a))
    temp = K*Tdiff
    # K*(T-T0)/(2*sqrt(Dt/pi))
    return abs(temp/b)

def ShannonEntropy(I):
    ''' Calculate base 2 Shannon Entropy of the given image

        I : matrix to calculate entropy
    '''
    from skimage.measure import shannon_entropy
    return shannon_entropy(I)

def ApplySobelOperator(I,sz=3):
    ''' Get gradient intensity of the supplied image using Sobel operator

        I : image source
        sz : size of the Sobel operator

        Applies the Sobel operator to find the gradient intensity in the x and y
        directions and combine the results together as 50% weighted matrices.

        Based off OpenCV tutorial https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html
    '''
    I_img = I.astype('uint16')
    # find the gradient using Sobel operator in the x direction
    grad_x= cv2.Sobel(I_img,cv2.CV_16S,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    # find the gradient using the Sobel operator in the y direction
    grad_y= cv2.Sobel(I_img,cv2.CV_16S,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    # take the absolute values of the gradient
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    # combine the results together each image contributing 50% of the magnitude
    return cv2.addWeighted(grad_x,0.5,grad_y,0.5,0),grad_x,grad_y

## distance metrics
metrics_str = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']

def distAllMetrics(I0,I1,mstr=metrics_str):
    for m in mstr:
        try:
            # attempt cdist function for given metric
            yield (m,cdist(I0,I1,metric=m))
        except:
            # if exception occurs, return zero
            yield (m,np.zeros(I0.shape,dtype=I0.dtype))

pixel_pitch = 20.0*10.0**-6 # metres
pitch_sq = pixel_pitch**2.0

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth # 1049

fc = 65.0
tc = 1/fc

T = np.zeros(frames_set_array.shape,dtype='float64')
I = np.zeros(frames_set_array.shape,dtype='float64')

## metrics for estimating activity
# average difference between one power density frame and the next
avg_diff_I = np.zeros(depth,dtype='float64')
# average difference between one raw frame and the next
avg_diff_raw = np.zeros(depth,dtype='float64')
# average power introduced to each pixel
avg_diff_P = np.zeros(depth,dtype='float64')
# entropy of each frame
shannon_entr = np.zeros(depth,dtype='float64')

## same metrics but for when relative to the first frame
# average difference between one power density frame and the next
avg_diff_I_rel = np.zeros(depth,dtype='float64')
# average difference between one raw frame and the next
avg_diff_raw_rel = np.zeros(depth,dtype='float64')
# average power introduced to each pixel
avg_diff_P_rel = np.zeros(depth,dtype='float64')
# entropy of each frame
shannon_entr_rel = np.zeros(depth,dtype='float64')

# create dictionary for results
# initial values are zero to represent zero distance
# future values are stacked onto the zero
dist_res_I = {k:np.zeros(frames_set_array.shape[:2]) for k in metrics_str}
dist_res_raw = {k:np.zeros(frames_set_array.shape[:2]) for k in metrics_str}

# dictionary for istances relative to first frame
dist_res_I_rel = {k:np.zeros(frames_set_array.shape[:2]) for k in metrics_str}
dist_res_raw_rel = {k:np.zeros(frames_set_array.shape[:2]) for k in metrics_str}

## image processing metrics
sobel_result = np.zeros(frames_set_array.shape,dtype=frames_set_array.dtype)
sobel_dx = np.zeros(frames_set_array.shape,dtype=frames_set_array.dtype)
sobel_dy = np.zeros(frames_set_array.shape,dtype=frames_set_array.dtype)

print("Evaluating up to ", target_f)

for f in range(0,target_f,1):
    #print("f=",f)
    if f==0:
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T0)
        I[:,:,f] = Temp2IApprox(T[:,:,f],T0,Kp_spline,Ds_spline,tc)
    else:
        # prev temp has to be T[:,:,f] to get expected behaviour for Tpeak
        # not sure why
        # if T0 is T[:,:,f-1], then Tpeak rises from 500K to 3200K
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T[:,:,f])
        I[:,:,f] = Temp2IApprox(T[:,:,f],T[:,:,f-1],Kp_spline,Ds_ispline,tc)
        
    if np.inf in I[:,:,f]:
        np.nan_to_num(I[:,:,f],copy=False)

    # calculate shannon entropy for the given frame
    # result is relative to the data type rather than the data set
    shannon_entr[f] = ShannonEntropy(frames_set_array[:,:,f])
    # apply sobel operator
    sobel_result[:,:,f],sobel_dx[:,:,f],sobel_dy[:,:,f] = ApplySobelOperator(I[:,:,f])
##  ax.contourf(sobel_result[:,:,f])
##  ax2.contourf(sobel_dx[:,:,f])
##  ax3.contourf(sobel_dy[:,:,f])
##  ff.savefig('C:/Users/DB/Desktop/BEAM/Scripts/ActivityMetriccs/SobelOperator/sobel-f{0}.png'.format(f))

# average differences
print("Finding average distances...")
for f in range(1,I.shape[2]):
   # calculate metrics with respet to the previous frame
     avg_diff_raw[f]=np.mean(frames_set_array[:,:,f]-frames_set_array[:,:,f-1],dtype='float64')
     avg_diff_I[f] = np.mean(I[:,:,f]-I[:,:,f-1],dtype='float64')
     avg_diff_P[f] = np.sum((I[:,:,f]-I[:,:,f-1])*(pitch_sq),dtype='float64')

     # calculate relative metrics
     avg_diff_raw_rel[f]=np.mean(frames_set_array[:,:,f]-frames_set_array[:,:,0],dtype='float64')
     avg_diff_I_rel[f] = np.mean(I[:,:,f]-I[:,:,0],dtype='float64')
     avg_diff_P_rel[f] = np.sum((I[:,:,f]-I[:,:,0])*(pitch_sq),dtype='float64')

## plot results
print("Plotting average distance results...")
plt.figure()
plt.plot(avg_diff_I)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W/m2)',title='Average Difference between Power Density Frames')
plt.gcf().savefig('avg-diff-I.png')

plt.figure()
plt.plot(avg_diff_raw)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W)',title='Average Difference between Radiative Heat Frames')
plt.gcf().savefig('avg-diff-Qr.png')

plt.figure()
plt.plot(avg_diff_P)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W)',title='Average Power added to each pixel between frames')
plt.gcf().savefig('avg-diff-P.png')

plt.figure()
plt.plot(shannon_entr)
plt.gca().set(xlabel='Frame Idx',ylabel='Entropy',title='Shannon Entropy of each frame')
plt.gcf().savefig('avg-diff-E.png')

## relative to first frame plots
plt.figure()
plt.plot(avg_diff_I_rel)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W/m2)',title='Average Difference between Power Density Frames and first Frame')
plt.gcf().savefig('avg-diff-I-rel.png')

plt.figure()
plt.plot(avg_diff_raw_rel)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W)',title='Average Difference between Radiative Heat Frames and first Frame')
plt.gcf().savefig('avg-diff-Qr-rel.png')

plt.figure()
plt.plot(avg_diff_P_rel)
plt.gca().set(xlabel='Frame Idx',ylabel='Average Difference (W)',title='Average Power added to each pixel between frames and first Frame')
plt.gcf().savefig('avg-diff-P-rel.png')

## attempt each metric string and search for which metrics will yield usable results
## used to trim down the ful list to save time in future runs
# invalid results are replaced with zeros
# due to the noise between frames, there is always some distance so zeros are impossible
# add key to list if value is not all zeros
valid_metrics = [k for k,v in distAllMetrics(I[:,:,0],I[:,:,1],metrics_str) if not np.all(v==0)]

if len(valid_metrics)==0:
   print("No valid metrics found for the data set")
else:
   print("Found ",len(valid_metrics), " valid metrics")
   for m in valid_metrics:
      print(m)
   # distance metrics
   print("Trying different distance metriccs...")
   for f in range(1,I.shape[2]):
         # update stack the new results for each distance metric to the current results in the results dictionary
        # metric X : current results => metric X : [current results,new_results]
        dist_res_I.update({k:np.dstack((dist_res_I[k],v)) for k,v in distAllMetrics(I[:,:,f-1],I[:,:,f],valid_metrics)})
        dist_res_raw.update({k:np.dstack((dist_res_raw[k],v)) for k,v in distAllMetrics(frames_set_array[:,:,f-1],frames_set_array[:,:,f],valid_metrics)})

         # update stack the new results for each distance metric to the current results in the results dictionary
        # metric X : current results => metric X : [current results,new_results]
        dist_res_I_rel.update({k:np.dstack((dist_res_I[k],v)) for k,v in distAllMetrics(I[:,:,0],I[:,:,f],valid_metrics)})
        dist_res_raw_rel.update({k:np.dstack((dist_res_raw[k],v)) for k,v in distAllMetrics(frames_set_array[:,:,0],frames_set_array[:,:,f],valid_metrics)})


   print("Creating folders for distance metric results")
   # get current working directory
   curr_cwd = os.getcwd()
   # for each distance metric create a folder for it
   # exist_ok flag is to stop the raising of an exception if the folder already exists
   for k in metrics_str:
      p = curr_cwd + "/DistanceMetrics/{0}".format(k)
      os.makedirs(p,exist_ok=True)

   # write the distance results to folders
   # separate loop from folder creation to save creating the same folders 
   print("Writing power density distance results to folders")
   f,ax = plt.subplots()
   f,ax2 = plt.subplots()
   for k in metrics_str:
      # create save path
      p = curr_cwd + "/DistanceMetrics/{0}".format(k)
      # calculate data limits to scale contour uniformly across data sets
      d_min = np.min(dist_res_I,axis=2)
      d_max = np.max(dist_res_I,ax=2)
      # iterate through saved data
      for df in range(dist_res_I[k].shape[2]):
         ax.contourf(dist_res_I[k][:,:,df],vmin=d_min,vmax=d_max)
         ax.set_title('Distance between Power Density Frames [{0},{1}]'.format(df,df+1))
         f.savefig(p)
         
      # calculate data limits to scale contour uniformly across data sets
      d_min = np.min(dist_res_raw,axis=2)
      d_max = np.max(dist_res_raw,ax=2)
      # iterate through saved data
      for df in range(dist_res_raw[k].shape[2]):
         ax.contourf(dist_res_raw[k][:,:,df],vmin=d_min,vmax=d_max)
         ax.set_title('Distance between Raw Value Frames [{0},{1}]'.format(df,df+1))
         f.savefig(p)
