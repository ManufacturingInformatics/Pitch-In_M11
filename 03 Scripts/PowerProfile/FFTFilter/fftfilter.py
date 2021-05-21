import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.signal import find_peaks
import cv2
from itertools import combinations

print("Starting")

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"

def readH264(path,flag='mask'):
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
    # combined frames together into a 3d array
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

def estPArea_Canny(I,PP):
    """ Estimating Laser Power by Area using Contours to estimate Area

        I : Power density image (W/pixel)
        PP : Pixel pitch (m)

        The power density image is filtered using Canny edge detection.
        The result is fed into findContours

        Returns estimated power or 0.0 if it can't find any contours
    """
    import cv2
    from numpy import zeros_like,where
    from numpy import sum as nsum
    # convert data to image
    # eq. to grayscale
    I_img = np.uint8(I)
    #print(I_img.dtype)
    # threshod using otsu, returns threshold and thresholded img
    low_threshold = 100
    ratio = 3
    # apply blur to image
    I_blur = cv2.blur(I_img,(3,3))
    # apply canny edge detection to create mask
    edges = cv2.Canny(I_blur,low_threshold,low_threshold*ratio,3)
    # find contours, returne contours and hierarchy
    ct=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    if len(ct)>0:
        # sort contours by area in ascending order
        ct = sorted(ct,key=lambda x : cv2.contourArea(x))
        ct_idx = 0
        # if the largest contour is likely the outer edge of the image
        if cv2.contourArea(ct[-1]) >= 0.9*(I.shape[0]*I.shape[1]):
            # use next largest
            largest_area = cv2.contourArea(ct[-2])
            ct_idx = -2
        else:
            largest_area = cv2.contourArea(ct[-1])
            ct_idx = -1
        #create mask
        mask = zeros_like(I_img)
        # draw contour of interest on mask filled
        cv2.drawContours(mask,ct,ct_idx,255,-1)

        # create color version of mask to return
        I_col = cv2.cvtColor(I_img,cv2.COLOR_GRAY2BGR)
        # draw masked area in bright green
        cv2.drawContours(I_col,ct,ct_idx,(0,255,0),-1)
        #print(largest_area,img_area)
        # sum the energy densities and multiply by area to get total energy
        # only sums the places where mask is 255
        return edges,I_col,nsum(I[where(mask==255)])*largest_area*(PP**2)
    else:
        return edges,zeros_like(I_img),0.0
        
pixel_pitch = 20.0*10.0**-6 # metres

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth #

fc = 65.0
tc = 1/fc

T = np.zeros(frames_set_array.shape,dtype='float64')
I = np.zeros(frames_set_array.shape,dtype='float64')
# array of fourier transforms of estimated power density
I_fftn = np.zeros(frames_set_array.shape,dtype='complex128')

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

def FFT2PlotData(data_fft2,hz=True):
    ''' Generates basic plot data from 2D fft data

        fft2_data : 2D fft image
        hz : Flag to convert freq data to Hertz. Default True.

        Returns ordered magnitude and ordered angle data ready for plotting
    '''
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
    ''' Filter the given power density matrix for the frequencies listed given the tolerance

        I : Power density matrix
        freq_list : List of frequencies to mask for
        freq_tol : Tolerance on searching for frequencies
        is_hz    : Flag indicating if the frequency list is in hz or rads
        Returns a masked copy of the power density matrix

        Performs a 2D FFT on the given power density matrix and extracts just the
        values that are within range of the given frequencies

        freq_list[i]-freq_tol <= np.angle(fftn(I)) <= freq_list[i]+freq_tol
    '''
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
    ''' Search the power density matrix for the freq to be used in the filter

        Qr : Data as processed by readH264
        mag_filt : Factor of maximum magnitude to filter the peaks by
        hz : Flag to return frequencies in terms of Hertz. Default True
        first_in : Index of the first frame where the laser is NOT on. Default 0 (i.e. first frame)

        Searches the data to identify the laser spatial frequencies to search for.
        As the peak frequency algorithm picks up lots of little peaks, only the
        frequencies whose magnitudes are above mag_filt times the maximum magnitude
        are returned. This gives helps return only the most dominant components.

        Returns a list of the frequencies of the most prominent components
    '''
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

print("Finding peak values of components introduced")
# get difference between fftn of one inactive frame and an active frame
# perform fft and store results
target_frames=[1000,1048]
I_diff = fftn(I[:,:,target_frames[1]])-fftn(I[:,:,target_frames[0]])
f,ax = plt.subplots()
ax.set_title('FFT Difference between Frame 1000 and 1048, Magnitude')
# plot the magnitude log and fftshifted
ax.imshow(np.log(1+fftshift(np.abs(I_diff))))
f.savefig('fft-diff-mag-2d-new-model.png')

ax.set_title('FFT Difference between Frame 1000 and 1048, Frequency')
ax.imshow(np.angle(I_diff))
f.savefig('fft-diff-freq-2d-new-model.png')

# convert Fourier difference to plot data
I_mag,I_freq = FFT2PlotData(I_diff)
# find the peaks in the magnitude data
mag_peaks = find_peaks(I_mag)[0]
# filter to only get peaks that are above a percentage of max
# helps filter out tiny peaks that are picked up
mag_peaks = mag_peaks[I_mag[mag_peaks]>(0.05*np.max(I_mag[mag_peaks]))]
peaks_mag = I_mag[mag_peaks]
peaks_freq = I_freq[mag_peaks]

# sort components by frequency so +ve/-ve freq components are next to each other
pk_comp = sorted(zip(peaks_freq,peaks_mag),key=lambda x : x[1])
# reform the list into the freq pairs
pk_comp = [(ii,jj) for ii,jj in zip(pk_comp[::2],pk_comp[1::2])]
# order the pairs by contribution (combined mag) of +ve and -ve components
pk_comp.sort(key=lambda x : x[0][1]+x[1][1])

# width of window
freq_tol = 0.5

print('Testing different frequency tolerances')

# get fft of an active frame
I_fft = fftn(I[:,:,1048])
# save min and ax vaues of original matrix so other contours can be scaled as well
o_min,o_max = np.min(I[:,:,1048]),np.max(I[:,:,1048])
mm,ff = FFT2PlotData(I_fft)

##fig,ax = plt.subplots(2,2,tight_layout=True,figsize=[x*1.5 for x in plt.rcParams['figure.figsize']])
### plot data as contour
##orig_ct = ax[0,0].contourf(I[:,:,1048])
##ax[0,0].set_title('Original Power Density f=1048')
##
##ax[0,1].plot(ff,mm)
##ax[0,1].set(xlabel='Frequency (Hz)',ylabel='Magnitude (W/m2)',title='Original Frequency Response, f=1048')
##filt_ct = [ax[1,0].contourf(I[:,:,1048])]
##filt_line, = ax[1,1].plot(ff,mm)
##ax[1,1].set(xlabel='Frequency (Hz)',ylabel='Magnitude (W/m2)')
##
##fig2,ax2 = plt.subplots(1,2,tight_layout=True,figsize=[x*1.5 for x in plt.rcParams['figure.figsize']])
##ax2[1].set_title('Largest Contour Found')
##
##fig3,ax3 = plt.subplots(1,2,tight_layout=True,figsize=[x*1.5 for x in plt.rcParams['figure.figsize']])
##ax3[1].set_title('Largest Canny Contour Found')
##
##fig4,ax4 = plt.subplots(1,2,tight_layout=True,figsize=[x*1.5 for x in plt.rcParams['figure.figsize']])
##ax4[1].set_title('Largest Canny Contour Found')
##for tol in np.arange(0.0,1.1,0.1):
##    # filter power density matrix with set tolerance
##    I_filt = filterFreq(I[:,:,1048],peaks_freq,freq_tol=tol)
##    # get fft of result
##    filt_fft = fftn(I_filt)
##    # generate plot data from it
##    m_filt,f_filt = FFT2PlotData(filt_fft)
##    ## update plots
##    # create contour using the limits of the original to scale the result
##    ax[1,0].clear()
##    ax[1,0].contourf(I_filt,vmin=o_min,vmax=o_max)
##    ax[1,0].set_title('Filtered Power Density, tol={:.1f}'.format(tol))
##    # update frequency response plot
##    filt_line.set_data(f_filt,m_filt)
##    ax[1,1].relim()
##    ax[1,1].autoscale_view(True,True,True)
##    #ax[1,1].plot(f_filt,m_filt)
##    ax[1,1].set_title('Filtered Frequency Response, tol={:.1f}'.format(tol))
##
##    # search for contours in the filtered image
##    ct = cv2.findContours(I_filt.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
##    # sort contours by contour area
##    ct.sort(key=cv2.contourArea,reverse=True)
##    # draw largest contour filled on a mask
##    filt_mask = cv2.drawContours(np.zeros(I_filt.shape),ct,0,255,-1)
##
##    # draw filtered results as a contour
##    ax2[0].contourf(I_filt,vmin=o_min,vmax=o_max)
##    ax2[0].set_title('Filtered Power Density, tol={:.1f}'.format(tol))
##    # show contour mask
##    ax2[1].contourf(filt_mask)
##
##    edges = cv2.Canny(I_filt.astype('uint8'),150,450,3)
##    ct = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
##    ct.sort(key=cv2.contourArea,reverse=True)
##    filt_mask = cv2.drawContours(np.zeros(I_filt.shape),ct,0,255,-1)
##    
##    ax3[0].contourf(I_filt,vmin=o_min,vmax=o_max)
##    ax3[0].set_title('Filtered Power Density, tol={:.1f}'.format(tol))
##    # show contour mask
##    ax3[1].contourf(filt_mask)
##
##    ax4[0].contourf(I_filt,vmin=o_min,vmax=o_max)
##    ax4[0].set_title('Filtered Power Density, tol={:.1f}'.format(tol))
##    # find the min circle that will enclose the largest contour
##    if len(ct)>0:
##        (x,y),r=cv2.minEnclosingCircle(ct[0])
##        # create meshgrid to search for points that are inside the circle
##        X,Y = np.meshgrid(np.arange(0,rows,1),np.arange(0,cols,1))
##        filt_mask = (X-int(x))**2.0+(Y-int(y))**2.0 <= int(r)
##    else:
##        filt_mask = np.full(I_filt.shape,False,dtype=np.bool)
##    ax4[1].contourf(filt_mask.astype('uint8'))
##    
##    # save plotted results
##    #maxCurrFig()
##    #fig.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/FrequencyTolerance/filt-impact-t{0}-{1}-new-model.png'.format(*'{:.2f}'.format(tol).split('.')))
##    #fig2.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/FrequencyTolerance/ct-filt-impact-t{0}-{1}-new-model.png'.format(*'{:.2f}'.format(tol).split('.')))
##    fig3.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/FrequencyTolerance/ct-canny-filt-impact-t{0}-{1}-new-model.png'.format(*'{:.2f}'.format(tol).split('.')))
##    fig4.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/FrequencyTolerance/ct-min-circle-filt-impact-t{0}-{1}-new-model.png'.format(*'{:.2f}'.format(tol).split('.')))

## contour filter timelapse
##fig,ax = plt.subplots(1,2)
##for f in range(depth):
##    ax[0].contourf(I[:,:,f])
##    ax[0].set_title('Original Frame,f={0}'.format(f))
##    # filter contour is scaled relative to original frame
##    # original frame has higher magnitude than the filtered frame
##    # contour limits help emphasise what is different between filtered and original
##    ax[1].contourf(filterFreq(I[:,:,f],peaks_freq,0.2),vmin=np.min(I[:,:,f]),vmax=np.max(I[:,:,f]))
##    ax[1].set_title('Filtered Frame,f={:d},t={:.2f}'.format(f,0.2))
##    fig.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/FilterTimelapse/filt-f{0}-t0-2-new-model.png'.format(f))

tol_range = np.arange(0.0,5.0,0.05)

## power estimate filter timelapse
f,ax = plt.subplots(1,2)
ax[0].set(xlabel='Frame Index',ylabel='Power (W)',title='Original')
ax[1].set(xlabel='Frame Index',ylabel='Power (W)',title='Filtered')
orig_line = ax[0].plot(tol_range,np.zeros(tol_range.shape))
filt_lint = ax[1].plot(tol_range,np.zeros(tol_range.shape))
f.suptitle('Power Estimate using Original and Filtered Frame')

# plot for all filtered results overlapping
f_all,ax_all = plt.subplots(figsize = [s*2.0 for s in plt.rcParams['figure.figsize']])
ax_all.set(xlabel='Frame Index',ylabel='Power (W)',title='Impact of Filtering Tolerances on Power Estimate')

# construct the power estimate results
print("Estimating power and the impact filtering has on it")
#metrics for quality
dist_filt = []

for tol in tol_range:
    print("Trying tolerance ",tol)
    pest_orig = [estPArea_Canny(I[:,:,ff],pixel_pitch)[2] for ff in range(depth)]
    pest_filt = [estPArea_Canny(filterFreq(I[:,:,ff],peaks_freq,tol),pixel_pitch)[2] for ff in range(depth)]

    ## caculate average distance between peaks and target power
    # find all peaks
    tol_peaks_idx = find_peaks(pest_filt)[0]
    # filter peaks to just get the most active ones
    tol_peaks = np.asarray(pest_filt)[tol_peaks_idx]
    tol_peaks = tol_peaks[tol_peaks>(max(pest_filt)*0.05)]
    # add mean distance to list
    # taking just the real component, discarded anyway on plot
    dist_filt.append(np.real(np.mean(500.0-tol_peaks)))
    
    # clear axes
    ax[0].cla()
    ax[1].cla()
    # plot results
    ax[0].plot(pest_orig)
    ax[1].plot(pest_filt)
    f.suptitle('Power Estimate using Original and Filtered Frame,tol={:.2f} Hz'.format(tol))
    ax[0].set(xlabel='Frame Index',ylabel='Power (W)')
    ax[1].set(xlabel='Frame Index')
    f.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/power-est-filt-t{0}-{1}-new-model.png'.format(*'{:.2f}'.format(tol).split('.')))

    # add new 'layer' to overlapping plot
    ax_all.plot(pest_filt,label='{:.2f}'.format(tol))

# add legend to plot
f_all.legend()
# save fig
f_all.savefig('C:/Users/DB/Desktop/BEAM/Scripts/FFTFilter/power-est-all-filt')

# plot average distance results
f,ax = plt.subplots()
ax.plot(tol_range,dist_filt)
ax.set(xlabel='Filter Tolerance (Hz)',ylabel='Average Peak Distance (W)',title='Frequency Tolerance impact on Avg. Distance to Target Power')
f.savefig('filt-tol-impact-avg-dist-new-model.png')

print("Best frequency tolerance for range [{:.2f},{:.2f}] is {:.2f} Hz".format(tol_range.min(),tol_range.max(),tol_range[np.asarray(dist_filt).argmin()]))

# plot fft difference with marked peaks
f,ax = plt.subplots()
ax.plot(I_freq,I_mag,'b-',label='FFT2 Diff.')
ax.plot(I_freq[mag_peaks],I_mag[mag_peaks],'yx',label='Peaks')
ax.set(xlabel='Frequency (Hz)',ylabel='Magnitude',title='Difference between FFTs of frame {0} and {1}'.format(target_frames[0],target_frames[1]))
ax.legend()
f.savefig("fft-diff-f{0}-f{1}-new-model.png".format(target_frames[0],target_frames[1]))

# perform the inverse fourier transform on the image to reveal the components introducted 
f,ax = plt.subplots()
ax.set_title('IFFT of difference between FFTs of frame {0} and {1}'.format(target_frames[0],target_frames[1]))
# create contour of inverse fft of the components introduced between inverse and inactive frames
diff_ct = ax.contourf(ifftn(I_diff),cmap='viridis')
# create divider for colorbar
div = make_axes_locatable(ax)
# add axes to divider
cax= div.append_axes("right",size="7%",pad="2%")
# add colorbar to divider axes
colorbar(diff_ct,cax=cax)
f.savefig("ifft-of-diff-bt-f{0}-f{1}-new-model.png".format(target_frames[0],target_frames[1]))

print("Plotting inverse fft of components")
# create array of axes to hold the original fft and the subsequent components
f,ax = plt.subplots(2,9,constrained_layout=True)
f.suptitle('IFFT of Components in FFT Difference between frame {0} and {1}'.format(target_frames[0],target_frames[1]))
# perform inverse fft of difference to see the components introduced
diff_ifftn = ifftn(I_diff)
# find min and max values to scale future contours
o_min,o_max = np.amin(diff_ifftn),np.amax(diff_ifftn)
ax[0,0].contourf(diff_ifftn)
ax[0,0].set_title='Original'
# axes for storing each component separately
f2,ax2 = plt.subplots()

# get angle of fft matrix to avoid having to recalculate it each time
diff_angle = np.angle(I_diff)*2.0*np.pi
for ii in range(1,len(peaks_freq)):
    # find where the freq is within range
    # [freq-(freq*freq_tol),freq+(freq*freq_tol)]
    x,y = np.where(np.logical_and(diff_angle>=(peaks_freq[ii-1]-freq_tol),diff_angle<=(peaks_freq[ii-1]+freq_tol)))
    # create blank copy of matrix
    I_copy = np.zeros(I_diff.shape,dtype=I_diff.dtype)
    # update values with ones of interest in matrix
    I_copy[x,y] = I_diff[x,y]
    # perform ifftn on just the values we're interested in
    # draw contour of ifftn results
    ax[(*np.unravel_index(ii,ax.shape),)].contourf(ifftn(I_copy),cmap='viridis')

    ax2.contourf(ifftn(I_copy),cmap='viridis')
    ax2.set_title('M={:.2f},F={:.2f}'.format(peaks_mag[ii-1],peaks_freq[ii-1]))
    f2.savefig('fftdiff-c{:d}-new-model.png'.format(ii))
    
f.savefig('fftdiff-ifft-components-new-model.png')

## scale contours to the same limits as the real component of the original contour
# scale via real components of the numbers
# as the original result is pure real numbers, it makes sense that the imaginary components of +ve and -ve components cancel each other out
# perform the inverse fourier transform on the image to reveal the components introducted 
f,ax = plt.subplots(2,9,constrained_layout=True)
f2,ax2 = plt.subplots()
f.suptitle('IFFT of Components in FFT Difference between frame {0} and {1}, Scaled, Real'.format(target_frames[0],target_frames[1]))
# create contour and save a copy of it
o_ct = ax[0,0].contourf(diff_ifftn)
ax[0,0].set_title='Original'
for ii in range(1,len(peaks_freq)):
    # find where the freq is within range
    # [freq-(freq*freq_tol),freq+(freq*freq_tol)]
    x,y = np.where(np.logical_and(diff_angle>=(peaks_freq[ii-1]-freq_tol),diff_angle<=(peaks_freq[ii-1]+freq_tol)))
    # create blank copy of matrix
    I_copy = np.zeros(I_diff.shape,dtype=I_diff.dtype)
    # update values with ones of interest in matrix
    I_copy[x,y] = I_diff[x,y]
    # perform ifftn on just the values we're interested in
    # draw contour of ifftn results
    # set the limits so the contour color is scaled the same way as o_ct
    ax[(*np.unravel_index(ii,ax.shape),)].contourf(ifftn(I_copy),cmap='viridis',vmin=np.real(o_min),vmax=np.real(o_max))

    ax2.contourf(ifftn(I_copy),cmap='viridis',vmin=np.real(o_min),vmax=np.real(o_max))
    ax2.set_title('M={:.2f},F={:.2f}'.format(peaks_mag[ii-1],peaks_freq[ii-1]))
    f2.savefig("fftdiff-c{:d}-scale-real-new-model.png".format(ii))

f.savefig('fftdiff-ifft-components-scale-real-new-model.png')
#/.show()

# create array of plots for cumulative sum
f,ax = plt.subplots(len(pk_comp),1,constrained_layout=True)
f.suptitle('Cumulative Sum of Ordered Components')
# create matrix to store current state of cumulative sum
I_copy = np.zeros(I_diff.shape,dtype=I_diff.dtype)
# iterate through each freq pairs
for ppi,pp in enumerate(pk_comp):
    # find where the freq is within range
    # [freq-(freq*freq_tol),freq+(freq*freq_tol)]
    xi,yi = np.where(np.logical_and(diff_angle>=(pk_comp[ppi][0][0]-freq_tol),diff_angle<=(pk_comp[ppi][0][0]+freq_tol)))
    xj,yj = np.where(np.logical_and(diff_angle>=(pk_comp[ppi][1][0]-freq_tol),diff_angle<=(pk_comp[ppi][1][0]+freq_tol)))
    # update values with ones of interest in matrix
    I_copy[xi,yi] += I_diff[xi,yi]
    I_copy[xj,yj] += I_diff[xj,yj]
    # draw contour of inverse using same scaling as the original complete plot created earlier
    ax[ppi].contourf(ifftn(I_copy),cmap='viridis',vmin=np.real(o_min),vmax=np.real(o_max))
    # update title to show what component it is, what the freq range is and the sum magnitude of the components
    # as the magnitude for each pair is the same, the 
    ax[ppi].set_title('Added Cp={:d},F=[{:.2f},{:.2f}],M={:,.2f}'.format(ppi,pk_comp[ppi][0][0],pk_comp[ppi][1][0],pk_comp[ppi][0][1]+pk_comp[ppi][1][1]))

#maxCurrFig()
f.savefig('cum-sum-diff-components-new-model.png')

f.suptitle('Canny Edge Detection of each component Pair')
for ppi,pp in enumerate(pk_comp):
    # find where the freq is within range
    # [freq-(freq*freq_tol),freq+(freq*freq_tol)]
    xi,yi = np.where(np.logical_and(diff_angle>=(pk_comp[ppi][0][0]-freq_tol),diff_angle<=(pk_comp[ppi][0][0]+freq_tol)))
    xj,yj = np.where(np.logical_and(diff_angle>=(pk_comp[ppi][1][0]-freq_tol),diff_angle<=(pk_comp[ppi][1][0]+freq_tol)))
    # update values with ones of interest in matrix
    I_copy = np.zeros(I_diff.shape,dtype=I_diff.dtype)
    I_copy[xi,yi] = I_diff[xi,yi]
    I_copy[xj,yj] = I_diff[xj,yj]
    # perform canny edge detection on the component pair
    edges = cv2.Canny(I_copy.astype('uint8'),100,300,3)
    # find largest contour
    ct = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    # if there are contours
    if len(ct)>0:
        # sort list by size
        ct.sort(key=cv2.contourArea,reverse=True)
        # draw largest contour onto a blank mask filled
        mask = cv2.drawContours(np.zeros((rows,cols)),ct,0,255,-1)
        # update plot
        ax[ppi].cla()
        ax[ppi].imshow(mask)
        
#maxCurrFig()
f.savefig('diff-components-canny-l-ct-new-model.png')

print("Finished")
