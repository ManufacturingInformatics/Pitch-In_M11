import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model
import WriteToHPF5.beamwritetohdf5 as hd
import numpy as np
import os
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani

def powerEstHoughCircle(circle,I,PP):
   ''' Estimating power using circle found in power density matrix by Hough Circle algorithm

       circle : Portion of the Hough Circle results matrix representing one circle. It's an array
                of a list of three values [x,y,radius]
       I : Power density matrix, W/m2
       PP : Pixel Pitch, m

       This is designed to be used with Numpy's apply along matrix command as applied to results

       Return sums the values within circle and multiplies by the area
   '''
   # create empty mask to draw results on 
   mask = np.zeros(I.shape[:2],dtype='uint8')
   # draw filled circle using given parameters
   cv2.circle(mask,(*circle[:2],),circle[2],(255),cv2.FILLED)
   # find where it was drawn
   i,j = np.where(mask==255)
   # sum the power density values in that area and multiply by area
   return np.sum(I[i,j])*(np.pi*(circle[2]*PP)**2.0)

def drawHoughCirclesOver(circles,img,**kwargs):
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

    # initially set the mask to be a 3d version of the target image if it's not 3d already
    if len(img.shape)<3:
        # the grayscale image is converted to a color 3d image by using the image for the other channels
        # the image looks exactly the same in color as it would grayscale
        mask = np.dstack((img,img,img))
    else:
        mask = img.copy()
    # iterate through circles array
    # array is [1xCx3]
    for c in range(len_c):
        # draw circle boundary
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), circles[0,c,2], bcol, 1)
    for c in range(len_c):
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), 1, ccol, 1)
    # return circle result
    return mask

def powerEstBestCircle(I,radii_range,pixel_pitch):
   # normalize image so it can be used by skimage
   I_norm = I/I.max(axis=(0,1))
   # search for circles
   res = hough_circle(I_norm,radii_range)
   # get the top three circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=1)
   # choose the highest rated circle to estimate power with
   return powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I,pixel_pitch)

def powerEstAccumCircle(I,radii_range,PP,min_accum=accum_tol,num_peaks=10):
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
      return powerEstHoughCircle([cx[ai][0],cy[ai][0],radii[ai][0]],I,pixel_pitch)

def remCirclesOutBounds(circles,shape):
   ''' Remove the circles form the set of circles whose area goes out of bounds in the image

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       shape : Shape of the data they were found in

       This function checks if given the postion and radius of the circle whether any portion of it
       goes outside the bounds of shape. It performs a Logical check of

       Returns filtered list of circles
   '''
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.Logical_or(
      np.Logical_or(circles[0,:,0]-circles[0,:,2]>128,circles[0,:,0]+circles[0,:,2]>128), # checking if any x coordinates are out of bounds
      np.Logical_or(circles[0,:,1]-circles[0,:,2]>128,circles[0,:,1]+circles[0,:,2]>128)))[0] # checking if any y coordinates are out of bounds
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

def collectHoughStats(accums,cx,cy,radii):
   return accums.max(),cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]

# paths to file
video_path = ""
xml_path = ""
# name of the end files based off the video filename
file_name = os.path.splitext(os.path.basename(video_path))[0]

### create folders
## Data as images
os.makedirs("{}-Results/DataImages/RadHeat",exist_ok=True)
os.makedirs("{}-Results/DataImages/Temperature",exist_ok=True)
os.makedirs("{}-Results/DataImages/PowerDensity",exist_ok=True)
## pre-processing results
os.makedirs("{}-Results/PreProcessing/HoughCircles",exist_ok=True)
os.makedirs("{}-Results/PreProcessing/EdgeDetection/Scharr",exist_ok=True)
os.makedirs("{}-Results/PreProcessing/EdgeDetection/Sobel",exist_ok=True)
## power estimates
os.makedirs("{}-Results/Plots/PowerEstimates",exist_ok=True)
## XML Data plots
os.makedirs("{}-Results/Plots/XMLData",exist_ok=True)
## Data Files
os.makedirs("{}-Results/DataFiles/HDF5Files",exist_ok=True)
os.makedirs("{}-Results/DataFiles/CSVFiles",exist_ok=True)
## Video Files
os.makedirs("{}-Results/VideoFiles",exist_ok=True)

## process settings
# room temperature
T0 = mat.CelciusToK(23.0)
# frame rate of the camera
fc = 65.0
# time between frames
tc = 1/fc
# pixel pitch
pixel_pitch=20.0e-6
# Laser radius
r0 = 0.00035 #m
# Laser radius in pixels
r0_p = int(np.ceil(r0/pixel_pitch))
# normalized accumulator score tolerance used in filtering circles
accum_tol = 0.6

## read in camera footage and convert it to temperature and power density
# get material data
e,K,D = mat.buildMaterialData()
# read in data
Qr = model.readH264(video_path)
xml_data = model.readXMLData(xml_data)
# predict temperature and power density
T = model.predictTemperature(Qr,e,T0,vectorize=True)
T = np.nan_to_num(T,copy=False)
I = model.predictPowerDensity(T,K,D,T0)
T = np.nan_to_num(T,copy=False)
# calculate global maximum values used in normalization
Qr_max = Qr.max(axis=(0,1,2))
T_max = T.max(axis=(0,1,2))
I_max = I.max(axis=(0,1,2))

# write the data to a hdf5 file
# power can be added later
with hd.initialize("{}-Results/DataFiles/HDF5Files/data.hdf5".format(file_name))as file:
    hd.updateData(file,createNew=True,Qr=Qr,T=T,Tp=T.max(axis=(0,1)),I=I,xml=xml_data)

# normalize the value based on global max
# used in skimage functions as float images
Qr_norm = Qr/Qr_max
T_norm = T/T_max
I_norm = I/I_max
# convert to 8-bit to be used in OpenCV functions
Qr_cv = (Qr_norm*255).astype('uint8')
T_cv = (T_norm*255).astype('uint8')
I_cv = (I_norm*255).astype('uint8')

## estimating power by using hough circle
## power matricies
# highest scoring circle
P_best = np.zeros(I.shape[2])
# high scorinng circle after filtering
P_accum = np.zeros(I.shape[2])
# highest scoring after filtering out of bounds circles
P_oob = np.zeros(I.shape[2])
# power estimate using the unfiltered Scharr sum result
P_best_scharr_raw = np.zeros(I.shape[2])
# power estimate using the unfiltered Sobel sum result
P_best_sobel_raw = np.zeros(I.shape[2])
# power estimate after denoising the Scharr results by <>
P_best_scharr_d = np.zeros(I.shape[2])
# power estimate after denoising the Sobel results by <>
P_best_sobel_d = np.zeros(I.shape[2])
# power estimate using the unfiltered Scharr sum result
P_accum_scharr_raw = np.zeros(I.shape[2])
# power estimate using the unfiltered Sobel sum result
P_accum_sobel_raw = np.zeros(I.shape[2])
# power estimate after denoising the Scharr results by <>
P_accum_scharr_d = np.zeros(I.shape[2])
# power estimate after denoising the Sobel results by <>
P_accum_sobel_d = np.zeros(I.shape[2])

## hough circle information
# raw data
best_accum = np.zeros(I.shape[2])
best_accum_filt = np.zeros(I.shape[2])
best_x = np.zeros(I.shape[2])
best_x_filt = np.zeros(I.shape[2])
best_y = np.zeros(I.shape[2])
best_y_filt = np.zeros(I.shape[2])
best_r = np.zeros(I.shape[2])
best_r_filt = np.zeros(I.shape[2])
# edge detection, scharr
best_scharr_accum = np.zeros(I.shape[2])
best_scharr_x = np.zeros(I.shape[2])
best_scharr_y = np.zeros(I.shape[2])
best_scharr_r = np.zeros(I.shape[2])
# edge detection, sobel
best_sobel_accum = np.zeros(I.shape[2])
best_sobel_x = np.zeros(I.shape[2])
best_sobel_y = np.zeros(I.shape[2])
best_sobel_r = np.zeros(I.shape[2])

# assuming gaussian behaviour, 99.7% of values are within 4 std devs of mean
# used as an upper limit in hough circles
rmax_gauss = 4*r0_p
# radius range
radius_range = np.arange(r0_p,rmax_gauss,1)

# iterate through frames
for ff in range(I.shape[2]):
    ###### DATA AS IMAGES ######
    # normalized data is colormapped and written to files
    cv2.imwrite("{}-Results/DataImages/RadHeat/qr-f{}.png".format(file_name,ff),cv2.applyColorMap(Qr_cv[:,:,ff],cv2.COLORMAP_JET))
    cv2.imwrite("{}-Results/DataImages/Temperature/temperature-f{}.png".format(file_name,ff),cv2.applyColorMap(T_cv[:,:,ff],cv2.COLORMAP_JET))
    cv2.imwrite("{}-Results/DataImages/PowerDensity/powerdensity-f{}.png".format(file_name,ff),cv2.applyColorMap(I_cv[:,:,ff],cv2.COLORMAP_JET))
    
    ###### HOUGH CIRCLE ######
    P_best[ff] = powerEstBestCircle(I[:,:,ff],radius_range,pixel_pitch)
    P_accum[ff] = powerEstAccumCircle(I[:,:,ff],radius_range,pixel_pitch,accum_tol,10)
    # collect the circles
    res = hough_circle(I[:,:,ff]/I[:,:,ff].max(),radius_range)
    accums,cx,cy,radii = hough_circle_peaks(res,radius_range,num_peaks=5)
    # package circles
    circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
    # draw the circles on a blank image
    mask = drawHoughCirclesOver(circles,I_cv[:,:,ff])
    # save the drawn circles
    cv2.imwrite("{}-Results/PreProcessing/HoughCircles/hough-circle-f{}.png".format(file_name,ff),mask)
    ## collect hough circle data
    # search for accumulator scores that are above filter tolerance
    aci = np.where(accums>=accum_tol)[0]
    # if there are no circles above the tolerance
    if aci.shape[0]==0:
        best_accum_filt[ff] = 0.0
    else:
        best_accum_filt[ff] = accums[aci].max()
    # get highest scoring tolerance
    best_accum[ff] = accums.max()
    # get best circle centre of highest scoring circle before and after filtering
    best_x[ff] = cx[accums.argmax()]
    best_x_filt[ff] = cx[aci][accums[aci].argmax()]
    best_y[ff] = cy[accums.argmax()]
    best_y_filt[ff] = cy[aci][accums[aci].argmax()]
    best_r[ff] = radii[accums.argmax()]
    best_r_filt[ff] = cy[aci][accums[aci].argmax()]
    
    # filter out of bounds hough circles
    circles = remCirclesOutBounds(circles,I.shape[:2])
    # calculate power of highest scoring circle after removing out of bounds circles
    P_oob[ff] = powerEstHoughCircle(circles[accums.argmax()],I[:,:,ff],pixel_pitch)
    
    ###### EDGE DETECTION ######
    # scharr sum
    img_edge = np.abs(cv2.Scharr(I_cv[:,:,ff],cv2.CV_8U,1,0))+np.abs(cv2.Scharr(I_cv[:,:,ff],cv2.CV_8U,0,1))
    cv2.imwrite("{}-Results/PreProcessing/EdgeDetection/Scharr/scharr-edge-f{}.png".format(file_name,ff),img_edge)
    res = hough_circle(img_edge,radius_range)
    accums,cx,cy,radii = hough_circle_peaks(res,radius_range,num_peaks=5)
    # package circles
    circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
    # calculate power using the highest scoring circle
    P_best_scharr_raw[ff] = powerEstHoughCircle(circles[accums.argmax()],I[:,:,ff],pixel_pitch)
    # draw the top 5 circles and save the image
    mask = drawHoughCirclesOver(circles,I_cv[:,:,ff])
    cv2.imwrite("{}-Results/PreProcessing/HoughCircles/hough-circle-edge-scharr-f{}.png".format(file_name,ff),mask)
    ## collect hough circle informationce
    # get highest scoring tolerance
    best_scharr_accum[ff] = accums.max()
    # get best circle centre of highest scoring circle before and after filtering
    best_scharr_x[ff] = cx[accums.argmax()]
    best_scharr_y[ff] = cy[accums.argmax()]
    best_scharr_r[ff] = radii[accums.argmax()]
    
    # sobel sum
    img_edge = np.abs(cv2.Sobel(I_cv[:,:,ff],cv2.CV_8U,1,0))+np.abs(cv2.Sobel(I_cv[:,:,ff],cv2.CV_8U,0,1))
    cv2.imwrite("{}-Results/PreProcessing/EdgeDetection/Sobel/sobel-edge-f{}.png".format(file_name,ff),img_edge)
    res = hough_circle(img_edge,radius_range)
    accums,cx,cy,radii = hough_circle_peaks(res,radius_range,num_peaks=5)
    # package circles
    circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
    P_best_sobel_raw[ff] = powerEstHoughCircle(circles[accums.argmax()],I[:,:,ff],pixel_pitch)
    mask = drawHoughCirclesOver(circles,I_cv[:,:,ff])
    cv2.imwrite("{}-Results/PreProcessing/HoughCircles/hough-circle-edge-sobel-f{}.png".format(file_name,ff),mask)

    ## collect hough circle information
    # get highest scoring tolerance
    best_sobel_accum[ff] = accums.max()
    # get best circle centre of highest scoring circle before and after filtering
    best_sobel_x[ff] = cx[accums.argmax()]
    best_sobel_y[ff] = cy[accums.argmax()]
    best_sobel_r[ff] = radii[accums.argmax()]

### plot results
print("Plotting data")
## set up axes
# 2d axes
f,ax = plt.subplots()
# 3d axes
f3d = plt.figure()
ax3d = f3d.add_subplot(111,projection='3d')
## power estimates
ax.plot(P_best)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate using the Highest Scoring Hough Circle')
f.savefig("{}-Results/Plots/pest-hough-best.png".format(file_name))

ax.clear()
ax.plot(P_accum)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate after filtering by accumulator score with a\n tolerance of accum_tol using the Highest Scoring Hough Circle')
f.savefig("{}-Results/Plots/pest-hough-filt-accum.png".format(file_name))

ax.clear()
ax.plot(P_oob)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate after Filtering Circles Whose Area Goes Out of\n the Image Area')
f.savefig("{}-Results/Plots/pest-hough-filt-oob.png".format(file_name))

## xml data
ax3d.plot(xml_data[4],xml_data[5],xml_data[6])
ax3d.set(xlabel='X Position (mm)',ylabel='Y Position (mm)',zlabel='Z Position (mm)')
f3d.suptitle('Laser Head Position According to Motor Log')
f3d.savefig("{}-Results/Plots/motor-position-3d.png".format(file_name))
# create and plot an array of several plots showing the xml data
fm = plt.Figure()
axm = fm.add_subplot(231)
axm.plot(xml_data[1],xml_data[2])
axm.set(xlabel='Time (s)',ylabel='Current to generate Torque (Amps)',title='Current supplied to generate the Laser Head Motor torque')

axm = fm.add_subplot(232)
axm.plot(xml_data[1],xml_data[3])
axm.set(xlabel='Time (s)',ylabel='Motor Velocity (mm/s)',title='Laser Head Motor velocity (mm/s)')

axm = fm.add_subplot(233)
axm.plot(xml_data[1],xml_data[4])
axm.set(xlabel='Time (s)',ylabel='Motor Head X Position (mm)',title='X Position of the Laser Head according to Motor Log')

axm = fm.add_subplot(234)
axm.plot(xml_data[1],xml_data[5])
axm.set(xlabel='Time (s)',ylabel='Motor Head Y Position (mm)',title='Y Position of the Laser Head according to Motor Log')

axm = fm.add_subplot(235)
axm.plot(xml_data[1],xml_data[6])
axm.set(xlabel='Time (s)',ylabel='Motor Head Z Position (mm)',title='Z Position of the Laser Head according to Motor Log')
fm.savefig("{}-Results/Plots/motor-log-combo-plot.png".format(file_name))
plt.close(fm)
# plot each dataset separately
ax.clear()
ax.plot(xml_data[1],xml_data[6])
ax.set(xlabel='Time (s)',ylabel='Motor Head Z Position (mm)')
f.suptitle('Z Position of the Laser Head according to Motor Log')
f.savefig("{}-Results/Plots/motor-position-z.png".format(file_name))

ax.clear()
ax.plot(xml_data[1],xml_data[5])
ax.set(xlabel='Time (s)',ylabel='Motor Head Y Position (mm)')
f.suptitle('Y Position of the Laser Head according to Motor Log')
f.savefig("{}-Results/Plots/motor-position-y.png".format(file_name))

ax.clear()
ax.plot(xml_data[1],xml_data[4])
ax.set(xlabel='Time (s)',ylabel='Motor Head X Position (mm)')
f.suptitle('X Position of the Laser Head according to Motor Log')
f.savefig("{}-Results/Plots/motor-position-x.png".format(file_name))

ax.clear()
ax.plot(xml_data[1],xml_data[3])
ax.set(xlabel='Time (s)',ylabel='Motor Velocity (mm/s)')
f.suptitle('Laser Head Motor velocity (mm/s)')
f.savefig("{}-Results/Plots/motor-velocity.png".format(file_name))

ax.clear()
ax.plot(xml_data[1],xml_data[2])
ax.set(xlabel='Time (s)',ylabel='Current to generate Torque (Amps)')
f.suptitle('Current supplied to generate the Laser Head Motor torque')
f.savefig("{}-Results/Plots/motor-torque-current.png".format(file_name))

## Hough circle data
ax.clear()
ax.plot(best_accum)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score found during Power Estimate')
f.savefig('{}-Results/Plots/highest-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(best_accum_filt)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after filtering for values above tol={}'.format(accum_tol))
f.savefig('{}-Results/Plots/highest-filt-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(best_x)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle')
f.savefig('{}-Results/Plots/hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(best_x_filt)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after\nfiltering for values above tol={}'.format(accum_tol))
f.savefig('{}-Results/Plots/filt-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(best_y)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle')
f.savefig('{}-Results/Plots/hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(best_y_filt)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after\nfiltering for values above tol={}'.format(accum_tol))
f.savefig('{}-Results/Plots/filt-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(best_r)


### Building animations as video files
print("Building animations")
## xml position data
ax3d.clear()
line3d, = ax3d.plot(xml_data[4],xml_data[5],xml_data[6])
ax3d.set(xlabel='X Position (mm)',ylabel='Y Position (mm)',zlabel='Z Position (mm)')
f3d.suptitle('Laser Head Position According to Motor Log')

def rotate(angle):
    ax3d.view_init(elev=35,azim=angle)
# save animation of a rotating fig showing the position data
plot_anim = ani.FuncAnimation(f3d,rotate,360,interval=10,blit=False)
plot_anim.save("{}-Results/VideoFiles/motor-3d-pos-rotate.gif".format(file_name),writer='imagemagick',fps=60)

# combine the data together into a single matrix
pos = np.row_stack((xml_data[4],xml_data[5],xml_data[6]))
def plot_data(i,pos,lobj):
    # set the x and y data for the line
    # set data method for 3d objects does not have a z component here
    lobj.set_data(pos[0,:i],pos[1,:i])
    # set the z components
    lobj.set_3d_properties(pos[2,:i])
    return lobj

# create the animation showing the line being drawn
plot_anim = ani.FuncAnimation(f3d,plot_data,len(xml_data[4]),fargs=(pos,line3d),interval=1,blit=False)
# create an ffmpeg writer for creating the animation
ffmpeg_writer = ani.writers['ffmpeg']
line_writer = ffmpeg_writer(fps=200,metadata=dict(artist='DBM'),bitrate=-1)
# save the animation using the created writer
plot_anim.save("{}-Results/VideoFiles/motor-3d-pos-line-draw.avi".format(file_name),writer=line_writer)

### save data
print("Saving data")
## power estimates
np.savetxt("{}-Results/DataFiles/CSVFiles/pest-hough-best.csv".format(file_name),P_best,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/pest-hough-filt-accum.csv".format(file_name),P_best,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/pest-hough-filt-oob.csv".format(file_name),P_oob,delimiter=',')
## hough circle data
# raw values
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-x.csv".format(file_name),best_x,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-y.csv".format(file_name),best_y,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-radius.csv".format(file_name),best_r,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-x-accum-filt.csv".format(file_name),best_x_filt,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-y-accum-filt.csv".format(file_name),best_y_filt,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/raw-circle-centre-radius.csv".format(file_name),best_r_filt,delimiter=',')
# scharr edge detection
np.savetxt("{}-Results/DataFiles/CSVFiles/scharr-circle-centre-x.csv".format(file_name),best_x_scharr,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/scharr-circle-centre-y.csv".format(file_name),best_y_scharr,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/scharr-circle-accum.csv".format(file_name),best_accum_scharr,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/scharr-circle-centre-radius.csv".format(file_name),best_r_scharr,delimiter=',')
# sobel edge detection
np.savetxt("{}-Results/DataFiles/CSVFiles/sobel-circle-centre-x.csv".format(file_name),best_x_sobel,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/sobel-circle-centre-y.csv".format(file_name),best_y_sobel,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/sobel-circle-accum.csv".format(file_name),best_accum_sobel,delimiter=',')
np.savetxt("{}-Results/DataFiles/CSVFiles/sobel-circle-centre-radius.csv".format(file_name),best_r_sobel,delimiter=',')
