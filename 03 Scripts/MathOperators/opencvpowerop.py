import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model
import numpy as np
import os
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt
import cv2

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


def powerEstBestCircle(I,radii_range,pixel_pitch):
   # normalize image so it can be used by skimage
   I_norm = I/I.max(axis=(0,1))
   # search for circles
   res = hough_circle(I_norm,radii_range)
   # get the top three circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=1)
   # choose the highest rated circle to estimate power with
   return powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I,pixel_pitch)


T0 = mat.CelciusToK(23.0)
e,K,D = mat.buildMaterialData()
print("Reading in power")
Qr = model.readH264("D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264")
T = model.predictTemperature(Qr,e,T0)
np.nan_to_num(T,copy=False)
I = model.predictPowerDensity(T,K,D,T0)
np.nan_to_num(I,copy=False)

# laser radius
r0 = 0.00035 #m
# laser radius in pixels
pixel_pitch = 20e-6
r0_p = int(np.ceil(r0/pixel_pitch))
# assuming gaussian behaviour, 99.7% of values are within 4 std devs of mean
# used as an upper limit in hough circles
rmax_gauss = 4*r0_p
radii_range = np.arange(r0_p,64,1)
os.makedirs("PowerOp",exist_ok=True)

# convert set to grayscale
I_norm_cv = ((I/I.max(axis=(0,1,2)))*255)
# multiplier to ensure that the log value covers the 0-255 range
# assuming the images cover the range
# comes from 255 = c*sqrt(rmax)
c = 255/(255**0.5)
f,ax = plt.subplots()
pest = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    print("\r{}".format(ff),end='')
    # interpret as 16-bit so the values can be incremented
    cvsqrt = c*(I_norm_cv[:,:,ff]**0.5)
    #cvsqrt=cv2.applyColorMap(cvsqrt.astype('uint8'),cv2.COLORMAP_JET)
    cv2.imwrite("PowerOp/sqrtop-f{}.png".format(ff),cvsqrt.astype('uint8'))
    res = hough_circle(cvsqrt.astype('uint8'),radii_range)
    accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
    if accums.shape[0]!=0:
        pest[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
    else:
        pest[ff]=0.0

ax.clear()
ax.plot(pest)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate After Filtering by Square Root Operator')
f.savefig('pest-sqrtop.png')
np.savetxt("pest-sqrtop-global-assumption.csv",pest,delimiter=',')

# trying a different constant
I_max = I_norm_cv.max(axis=(0,1,2))
c = 255/(I_max**0.5)
os.makedirs("PowerOpDiffC",exist_ok=True)
print("\nTrying different constant")
for ff in range(I.shape[2]):
    print("\r{}".format(ff),end='')
    # interpret as 16-bit so the values can be incremented
    cvsqrt = c*(I_norm_cv[:,:,ff]**0.5)
    #cvsqrt=cv2.applyColorMap(cvsqrt.astype('uint8'),cv2.COLORMAP_JET)
    cv2.imwrite("PowerOpDiffC/sqrtop-diffc-f{}.png".format(ff),cvsqrt.astype('uint8'))
    res = hough_circle(cvsqrt.astype('uint8'),radii_range)
    accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
    if accums.shape[0]!=0:
        pest[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
    else:
        pest[ff]=0.0

ax.clear()
ax.plot(pest)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate After Filtering by Square Root Operator where\nScaling Constant is based off Global Maxima')
f.savefig('pest-sqrtop-global-maxima.png')
np.savetxt("pest-sqrtop-global-maxima.csv",pest,delimiter=',')

os.makedirs("SqrtOpLocalMax",exist_ok=True)
print("Trying scaling constant based off local maxima")
for ff in range(I.shape[2]):
    print("\r{}".format(ff),end='')
    # interpret as 16-bit so the values can be incremented
    c = 255/(I_norm_cv[:,:,ff].max((0,1))**0.5)
    cvsqrt = c*(I_norm_cv[:,:,ff]**0.5)
    #cvsqrt=cv2.applyColorMap(cvsqrt.astype('uint8'),cv2.COLORMAP_JET)
    cv2.imwrite("SqrtOpLocalMax/sqrtop-local-f{}.png".format(ff),cvsqrt.astype('uint8'))
    res = hough_circle(cvsqrt.astype('uint8'),radii_range)
    accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
    if accums.shape[0]!=0:
        pest[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
    else:
        pest[ff]=0.0

ax.clear()
ax.plot(pest)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate After Filtering by Square Root Operator where\nScaling Constant is based off Local Maxima')
f.savefig('pest-sqrtop-local-maxima.png')
np.savetxt("pest-sqrtop-local-maxima.csv",pest,delimiter=',')

## trying different tolerances and power estimates
tol = 20.0
tol_res = 1.0
print("Trying with different tolerances")
pest = np.zeros(len(list(range(-tol,tol,tol_res))))
for pi,pc in enumerate(range(-tol,tol,tol_res)):
    path = "ConstantTol/SquareRoot/{}".format(int(pc))
    os.makedirs(path,exist_ok=True)

    c = 255/(I_norm_cv[:,:,ff].max((0,1))**0.5)
    c += (1/pc)*c
    cvsqrt = c*(I_norm_cv[:,:,ff]**0.5)
    cv2.imwrite(os.path.join(path,"sqrtop-local-tol{}-f{}.png".format(int(pc),ff)),cvsqrt.astype('uint8'))
    res = hough_circle(cvsqrt.astype('uint8'),radii_range)
    accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
    if accums.shape[0]!=0:
        pest[pi,ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
    else:
        pest[pi,ff]=0.0

fs,axs = plt.subplots()
ax.clear()
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
for pi,pc in enumerate(range(-tol,tol,tol_res)):
    ax.plot(pest[pi,:],label="{}%".format(int(pc)))

    axs.clear()
    axs.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
    fs.suptitle("Power Estimate After Filtering by a Square Root Operator where\nScaling Constant is based off Local Maxima plus {}%".format(int(pc)))
    fs.savefig("pest-sqrtop-local-max-diff-tol-{}.png".format(int(pc)))

ax.legend()
f.suptitle("Power Estimate After Filtering by a Logarithm Operator where\nScaling Constant is based off Local Maxima plus a percentage")
f.savefig("pest-sqrtop-local-max-diff-tol.png")
