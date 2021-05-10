import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model
import numpy as np
import os
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
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

rmin_area = np.pi*r0_p**2.0
radii_range = np.arange(r0_p,64,1)
print("Normalizing values")
I_norm = I/I.max(axis=(0,1))
I_norm_cv = (I_norm*255).astype('uint8')

# ellipse stats
ellipse_cx = np.zeros(I.shape[2])
ellipse_cx = np.zeros(I.shape[2])
ellipse_a = np.zeros(I.shape[2])
ellipse_b = np.zeros(I.shape[2])
ellipse_theta = np.zeros(I.shape[2])

os.makedirs("CannySobel",exist_ok=True)
os.makedirs("CannySobel/EdgeResults",exist_ok=True)
os.makedirs("CannySobel/Contours",exist_ok=True)
os.makedirs("CannySobel/Ellipse",exist_ok=True)
print("Starting run")
for ff in range(I.shape[2]):
    sobel_res = np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5))+np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,0,1,ksize=5))
    canny = cv2.Canny(sobel_res,sobel_res.mean(),sobel_res.mean()*2)
    cv2.imwrite("CannySobel/EdgeResults/canny-sobel-f{}.png".format(ff),canny)
    # find contours
    ct,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # find ellipse
    minEllipse = []
    # if contours were found
    if ct:
        # if there is a contour
        #print("Found {} cts".format(len(ct)))
        if len(ct) >0:
            mask = np.zeros((*I.shape[:2],3),np.uint8)
            for c in ct:
                cv2.drawContours(mask,c,0,(255,0,0),1)
                cv2.imwrite("CannySobel/Contours/canny-sobel-contours-f{}.png".format(ff),mask)
                if c.shape[0]>5:
                    minEllipse.append(cv2.fitEllipse(c))
            #print("Found {} ellipses".format(len(minEllipse)))
            mask[...]=0
            for e in minEllipse:
                cv2.ellipse(mask,e,(0,255,0),2)
            cv2.imwrite("CannySobel/Ellipse/canny-sobel-ellipses-f{}.png".format(ff),mask)


            
