import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model
import os
import cv2
import numpy as np
from scipy.signal import find_peaks

T0 = mat.CelciusToK(23.0)
e,K,D = mat.buildMaterialData()
print("Reading in power")
Qr = model.readH264("D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264")
T = model.predictTemperature(Qr,e,T0)
np.nan_to_num(T,copy=False)
I = model.predictPowerDensity(T,K,D,T0)
np.nan_to_num(I,copy=False)

del T
del Qr
del e
del K
del D

# convert frames to 8-bit
I_norm_cv = ((I/I.max(axis=(0,1,2)))*255)
# find the dullest frames
peaks,_ = find_peaks(I.max(axis=(0,1)))
tol = 0.05
I_abs_max = I.max(axis=(0,1,2))
# searching for peak values within the tolerancce of the power density max
pki = peaks[np.where(I[peaks]<=(1-tol)*I_abs_max)]

print("Found {} dull frames".format(pki.shape[0]))

os.makedirs("LogHistDull",exist_ok=True)
f,ax = plt.subplots()
print("Starting his run of dull frames")
for ff in pki:
    print("\r{}".format(ff),end='')
    # interpret as 16-bit so the values can be incremented
    c = 255/np.log10(I_norm_cv[:,:,ff].max((0,1))+1)
    cvlog = c*np.log10(I_norm_cv[:,:,ff]+1)
    pop,edges = np.histogram(I_norm_cv[:,:,ff],bins=5)
    ax.clear()
    ax.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
    ax.set_xlim(edges.min(),edges.max())
    ax.set_title("Histogram of Log Scaled Values, f={}".format(ff))
    f.savefig("LogHistDull/loghist-dull-f{}.png".format(ff))
