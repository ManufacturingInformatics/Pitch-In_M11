import os
import shutil
import glob
import matplotlib.pyplot as plt
from skimage.io import imread as sk_imread
from skimage.io import imsave as sk_imsave
from skimage.util import img_as_ubyte,img_as_float
import cv2

# location of scaled results
target = r"D:\BEAM\Scripts\CallWaifu2x\arrowshape-temperature-HDF5-denoise\DenoisedResults"
# create formatted version of target loc to be used in formatting of the new filename
target_fmt = target.replace('\\','-').replace(':','-')
# create local directory for results
os.makedirs('ExampleImages',exist_ok=True)
dest = 'ExampleImages'
# counter for how many frames were copied
ct = 0
# frame number to use as example
target_f = 1048
# paths of copied files
paths = []
print("Starting walk")
for root,dirs,files in os.walk(target):
    #print(root)
    # for each ff found
    for ff in files:
        #print(os.path.join(root,ff))
        # check if the ff matches our target
        if glob.fnmatch.fnmatch(ff,'*-{}.png'.format(target_f)) or glob.fnmatch.fnmatch(ff,'*-{}.tif'.format(target_f)):
            ct +=1
            # create new variable name based on the old one
            #  - replace slashes with dashes
            #  - replace colon with dash
            #  - delete target_fmt from the file path
            #  - replace underscores with dashes
            pathnew_fmt=os.path.join(root,ff).replace('\\','-').replace(':','-').replace(target_fmt,'')[1:]
            pathnew_fmt = pathnew_fmt.replace('_','-')
            # copy and rename ff
            shutil.copyfile(os.path.join(root,ff),os.path.join(dest,pathnew_fmt))
            paths.append(os.path.join(dest,pathnew_fmt))

print("Found and copied {} files!".format(ct))
print("Generating contours for scaled images")
os.makedirs('Contours',exist_ok=True)
os.makedirs('ContoursBase',exist_ok=True)
f,ax = plt.subplots()
for ff in paths:
# get ff extension and name
    # get ff extension and name
    ffbase = os.path.basename(ff)
    name,ext = os.path.splitext(ffbase)
    if '.png' in ff:
        frame = cv2.imread(ff,cv2.IMREAD_ANYDEPTH)
        if frame.shape[0]==0:
            print("Cannot read in ",ff)
            continue
    elif '.tif' in ff:
        try:
            frame = sk_imread(ff,as_gray=True,plugin='pil')
        except:
            print("Cannot read in ",ff)
            continue
    ax.clear()
    #print(ff)
    ax.imshow(frame,cmap='jet')
    f.savefig(os.path.join('Contours',name+'-contours.png'))

## copy reference images
target_orig = "D:\BEAM\Scripts\CallWaifu2x\OriginalSaves"
# get parent directory
prnt = target[:target.rfind('\\')]
# form path to dataset images
prnt = os.path.join(prnt,"DatasetImages")
print("Copying reference images from ",prnt)
# collect and copy images
for root,dirs,files in os.walk(prnt):
    for ff in files:
        if glob.fnmatch.fnmatch(ff,'*-{}.png'.format(target_f)) or glob.fnmatch.fnmatch(ff,'*-{}.tif'.format(target_f)):
                # create variable name based on 
                local_prnt = root[root.rfind('\\')+1:]
                # get name and ext
                name,ext = os.path.splitext(ff)
                pathnew_fmt = name+"-"+local_prnt+ext
                # copy and rename ff
                shutil.copyfile(os.path.join(root,ff),os.path.join(target_orig,pathnew_fmt))

print("Generating contours for base images")
for root,dirs,files in os.walk(target_orig):
    print(dirs)
    for ff in files:
        # get ff extension and name
        ffbase = os.path.basename(ff)
        name,ext = os.path.splitext(ffbase)
        if '.png' in ff:
            frame = cv2.imread(os.path.join(root,ff),cv2.IMREAD_ANYDEPTH)
            if frame.shape[0]==0:
                print("Cannot read in ",ff)
                continue
        elif '.tif' in ff:
            try:
                frame = sk_imread(os.path.join(root,ff),as_gray=True,plugin='pil')
            except:
                print("Cannot read in ",ff)
                continue
        ax.clear()
        ax.imshow(frame,cmap='jet')
        f.savefig(os.path.join('ContoursBase',name+'-contours.png'))
