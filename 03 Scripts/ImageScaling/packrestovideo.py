import cv2
import os
import fnmatch
from skimage.io import imread as sk_imread
from skimage import img_as_ubyte

path = "D:\BEAM\Scripts\CallWaifu2x\pi-camera-data-192168146-2019-09-05T10-45-18-Run1\ScaledResults"

font = cv2.FONT_HERSHEY_SIMPLEX
font_col = (255,0,0)
size_target = 0.1
line_thick = 2
line_type=2

fnames = []
vw = None
for root,_,filenames in os.walk(path):
    print("Trying", root)
    #create filename for video
    foname = '-'.join(root.split('\\')[-3:]).replace('_','-')+'.avi'
    fn=1 # counter for number of files
    # sort files for correct frame order
    if len(filenames)>0:
        filenames.sort(key=lambda x : int(os.path.splitext(x)[0].split('-')[-1]))
    for ff in filenames:
        if fnmatch.fnmatch(ff,'pi-frame-*'):
            if 'png' in os.path.splitext(ff)[1]:
                frame = cv2.imread(os.path.join(root,ff),cv2.IMREAD_ANYCOLOR)
            elif 'tif' in os.path.splitext(ff)[1]:
                try:
                    frame = img_as_ubyte(sk_imread(os.path.join(root,ff),as_gray=False,plugin='pil'))
                except:
                    #print("Failed to read in frame",ff)
                    continue
            if frame.shape[0]!=0:
                if vw is None:
                    vw = cv2.VideoWriter(foname,cv2.VideoWriter_fourcc('M','J','P','G'),32,(frame.shape[0],frame.shape[1]),isColor=True)
                    if vw.isOpened():
                        print("Video writer opened",foname)
                    else:
                        print("Failed to open video writer!")
                        break
                    
                sz = cv2.getTextSize(str(ff),font,1,line_thick)[0]
                scale_f = (frame.shape[0]*size_target/sz[0])
                sz = cv2.getTextSize(str(ff),font,scale_f,line_thick)[0]
                cv2.putText(frame,str(ff),(sz[0],0),font,scale_f,font_col,line_thick,line_type)
                # check if writer is open, if so write frame
                if vw.isOpened():
                    vw.write(frame)
                    print("Failed to write frame ",ff)
                    fn+=1
            else:
                print("Failed to read in frame",ff)

    if len(filenames)>0:                    
        if vw is not None:
            if vw.isOpened():
                vw.release()
                print("Successfully written {} to {}".format(fn,foname))
                fnames.append([foname,fn])

print("Frames written")
for ff,fct in fnames:
    print("Name: {}, Frames: {}".format(ff,fct))
# ensure resources for writer have been released
vw.release()
