import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import BuildNetVgg16_v2
import TensorflowUtils
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import CheckVGG16Model
import cv2
from PIL import Image

Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Test image folder
w=0.7# weight of overlay on image
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 4 # Number of classes
logs_dict = {4 : "logs/",
             2 : "logs_2c/"}
logs_dir = logs_dict[NUM_CLASSES]
#-------------------------------------------------------------------------------------------------------------------------
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

# convert image into a 1-image dataset batch
def processImg(img):
    Sy,Sx = img.shape[:2]
    Images = np.zeros([1,Sy,Sx,3],dtype=np.float)
    Img = np.array(Image.fromarray(img).resize((Sx,Sy)))
    Images[0] = Img
    return Images

# color scheme used for labelling
TR = [0,1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5]
TB = [0,0, 1, 0,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  0.5]
TG = [0,0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5]
# text for each label
labelDict = {2:["Background","Vessel"],
             3:["Background","Empty Vessel","Filled Vessel"],
             4:["Background","Empty Vessel","Liquid","Solid"],
             15:["Background","Empty Vessel","Liquid","Liquid p2","Suspension","Emulsion","Foam","Solid","Gel","Powder","Granular","Solid Bulk","Solid Liquid Mixture","Solid Phase 2","Vapor"]}
             
# function for building the label key explaining what each of the colors does
# based off the color table above
# user sets size of it and the max number of labels
def generateLabelKey(labelShape,keySize=None,numLabels=NUM_CLASSES):
    offset = 5
    # create key image
    key = np.zeros((labelShape[0],labelShape[1],3),np.uint8)
    # if the size of each key is not given
    # estimate from number of labels and offset
    if keySize is None:
        keySize = labelShape[0]//(numLabels) - (numLabels*offset)
    # get key dictionary
    keyDict = labelDict[numLabels]
    # iterating over labels generating and labelling colors
    for i in range(1,numLabels):
        col = (0,)*3
        ## build color
        # load color from table if the index is supported
        if i<len(TR):
            col = (TR[i]*255,TG[i]*255,TB[i]*255)
        else:
            col = (np.mod(i*i+4*i+5,255),np.mod(i*10,255),np.mod(i*i*i+7*i*i+3*i+30,255))
        # draw color on image
        key = cv2.rectangle(key,(offset,offset+((i-1)*keySize)),(offset+keySize,offset+(i*keySize)),col,-1)
        # draw text next to it
        cv2.putText(key,keyDict[i],(offset+keySize,offset+(i*keySize)-(keySize//2)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    return key

# run the trained model passing each image from the camera as the color input
# and a dummy ROI as the ROI map
def useCamera():
    print("Getting camera")
    # get camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Failed to open camera")
        cam.release()
        return
    # get test frame
    ret,frame = cam.read()
    if not ret:
        print("Failed to get test frame")
        cam.release()
        return
    ## create dummy region of interest for the purposes of testing
    # ROI is a box based around the centre of the image
    # this variable controls what proportion of the image is a ROI
    p = 0.6
    # get frame size
    sz = frame.shape
    # frame
    roi = np.zeros(sz,np.int32)
    # calculate rectangle size
    h,w,_ = [int(p*ss) for ss in sz]
    # get centre of the image
    c = sz[1]//2,sz[0]//2
    # draw filled rectangle
    # roi is marked by the ones
    roi = cv2.rectangle(roi,(c[0]-w//2,c[1]-h//2),(c[0]+w//2,c[1]+h//2),(1,1,1),-1)
    # as all channels are the same
    # just get the first channel
    roi = roi[:,:,0]
    # convert to batch
    ROIBatch = np.zeros([1,sz[0],sz[1],1],dtype=np.int)
    roi = np.array(Image.fromarray(roi).resize((sz[1],sz[0])))
    ROIBatch[0,:,:,0]=roi
    print(f"roi shape, {ROIBatch.shape}")

    # generate label key
    #annoKey = generateLabelKey(sz)
    
    # setup placeholders
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    # data is fed in as batches
    # data size is set as batch size x height x width x channels
    # e.g. 1 x 480 x 640 x 3
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    ROIMap = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="ROIMap")  # ROI input map
    # build encoder ?
    print("building encoder")
    try:
        Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
        Net.build(image, ROIMap, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)
    except:
        cam.release()
        
    # start tensorflow session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # load model from last checkpoint
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        # if train model exists, restore it
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
            cam.release()
            sys.exit()

        print("Starting...")
        while True:
            # get frame from camera
            ret,frame = cam.read()
            if not ret:
                print("Failed to get test frame")
                cam.release()
                return
            # process image
            # channels are reversed to be RGB order
            imgBatch = processImg(frame[:,:,:])
            # feed data into model
            # get output
            pred = sess.run(Net.Pred,feed_dict={image:imgBatch,keep_prob:1.0,ROIMap:ROIBatch})
            # show results
            cv2.imshow("Frame",frame)
            # overlay the labelled output on top of the input frame to demonstrate labels
            cv2.imshow("Overlay",Overlay.OverLayLabelOnImage(frame,pred[0], w))
            cv2.imshow("Labels",cv2.normalize(pred[0].astype("uint8"),pred[0].astype("uint8"),0,255,cv2.NORM_MINMAX))
            #cv2.imshow("Key",annoKey)
            # press ESC to exit
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

    cam.release()
    return pred

if __name__ == "__main__":
    pred = useCamera()
    cv2.destroyAllWindows()
        
