# Run prediction and genertae pixelwise annotation for every pixels in the image based on image and ROI mask
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set folder of ROI maps (for train images) in Label_Dir
#    the ROI Maps should be saved as png image with same name pixel of ROI should be 1 and all other pixels should be zero
# 4) Set number of classes number in NUM_CLASSES
# 5) Set Pred_Dir the folder where you want the output annotated images to be save
# 6) Run script
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import sys
import BuildNetVgg16
import TensorflowUtils
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import CheckVGG16Model
from PIL import Image
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
ROIMap_Dir="Data_Zoo/Materials_In_Vessels/VesselLabels/" # Folder where ROI map are save in png format (same name as coresponding image in images folder)
Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Test image folder
w=0.7# weight of overlay on image
Pred_Dir="Output_Prediction/" # Library where the output prediction will be written
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 4 # Number of classes
#-------------------------------------------------------------------------------------------------------------------------
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

################################################################################################################################################################################
def main(argv=None):
      # .........................Placeholders for input image and labels........................................................................
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    ROIMap = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None, 1], name="ROIMap")  # ROI input map
    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image, ROIMap, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)
    # -------------------------Data reader for validation/testing images-----------------------------------------------------------------------------------------------------------------------------
    ValidReader = Data_Reader.Data_Reader(Image_Dir, ROIMap_Dir, BatchSize=1)
    #-------------------------Load Trained model if you dont have trained model see: Train.py-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.compat.v1.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.compat.v1.train.Saver()

    sess.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()

#--------------------Create output directories for predicted label, one folder for each granulairy of label prediciton---------------------------------------------------------------------------------------------------------------------------------------------

    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")

    
    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)
 #----------------------Go over all images and predict semantic segmentation in various of classes-------------------------------------------------------------
    fim = 0
    print("Start Predicting " + str(ValidReader.NumFiles) + " images")
    while (ValidReader.itr < ValidReader.NumFiles):
        print(str(fim * 100.0 / ValidReader.NumFiles) + "%")
        fim += 1
        # ..................................Load image.......................................................................................
        FileName=ValidReader.OrderedFiles[ValidReader.itr] #Get input image name
        Images, ROIMaps = ValidReader.ReadNextBatchClean()  # load testing image and ROI_Map

        # Predict annotation using net
        LabelPred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0, ROIMap: ROIMaps})
        #------------------------Save predicted labels overlay on images---------------------------------------------------------------------------------------------
        ov = Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)
        # convert to pillow image
        Image.fromarray(ov.astype("uint8")).save(Pred_Dir + "/OverLay/"+ FileName+NameEnd)
        Image.fromarray(LabelPred[0].astype(np.uint8)).save(Pred_Dir + "/Label/" + FileName[:-4]+".png" + NameEnd)
        ##################################################################################################################################################
main()#Run script
print("Finished")
