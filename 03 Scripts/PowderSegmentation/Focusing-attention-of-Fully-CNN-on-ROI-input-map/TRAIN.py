# Train fully convolutional neural net with valve filters and ROI map as input
# Instructions:
# a) Set folder of train images in Train_Image_Dir
# b) Set folder of ROI maps (for train images) in Label_Dir (
#    the ROI Maps should be saved as png image with same name as the corresponding image,
#    pixel of ROI should be 1 and all other pixels should be zero
# c) Set folder for ground truth labels in Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# d) Download pretrained vgg16 model and put in model_path (should be done autmatically if you have internet connection)
#    Vgg16 pretrained Model can be download from ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
#    or https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing
# e) Set number of classes number in NUM_CLASSES
# g) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir (assume that the labels and ROI maps for the validation image are also in ROIMap_Dir and Label_Dir)
# h) Run scripty
##########################################################################################################################################################################
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import Data_Reader
import BuildNetVgg16
import os
import CheckVGG16Model
#...........................................Input and output folders.................................................
Train_Image_Dir="Data_Zoo/Materials_In_Vessels/Train_Images/" # Images and labels for training
ROIMap_Dir="Data_Zoo/Materials_In_Vessels/VesselLabels/" # Folder where ROI map are save in png format (same name as coresponding image in images folder)
Label_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
UseValidationSet=False# do you want to use validation set in training
Valid_Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Validation images that will be used to evaluate training (the ROImap and Labels are in same folder as the training set)
logs_dir= "test/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
learning_rate=1e-5 #Learning rate for Adam Optimizer
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
#-----------------------------Other Paramters------------------------------------------------------------------------
TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Batch_Size=2 # Number of files per training iteration
Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = int(100010) # Max  number of training iteration
NUM_CLASSES = 4#Number of class for fine grain +number of class for solid liquid+Number of class for empty none empty +Number of class for vessel background
######################################Solver for model   training#####################################################################################################################
def train(loss_val, var_list):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
################################################################################################################################################################################
def main(argv=None):
    tf.compat.v1.reset_default_graph()
    keep_prob= tf.compat.v1.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    ROIMap= tf.compat.v1.placeholder(tf.int32, shape=[None, None, None, 1], name="ROIMap")  # ROI input map
    GTLabel = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")#Ground truth labels for training
  #.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    Net.build(image, ROIMap,NUM_CLASSES,keep_prob)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................
    Loss = tf.reduce_mean(input_tensor=(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(GTLabel, axis=[3]), logits=Net.Prob,name="Loss")))  # Define loss function for training
   #....................................Create solver for the net............................................................................................
    trainable_var = tf.compat.v1.trainable_variables() # Collect all trainable variables for the net
    train_op = train(Loss, trainable_var) #Create Train Operation for the net
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir, ROIMap_Dir, GTLabelDir=Label_Dir,BatchSize=Batch_Size) #Reader for training data
    if UseValidationSet:
        ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir, ROIMap_Dir, GTLabelDir=Label_Dir,BatchSize=Batch_Size) # Reader for validation data
    sess = tf.compat.v1.Session() #Start Tensorflow session
# -------------load trained model if exist-----------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer()) #Initialize variables
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------

    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tloss\t Learning Rate="+str(learning_rate))
    f.close()
    if UseValidationSet:
       f = open(ValidLossTxtFile, "w")
       f.write("Iteration\tloss\t Learning Rate=" + str(learning_rate))
       f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        Images, ROIMaps, GTLabels =TrainReader.ReadAndAugmentNextBatch() # Load  augmeted images and ground true labels for training
        feed_dict = {image: Images,GTLabel:GTLabels,ROIMap:ROIMaps, keep_prob: 0.5}
        sess.run(train_op, feed_dict=feed_dict) # Train one cycle
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 500 == 0 and itr>0:
            print("Saving Model to file in"+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) #Save model
#......................Write and display train loss..........................................................................
        if itr % 10==0:
            # Calculate train loss
            feed_dict = {image: Images, GTLabel: GTLabels, ROIMap: ROIMaps, keep_prob: 1}
            TLoss=sess.run(Loss, feed_dict=feed_dict)
            print("Step "+str(itr)+" Train Loss="+str(TLoss))
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(TLoss))
                f.close()
#......................Write and display Validation Set Loss by running loss on all validation images.....................................................................
        if UseValidationSet and itr % 2000 == 0:
            SumLoss=np.float64(0.0)
            NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
            print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
            for i in range(NBatches):# Go over all validation image
                Images, ROIMaps,GTLabels= ValidReader.ReadNextBatchClean() # load validation image and ground true labels
                feed_dict = {image: Images,ROIMap:ROIMaps, GTLabel: GTLabels ,keep_prob: 1.0}
                # Calculate loss for all labels set
                TLoss = sess.run(Loss, feed_dict=feed_dict)
                SumLoss+=TLoss
                NBatches+=1
            SumLoss/=NBatches
            print("Validation Loss: "+str(SumLoss))
            with open(ValidLossTxtFile, "a") as f:
                f.write("\n" + str(itr) + "\t" + str(SumLoss))
                f.close()


##################################################################################################################################################
main()#Run script
print("Finished")
