import h5py
import numpy as np
import os
import cv2
from skimage.io import imread as sk_imread
from skimage.io import imsave as sk_imsave
from skimage.util import img_as_float
from skimage.color import rgb2gray
import subprocess as sp
import matplotlib.pyplot as plt
import datetime as dt
import csv

def dur2str(dur):
    """ Convert datetime duration to string of fixed format

        dur : datetime duration object

        Returns formatted string
        
        Converts the given duration object to the following format:
        {0} days, {1} hours, {2} mins, {3} secs
    """
    dur_s = dur.total_seconds()
    d = divmod(dur_s,86400)
    h = divmod(d[1],3600)
    m = divmod(h[1],60)
    s = divmod(m[1],1)
    return "{0} days, {1} hours, {2} mins, {3} secs".format(d[0],h[0],m[0],s[0])

## paths to resources
# path to target file
#path = "arrowshape-temperature-HDF5.hdf5"
#path = "pi-camera-data-192168146-2019-09-05T10-45-18.hdf5"
path = r"D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
# path to waifu command line utility executable
waifu_path = "D:\waifu2x-caffe\waifu2x-caffe\waifu2x-caffe-cui.exe"

# thermal camera limits, Raspberry Pi, deg C
temp_min = -30
temp_max = 300

# temperature limits of arrow set data, deg C
#temp_min = 23.0
#temp_max = 342.84793

# limit of scale factor to search for, 2,3,4 etc to limit
scale_lim = 10.0

# force recreate images flag
# if false, it checks whether there are images present in the reference folders and if so
# uses them instead
create_refs = False
rebuild_res = False
# get filename
filename = os.path.basename(path)
# get it without file extension
# used as the basis for foldernames
foldername = os.path.splitext(filename)[0]
# record start time of program
prog_start = dt.datetime.now()
# open program log file
with open("superlog-{}.txt".format(foldername),'w') as log:
    # define short cut for writing timestamped entries to file
    def logwrite(entry):
        log.write("[{0}] {1}\n".format(dt.datetime.now().time().isoformat(),entry))
    # writing start time to file
    logwrite("Starting program at {}".format(prog_start.isoformat()))
    # generate range of scale values to try
    # scale factor of 1.0 doesn't effect the image
    # scale factor is applied uniformally across height and width
    scale_vals = np.arange(2.0,scale_lim+1,1)
    # check if the data set path exists
    if os.path.isfile(path):
        logwrite("Confirmed {} dataset exists".format(path))
        print("Reading in data")
        logwrite("Reading in dataset")
        # read in data
        with h5py.File(path,'r') as f:
            dset = f[list(f.keys())[0]]
            logwrite("Finished reading in dataset {}".format(os.path.basename(path)))
            logwrite("Found {0} frames of type {1}".format(str(dset.shape),dset.dtype.name))
            # get size of dataset
            rows,cols,depth = dset.shape
            # different image data types to try and record the behaviour of
            img_depth = [np.dtype('uint8'),np.dtype('uint16'),np.dtype('float32'),np.dtype('float64')]
            # get depth type information
            depth_info = [np.iinfo(d) if 'int' in d.name else np.finfo(d) for d in img_depth]
            print("Encoding dataset frames for the different datatypes")
            logwrite("Encoding data for the different image datatypes")
            # list of folderpaths to saved images
            dset_imgd = []
            for imgd,di in zip(img_depth,depth_info):
                print("Using datatype ",imgd.name)
                logwrite("Starting with data type {}".format(imgd.name))
                # create folder as <filename>+-images-<target datatype>
                target_folder = os.path.join(foldername,"DatasetImages",imgd.name)
                # if creation flag is not set
                if not create_refs:
                    print("Checking for existing reference images...")
                    logwrite("Checking for existing reference images...")
                    # if the folder contains the same number of frames as the input dataset
                    num_files = len([name for name in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder,name))])
                    if num_files==dset.shape[2]:
                        # print and write to log
                        logwrite("{} already contains reference images".format(target_folder))
                        print("Skipping creation of reference images for {}".format(imgd.name))
                        # add folder to list of reference image locations
                        dset_imgd.append(target_folder)
                        continue
                    else:
                        print("Number of files does not match input. Recreating reference images for {}".format(imgd.name))
                        logwrite("Only found {0} images! Starting recreation of images for dtype {1}".format(num_files,imgd.name))
                # create folder name for generated images
                logwrite("Making folder for results : {}".format(target_folder))
                os.makedirs(target_folder,exist_ok=True)
                # iterate through frames
                logwrite("Converting temperature frames to images")
                for ff in range(dset.shape[2]):
                    #print(ff)
                    # normalize the data frame to [0,1]
                    norm_f = (dset[:,:,ff]-temp_min)/(temp_max-temp_min)
                    # remove nans
                    norm_f[np.isnan(norm_f)]=0.0
                    # convert to an n-bit image if an integer type
                    # if it's a float target type, then it is left as 0,1
                    # attempts to scale it to data type limits turns it to inf
                    if type(di)==np.iinfo:
                        # as the current range of 0-1 the denominator is 
                        norm_f = ((di.max-di.min)*norm_f + di.min).astype(imgd)
                    # the dnn acts on RGB images so we add duplicates of the image
                    # to replicate it
                    norm_f = np.dstack((norm_f,norm_f,norm_f))
                    
                    # if it's an integer datatype, use opencv to save the image
                    # skimage can do it but tends to print warnings
                    if type(di)==np.iinfo:
                        # create folder path for image
                        p = os.path.join(target_folder,"pi-frame-{0}.png".format(ff))
                        # if failed to write images inform user
                        if not cv2.imwrite(p,norm_f):
                            print("Failed to write image {} to folder!".format(p))
                    # if the target is a float datatype, use skimage as we have to save as tifs
                    elif type(di)==np.finfo:
                        # create folder path for image
                        p = os.path.join(target_folder,"pi-frame-{0}.tif".format(ff))
                        # save float image as tif
                        try:
                            sk_imsave(p,norm_f,check_contrast=False,plugin='pil')
                        except ValueError:
                            print(f"Failed to save frame {ff} to {p}!")
                            logwrite(f"Failed to save frame {ff} to {p}!")
                            
                dset_imgd.append(target_folder)
            # check if waifu2x cui still exists
            if os.path.isfile(waifu_path):
                logwrite("Waifu2x command line utility exists")
                # get models directory
                waifu_models = os.path.join(os.path.dirname(waifu_path),"models")
                # get the list of models supported
                _,wmodels,_ = next(os.walk(waifu_models))
                for model in wmodels:
                    logwrite("Starting run with model {}".format(model))
                    print("Starting run with model {}".format(model))
                    # iterate through different datatypes
                    logwrite("Starting iteration through image data types...")
                    for dsetp,imgd,di in zip(dset_imgd,img_depth,depth_info):
                        print("Starting {}".format(imgd.name))
                        logwrite("Starting {}".format(imgd.name))
                        # list of folders created
                        scale_dir = []
                        # iterate through each scaling factor
                        for scale in scale_vals:
                            # create folder for scaled output
                            target_folder = os.path.join(foldername,"ScaledResults",model,imgd.name,"scale-{}".format(int(scale)))
                            # if the flag to rebuild the results is not set
                            if not rebuild_res:
                                print(f"Checking number of files in {target_folder}")
                                logwrite(f"Checking number of files in {target_folder}")
                                if not os.path.isdir(target_folder):
                                    print("Target folder does not exist. Moving on with processing")
                                else:
                                    logwrite("Counting files")
                                    num_files = len([name for name in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder,name))])
                                    if num_files==dset.shape[2]:
                                        print("Number of output files matches number of input files.")
                                        continue
                                        
                            # create folder
                            os.makedirs(target_folder,exist_ok=True)
                            # add to list
                            scale_dir.append(target_folder)
                            # arguments for waifu2x
                            args = ["-i {}".format(dsetp), # source of images
                                    #"-l :png", # file format of images to search for
                                    # file extension of the output images, ensuring the output matches input
                                    "-e {}".format('png' if type(di)==np.iinfo else 'tif'),
                                    "-m scale", # type of operation
                                    '--model_dir "{}"'.format(os.path.join('models',model)),
                                    "-s {:.1f}".format(scale), # scale factor
                                    "-p gpu", # use gpu
                                    "--gpu 0", # use first gpu (idx 0)
                                    "-d {}".format(di.bits), # size of image depth
                                    # name of folder to hold results
                                    # automatically creates it
                                    "-o {}".format(target_folder)]
                            logwrite("Calling Waifu2x with the args:\n{}".format("    \n".join(args)))
                            # windows requires a single command string to be passed#
                            # linux can accept the path and set of arguments
                            cmd_str = "{0} {1}".format(waifu_path,' '.join(args))
                            print("Starting {:.2f}x scaling".format(scale))
                            # record start of model call
                            model_start = dt.datetime.now()
                            # start process, popen is non-blocking
                            with sp.Popen(cmd_str,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True) as waifu:
                                # print output of command
                                for ln in waifu.stdout.readlines():
                                    logwrite("waifu2x-cui output: {}".format(ln))
                            #logwrite("waifu2x-caffe-cui.exe output: "+str(sp.check_output(cmd_str,stderr=sp.STDOUT,universal_newlines=True)))
                            logwrite("Finished Waifu2x {0} {1}x run".format(imgd.name,int(scale)))
                            logwrite("Runtime {}".format(dur2str(dt.datetime.now()-model_start)))
                        ## read results back in and process results
                        # create matrix to hold results
                        max_var = np.zeros((scale_vals.shape[0],depth))
                        min_var = np.zeros((scale_vals.shape[0],depth))
                        diff_var = np.zeros((scale_vals.shape[0],depth))
                        diff_range = np.zeros((scale_vals.shape[0],depth))
                        ## calculate stats about original dataset before hand
                        # maximum values
                        dset_max = dset[()].max(axis=(0,1))
                        # minimum values
                        dset_min = dset[()].min(axis=(0,1))
                        # variance of each frame
                        dset_var = dset[()].var(axis=(0,1),dtype='float32')
                        # range of temperatures in each frame
                        dset_range = dset_max-dset_min
                        logwrite("Generating stats for {}".format(imgd.name))
                        # populate metrics matricies by iterating through each scale factor used
                        for ctr,scale in enumerate(scale_vals):
                            logwrite("Iterating through results for {} data type".format(imgd.name))
                            # check if it exists, just to be sure
                            if os.path.isdir(scale_dir[ctr]):
                                logwrite("Found directory for {} results".format(imgd.name))
                                filenames = next(os.walk(scale_dir[ctr]))[2]
                                # sort by frame order to guarantee they are read in the correct order
                                filenames.sort(key=lambda x: int(os.path.splitext(x)[0].split('-')[-1]))
                                print("Found {} files".format(len(filenames)))
                                # iterate through every image in the folder
                                for fct,ff in enumerate(filenames):
                                    # read in frame
                                    # flags are set to read it in as color and any depth
                                    if '.png' in ff:
                                        frame = cv2.imread(os.path.join(scale_dir[ctr],ff),cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                                        if frame is None:
                                            print(f"Error while handling {ff}! Failed to read in image! Image likely corrupt!")
                                            continue
                                    elif '.tif' in ff:
                                        try:
                                            frame = img_as_float(sk_imread(os.path.join(scale_dir[ctr],ff),plugin='pil'))
                                        except IOError as e:
                                            print("Error while handling {}! Image likely corrupted in some way".format(ff))
                                            logwrite("Error accessing {0} while using model {1} and dtype {2}!\n{3}".format(ff,model,imgd.name,str(e)))
                                            # force continue to the next loop
                                            # leave metrics for entry at 0
                                            continue
                                    ## normalize frame according to image type
                                    # if it's an integer, convert BGR to grayscale and scale result
                                    if type(di) == np.iinfo:
                                        frame = ((temp_max-temp_min)/(di.max-di.min))*cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) + temp_min
                                    # if it's a float, normalize image as is, already single channel image
                                    elif type(di) == np.finfo:
                                        frame = (temp_max-temp_min)*rgb2gray(frame) + temp_min
                                    # replace nans with 0 and inf with large numbers
                                    frame[np.isnan(frame)]=0
                                    frame[np.isinf(frame)]=0
                                    # collect metrics
                                    #print("dset max shape: ",dset_max.shape)
                                    #print("ctr: ",ctr)
                                    #print("fct: ",fct)
                                    #print("max var: ",max_var.shape)
                                    max_var[ctr,fct] = frame.max(axis=(0,1))-dset_max[fct]
                                    min_var[ctr,fct] = frame.min(axis=(0,1))-dset_min[fct]
                                    diff_var[ctr,fct] = frame.var(dtype='float32')-dset_var[fct]
                                    diff_range[ctr,fct] = (frame.max(axis=(0,1))-frame.min(axis=(0,1)))-diff_range[ctr,fct]
                            else:
                                print("{} is not a folder!".format(scale_dir[ctr]))
                                logwrite("ERROR: {} folder of scaled images not found!".format(scale_dir[ctr]))
                            ## plot data
                            print("Plotting data for {}".format(scale))
                            logwrite("Generating plots for {}".format(imgd))
                            # creating folder for results
                            target_folder = os.path.join(foldername,"Plots",model,imgd.name)
                            os.makedirs(target_folder,exist_ok=True)
                            # global data
                            f,ax = plt.subplots()
                            ax.plot(scale_vals,max_var,'rx')
                            ax.set(xlabel='Image Scale Factor',ylabel='Difference between Max of Scaled and Original (C)')
                            f.suptitle('Difference Between the Maximum Temperatures of the Original\n and the Scaled Images for Different Scale Factors ({})'.format(model))
                            f.savefig(os.path.join(target_folder,'diff-max-temp-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,min_var,'rx')
                            ax.set(xlabel='Image Scale Factor',ylabel='Difference between Min of Scaled and Original (C)')
                            f.suptitle('Difference Between the Minimum Temperatures of the Original\n and the Scaled Images for Different Scale Factors ({})'.format(model))
                            f.savefig(os.path.join(target_folder,'diff-min-temp-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,max_var,'rx')
                            ax.set(xlabel='Image Scale Factor',ylabel='Difference between Variance of Scaled and Original (C)')
                            f.suptitle('Difference Between the Temperatrure Variance of the Original\n and the Scaled Images for Different Scale Factors ({})'.format(model))
                            f.savefig(os.path.join(target_folder,'diff-var-temp-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,diff_range,'rx')
                            ax.set(xlabel='Image Scale Factor',ylabel='Difference between Temp. Ranges of Scaled and Original (C)')
                            f.suptitle('Difference Between the Temperatrure Ranges of the Original\n and the Scaled Images for Different Scale Factors ({})'.format(model))
                            f.savefig(os.path.join(target_folder,'diff-range-temp-all-scale-{}.png'.format(imgd.name)))
                            
                            # individual scale values
                            for ctr,scale in enumerate(scale_vals):
                                ax.clear()
                                ax.plot(max_var[ctr,:])
                                ax.set(xlabel='Frame Index',ylabel='Difference between Max Temperatures (C)')
                                f.suptitle('Difference Between the Maximum Temperatures of the Original\n and the Scaled Image,f={0},m={1}'.format(int(scale),model))
                                f.savefig(os.path.join(target_folder,'diff-max-temp-scale-{0}-{1}.png'.format(int(scale),imgd.name)))

                                ax.clear()
                                ax.plot(min_var[ctr,:])
                                ax.set(xlabel='Frame Index',ylabel='Difference between Min Temperatures (C)')
                                f.suptitle('Difference Between the Minimum Temperatures of the Original\n and the Scaled Image,f={0},m={1}'.format(int(scale),model))
                                f.savefig(os.path.join(target_folder,'diff-min-temp-scale-{0}-{1}.png'.format(int(scale),imgd.name)))

                                ax.clear()
                                ax.plot(diff_var[ctr,:])
                                ax.set(xlabel='Frame Index',ylabel='Difference between Temperature Variance of Scaled and Original (C)')
                                f.suptitle('Difference Between the Temperature Variances of the Original\n and the Scaled Image,f={0},m={1}'.format(int(scale),model))
                                f.savefig(os.path.join(target_folder,'diff-var-temp-scale-{0}-{1}.png'.format(int(scale),imgd.name)))

                                ax.clear()
                                ax.plot(diff_range[ctr,:])
                                ax.set(xlabel='Image Scale Factor',ylabel='Difference between Variance of Scaled and Original (C)')
                                f.suptitle('Difference Between the Temperatrure Ranges of the Original\n and the Scaled Image,f={0},m={1}'.format(int(scale),model))
                                f.savefig(os.path.join(target_folder,'diff-var-temp-scale-{0}-{1}.png'.format(int(scale),imgd.name)))

                            # plotting the variances in the values
                            var_diff_var = diff_var.var(axis=1)
                            var_diff_max = max_var.var(axis=1)
                            var_diff_min = min_var.var(axis=1)
                            var_diff_range = diff_range.var(axis=1)
                            
                            ax.clear()
                            ax.plot(scale_vals,var_diff_var)
                            ax.set(xlabel='Image Scale Factor',ylabel='Variance of the Temperature Variance (C)')
                            f.suptitle('Variance of the Temperature Variance for Different Image Scale Factors\n Model: {}'.format(model))
                            f.savefig(os.path.join(target_folder,'var-diff-var-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,var_diff_max)
                            ax.set(xlabel='Image Scale Factor',ylabel='Variance of the Difference Between Max Temp. (C)')
                            f.suptitle('Variance of the Difference Between Max. Temperatures \nfor Different Image Scale Factors, Model: {}'.format(model))
                            f.savefig(os.path.join(target_folder,'var-diff-max-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,var_diff_min)
                            ax.set(xlabel='Image Scale Factor',ylabel='Variance of the Difference Between Min Temp. (C)')
                            f.suptitle('Variance of the Difference Between Min. Tempperatures \nfor Different Image Scale Factor, Model: {}'.format(model))
                            f.savefig(os.path.join(target_folder,'var-diff-min-all-scale-{}.png'.format(imgd.name)))

                            ax.clear()
                            ax.plot(scale_vals,var_diff_range)
                            ax.set(xlabel='Image Scale Factor',ylabel='Variance of the Difference Between Temp. Ranges (C)')
                            f.suptitle('Variance of the Difference Between Temperature Ranges \nfor Different Image Scale Factors, Model: {}'.format(model))
                            f.savefig(os.path.join(target_folder,'var-diff-range-all-scale-{}.png'.format(imgd.name)))

                            ## the ideal scale factor is the one that causes minimum change in the respective values
                            best_str = ["Scale factor that causes min change in max vals {}\n".format(scale_vals[var_diff_max.argmin()]),
                            "Scale factor that causes min change in min vals {}\n".format(scale_vals[var_diff_min.argmin()]),
                            "Scale factor that causes min change in variance {}\n".format(scale_vals[var_diff_var.argmin()]),
                            "Scale factor that causes the min change in temperature ranges {}\n".format(scale_vals[var_diff_range.argmin()])]

                            with open(os.path.join(target_folder,"best-scale-{}.txt".format(imgd.name)),'w') as f:
                                f.writelines(best_str)

                            # save data to csv
                            # variance is not saved as it can be calculated from the data
                            np.savetxt(os.path.join(target_folder,'diff-max-{0}-{1}.csv'.format(model,imgd.name)),max_var,delimiter=',',newline='\n')
                            np.savetxt(os.path.join(target_folder,'diff-min-{0}-{1}.csv'.format(model,imgd.name)),min_var,delimiter=',',newline='\n')
                            np.savetxt(os.path.join(target_folder,'diff-var-{0}-{1}.csv'.format(model,imgd.name)),diff_var,delimiter=',',newline='\n')
                            np.savetxt(os.path.join(target_folder,'diff-range-{0}-{1}.csv'.format(model,imgd.name)),diff_range,delimiter=',',newline='\n')
                            
                            logwrite("Finished run for {}".format(imgd.name))
                            # close figures
                            plt.close(f)
            else:
                print("Waifu2x CUI does not exist")
                logwrite("ERROR: Waifu2x CUI does NOT exist!")
    else:
        print("{} is not a file!".format(path))
        logwrite("ERROR: Cannot find dataset at {}".format(path))

    # get total runtime
    prog_end = dt.datetime.now()
    dur = prog_end-prog_start
    print("Finished program")
    logwrite("Finished program run. Total runtime: {}".format(dur2str(dur)))
    
