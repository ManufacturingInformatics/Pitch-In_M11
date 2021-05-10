import cv2

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__
from matplotlib.lines import Line2D

import numpy as np
import os,os.path
import xml.etree.ElementTree as ET
from math import log2,ceil

# count the number of images in the folder
# not going through nested directories
DIR_IMG = 'C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/img-rename'
# for each of the items in the given folder
num_of_frames = len([name for name in os.listdir(DIR_IMG)
                     # if the item is a file add it to the list
                     # return length of list, i.e. number of items in list
                     if os.path.isfile(os.path.join(DIR_IMG,name))])
print("Number of frames: ", num_of_frames)

# paths
p_img = 'C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/img-rename/image_ (%1d).bmp'
p_xml = 'C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/tree2.xml'

# parse xml file
tree = ET.parse(p_xml)
root  = tree.getroot()
log = tree.findall("traceData")

# data sets for position
pos1 = []
pos2 = []
pos3 = []


def cventropy(frame):
    # get size
    w,h=frame.shape[:2]
    # get area
    total_size = w*h
    #get histogram
    # list where index is bin and value is population
    hist=cv2.calcHist([frame],[0],None,[256],[0,256])
    # iterate through filtered list
    entropy=0
    for b in hist:
        if b>0:
            entropy+=(b/total_size)*(log2(b/total_size))
    return entropy*-1

# for each traceData in log
for f in log:
    # for the dataFrame in f
    for d in f:
        # get date stamp
        frame_header = d.findall("frameHeader")
        data_frame_date = frame_header[0].attrib.get('startTime').split('T')[0]

        # get all recordings in data frame
        rec = d.findall("rec")
        num_rec = len(rec)
        print("Found ",len(rec), " recordings in data frame")
        print("Last timestamp :", rec[-1].get("time"))
        print("Parsing data")
        # parse data and separate it into respective lists
        for ri,r in enumerate(rec):
            # get pos1 value
            f3 = r.get("f3")
            # if there is no torque value, then it hasn't changed
            if f3==None:
                # if there is a previous value, use it
                if ri>0:
                    pos1.append(float(pos1[ri-1]))
            else:
                pos1.append(float(f3))

            # get pos2 value
            f4 = r.get("f4")
            # if there is no torque value, then it hasn't changed
            if f4==None:
                # if there is a previous value, use it
                if ri>0:
                    pos2.append(float(pos2[ri-1]))
            else:
                pos2.append(float(f4))

            # get pos3 value
            f5 = r.get("f5")
            # if there is no torque value, then it hasn't changed
            if f5==None:
                # if there is a previous value, use it
                if ri>0:
                    pos3.append(float(pos3[ri-1]))
            else:
                pos3.append(float(f5))

# open image sequence
cap = cv2.VideoCapture(p_img)
# if failed to open, inform user
if not cap.isOpened():
    print("Failed to get frames")
else:
    # read first frame
    ret,frame = cap.read()

    ## initialize variables
    # counter for number of read frames
    counter=0

    # initialize idx list
    plume_frame_idx_list = []

    print("Processing frames")
    # loop through frames
    while ret:
        # scale frame to 512
        frame = cv2.resize(frame,(0,0),fx=4.0,fy=4.0)

        # get gray version of frame
        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #print(cventropy(gray))
        # calculate entropy for grayscale image
        # if entropy is above threshold, add frame idx to list
        if cventropy(gray)>=3.0:
            plume_frame_idx_list.append(counter)

        # increment counter
        counter +=1
        # read next frame, if sucessful the while loop iterates
        ret,frame = cap.read()

    # capture object
    cap.release()

    print("Finished processing ", counter+1, " : ", num_of_frames," frames")
    # print number of frames found
    print("Found ", len(plume_frame_idx_list), " frames with plumes in them")
    #print(plume_frame_idx_list)
    # factor to scale frame idx to rec list
    frame_rec_scale = ceil(num_rec/num_of_frames)
    print("Scale factor between frame idx and data idx: ", frame_rec_scale)
    
    # scale frame idx [0 num_of_frames] to rec idx [0 num_rec]
    plume_rec_idx_list = [i*frame_rec_scale for i in plume_frame_idx_list]
    # check that no idx exceeds the number of recordings
    # set to limit if they do
    plume_rec_idx_list = [num_rec-1 if i>(num_rec-1) else i for i in plume_rec_idx_list]
    
    #print(plume_rec_idx_list)
    # get points where the laser was turned on
    pos1_plume = list(np.array(pos1)[plume_rec_idx_list])
    pos2_plume = list(np.array(pos2)[plume_rec_idx_list])
    pos3_plume = list(np.array(pos3)[plume_rec_idx_list])

    print("Generating plots")
    # plot 3d position data
    fig3d = plt.figure()
    ax = fig3d.gca(projection='3d')
    ax.view_init(elev=19,azim=-15)
    #plot position data
    ax.plot(pos1,pos2,pos3,'-b',label='Laser OFF')
    # scatter plot indicating idxs where laser was turned on
    ax.scatter(pos1_plume,pos2_plume,pos3_plume,c='r',marker="^",s=40, label='Laser ON')
    ax.legend()

    # 3d position data, laser line plot 2
    fig3dl2 = plt.figure()
    ax3 = fig3dl2.gca(projection='3d')
    ax3.view_init(elev=19,azim=-15)
    # plot 3d position data
    ax3.plot(pos1,pos2,pos3,'-b')

    # for each idx in plume idx, draw a line using indexes to define range, skip 0
    for i in range(1,len(plume_rec_idx_list),1):
        # if the indexes are grouped together, diff <10 then draw a line between indexes
        if abs(plume_frame_idx_list[i-1] - plume_frame_idx_list[i])<10:
            ax3.plot(pos1[plume_rec_idx_list[i-1]:plume_rec_idx_list[i]],
            pos2[plume_rec_idx_list[i-1]:plume_rec_idx_list[i]],
            pos3[plume_rec_idx_list[i-1]:plume_rec_idx_list[i]],'-r')

    # define custom legend to avoid creating entries for each red line
    custom_lines = [Line2D([0],[0],color='b'),
                    Line2D([0],[0],color='r')]

    ax3.legend(custom_lines,['Laser OFF','Laser ON'])
    ax3.set_title('3D Position data & Estimated Laser Activation')

    # save figures before showing
    fig3d.savefig('position-laser-on-scatter-' + data_frame_date + '.png')
    fig3dl2.savefig('position-laser-on-line-' + data_frame_date + '.png')

    plt.show()