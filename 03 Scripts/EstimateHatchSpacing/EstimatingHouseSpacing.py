import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn
from scipy.signal import find_peaks

def readXMLPos(path):
    import xml.etree.ElementTree as ET
    """ Reads data from the given laser machine XML file

        =====ONLY READS FIRST DATA FRAME=====

        path : file path to xml file

        Returns time,torque,vel,x,y,z

        time,torque,vel,x,y,z are lists of data
    """
    # parse xml file to a tree structure
    tree = ET.parse(path)
    # get the root/ beginning of the tree
    root  = tree.getroot()
    # get all the trace data
    log = tree.findall("traceData")
    # get the number of data frames/sets of recordings
    log_length = len(log)

    # data sets to record
    x = []
    y = []
    z = []
    torque = []
    vel = []
    time = []
    
    ## read in log data
    # for each traceData in log
    for f in log:   
        # get all recordings in data frame
        rec = f[0].findall("rec")
        # parse data and separate it into respective lists
        for ri,r in enumerate(rec):
            # get timestamp value
            time.append(float(r.get("time")))
            
            # get torque value
            f1 = r.get("f1")
            # if there is no torque value, then it hasn't changed
            if f1==None:
                # if there is a previous value, use it
                if ri>0:
                    torque.append(float(torque[ri-1]))
            else:
                torque.append(float(f1))

            # get vel value
            f2 = r.get("f2")
            # if there is no torque value, then it hasn't changed
            if f2==None:
                # if there is a previous value, use it
                if ri>0:
                    vel.append(float(vel[ri-1]))
            else:
                vel.append(float(f2))

            # get pos1 value
            val = r.get("f3")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    x.append(float(x[ri-1]))
            else:
                x.append(float(val))

            # get pos2 value
            val = r.get("f4")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    y.append(float(y[ri-1]))
            else:
                y.append(float(val))

            # get pos3 value
            val = r.get("f5")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    z.append(float(z[ri-1]))
            else:
                z.append(float(val))

    return time,torque,vel,x,y,z

p='C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/tree2.xml'
print("reading in data")
_,_,_,x,y,z = readXMLPos(p)

# find element wise difference
x_dist = np.diff(np.array(x))
y_dist = np.diff(np.array(y))
z_dist = np.diff(np.array(z))

# find element wise difference of difference
# eq to d2
x_dist2 = np.diff(np.array(x),n=2)
y_dist2 = np.diff(np.array(y),n=2)
z_dist2 = np.diff(np.array(z),n=2)

# range of values to inspect
# for a more general soln or fn, means of detecting periods of activity are required
z_range = [11732,23000]

# scale default figure size by a factor of 2 so subplot labels don't overlap with each other
fig_size = [x**1.5 for x in plt.rcParams["figure.figsize"]]
# scale font to compensate for figure scaling
font_scale = 2.0
font_newsz = plt.rcParams['font.size']*font_scale

def incrFontSize(ax,f=1.5):
    ''' Increase the font size from default of labels within the specified axes by a factor

        ax : specified axis
        f  : factor to scale fonts by from default value

        Default font size pulled from pyplot.rcParams['font.size']

        Labels updated:
            - title
            - x-axis label
            - y-axis label
            - x-axis tick label
            - y-axis tick label
    '''
    import matplotlib.pyplot as plt
    # for each of the following items
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(plt.rcParams['font.size']*f)

# plot x,y,z data
f,(ax,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=fig_size)
f.suptitle("XML Position Data",fontsize=font_newsz)
ax.plot(x,'b-')
ax.set(xlabel="Frame Idx",ylabel="X (mm)")

ax2.plot(y,'g-')
ax2.set(xlabel="Frame Idx",ylabel="Y (mm)")
ax3.plot(z,'k-')
ax3.set(xlabel="Frame Idx",ylabel="Z (mm)")

#incrFontSize(ax,font_scale)
#incrFontSize(ax2,font_scale)
#incrFontSize(ax3,font_scale)

f.savefig("xml-data.png")

# plot distance between values
f,(ax,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=fig_size)
f.suptitle("1st difference between points",fontsize=font_newsz)
ax.plot(x_dist,'b-')
ax.set(xlabel="Frame Idx",ylabel="Difference,X (mm)")
ax2.plot(y_dist,'g-')
ax2.set(xlabel="Frame Idx",ylabel="Difference,Y (mm)")
ax3.plot(z_dist,'k-')
ax3.set(xlabel="Frame Idx",ylabel="Difference,Z (mm)")
ax4.plot(x,'b-')
ax4.set(xlabel="Frame Idx",ylabel="X (mm)",title="X Position of Motor")

#incrFontSize(ax,font_scale)
#incrFontSize(ax2,font_scale)
#incrFontSize(ax3,font_scale)
#incrFontSize(ax4,font_scale)

f.savefig("xml-diff-w-x.png")

# plot 2nd distance between values
f,(ax,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=fig_size)
f.suptitle("2nd difference between points",fontsize=font_newsz)
ax.plot(x_dist2,'b-')
ax.set(xlabel="Frame Idx",ylabel="Difference,X (mm)")
ax2.plot(y_dist2,'g-')
ax2.set(xlabel="Frame Idx",ylabel="Difference,Y (mm)")
ax3.plot(z_dist2,'k-')
ax3.set(xlabel="Frame Idx",ylabel="Difference,Z (mm)")
ax4.plot(x,'b-')
ax4.set(xlabel="Frame Idx",ylabel="X (mm)",title="X Position of Motor")

#incrFontSize(ax,font_scale)
#incrFontSize(ax2,font_scale)
#incrFontSize(ax3,font_scale)
#incrFontSize(ax4,font_scale)
f.savefig("xml-second-diff-w-x.png")

# plot region of interest
f,(ax,ax2) = plt.subplots(2,1,sharex=True)
f.suptitle("Zoomed in Region of Data and 1st Difference of Y")
ax.plot(x[z_range[0]:z_range[1]],'b-')
ax.set(xlabel="Frame Idx",ylabel="X (mm)")
ax2.plot(y_dist[z_range[0]:z_range[1]],'r-')
ax2.set(xlabel="Frame Idx",ylabel="Diff Y (mm)",ylim=[0.0,0.028])

f.savefig("x-y-diff-activity.png")

# zoomed in range of data
ydiff_z = y_dist[z_range[0]:z_range[1]]
# find peaks in region of data
# second element would have been a dictionary of features of the peak
yd_peaks = find_peaks(ydiff_z)[0]
# find most distinctive peaks, filtered by magnitude
yd_peaks = yd_peaks[ydiff_z[yd_peaks]>0.001]

# plot peaks of interest
f,(ax,ax2) = plt.subplots(2,1,sharex=True)
f.suptitle("Zoomed in Region of Data and 1st Difference of Y,[{:d},{:d}]".format(z_range[0],z_range[1]))
ax.plot(x[z_range[0]:z_range[1]],'b-')
ax.set(xlabel="Frame Idx",ylabel="X (mm)")
ax2.plot(ydiff_z,'r-')
ax2.plot(yd_peaks,ydiff_z[yd_peaks],'kx')
ax2.set(xlabel="Frame Idx",ylabel="Diff Y (mm)",ylim=[0.0,0.028])

f.savefig("ydiff-with-peaks.png")

# calculate average value of peaks aka. hatch spacing
h_space = np.mean(ydiff_z[yd_peaks])
# region to set y-lim to so the line indicating hatch spacing is clear
hs_range = [0.020,0.025]

f,ax = plt.subplots()
ax.plot(ydiff_z,'r-',label='Y. Diff.')
ax.plot(yd_peaks,ydiff_z[yd_peaks],'kx',label='Peaks')
# draw hatch spacing line
ax.plot(np.full(len(ydiff_z),h_space),'g--',label='Avg. Peaks')
# update plot labels including ylim
ax.set(xlabel='Frame Idx',ylabel='Y diff (mm)',title='Zoomed in region of Y difference, hatch spacing = {:.4f}mm'.format(h_space),ylim=hs_range)
f.savefig("ydiff-peaks-avg-line.png")

plt.show()
