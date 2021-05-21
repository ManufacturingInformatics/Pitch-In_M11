import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__

import matplotlib.animation as ani
import numpy as np

from decimal import getcontext, Decimal

p='C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/tree2.xml'
tree = ET.parse(p)
root  = tree.getroot()

print("Found ",len(root[0]), " attributes")
# print name of all elements in xml tree
#for elem in root:
    #for subelem in elem:
        #print(subelem.attrib)

# iterate through tracedata element
print("\n\nIterating through logged data")
log = tree.findall("traceData")
log_length = len(log)
print("Found ", log_length, " data frames")

# data sets to record
torque = []
vel = []
pos1 = []
pos2 = []
pos3 = []
time = []

# for each traceData in log
for f in log:
    # for the dataFrame in f
    for d in f:
        # get ad print header for data frame
        frame_header = d.findall("frameHeader")
        print("Data Frame Header")
        print(frame_header[0].attrib)

        # get number of signals logged
        signals = d.findall("dataSignal")
        print("Data frame recorded ", len(signals), " signals")
        # print descriptions
        print("Signals are: ")
        for s in signals:
            print("\t",s.get("description"))
            
        # get all recordings in data frame
        rec = d.findall("rec")
        print("Found ",len(rec), " recordings in data frame")
        print("Last timestamp :", rec[-1].get("time"))
        print("Parsing data")
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

        print("Got ", len(torque), " torque values")
        print("Got ", len(vel), " vel values")
        print("Got ", len(pos1), " pos1 values")
        print("Got ", len(pos2), " pos2 values")
        print("Got ", len(pos3), " pos3 values")
        print("Got ", len(time), " time values")

        # find max difference in z value and where it occurs
        maxdiff = 0
        maxdiff_idx1 = 0
        maxdiff_idx2 = 0
        diff = 0
        for i in range(1,len(pos3)):
            diff = abs(pos3[i]-pos3[i-1])
            if diff>maxdiff:
                maxdiff = diff
                maxdiff_idx1 = i-1
                maxdiff_idx2 = i

        print("Max Z diff: ",maxdiff," occurs between ", maxdiff_idx1, " and ", maxdiff_idx2)

        # convert z pos to integers
        z_int = np.array([int(z) for z in pos3])
        # get unique values
        z_u_int = np.unique(z_int)
        print("Z Unique values ", z_u_int)

        z_dec = np.array([round(z,2) for z in pos3])
        z_u_dec = np.unique(z_dec)
        print("Z Dec Unique values ", z_u_dec)

        ## Plotting data
        # create subplots
        f,( (ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2,3,figsize=(12,10),sharex=True)

        # plot torque data
        ax1.plot(torque,'-r')
        ax1.set_title(signals[0].get("description"))
        ax1.set_xlabel('Data Idx')
        ax1.set_ylabel('Torque')

        # plot vel data
        ax2.plot(vel,'-b')
        # plot avg line
        ax2.set_title(signals[1].get("description"))
        ax2.set_xlabel('Data Idx')
        ax2.set_ylabel('Velocity')

        # pos 1
        ax3.plot(pos1,'-g')
        ax3.set_title('X-Position')
        ax3.set_xlabel('Data Idx')
        ax3.set_ylabel('Pos')

        # pos2
        ax4.plot(pos2,'-m')
        ax4.set_title('Y-Position')
        ax4.set_xlabel('Data Idx')
        ax4.set_ylabel('Pos')

        # pos 3
        ax5.plot(pos3,'-k')
        ax5.set_title('Z-position')
        ax5.set_xlabel('Data Idx')
        ax5.set_ylabel('Pos')

        #set tight layout
        plt.tight_layout(pad=1.0,w_pad=0.5,h_pad=1.0)

        # save 2d data plot using date stamp from xml
        data_frame_date = frame_header[0].attrib.get('startTime').split('T')[0]
        #f.savefig(data_frame_date+"-motorControl.png")
        #print("Saved 2d data plot")

        # create plot for 3d positioning
        fig3d = plt.figure()
        ax = fig3d.gca(projection='3d')
        ax.plot(pos1,pos2,pos3,label='position of laser')
        #ax.set_xlabel(signals[2].get("description"))
        #ax.set_ylabel(signals[3].get("description"))
        #ax.set_zlabel(signals[4].get("description"))
        ax.legend()
        fig3d.suptitle('Laser Position')

        #plt.show()
        # rotation animation update fn
        def rotate(angle):
            ax.view_init(elev=35,azim=angle)

        # animation object
        #rotate_ani = ani.FuncAnimation(fig3d,rotate,360,interval=10,blit=False)

        #print("Saving rotating plot to gif")
        #rotate_ani.save('motor-3d-pos-rotate.gif',writer='imagemagick',fps=60)
        #print("Finished saving plot")

        ## XYZ 3D plot animation
        # setup figure
        print("Creating figure for 3D position animation")
        figlive = plt.figure()
        axlive = figlive.gca(projection='3d')
        # set viewing angle for viewing animation
        # 
        axlive.view_init(elev=35,azim=30)
        style = '-b'

        # create click release event handler for figure
        def onclick(event):
            azim,elev = axlive.azim, axlive.elev
            # print viewing angle and elevation
            print("azim: ",azim," elev: ",elev)

        # assign click event to figure
        cid = figlive.canvas.mpl_connect('button_release_event',onclick)

        # combine data into an array
        pos = np.row_stack((pos1,pos2,pos3))

        # create line3d object
        line = axlive.plot(pos[0,0:1],pos[1,0:1],pos[2,0:1],style)[0]
        #update function for animation
        skip = 10
        def animate(i,pos,line):
            # set x,y
            line.set_data(pos[0,:i],pos[1,:i])
            #set z
            line.set_3d_properties(pos[2,:i])
            return line

        line_ani = ani.FuncAnimation(figlive,animate,int(len(rec)/skip),fargs=(pos[:,::skip],line),interval=1,blit=False)
        
        #print("Writing animation to file")
        #Writer = ani.writers['ffmpeg']
        #mpfps = 200
        #writer = Writer(fps=mpfps,metadata=dict(artist='DBM'),bitrate=-1)
        #line_ani.save("motor-3d-plot-f" + str(mpfps)+".mp4",writer=writer)
        #print("Finished saving 3d move plot animation")
        #writer=None

        # display animation
        # requires previous windows to be closed before starting 3d plot
        plt.show()

        ## XYZ surface plots
        print("Creating 3D surface plot")
        fthree = plt.figure()
        surfskip = 1
        #figsurf = plt.figure()
        thresh = 6.18
        axsurf = fthree.add_subplot(1,1,1,projection='3d')
        axsurf.plot_trisurf(pos1[::surfskip],pos2[::surfskip],pos3[::surfskip],edgecolor='none',linewidth=0.2,antialiased=False)

        plt.show()

print("Cleaning up")
# close all plt windows
plt.close('all')