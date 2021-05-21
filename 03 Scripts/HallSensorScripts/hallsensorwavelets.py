import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pywt
import scaleogram as scg
import os

def iter_wavelets(data):
    # make axes
    f,ax = plt.subplots()
    print("iterating over channels")
    ## iterate over the good channels
    for ff in range(1,5):
        print(f"channel {ff}")
        for w in pywt.wavelist():
            print(f"Trying {w}")
            try:
                ax.clear()
                scg.cws(data[:,0],data[:,ff]-data[:,ff].mean(),scales=np.arange(520,570),wavelet=w,ax=ax,cmap='jet',cbar=None,ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using {w.replace('.','-')}")
                f.savefig(f"hall-sensor-wavelets-c{ff}-w{w.replace('.','-')}.png")
            except AttributeError:
                continue

def find_nearest_idx(array, value):
        # ensures input is an array
        array = np.asarray(array)
        # get the location of the value that has the smallest
        # difference to the target value
        return (np.abs(array - value)).argmin()

def wavelets_narrow(data):
    # construct periods of activity
    dist = 512400/2    # 256,200 ms
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]
    # get the indicies
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]
    t = [xt0,xt1,xt2,xt3]
    # make folder for results
    os.makedirs("WaveletTracks",exist_ok=True)
    # make figure and axes
    f,ax = plt.subplots()
    # iterate over channels
    for ff in range(1,6):
        print(f"channel {ff}")
        # iterate over data limits
        for ti,(t0,t1) in enumerate(t):
            print(f"For {t0},{t1}")
            # get data
            time = data[t0:t1,0]
            y = data[t0:t1,ff]-data[t0:t1,ff].mean()
            try:
                ax.clear()
                scg.cws(time,y,scales=np.arange(520,570),wavelet="shan",ax=ax,cmap='jet',cbar=None,ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {t0} to {t1}, Track {ti}")
                f.savefig(os.path.join("WaveletTracks",f"hall-sensor-wavelets-c{ff}-shan-track-{ti}-{t0}-{t1}.png"))
            except AttributeError:
                continue

def wavelets_narrow_scale(data,scale):
    # construct periods of activity
    dist = 512400/2    # 256,200 ms
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]
    # get the indicies
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]
    t = [xt0,xt1,xt2,xt3]
    # make folder for results
    os.makedirs("WaveletTracksScale",exist_ok=True)
    # iterate over channels
    for ff in range(1,6):
        print(f"channel {ff}")
        # iterate over data limits
        for ti,(t0,t1) in enumerate(t):
            plt.close('all')
            print(f"For {t0},{t1}")
            # get data
            time = data[t0:t1,0]
            y = data[t0:t1,ff]-data[t0:t1,ff].mean()
            try:
                # make figure and axes
                f,ax = plt.subplots()
                scg.cws(time,y,scales=scale,wavelet="shan",ax=ax,cmap='hsv',ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {t0} to {t1}, Track {ti}")
                f.savefig(os.path.join("WaveletTracksScale",f"hall-sensor-wavelets-c{ff}-shan-track-{ti}-{t0}-{t1}-{scale.min()}-{scale.max()}.png"))
            except AttributeError:
                continue
            
def wavelets_narrow_cmap(data,cmap='jet'):
    # construct periods of activity
    dist = 512400/2    # 256,200 ms
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]
    # get the indicies
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]
    t = [xt0,xt1,xt2,xt3]
    # make folder for results
    os.makedirs("WaveletTracksColormap",exist_ok=True)
    # make figure and axes
    f,ax = plt.subplots()
    # iterate over channels
    for ff in range(1,6):
        print(f"channel {ff}")
        # iterate over data limits
        for ti,(t0,t1) in enumerate(t):
            print(f"For {t0},{t1}")
            # get data
            time = data[t0:t1,0]
            y = data[t0:t1,ff]-data[t0:t1,ff].mean()
            try:
                #ax.clear()
                scg.cws(time,y,scales=np.arange(520,570),wavelet="shan",cmap=cmap,ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {t0} to {t1}, Track {ti}")
                plt.gcf().savefig(os.path.join("WaveletTracksColormap",f"hall-sensor-wavelets-c{ff}-shan-track-{ti}-{t0}-{t1}-{cmap}.png"))
            except AttributeError:
                continue

def wavelets_period(data,p=8.6,c='k'):
    p *= 1000
    # construct periods of activity
    dist = 512400/2    # 256,200 ms
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]
    # get the indicies
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]
    t = [xt0,xt1,xt2,xt3]
    # make folder for results
    os.makedirs("WaveletTracksLine",exist_ok=True)
    # make figure and axes
    f,ax = plt.subplots()
    # iterate over channels
    for ff in range(1,6):
        print(f"channel {ff}")
        # iterate over data limits
        for ti,(t0,t1) in enumerate(t):
            print(f"For {t0},{t1}")
            # get data
            time = data[t0:t1,0]
            y = data[t0:t1,ff]-data[t0:t1,ff].mean()
            try:
                ax.clear()
                scg.cws(time,y,scales=np.arange(520,570),wavelet="shan",ax=ax,cmap='jet',cbar=None,ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {t0} to {t1}, Track {ti}")
                if ti!=0:
                    ## draw lines at each period
                    # get axes limits
                    xmin,xmax = ax.get_xlim()
                    ymin,ymax = ax.get_ylim()
                    print(xmax,xmin)
                    print(ymin,ymax)
                    # iterate over the time placing black lines
                    print(f"{(xmax-xmin)//p} black lines")
                    for x in range(1,int((xmax-xmin)//p)):
                        #print(f"time {xmin+(p*x)}")
                        l = mlines.Line2D([xmin+(p*x),xmin+(p*x)],[ymin,ymax],linewidth=2,color=c)
                        ax.add_line(l)
                f.savefig(os.path.join("WaveletTracksLine",f"hall-sensor-wavelets-c{ff}-shan-track-{ti}-{t0}-{t1}.png"))
            except AttributeError:
                continue

def wavelets_all_period(data,p=8.6,c='k'):
    p *= 1000
    # construct periods of activity
    dist = 512400/2    # 256,200 ms
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]
    # get the indicies
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]
    t = [xt0,xt1,xt2,xt3]
    # make folder for results
    os.makedirs("WaveletTracksPeriod",exist_ok=True)
    # make figure and axes
    f,ax = plt.subplots()
    # iterate over channels
    for ff in range(1,6):
        print(f"channel {ff}")
        # iterate over data limits
        for ti,(t0,t1) in enumerate(t):
            print(f"For {t0},{t1}")
            # get data
            time = data[t0:t1,0]
            y = data[t0:t1,ff]-data[t0:t1,ff].mean()
            try:
                ax.clear()
                scg.cws(time,y,scales=np.arange(520,570),wavelet="shan",ax=ax,cmap='jet',cbar=None,ylabel="Period [ms]",xlabel="Time [ms]",title=f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {t0} to {t1}, Track {ti}")
                if ti!=0:
                    ## draw lines at each period
                    # get axes limits
                    xmin,xmax = ax.get_xlim()
                    ymin,ymax = ax.get_ylim()
                    print(xmax,xmin)
                    print(ymin,ymax)
                    # iterate over the time placing black lines
                    print(f"{(xmax-xmin)//p} black lines")
                    for x in range(1,int((xmax-xmin)//p)):
                        #print(f"time {xmin+(p*x)}")
                        # update limits
                        ax.set_xlim(xmin+((p-1)*x),xmin+(p*x))
                        ax.set_title(f"Scaleogram for Hall Sensor Channel {ff} using Shannon\nfor period {xmin+((p-1)*x)} to {xmin+(p*x)}, Track {ti}")
                        # save plot
                        f.savefig(os.path.join("WaveletTracksPeriod",f"hall-sensor-wavelets-c{ff}-shan-track-{ti}-pass-{x}.png"))
                        ax.set_xlim(xmin,xmax)
            except AttributeError:
                continue
        
if __name__ == "__main__":
    print("Getting data")
    # generate the data from the CSV file
    fname = "h0000000.csv"
    data = np.genfromtxt("h0000000.csv",delimiter=',')
    
        
    
