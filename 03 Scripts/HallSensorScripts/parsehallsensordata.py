import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as anim
from scipy.signal import find_peaks, peak_prominences

def plot_all(data):
    # generate basic plots of the different channels and global data
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    for i in range(1,data.shape[1]):
        ax.plot(data[:,0],data[:,i],label=str(i))
        axc.clear()
        axc.plot(data[:,0],data[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title=f'Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-channel-{i-1}.png")
        
    # display legend
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title='Data Recorded Using Hall Sensors During Three Track Run')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-all-channels.png")

    # limited channel set
    f,ax = plt.subplots()
    for i in range(1,data.shape[1]-2):
        ax.plot(data[:,0],data[:,i],label=str(i))

    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title=f'Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-limited-channels.png")

def plot_tracks(data,t):
    # setup plots
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    # create list to hold the matrix of each track
    trackdata = []
    # iterate over the 
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,data.shape[1]-2):
            ax.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],i],label=str(i))
            axc.clear()
            axc.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],i],label=str(i))
            axc.set(xlabel='Time (s)',ylabel='Recorded Voltage x100',title=f"Data Recorded for Track {ti} by Hall Sensors \n During Three Track Run, Channel {i-1}")
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-track-{ti}-channel-{i-1}.png")
        trackdata.append(data[tt[0]:tt[1],1:-2])
        #print(trackdata[-1].shape)
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Recorded Voltage x100',title=f"Data Recorded for Track {ti} by Hall Sensors \n During Three Track Run")
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-track-{ti}-limited-channels.png")
    return trackdata

def plot_contourf(trackdata):
    f,ax = plt.subplots()
    ## contour plot
    # plot data as contour to show history across the channels for the specific track
    for ti,td in enumerate(trackdata):
        f,ax = plt.subplots()
        im = ax.contourf(td.T)
        f.colorbar(im)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set(xlabel='Time ',ylabel='Channel',title=f'Contour of the Hall Sensor Channel History of Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-track-{ti}-contour-limited-channels.png")
        plt.close(f)

def sub_avg_plot(data):
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    ## subtract the average from the data
    # make a copy of the data to manipulate
    dataavg = np.array(data)
    # find the average of the non-zero start values
    avg = np.mean(dataavg[0,1:-2])
    # subtract it
    dataavg[:,1:-2]-=avg
    # plot the channel data
    f,ax = plt.subplots()
    for i in range(1,dataavg.shape[1]-2):
        ax.plot(dataavg[:,0],dataavg[:,i],label=str(i))
        axc.clear()
        axc.plot(dataavg[:,0],dataavg[:,i],label=str(i))
        ax.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title=f'Data Recorded Using Hall Sensors During Three Track Run\n Minus the Average, Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-channel-{i-1}.png")

    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title=f'Data Recorded Using Hall Sensors During Three Track Run\n Minus the Average, Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-avg-limited-channels.png")

def est_channel_offset(trackdata):
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    ## estimate offset between channels
    # array to store all the offset data
    # if will have 3 element each containing the channel offset for the specifc track
    channel_offset = []
    for ti,tt in enumerate(trackdata[1:],start=1):
        # add an array to hold the track data
        channel_offset.append([])
        #print("track: ",ti)
        for i in range(tt.shape[1]):
            #print("channel: ",i)
            # search for peaks in channel i, track ti
            pks = find_peaks(tt[:,i])[0]
            # just get the peaks that are above the second channel min
            # means we just get the interesting peaks
            pks = np.intersect1d(pks,np.where(tt[:,i]>np.unique(np.sort(tt[:,i]))[1]))
            axc.clear()
            # plot the channel data as is
            axc.plot(tt[:,i],label=str(i))
            axc.plot(pks,tt[pks,i],'ro',label='Peaks')
            axc.set(xlabel='Time (ms)',ylabel='Recorded Voltage x100',title=f"Data Recorded for Track {ti} by Hall Sensors \n During Three Track Run, Channel {i}")
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-marked-peaks-track-{ti}-channel-{i}.png")
            # save peaks data, for the 
            channel_offset[-1].append(pks)

    # create array for hold offset data for each track
    # not sure if it works??
    offset_data = []
    # iterate over peaks data for each track 
    for ti,track in enumerate(channel_offset):
        # as the peaks are different length
        # we're using the shortest set as the number of indicies to travel over
        sz = min([len(c) for c in track])
        # combine them together into a 2d array of no. channels x sz
        pks = np.vstack([c[:sz] for c in track])
        # find the difference in where the peak values occur across the different channels
        # so the first element is the difference between channel 1 and 2, then 2 and 3 etc.
        ax.clear()
        # find the row wise difference between the peaks data for track ti
        # row[i] = channel peaks[i+1] - channel peaks[i]
        peakdiff = np.diff(pks,axis=0)
        # iterate over each row and add to plot
        # set time data as the time period for the 
        for i in range(peakdiff.shape[0]):
            #print(i)
            ax.plot(peakdiff[i,:],label=f'Diff. Channel {i+1} & {i}')
        f.legend()
        ax.set(xlabel='Time (ms)',ylabel='Difference in Peak Positions (ms)',title=f"Estimated Offset between Peaks in Channel Hall Sensor Data for Track {ti}")
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-offsetdata-track-{ti}.png")

def anim_line(data,trackdata,t,targettime=25.0):
    ## timelapses
    # timelapse of the magnetic field across the different channels, line plot
    # clear plot
    f,ax = plt.subplots()
    # get the max change in the data
    maxdiff = np.diff(data[:,1:-2],axis=1).max()
    # set limits as the animation doesn't seem to rescale l
    ax.set_ylim(bottom=data[:,1:-2].min()-maxdiff,top=data[:,1:-2].max()+maxdiff)
    # set the labels
    ax.set(xlabel='Channel',ylabel='Recorded Voltage x100')
    # set x axis to display as integers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # create line plot to update
    # set the default y data as just zeros
    ll, = ax.plot(np.arange(0,trackdata[0].shape[1]),np.full(trackdata[0].shape[1],500.0),'b-')

    # title variables
    title = 'Timelapse of Recorded Hall Sensor Data Across Channels\n'
    track_title = ''
    starttidx = 0

    # time variables for title
    startt = 0

    # function for updating the line plot
    def lineTimelapse(idx,d):
        # update the y data of the line
        # track data is arranged as channel x time
        # so the index gets the time slice
        ll.set_ydata(d[:,idx])
        # update title
        # time portion shows local and global time
        ax.set_title(title + track_title + f" ,Time l:{(data[starttidx+idx,0]-startt)/1000.0} secs, g:{data[starttidx+idx,0]/1000.0} secs ")

    # iterate over each track
    for ti,track in enumerate(trackdata):
        #print(f"Generating animation for track {ti}")
        # update time variable used in title
        starttidx = t[ti][0]
        startt = data[t[ti][0],0]
        # update track number used in title
        track_title = f"Track {ti}"
        # transpose matrix to channels x time shape
        track = track.T
        # construct animation object
        # pass data
        linetl = anim.FuncAnimation(f,lineTimelapse,frames=track.shape[1]-1,fargs=(track,),interval=20,blit=False)
        # save animation
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=track.shape[1]/targettime,metadata=dict(artist='DBM'),bitrate=1800)
        linetl.save(f"{os.path.splitext(os.path.basename(fname))[0]}-line-timelapse-track-{ti}.mp4",writer=writer)

def plot_offset_normed(data,t):
    channelpks = []
    # set labels
    for ti,tt in enumerate(t):
        f,ax = plt.subplots()
        fc,axc = plt.subplots()
        # data is reset in case the track time portions overlaps
        # save a copy of the data to norm
        datanorm = np.array(data)
        # remove last two columns
        # known to be nans and all zeros
        datanorm = datanorm[:,:-2]
        # found peaks for each channel
        channelpks.clear()
        # iterate over each channel
        for i in range(1,datanorm.shape[1]):
            # get local min and max
            datamin = datanorm[tt[0]:tt[1],i].min()
            datamax = datanorm[tt[0]:tt[1],i].max()
            # normalize channel i
            datanorm[tt[0]:tt[1],i] = (datanorm[tt[0]:tt[1],i]-datamin)/(abs(datamax-datamin))
            # search for peaks, return indicies
            pki = find_peaks(datanorm[tt[0]:tt[1],i])[0]
            # plot each channel
            axc.clear()
            # plot track data
            axc.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],i])
            # plot found peaks
            axc.plot(data[tt[0]:tt[1],0][pki]/1000.0,data[tt[0]:tt[1],i][pki],'ro')
            axc.set(xlabel="Time (s)",ylabel="Measured Voltage x100",title=f"Locally Normalized Hall Sensor Data, Marked Peaks\n Track {ti}, Channel {i}")
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-locally-normed-marked-peaks-track-{ti}-channel-{i}.png")

            ## add to global plot
            # add data
            ax.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],i],label=f"Channel {i}")
            # add marked peaks
            ax.plot(data[tt[0]:tt[1],0][pki]/1000.0,data[tt[0]:tt[1],i][pki],'ro',label="Found Peaks")
            # add to list
            channelpks.append(pki)
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Measured Voltage x100',title=f"Locall Normalized Hall Sensor Data, Marked Peaks\n Track {ti}")
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-locally-normed-marked-peaks-track-{ti}-limited-channels.png")

        plt.close(f)
        plt.close(fc)
        # as each array of peaks is likely different sizes
        # trim them each to the shortest subset
        minsz = min([len(c) for c in channelpks])
        for ci in range(len(channelpks)):
            channelpks[ci] = channelpks[ci][:minsz]

        f,ax = plt.subplots()
        fc,axc = plt.subplots()
        # convert channel peak index data to a 2D array
        # channels x number of peaks
        channelpksarr = np.array(channelpks)
        # iterate over each channel's trimmed peak data
        # zip is so that we get pairs, channel c[0],c[1] then c[1],c[2] etc.
        for ci,(c0,c1) in enumerate(zip(channelpks,channelpks[1:])):
            # add the two channels
            ax.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],ci+1],label=f"Channel {ci+1}")
            ax.plot(data[tt[0]:tt[1],0]/1000.0,data[tt[0]:tt[1],ci+2],label=f"Channel {ci+2}")
            # add the marked peaks
            ax.plot(data[tt[0]:tt[1],0][c0]/1000.0,data[tt[0]:tt[1],i][c0],'ro',label="Marked Peaks")
            ax.plot(data[tt[0]:tt[1],0][c1]/1000.0,data[tt[0]:tt[1],i][c1],'ro')
            # draw a line between pairs of values
            for p0,p1 in zip(c0,c1):
                #print("channel: ",ci,f"({len(channelpks)})")
                # draw line between peak i of channel ci and peak i of channel ci+1
                #print("data idx: ",i,f"({minsz})")
                ax.plot(# x points
                [data[tt[0]:tt[1],0][p0]/1000.0, # time index of peak i in channel ci
                data[tt[0]:tt[1],0][p1]/1000.0], # time index of peak i in channel ci+1
                # y points
                # channel is ci+1 and ci+2 as data matrix includes column on time
                [data[tt[0]:tt[1],ci+1][p0],
                 data[tt[0]:tt[1],ci+2][p1]],'k')
        f.legend()
        # set labels and save figure
        ax.set(xlabel="Time (s)",ylabel="Measured Voltage x100",title=f"Locally Normalized Hall Sensor Data, Marked Peaks\n Track {ti}, Channel {i}")
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-locally-normed-marked-peaks-lines-track-{ti}.png")
        # clear plot as drawing alot of lines eats up a lot of memory
        plt.close(f)
        plt.close(fc)
        
        f,ax = plt.subplots()
        fc,axc = plt.subplots()
        # find the difference between the different channels
        # diff[:,i] = channel[:,i+1] - channel[:,i]
        # e.g. channel[1]-channel[0]
        diffpks = np.diff(channelpksarr,n=1,axis=0)
        # iterate and plot
        for ci in range(diffpks.shape[0]):
            # plot each one
            axc.clear()
            axc.plot(diffpks[ci,:],label=f"Diff. Channel ({ci+1},{c})")
            axc.set(xlabel="Data Index",ylabel=f"Difference in Normed Peak Locations",title=f"Difference in Peak Locations of Normalized Hall Sensor Data\nChannel ({ci+1},{ci}), Track {ti}")
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-locally-normed-diff-peaks-locs-channels-{ci+1}-{ci}-track-{ti}.png")
            # plot global
            ax.plot(diffpks[ci,:])
        ax.set(xlabel="Data Index",ylabel=f"Difference in Normed Peak Locations",title=f"Difference in Peak Locations of Normalized Hall Sensor Data\nChannel ({ci+1},{ci}), Track {ti}")
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-locally-normed-diff-peaks-locs-limited-channels-track-{ti}.png")
        # clear plots
        plt.close(f)
        plt.close(fc)

def plot_normed_data(data,t):
    # reset plots to ensure legends can be redrawn
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    ## normalize data with respect to their limits
    # make a copy of the original data
    datanorm = np.array(data)
    # remove the last two channels as we know they're bad or can be ignored
    datanorm = datanorm[:,:-2]
    # clear the global
    ax.clear()
    # clear the individual channel figures
    axc.clear()
    # for each channel, normalize
    for i in range(1,datanorm.shape[1]):
        datamin = datanorm[:,i].min()
        datamax = datanorm[:,i].max()
        # divide by difference in data
        datanorm[:,i] = (datanorm[:,i]-datamin)/(abs(datamax-datamin))
        # plot the channel figure
        axc.clear()
        axc.plot(datanorm[:,0],datanorm[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-norm-channel-{i-1}.png")
        # add to global plot
        ax.plot(datanorm[:,0],datanorm[:,i],label=str(i))
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-limited-channels.png")

    # individual tracks
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,datanorm.shape[1]):
            # update the global plot
            ax.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i],label=str(i))
            # update the local channel plot
            axc.clear()
            axc.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i])
            axc.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Track {ti}')
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-norm-channel-{i-1}-track-{ti}.png")
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1}), Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-limited-channels-track-{ti}.png")

    ## normalize data using local min and global max
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    datanorm = np.array(data)
    datanorm = datanorm[:,:-2]
    globalrng = np.abs((datanorm[:,1:].max() - datanorm[:,1:].min()))
    # for each channel, normalize
    for i in range(1,datanorm.shape[1]):
        datamin = datanorm[:,i].min()
        datamax = datanorm[:,i].max()
        # divide by difference in data
        datanorm[:,i] = (datanorm[:,i]-datamin)/(globalrng)
        # plot the channel figure
        axc.clear()
        axc.plot(datanorm[:,0],datanorm[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Local Min, Global Range')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-norm-local-min-global-range-channel-{i-1}.png")
        # add to global plot
        ax.plot(datanorm[:,0],datanorm[:,i],label=str(i))
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-local-min-global-range-limited-channels.png")

    # individual tracks
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,datanorm.shape[1]):
            # update the global plot
            ax.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i],label=str(i))
            # update the local channel plot
            axc.clear()
            axc.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i])
            axc.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Track {ti}')
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-norm-local-min-global-range-channel-{i-1}-track-{ti}.png")
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1}), Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-local-min-global-range-limited-channels-track-{ti}.png")


    ## normalizing using method #2
    # reset plots to ensure legends can be redrawn
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    # reset copy of data
    datanorm = np.array(data)
    # remove the last two channels as we know they're bad or can be ignored
    datanorm = datanorm[:,:-2]
    # clear the global
    ax.clear()
    # clear the individual channel figures
    axc.clear()
    # for each channel, normalize
    for i in range(1,datanorm.shape[1]):
        # divide by difference in data
        datanorm[:,i] = (datanorm[:,i]-datanorm[:,i].mean(axis=0))/(np.abs(datanorm[:,i]).max())
        # plot the channel figure
        axc.clear()
        axc.plot(datanorm[:,0],datanorm[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-mean-method-channel-{i-1}.png")
        # add to global plot
        ax.plot(datanorm[:,0],datanorm[:,i],label=str(i))
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-mean-method-channel-limited-channels.png")

    # individual tracks
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,datanorm.shape[1]):
            # update the global plot
            ax.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i],label=str(i))
            # update the local channel plot
            axc.clear()
            axc.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i])
            axc.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Track {ti}')
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-mean-method-channel-channel-{i-1}-track-{ti}.png")
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1}), Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-mean-method-channel-limited-channels-track-{ti}.png")

    ## normalizing using method #3, using global starter mean, local max
    # reset plots to ensure legends can be redrawn
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    # reset copy of data
    datanorm = np.array(data)
    # remove the last two channels as we know they're bad or can be ignored
    datanorm = datanorm[:,:-2]
    # clear the global
    ax.clear()
    # clear the individual channel figures
    axc.clear()
    # calculate global starter mean
    gstartmean = datanorm[0,1:].mean(axis=0)
    # for each channel, normalize
    for i in range(1,datanorm.shape[1]):
        # divide by difference in data
        datanorm[:,i] = (datanorm[:,i]-gstartmean)/(np.abs(datanorm[:,i]).max())
        # plot the channel figure
        axc.clear()
        axc.plot(datanorm[:,0],datanorm[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-method-channel-{i-1}.png")
        # add to global plot
        ax.plot(datanorm[:,0],datanorm[:,i],label=str(i))
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-method-channel-limited-channels.png")

    # individual tracks
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,datanorm.shape[1]):
            # update the global plot
            ax.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i],label=str(i))
            # update the local channel plot
            axc.clear()
            axc.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i])
            axc.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Track {ti}')
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-method-channel-channel-{i-1}-track-{ti}.png")
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1}), Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed--global-start-mean-method-channel-limited-channels-track-{ti}.png")

    ##
    ## normalizing using method #4, using global starter mean, global max
    # reset plots to ensure legends can be redrawn
    f,ax = plt.subplots()
    fc,axc = plt.subplots()
    # reset copy of data
    datanorm = np.array(data)
    # remove the last two channels as we know they're bad or can be ignored
    datanorm = datanorm[:,:-2]
    # clear the global
    ax.clear()
    # clear the individual channel figures
    axc.clear()
    # calculate global starter mean
    gstartmean = datanorm[0,1:].mean(axis=0)
    # global max
    gmax = datanorm[:,1:].max()
    # for each channel, normalize
    for i in range(1,datanorm.shape[1]):
        # divide by difference in data
        datanorm[:,i] = (datanorm[:,i]-gstartmean)/(gmax)
        # plot the channel figure
        axc.clear()
        axc.plot(datanorm[:,0],datanorm[:,i])
        axc.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}')
        fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-global-max-method-channel-{i-1}.png")
        # add to global plot
        ax.plot(datanorm[:,0],datanorm[:,i],label=str(i))
    f.legend()
    ax.set(xlabel='Time (ms)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1})')
    f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-global-max-method-channel-limited-channels.png")

    # individual tracks
    for ti,tt in enumerate(t):
        ax.clear()
        for i in range(1,datanorm.shape[1]):
            # update the global plot
            ax.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i],label=str(i))
            # update the local channel plot
            axc.clear()
            axc.plot(datanorm[tt[0]:tt[1],0]/1000.0,datanorm[tt[0]:tt[1],i])
            axc.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channel {i-1}, Track {ti}')
            fc.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed-global-start-mean-global-max-method-channel-channel-{i-1}-track-{ti}.png")
        f.legend()
        ax.set(xlabel='Time (s)',ylabel='Normalized Measurement',title=f'Normalized Data Recorded Using Hall Sensors During Three Track Run\n Channels (0-{i-1}), Track {ti}')
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-normed--global-start-mean-global-max-method-channel-limited-channels-track-{ti}.png")

# utility function for plotting specific tracks afterwards for inspection etc
def plot_track(ti,data,t,ci='all'):
    # create plot
    ft,axt = plt.subplots()
    # if targetted channels is all, plot all of them
    if ci=='all':
        for cc in range(1,data.shape[1]):
            axt.plot(data[t[ti][0]:t[ti][1],0]/1000.0,data[t[ti][0]:t[ti][1],cc],label=str(cc))
        ft.legend()
        axt.set(xlabel='Time (s)',ylabel='Measured Voltage x100',title=f"Track {ti}, All Channels")
        return ft,axt
    elif type(ci)==int:
        axt.plot(data[t[ti][0]:t[ti][1],0]/1000.0,data[t[ti][0]:t[ti][1],ci],label=str(ci))
        ft.legend()
        axt.set(xlabel='Time (s)',ylabel='Measured Voltage x100',title=f"Track {ti}, Channel {ci}")
        return ft,axt
    else:
        # see if targeted channels were given as an iterable sequence
        try:
            # attempt to form iterator of channel list
            # if it cannot be iterated, a value error is raides 
            citer = iter(ci)
            # iterate over targeted channels
            for cc in citer:
                axt.plot(data[t[ti][0]:t[ti][1],0]/1000.0,data[t[ti][0]:t[ti][1],int(cc)],label=str(cc))
            axt.set(xlabel='Time (s)',ylabel='Measured Voltage x100',title=f"Track {ti}, Channels {list(citer)}")
            return ft,axt
        except ValueError:
            print("Cannot iterate over target channels")
            plt.close(ft)
            return None

# utility function to get track data
def get_track(ti,data,t,ci='all'):
    if ci=='all':
        return data[t[ti][0]:t[ti][1],:]
    elif type(ci) == int:
        if ci>=0:
            return data[t[ti][0]:t[ti][1],ci]
        elif ci<0:
            return data[t[ti][0]:t[ti][1],:ci]
    elif (type(ci) == list) or (type(ci) == tuple):
        return [data[t[ti][0]:t[ti][1],c] for c in ci]

def get_peak_data_prom(ti,data,t,pi=-5,ci='all'):
    # get data for specific track
    # and the time data
    td = [get_track(ti,data,t,0),get_track(ti,data,t,ci)]
    # find the peaks in the data
    pks = find_peaks(td[1])[0]
    # find the prominence of the peaks
    pkp = peak_prominences(td[1],pks)[0]
    # get the unique peak prominences values
    pku = np.unique(pkp)
    print(f"Channel {ci}, Track {ti} : {pku.shape}, {pku}")
    # find the peaks above a specific prominence value
    if type(pi) == int:
        pks = pks[pkp>pku[pi]]
    elif pi == 'half':
        pks = pks[pkp>pku[pku.shape[0]//2]]
    elif pi == 'nhalf':
        pks = pks[pkp>pku[-pku.shape[0]//2]]
    # plot which peaks were selected and the values in a separate plot
    f,ax = plt.subplots(1,2,constrained_layout=True)
    # plot the data with the peaks marked
    ax[0].plot(td[0],td[1],'b-',label=f"Channel {ci}")
    ax[0].plot(td[0][pks],td[1][pks],'rx',label="Filtered Peaks")
    ax[0].set(xlabel='Time (ms)',ylabel='Measured Voltage x100')
    # plot the values for the found peaks as a line to shown change
    ax[1].plot(td[0][pks]/1000.0,td[1][pks],'b-',label="Filtered Peak Values")
    ax[1].set(xlabel='Time (s)',ylabel="Measured Voltage x100")
    f.suptitle(f"Track {ti}, Channels {ci}, Filtered Peaks According to Prominence")
    # return the data and the figure and axes
    return td[1][pks],f,ax

def get_peak_data_width(ti,data,t,period=10000,sf=0.7,ci='all'):
    td = get_track(ti,data,t,ci=ci)
    # find peaks
    # restriction is based on the given period between peaks
    # sf is a scaling factoro to account for the stated track range to be larger than the actual track range
    pks = find_peaks(td[1],distance=(sf*period)/np.diff(td[0]).min())[0]
    # plot results
    f,ax = plt.subplots(1,2,constrained_layout=True)
    # plot the data with the peaks marked
    ax[0].plot(td[0],td[1],'b-',label=f"Channel {ci}")
    ax[0].plot(td[0][pks],td[1][pks],'rx',label="Filtered Peaks")
    ax[0].set(xlabel='Time (ms)',ylabel='Measured Voltage x100')
    # plot the values for the found peaks as a line to shown change
    ax[1].plot(td[0][pks]/1000.0,td[1][pks],'b-',label="Filtered Peak Values")
    ax[1].set(xlabel='Time (s)',ylabel="Measured Voltage x100")
    f.suptitle(f"Track {ti}, Channels {ci}, Filtered Peaks According to Period")
    # return the data and the figure and axes
    return td[1][pks],f,ax

def get_peak_offset_width(ti,data,t,ci,period=10000,sf=0.7,):
    # get the data for each of the target channels
    # assuming first channel is time
    td = get_track(ti,data,t,ci=ci)
    pks = []
    for cd in td[1:]:
        # find peaks
        # restriction is based on the given period between peaks
        # sf is a scaling factoro to account for the stated track range to be larger than the actual track range
        pks.append(find_peaks(cd,distance=(sf*period)/np.diff(td[0]).min())[0])
    # as the different channels are likely to have different lengths
    # we need to get a subset of points to make them the same length
    sz = min([len(p) for p in pks])
    pksf = [p[:sz] for p in pks]
    
    # find difference between channels
    dpks = np.diff(np.vstack(pksf),axis=0)
    # generate plots of the peaks used
    fp,axp = plt.subplots()
    fcp,axcp = plt.subplots()
    # iterate over channel index, channel data, unfiltered peaks,sub set of peaks
    for cc,cd,pc,pcf in zip(ci[1:],td[1:],pks,pksf):
        # global plot for all channels
        axp.plot(td[0],cd,'-',label=f"Channel {cc}")
        axp.plot(td[0][pc],cd[pc],'x',label=f"Peaks, {cc}")
        axp.plot(td[0][pcf],cd[pcf],'o',label=f"Subset Peaks, {cc}")

        # individual channels
        axcp.clear()
        axcp.plot(td[0],cd,'b-',td[0][pc],cd[pc],'ro',td[0][pcf],cd[pcf],'gx')
        axcp.set(xlabel='Time (ms)',ylabel='Measured Voltage x100',title=f"Filtered peaks by Period and Data Length, Track {ti}, Channel {cc}")
        fcp.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-peaks-period-datalength-track-{ti}-channel-{cc}.png")

    axp.set(xlabel='Time (ms)',ylabel='Measured Voltage x100',title=f"Filtered peaks by Period and Data Length, Track {ti}")
    fp.legend()    
    fp.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-peaks-period-datalength-track-{ti}-channel-{'-'.join([str(c) for c in ci])}.png")

    plt.close(fcp)
    # generate plots for the differences between the different channels
    for c in range(dpks.shape[0]):
        axcp.clear()
        axcp.plot(dpks[c,:],'b-')
        axcp.set(xlabel='Peak Index',ylabel="Difference in Peak Locations (Samples)",title=f"Estimated Offset Between Channels [{ci[c+2]},{ci[c+1]}], Track {ti}")
        fcp.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-peaks-period-offset-datalength-track-{ti}-channel-{ci[c+2]}-{ci[c+1]}.png")
    plt.close(fcp)
    
if __name__ == "__main__":
    print("Getting data")
    # generate the data from the CSV file
    fname = "h0000000.csv"
    data = np.genfromtxt("h0000000.csv",delimiter=',')

    # create directory for results
    os.makedirs("Plots",exist_ok=True)
    print("Generating basic plots")
    # plot the data and channels
    #plot_all(data)

    # track time ranges, estimated by eye
    # length of each track run
    dist = 512400/2

    ### fixed length track histories
    # track t0 is different as it was stopped before a complete run was finished
    t0 = [2275000,2290000]
    t1 = [2567220,2567220+dist]
    t2 = [3136810,3136810+dist]
    t3 = [3534450,3534450+dist]

    ## find closest indicies to time value
    # searches array by distance
    def find_nearest(array, value):
        # ensures input is an array
        array = np.asarray(array)
        # get the location of the value that has the smallest
        # difference to the target value
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def find_nearest_idx(array, value):
        # ensures input is an array
        array = np.asarray(array)
        # get the location of the value that has the smallest
        # difference to the target value
        return (np.abs(array - value)).argmin()

    # searches array for target value
    # from https://stackoverflow.com/a/26026189
    def find_nearest_sorted(array,value):
        # search assumed sorted array and return index of closest value
        idx = np.searchsorted(array, value, side="left")
        # return either use the index or the value just before it
        # whichever if closer
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]

    def find_nearest_idx_sorted(array,value):
        # search assumed sorted array and return index of closest value
        idx = np.searchsorted(array, value, side="left")
        # return either use the index or the value just before it
        # whichever if closer
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
    
    # find the indicies of the target time ranges
    # ensure result is integer so it can be used as an index
    xt0 = [find_nearest_idx(data[:,0],float(tt)) for tt in t0]
    xt1 = [find_nearest_idx(data[:,0],float(tt)) for tt in t1]
    xt2 = [find_nearest_idx(data[:,0],float(tt)) for tt in t2]
    xt3 = [find_nearest_idx(data[:,0],float(tt)) for tt in t3]

    t = [xt0,xt1,xt2,xt3]
    
    # iterate over limits to check that they're the same length
    # if not, move the last time index so the vector is the same length
    d = t[1][1]-t[1][0]
    for tt in t[1:]:
        if (tt[1]-tt[0])!=d:
            tt[1] += (d-(tt[1]-tt[0]))

    
    # plot tracks
    print("Generating plots for each track")
    trackdata=plot_tracks(data,t)
    # plot the contourf plots
    #plot_contourf(trackdata)

    # combine track histories into a 3d history
    # number of channels x number of tracks x time
    # ignoring the track 0 as it does not have the same length as the others
    #trackhist = np.dstack(trackdata[1:]).swapaxes(1,2).swapaxes(2,0)

    # plot data with the average subtracted
    print("Creating minus average plot")
    #sub_avg_plot(data)

    # estimate offset between channel peaks
    #est_channel_offset(trackdata)

    # get the filtered peak plots
    # shows how the peaks vary over time
    for ci in range(1,6,1):
        _,f,ax = get_peak_data_prom(1,data,t,pi='nhalf',ci=ci)
        f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-filtered-peaks-prominence-track-{1}-channel-{ci}.png")
        plt.close(f)

    for ti in [0,2,3]:
        for ci in range(1,6,1):
            _,f,ax = get_peak_data_prom(ti,data,t,pi='nhalf',ci=ci)
            f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-filtered-peaks-prominence-track-{ti}-channel-{ci}.png")
            plt.close(f)

    # plot filtered peaks as filtered by time between peaks
    for ti in range(len(t)):
        for ci in range(1,6,1):
            _,f,ax = get_peak_data_width(ti,data,t,period=10000,sf=0.7,ci=ci)
            f.savefig(f"Plots/{os.path.splitext(os.path.basename(fname))[0]}-filtered-peaks-period-track-{ti}-channel-{ci}.png")
            plt.close(f)

    # create the line animations for the different tracks
    # set target time based on captured time period times by a factor
    print("Building animation")
    speed_f = 2.0**-1.0
    #anim_line(data,trackdata,t,(dist/1000.0)*speed_f)

    # clear all plots
    # plot_norm tends to generate alot of plots
    plt.close('all')

    # norm data different ways and plot it
    print("Creating ALL normed data")
    #plot_normed_data(data,t)
