import numpy as np
from numpy.polynomial.polynomial import polyfit as poly_fit
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from scipy.signal import find_peaks
import cv2
from scipy.stats import gaussian_kde

def dbscanplot(x,e,min_sp):
   ## scanning and plotting
   db = DBSCAN(eps=ee,min_samples=min_sp,algorithm='auto').fit(data)
   # construct masks for plotting
   #print("Constructing masks for plotting")
   core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
   core_samples_mask[db.core_sample_indices_]=True
   labels = db.labels_
   # number of clusters in labels, -1 indicates noise
   nc = len(set(labels)) - (1 if -1 in labels else 0)
   # number of points that are labelled as noise
   n_noise = list(labels).count(-1)
   ## print stats
   print("Estimated number of clusters : {}".format(nc))
   print("Estimated number of noise points: {}".format(n_noise))
   ## plot results
   print("Plotting results")
   # get the list of unique labels
   unique_labels = set(labels)
   # generate colors for each label
   colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]

   # iterte through labels and colors
   for k,col in zip(unique_labels,colors):
       # if it's labelled as noise
       if k == -1:
           col = [0,0,0,1]

       # create a mask to filter values by the current label
       class_mask = (labels==k)
       # combine the core samples mask and the class mask together
       # use too get the relevant data points
       xy = data[class_mask & core_samples_mask]
       # plot the core samples with enlarged markers
       # markerfacecolor is the color to fill in the markers with
       plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=14)

       xy = data[class_mask & ~core_samples_mask]
       # mark everything else with smaller size markers
       plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6)

# read in data
local_data = np.genfromtxt("D:\\BEAM\\Scripts\\RelatedToAbsorbtivity\\thermal-hough-circle-metrics-local.csv",
                           delimiter=',',
                           dtype='float',
                           names=True)

global_data = np.genfromtxt("D:\\BEAM\\Scripts\\RelatedToAbsorbtivity\\thermal-hough-circle-metrics-global.csv",
                            delimiter=',',
                            dtype='float',
                            names=True)
# extract data into easier to use datasets
qr = local_data['Radiated_Heat_Times_Area_W']
x = local_data["Best_Circle_X_Position_Pixels"]
y = local_data["Best_Circle_Y_Position_Pixels"]
r = local_data['Best_Circle_Radius_Pixels']
# index array
ii = np.arange(qr.shape[0])

# filter data to remove outliers
def filt3stdIdx(x):
   return np.where((x>=(x.mean()-3*x.std())) & (x<=(x.mean()+3*x.std())))[0]

stdi = filt3stdIdx(qr)
qr = qr[stdi]
x = x[stdi]
y = y[stdi]
r = r[stdi]

###### 10 10 is pretty good for x

# put datasets into list
dsets = [x,y,r]
dnames = ['x','y','r']
dtitles = ["Best Circle X Position Pixels","Best Circle Y Position Pixels","Best Circle Radius Pixels"]
os.makedirs("DBSCAN\\x",exist_ok=True)
os.makedirs("DBSCAN\\y",exist_ok=True)
os.makedirs("DBSCAN\\r",exist_ok=True)

print("Performing db scan")
f,ax = plt.subplots()
##
##for di,dd in enumerate(dsets):
##   data = np.array([(qq,xx) for qq,xx in zip(qr,dd)])
##   print("Running for {}".format(dnames[di]))
##   for ee in np.arange(1,100,10):
##       for min_sp in np.arange(1,100,10):
##           ## scanning and plotting
##           db = DBSCAN(eps=ee,min_samples=min_sp,algorithm='auto').fit(data)
##           # construct masks for plotting
##           #print("Constructing masks for plotting")
##           core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
##           core_samples_mask[db.core_sample_indices_]=True
##           labels = db.labels_
##           # number of clusters in labels, -1 indicates noise
##           nc = len(set(labels)) - (1 if -1 in labels else 0)
##           # number of points that are labelled as noise
##           n_noise = list(labels).count(-1)
##           ## print stats
##           print("Estimated number of clusters : {}".format(nc))
##           print("Estimated number of noise points: {}".format(n_noise))
##           
##           # plot information if we have at least one cluster and not all the data is classified as noise
##           if n_noise == len(data) or nc==0:
##               continue
##           else:
##               ## plot results
##               print("Plotting results")
##               ax.clear()
##               # get the list of unique labels
##               unique_labels = set(labels)
##               # generate colors for each label
##               colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]
##
##               # iterte through labels and colors
##               for k,col in zip(unique_labels,colors):
##                   # if it's labelled as noise
##                   if k == -1:
##                       col = [0,0,0,1]
##
##                   # create a mask to filter values by the current label
##                   class_mask = (labels==k)
##                   # combine the core samples mask and the class mask together
##                   # use too get the relevant data points
##                   xy = data[class_mask & core_samples_mask]
##                   # plot the core samples with enlarged markers
##                   # markerfacecolor is the color to fill in the markers with
##                   ax.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=14)
##
##                   xy = data[class_mask & ~core_samples_mask]
##                   # mark everything else with smaller size markers
##                   ax.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6)
##
##               f.suptitle('Best {}\n as Sorted by Density Based Clustering, eps={},min_s={}'.format(dtitles[di],ee,min_sp))
##               f.savefig('DBSCAN\\{}\\dbscan-best-hough-circle-centre-{}-e{}-minsp{}.png'.format(dnames[di],dnames[di],ee,min_sp))

print("Running narrowed range scan")
## performm a density clustering run using a narrowed range of parameters
# narrowed range of parameters to try
x_narrow_eps = np.arange(10,15,1)
x_narrow_mins = np.arange(40,50,1)

y_narrow_eps = np.arange(10,15,1)
y_narrow_mins = np.arange(30,50,1)

r_narrow_eps = np.arange(10,15,1)
r_narrow_mins = np.arange(80,90,1)

# package the data so it can be iterated
narrow_eps = [x_narrow_eps,y_narrow_eps,r_narrow_eps]
narrow_mins = [x_narrow_mins,y_narrow_mins,r_narrow_mins]

os.makedirs("DBSCAN\\Narrow\\x",exist_ok=True)
os.makedirs("DBSCAN\\Narrow\\y",exist_ok=True)
os.makedirs("DBSCAN\\Narrow\\r",exist_ok=True)

fouts = ["DBSCAN\\Narrow\\x","DBSCAN\\Narrow\\y","DBSCAN\\Narrow\\r"]
def dbscan_custom(data,eps_range,mins_range,dtitles,dnames,fout,legends=False):
    f,ax = plt.subplots()
    for ee in eps_range:
       for min_sp in mins_range:
           ## scanning and plotting
           db = DBSCAN(eps=ee,min_samples=min_sp,algorithm='auto').fit(data)
           # construct masks for plotting
           #print("Constructing masks for plotting")
           core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
           core_samples_mask[db.core_sample_indices_]=True
           labels = db.labels_
           # number of clusters in labels, -1 indicates noise
           nc = len(set(labels)) - (1 if -1 in labels else 0)
           # number of points that are labelled as noise
           n_noise = list(labels).count(-1)
           
           # plot information if we have at least one cluster and not all the data is classified as noise
           if n_noise == len(data) or nc==0:
               continue
           else:
               ax.clear()
               # get the list of unique labels
               unique_labels = set(labels)
               # generate colors for each label
               colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]

               # iterte through labels and colors
               for k,col in zip(unique_labels,colors):
                   # if it's labelled as noise
                   if k == -1:
                       col = [0,0,0,1]
                       label_add = " (noise)"
                   else:
                       label_add = ""

                   # create a mask to filter values by the current label
                   class_mask = (labels==k)
                   # combine the core samples mask and the class mask together
                   # use too get the relevant data points
                   xy = data[class_mask & core_samples_mask]
                   # plot the core samples with enlarged markers
                   # markerfacecolor is the color to fill in the markers with
                   ax.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=14,label="Core Samples {}".format(k)+ label_add)

                   xy = data[class_mask & ~core_samples_mask]
                   # mark everything else with smaller size markers
                   ax.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6,label="Cluster Members {}".format(k)+ label_add)

               if legends:
                  ax.legend()
               f.suptitle('Best {}\n as Sorted by Density Based Clustering, eps={},min_s={}'.format(dtitles,ee,min_sp))
               f.savefig(os.path.join(fout,'dbscan-best-hough-circle-centre-{}-e{}-minsp{}.png'.format(dnames,ee,min_sp)))

##for di,dd in enumerate(dsets):
##    data = np.array([(qq,xx) for qq,xx in zip(qr,dd)])
##    dbscan_custom(data,narrow_eps[di],narrow_mins[di],dtitles[di],dnames[di],fouts[di])

# filtering radius data
##r_filt = r[r<64.0]
##os.makedirs("DBSCAN/r_lim",exist_ok=True)
##data = np.array([(qq,xx) for qq,xx in zip(qr,r_filt)])
##
##dbscan_custom(data,np.arange(1,100,10),np.arange(1,100,10),dtitles[2],dnames[2],"DBSCAN/r_lim")
##
##os.makedirs("DBSCAN/r_lim/Narrow",exist_ok=True)
##dbscan_custom(data,np.arange(1,10,1),np.arange(1,30,1),dtitles[2],dnames[2],"DBSCAN/r_lim/Narrow")
##
##ax.clear()
##kde_data = np.vstack((qr[r<64.0],r_filt))
##kde_r_filt = gaussian_kde(kde_data)(kde_data)
##ax.scatter(qr[r<64.0],r_filt,c=kde_r_filt,edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \n Best Circle Radius for values less than 64.0')
##f.savefig('DensityPlots/density-head-rad-best-circle-radius-filt-lt.png')

## clustering radius after filtering by peak
peaks,_ = find_peaks(qr)
tol = 0.05
Qr_abs_max = qr.max()
# searching for peak values within the tolerancce of the power density max
pki = peaks[np.where(qr[peaks]>=(1-tol)*Qr_abs_max)]

##os.makedirs("DBSCAN/r_lim/Peaks",exist_ok=True)
### use peak indicies to find corresponding data
##r_pk_filt = r[pki]
### get the radii values that are less than 64.0 pixels, half image size
##r_pk_filt = r_pk_filt[r_pk_filt<64.0]
### create a boolean array from the peak indicies
##fill = np.zeros(r.shape[0],dtype='bool')
##fill[pki]=True
### look at element wise combination of the two
##mask = np.all([fill,r<64.0],axis=0)
##data = np.array([(qq,xx) for qq,xx in zip(qr[mask],r_pk_filt)])
##dbscan_custom(data,np.arange(1,100,10),np.arange(1,100,10),dtitles[2],dnames[2],"DBSCAN/r_lim/Peaks")
##
##ax.clear()
##kde_data = np.vstack((qr[mask],r_pk_filt))
##kde_r_filt = gaussian_kde(kde_data)(kde_data)
##ax.scatter(qr[mask],r_pk_filt,c=kde_r_filt,edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs Best Circle Radius\n Filtered by Power Density Peaks and Values less than 64.0')
##f.savefig('DensityPlots/density-head-rad-best-circle-radius-filt-lt-peaks.png')

##os.makedirs("DBSCAN\\r-peak",exist_ok=True)
##data = np.array([(qq,xx) for qq,xx in zip(qr[pki],r[pki])])
##print("Starting run for radius, peak power only")
##dbscan_custom(data,np.arange(1,100,10),np.arange(1,100,10),"Best Circle Radius Pixels (Power Peak)",'r',"DBSCAN\\r-peak")
##
##os.makedirs("DBSCAN\\r-filt",exist_ok=True)
##data = np.array([(qq,xx) for qq,xx in zip(qr[r>=64.0],r[r>64.0])])
##print("Starting run for radius, filt limit")
##dbscan_custom(data,np.arange(1,100,10),np.arange(1,100,10),"Best Circle Radius Pixels (Limit)",'r',"DBSCAN\\r-filt")

## finding and fitting limits to range
# cluster the data to filter it
##ee = 11
##min_sp = 49
##data = np.array([(qq,xx) for qq,xx in zip(qr,x)])
##db = DBSCAN(eps=ee,min_samples=min_sp,algorithm='auto').fit(data)
### construct masks for plotting
##core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
##core_samples_mask[db.core_sample_indices_]=True
##labels = db.labels_
### get the points that are not noise
##class_mask = (db.labels_ != -1) & core_samples_mask
##xy = data[class_mask]
### save the plot of the filtered data
##ax.clear()
##ax.plot(qr,x,'ro',xy[:,0],xy[:,1],'bo')
##ax.set(xlabel='Radiated Heat of Area (W)',ylabel='Best Circle Centre X Position (Pixels)')
##ax.legend(['Original Data','Filtered Data'])
##f.suptitle('Best Circle Centre X filtered by DBSCAN. eps={}, min samples={}'.format(ee,min_sp))
##f.savefig('best-circle-centre-x-filt-dbscan.png')
##
##os.makedirs('DBSCAN\\StripsFitting',exist_ok=True)
##x_xs_min = []
##x_xs_max = []
##for num_strips in range(10,50):
##   # create strips
##   qr_strips = np.linspace(qr[class_mask].min(),qr[class_mask].max(),num_strips)
##   # lists for storing min max values
##   xs_min = []
##   xs_max = []
##   # get the middle values between strips
##   qrs = (qr_strips[1:]+qr_strips[:-1])/2
##   # find max and min value within each strip
##   for qi in range(0,qr_strips.shape[0]-1):
##       i = np.where((qr[class_mask]>=qr_strips[qi]) & (qr[class_mask]<=qr_strips[qi+1]))[0]
##       # if there are values within the strip, add the middle value and respective limit to the list
##       if i.shape[0]!=0:
##           xs_min.append((qrs[qi],xy[i,1].min()))
##           xs_max.append((qrs[qi],xy[i,1].max()))
##   # convert to arrays
##   xs_min = np.asarray(xs_min)
##   xs_max = np.asarray(xs_max)
##   # add to super list
##   x_xs_min.append(np.poly1d(poly_fit(xs_min[:,0],xs_min[:,1],3)[::-1]))
##   x_xs_max.append(np.poly1d(poly_fit(xs_max[:,0],xs_max[:,1],3)[::-1]))
##   ## plot result
##   ax.clear()
##   ax.plot(qr[class_mask],x[class_mask],'rx',# plot data
##            xs_min[:,0],xs_min[:,1],'mo-', # plot x min
##            xs_max[:,0],xs_max[:,1],'ko-') # plot x max
##
##   # add lines representing the strip limits
##   ylim = ax.get_ybound()
##   for q in qr_strips:
##      ax.add_line(Line2D([q]*5,np.linspace(ylim[0],ylim[1],num=5),
##                         color='b',
##                         linestyle='--'))
##
##   ax.legend(['Data','Strip Min','Strip Max','Strips'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Radiated Heat Times Area vs Best Circle X Position, Strips={}'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-fit-heat-rad-best-circle-centre-x-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Minimum Best Circle X Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-min-best-circle-centre-x-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Maximum Best Circle X Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-max-best-circle-centre-x-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-',xs_min[:,0],x_xs_min[-1](xs_min[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Minimum Best Circle X Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-min-best-circle-centre-x-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-',xs_max[:,0],x_xs_max[-1](xs_max[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Maximum Best Circle X Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-max-best-circle-centre-x-s{}.png'.format(num_strips))
##
#### y 
##ee = 10
##min_sp = 46
##data = np.array([(qq,xx) for qq,xx in zip(qr,y)])
##db = DBSCAN(eps=ee,min_samples=min_sp,algorithm='auto').fit(data)
### construct masks for plotting
##core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
##core_samples_mask[db.core_sample_indices_]=True
##labels = db.labels_
### get the points that are not noise
##class_mask = (db.labels_ != -1) & core_samples_mask
##xy = data[class_mask]
### save the plot of the filtered data
##ax.clear()
##ax.plot(qr,y,'ro',xy[:,0],xy[:,1],'bo')
##ax.set(xlabel='Radiated Heat of Area (W)',ylabel='Best Circle Centre Y Position (Pixels)')
##ax.legend(['Original Data','Filtered Data'])
##f.suptitle('Best Circle Centre Y filtered by DBSCAN. eps={}, min samples={}'.format(ee,min_sp))
##f.savefig('best-circle-centre-y-filt-dbscan.png')
##
##y_xs_min = []
##y_xs_max = []
### break the data into strips
##for num_strips in range(10,50):
##   # create strips
##   qr_strips = np.linspace(qr[class_mask].min(),qr[class_mask].max(),num_strips)
##   # lists for storing min max values
##   xs_min = []
##   xs_max = []
##   # get the middle values between strips
##   qrs = (qr_strips[1:]+qr_strips[:-1])/2
##   # find max and min value within each strip
##   for qi in range(0,qr_strips.shape[0]-1):
##       i = np.where((qr[class_mask]>=qr_strips[qi]) & (qr[class_mask]<=qr_strips[qi+1]))[0]
##       # if there are values within the strip, add the middle value and respective limit to the list
##       if i.shape[0]!=0:
##           xs_min.append((qrs[qi],xy[i,1].min()))
##           xs_max.append((qrs[qi],xy[i,1].max()))
##   # convert to arrays
##   xs_min = np.asarray(xs_min)
##   xs_max = np.asarray(xs_max)
##   # add to super list
##   y_xs_min.append(np.poly1d(poly_fit(xs_min[:,0],xs_min[:,1],3)[::-1]))
##   y_xs_max.append(np.poly1d(poly_fit(xs_max[:,0],xs_max[:,1],3)[::-1]))
##   ## plot result
##   ax.clear()
##   ax.plot(qr[class_mask],x[class_mask],'rx',# plot data
##            xs_min[:,0],xs_min[:,1],'mo-', # plot x min
##            xs_max[:,0],xs_max[:,1],'ko-') # plot x max
##
##   # add lines representing the strip limits
##   ylim = ax.get_ybound()
##   for q in qr_strips:
##      ax.add_line(Line2D([q]*5,np.linspace(ylim[0],ylim[1],num=5),
##                         color='b',
##                         linestyle='--'))
##
##   ax.legend(['Data','Strip Min','Strip Max','Strips'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Y Position (Pixels)')
##   f.suptitle('Radiated Heat Times Area vs Best Circle Y Position, Strips={}'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-fit-heat-rad-best-circle-centre-y-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Y Position (Pixels)')
##   f.suptitle('Minimum Best Circle Y Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-min-best-circle-centre-y-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Y Position (Pixels)')
##   f.suptitle('Maximum Best Circle X Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-max-best-circle-centre-y-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-',xs_min[:,0],y_xs_min[-1](xs_min[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Y Position (Pixels)')
##   f.suptitle('Minimum Best Circle Y Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-min-best-circle-centre-y-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-',xs_max[:,0],y_xs_max[-1](xs_max[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Y Position (Pixels)')
##   f.suptitle('Maximum Best Circle Y Position for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-max-best-circle-centre-y-s{}.png'.format(num_strips))
##   

####r
##class_mask = np.array(r<64.0)
##data = np.array([(qq,xx) for qq,xx in zip(qr,r)])
##xy = data[class_mask]
##r_xs_min = []
##r_xs_max = []
##print("Breaking filtered radius data into strips")
##for num_strips in range(10,50):
##   # create strips
##   qr_strips = np.linspace(qr[class_mask].min(),qr[class_mask].max(),num_strips)
##   # lists for storing min max values
##   xs_min = []
##   xs_max = []
##   # get the middle values between strips
##   qrs = (qr_strips[1:]+qr_strips[:-1])/2
##   # find max and min value within each strip
##   for qi in range(0,qr_strips.shape[0]-1):
##       i = np.where((qr[class_mask]>=qr_strips[qi]) & (qr[class_mask]<=qr_strips[qi+1]))[0]
##       # if there are values within the strip, add the middle value and respective limit to the list
##       if i.shape[0]!=0:
##           xs_min.append((qrs[qi],xy[i,1].min()))
##           xs_max.append((qrs[qi],xy[i,1].max()))
##   # convert to arrays
##   xs_min = np.asarray(xs_min)
##   xs_max = np.asarray(xs_max)
##   # add to super list
##   r_xs_min.append(np.poly1d(poly_fit(xs_min[:,0],xs_min[:,1],3)[::-1]))
##   r_xs_max.append(np.poly1d(poly_fit(xs_max[:,0],xs_max[:,1],3)[::-1]))
##   ## plot result
##   ax.clear()
##   ax.plot(qr[class_mask],x[class_mask],'rx',# plot data
##            xs_min[:,0],xs_min[:,1],'mo-', # plot x min
##            xs_max[:,0],xs_max[:,1],'ko-') # plot x max
##
##   # add lines representing the strip limits
##   ylim = ax.get_ybound()
##   for q in qr_strips:
##      ax.add_line(Line2D([q]*5,np.linspace(ylim[0],ylim[1],num=5),
##                         color='b',
##                         linestyle='--'))
##
##   ax.legend(['Data','Strip Min','Strip Max','Strips'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##   f.suptitle('Radiated Heat Times Area vs Best Circle Radius, Strips={}'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-fit-heat-rad-best-circle-radius-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##   f.suptitle('Minimum Best Circle Radius for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-min-best-circle-radius-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##   f.suptitle('Maximum Best Circle Radius for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\strips-max-best-circle-radius-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_min[:,0],xs_min[:,1],'bo-',xs_min[:,0],r_xs_min[-1](xs_min[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##   f.suptitle('Minimum Best Circle Radius for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-min-best-circle-radius-s{}.png'.format(num_strips))
##
##   ax.clear()
##   ax.plot(xs_max[:,0],xs_max[:,1],'bo-',xs_max[:,0],r_xs_max[-1](xs_max[:,0]),'ro-')
##   ax.legend(['Data','Fitted Cubic'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
##   f.suptitle('Maximum Best Circle Radius for {} Strips'.format(num_strips))
##   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-max-best-circle-radius-s{}.png'.format(num_strips))

##r double filt
r_pk_filt = r[pki]
# get the radii values that are less than 64.0 pixels, half image size
r_pk_filt = r_pk_filt[r_pk_filt<64.0]
# create a boolean array from the peak indicies
fill = np.zeros(r.shape[0],dtype='bool')
fill[pki]=True
# look at element wise combination of the two
class_mask = np.all([fill,r<64.0],axis=0)
data = np.array([(qq,xx) for qq,xx in zip(qr,r)])
xy = data[class_mask]
r_xs_min = []
r_xs_max = []
print("Breaking double filtered radius data into strips")
for num_strips in range(10,50):
   # create strips
   qr_strips = np.linspace(qr[class_mask].min(),qr[class_mask].max(),num_strips)
   # lists for storing min max values
   xs_min = []
   xs_max = []
   # get the middle values between strips
   qrs = (qr_strips[1:]+qr_strips[:-1])/2
   # find max and min value within each strip
   for qi in range(0,qr_strips.shape[0]-1):
       i = np.where((qr[class_mask]>=qr_strips[qi]) & (qr[class_mask]<=qr_strips[qi+1]))[0]
       # if there are values within the strip, add the middle value and respective limit to the list
       if i.shape[0]!=0:
           xs_min.append((qrs[qi],xy[i,1].min()))
           xs_max.append((qrs[qi],xy[i,1].max()))
   # convert to arrays
   xs_min = np.asarray(xs_min)
   xs_max = np.asarray(xs_max)
   # add to super list
   r_xs_min.append(np.poly1d(poly_fit(xs_min[:,0],xs_min[:,1],3)[::-1]))
   r_xs_max.append(np.poly1d(poly_fit(xs_max[:,0],xs_max[:,1],3)[::-1]))
   ## plot result
   ax.clear()
   ax.plot(qr[class_mask],x[class_mask],'rx',# plot data
            xs_min[:,0],xs_min[:,1],'mo-', # plot x min
            xs_max[:,0],xs_max[:,1],'ko-') # plot x max

   # add lines representing the strip limits
   ylim = ax.get_ybound()
   for q in qr_strips:
      ax.add_line(Line2D([q]*5,np.linspace(ylim[0],ylim[1],num=5),
                         color='b',
                         linestyle='--'))

   ax.legend(['Data','Strip Min','Strip Max','Strips'])
   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
   f.suptitle('Radiated Heat Times Area vs Best Circle Radius\nFiltered by Peaks, Strips={}'.format(num_strips))
   f.savefig('DBSCAN\\StripsFitting\\strips-fit-heat-rad-best-circle-radius-s{}-filt-peaks.png'.format(num_strips))

   ax.clear()
   ax.plot(xs_min[:,0],xs_min[:,1],'bo-')
   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
   f.suptitle('Minimum Best Circle Radius for {} Strips Filtered by Peaks'.format(num_strips))
   f.savefig('DBSCAN\\StripsFitting\\strips-min-best-circle-radius-s{}-filt-peaks.png'.format(num_strips))

   ax.clear()
   ax.plot(xs_max[:,0],xs_max[:,1],'bo-')
   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
   f.suptitle('Maximum Best Circle Radius for {} Strips Filtered by Peaks'.format(num_strips))
   f.savefig('DBSCAN\\StripsFitting\\strips-max-best-circle-radius-s{}-filt-peaks.png'.format(num_strips))

   ax.clear()
   ax.plot(xs_min[:,0],xs_min[:,1],'bo-',xs_min[:,0],r_xs_min[-1](xs_min[:,0]),'ro-')
   ax.legend(['Data','Fitted Cubic'])
   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
   f.suptitle('Minimum Best Circle Radius for {} Strips Filtered by Peaks'.format(num_strips))
   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-min-best-circle-radius-s{}-filt-peaks.png'.format(num_strips))

   ax.clear()
   ax.plot(xs_max[:,0],xs_max[:,1],'bo-',xs_max[:,0],r_xs_max[-1](xs_max[:,0]),'ro-')
   ax.legend(['Data','Fitted Cubic'])
   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Radius (Pixels)')
   f.suptitle('Maximum Best Circle Radius for {} Strips'.format(num_strips))
   f.savefig('DBSCAN\\StripsFitting\\fitted-strips-max-best-circle-radius-s{}-filt-peaks.png'.format(num_strips))


#### plot search range in image
##print("Iterating through search regions")
##os.makedirs("DBSCAN\\StripsFitting\\ImageScan",exist_ok=True)
##for ni,num_strips in enumerate(range(10,50)):
##    for ff in range(qr.shape[0]):
##        # create empty mask
##        mask = np.zeros((128,128,3),dtype='uint8')
##        # get limits of search area
##        x_min = int(x_xs_min[ni](qr[ff]))
##        x_max = int(x_xs_max[ni](qr[ff]))
##        y_min = int(y_xs_min[ni](qr[ff]))
##        y_max = int(y_xs_max[ni](qr[ff]))
##        # draw search box on image
##        cv2.rectangle(mask,(y_min,x_min),(y_max,x_max),(0,0,255),cv2.FILLED)
##        cv2.imwrite("DBSCAN\\StripsFitting\\ImageScan\\search-area-s{}-f{}.png".format(num_strips,ff),mask)        

##os.makedirs("DBSCAN/StripsFitting/RadiusRange",exist_ok=True)
##for ni,num_strips in enumerate(range(10,50)):
##   for ff in range(qr.shape[0]):
##      mask = np.zeros((128,128),dtype='uint8')
##
##      r_min = int(r_xs_min[ni](qr[ff]))
##      r_max = int(r_xs_max[ni](qr[ff]))
##
##      cv2.circle(mask,(64,64),r_max,255,1,cv2.FILLED)
##      cv2.circle(mask,(64,64),r_min,0,1,cv2.FILLED)
##      cv2.imwrite("DBSCAN/StripsFitting/RadiusRange/radius-range-s{}-f{}.png".format(num_strips,ff),mask)
##      
