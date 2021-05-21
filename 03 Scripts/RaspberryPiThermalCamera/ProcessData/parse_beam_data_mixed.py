import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

path = "D:\BEAM\Trial1_14.10.19-20191101T160322Z-001\Trial1_14.10.19\BeAMON-Contains Tree and 9welds.csv"
fname = os.path.splitext(os.path.basename(path))[0]
# read in data from target csv
data = np.genfromtxt(path,delimiter=',',names=True)
# it's an array of tuples of a custom data type determined by headers
# each tuple is a row of data
#data_s = np.array([list(x) for x in pp])

# create folder for results
plot_folder = "{}-Plots".format(fname.replace(" ",'-'))
os.makedirs(plot_folder,exist_ok=True)

# iterate over columns starting at 5th row

def gen_plots(fpath,data):
    f,ax = plt.subplots()
    # iterate over data names
    for nn in data.dtype.names:
        # get data
        pp = data[nn]
        # check if data is all NaNs
        if not np.all(np.isnan(pp)):
            print(nn)
            ax.clear()
            ax.plot(pp,'b-')
            #ax.plot(pp,'rx')
            ax.set(xlabel='Index',ylabel=nn,title="Plot of {}".format(nn))
            f.savefig(os.path.join(fpath,"{}-{}.png".format(fname,nn)))
        else:
            print(nn," skipping!")
    plt.close(f)

def save_data(fpath,data):
    for nn in data.dtype.names:
        # get data
        pp = data[nn]
        # check if data is all NaNs
        if not np.all(np.isnan(pp)):
            print(nn)
            # save data to file
            np.savetxt(os.path.join(fpath,"{}-{}.csv".format(fname,nn)),pp)

# call function to iterate over names in structured data type, produce and save plots
# x data is the index
gen_plots(plot_folder,data)
# create folder to store data in
data_folder = os.path.join(plot_folder,"Data")
os.makedirs(data_folder,exist_ok=True)
# save data as csvs for future processing
save_data(data_folder,data)

print("Generating 3d axis plot")
f3d = plt.figure()
ax3d = f3d.add_subplot(111,projection='3d')
x = data['xAxis']
y = data['yAxis']
z = data['zAxis']
ax3d.plot(x,y,z,'b-')
#ax3d.plot(x,y,z,'rx')
ax3d.set(xlabel='X',ylabel='Y',zlabel='Z',title='Dataset Axis Plot')
f3d.savefig(os.path.join(plot_folder,'axis-3d-plot.png'))

################
print("Moving onto next file")
path = 'D:\\BEAM\\Trial1_14.10.19-20191101T160322Z-001\\Trial1_14.10.19\\tree4_2_CSV.csv'
fname = os.path.splitext(os.path.basename(path))[0]
# read in file and determine data type automatically
data = np.genfromtxt(path,delimiter=',',dtype=None,skip_header=10)
# generate folder path for results
plot_folder = "{}-Plots".format(fname.replace(" ",'-'))
# create folder
os.makedirs(plot_folder,exist_ok=True)
# generate plots by iterating over data
gen_plots(plot_folder,data)

# create folder to store data in
data_folder = os.path.join(plot_folder,"Data")
os.makedirs(data_folder,exist_ok=True)
# save data as csvs for future processing
save_data(data_folder,data)

print("Generating 3d plot")
ax3d.clear()
x = data['f3']
y = data['f4']
z = data['f5']
ax3d.plot(x,y,z,'b-')
ax3d.plot(x,y,z,'rx')
ax3d.set(xlabel='X',ylabel='Y',zlabel='Z',title='Dataset Axis Plot')
f3d.savefig(os.path.join(plot_folder,'axis-3d-plot.png'))

###############
print("Moving onto next file")
path = 'D:\\BEAM\\Trial1_14.10.19-20191101T160322Z-001\\Trial1_14.10.19\\9welds_2_csv.csv'
fname = os.path.splitext(os.path.basename(path))[0]
# read in file and determine data type automatically
data = np.genfromtxt(path,delimiter=',',dtype=None,skip_header=10)
# generate folder path for results
plot_folder = "{}-Plots".format(fname.replace(" ",'-'))
# create folder
os.makedirs(plot_folder,exist_ok=True)
# generate plots by iterating over data
gen_plots(plot_folder,data)
# create folder to store data in
data_folder = os.path.join(plot_folder,"Data")
os.makedirs(data_folder,exist_ok=True)
# save data as csvs for future processing
save_data(data_folder,data)

print("Generating 3d plot")
ax3d.clear()
x = data['f3']
y = data['f4']
z = data['f5']
ax3d.plot(x,y,z,'b-')
ax3d.plot(x,y,z,'rx')
ax3d.set(xlabel='X',ylabel='Y',zlabel='Z',title='Dataset Axis Plot')
f3d.savefig(os.path.join(plot_folder,'axis-3d-plot.png'))

#################
print("Parsing laser log")
ll = []
path = r"D:\BEAM\Trial1_14.10.19-20191101T160322Z-001\Trial1_14.10.19\LaserNet Log.txt"
with open(path,'r') as file:
    while True:
        line = file.readline()
        # if a line was not retrieved, exit from loop
        if not line:
            break
        ll.append(line)

# it's known that there are two lines of stars separating the header 
i = 0
# iterate over the lines in list
for li,l in enumerate(reversed(ll)):
    # if the string in line is just stars
    if l == ((len(l)-1)*'*')+'\n':
        break

# index of first non star line
# headers for the data columns
li = len(ll)-li

## generate better header string
# known headers before hand
# from earlier in the document
col_headers = {'[1]':'Laser ON',
'[2]':'Emission',
'[3]':'Set Power',
'[4]':'Power (kW)',
'[5]':'Temperature (mean)'}
# remove newline character and split into components
header = ll[li].rstrip().split()

for hi,hh in enumerate(header):
    # if there's a corresponding replacement in the dictionary
    # replace entry in header with corresponding string
    if hh in col_headers.keys():
        header[hi] = col_headers[hh]

# convert rest of data to float values
data = []
dtime = []
for l in ll[li+1:]:
    # remove the new line character from string, rstrip
    # split result into parts, split
    # get last 5 parts (known float values),[-5:]
    # reform list by converting each value to float
    # append list of floats to list
    data.append([float(x) for x in l.rstrip().split()[-5:]])
    # add time to list
    dtime.append(datetime.strptime(l.rstrip().split()[1],'%H:%M:%S.%f'))

# update time vector as relative to first time stamp
time = np.array([(dd-dtime[0]).total_seconds() for dd in dtime])

# convert to array
data = np.asarray(data,dtype='float')
# generate folder for results
fname = os.path.splitext(os.path.basename(path))[0]
# generate folder path for results
plot_folder = "{}-Plots".format(fname.replace(" ",'-'))
# create folder
os.makedirs(plot_folder,exist_ok=True)

# save data as separate CSVs
# create folder to store data in
data_folder = os.path.join(plot_folder,"Data")
os.makedirs(data_folder,exist_ok=True)

f,ax = plt.subplots()
for ii in range(data.shape[1]):
    print(header[-5+ii])
    ax.clear()
    # plot column
    ax.plot(time,data[:,ii],'b-')
    # set labels to modified column header
    ax.set(xlabel='Time (s)',ylabel=header[-5+ii],title='Plot of {}'.format(header[-5+ii]))
    # save plot where the filename is the header with the spaces replaced with dashes and remove brackets
    f.savefig(os.path.join(plot_folder,"{}-{}.png".format(fname,header[-5+ii].replace(' ','-').replace(')','').replace('(',''))))
    # save time and data a 2 column dataset
    np.savetxt(os.path.join(data_folder,"{}-{}.csv".format(fname,header[-5+ii].replace(' ','-').replace(')','').replace('(',''))),np.vstack((time,data[:,ii])),delimiter=',')

    
