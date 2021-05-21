import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

## setup argument parser
parser = argparse.ArgumentParser(description="Plot 3-col VTk mass properties data in the received CSV file. Assumes comma delimiter")
parser.add_argument("-f",help="File path",type=str,required=True)
parser.add_argument("--o",help="Where to put the plots. Default <filename \wo ext>-plots",type=str,action='store')
parser.add_argument("--superfig",help="Generate plot of all the data. Default False",type=bool,nargs='?',const=True,default=False,required=False)
parser.add_argument("--id_offset",help="Offset applied to index for x-axis. Default 91399",type=int,nargs='?',const=True,default=91399,required=False)

# parse arguments
args = parser.parse_args()
path = args.f
plot_path = args.o
gen_super = args.superfig
ii = args.id_offset

def shortenTitle(title,ll=4):
    return ' '.join([w[:ll] for w in title.split(' ')])

# labels for y axes
ylabels = ["Volume $(pixels^3)$","Projected Volume $(pixels^3)$","Surface Area $(pixels^2)$"]

# check if path is empty
# if it is, raise error
if (not path) or (not path.strip()):
    raise ValueError("Path is empty!")
# check to see if the path points towards a file
elif not os.path.isfile(path):
    raise ValueError("Path doesn't point towards a file!")
else:
    # extract filename without extension
    fname = os.path.splitext(os.path.basename(path))[0]
    # if a plot location wasn't given, inform user and generate location
    if (not plot_path) or (not plot_path.strip()):
        print("No output path specified! Building based off filename")
        # create path for plots based off
        plot_path = fname+"-plots"
        # create folder for results
        os.makedirs(plot_path,exist_ok=True)
    # create flag to make axes only in first loop
    ax_made = False
    # open file in read mode
    with open(path,'r') as file:
        # read in line counter and respective line
        # counter used as x data
        for dataln in file:
            # create in a line of data converting each element to float assuming comma delimiter
            dataln = [float(dd) for dd in dataln.strip().split(',')]
            # if the axes have not been created yet
            if not ax_made:
                # number of cols is number of floats
                num_cols = len(dataln)
                # create set of figures and plots
                f,ax = zip(*[plt.subplots() for i in range(num_cols)])
                # set labels
                for ai,aa in enumerate(ax):
                    # title is set as parsed version of ylabels set
                    aa.set(xlabel='Index',ylabel=ylabels[ai],title=f"Plot of {ylabels[ai].strip().split(' $')[0]} from\n{fname}")
                # create line objects to update
                # initialize with first data point at Index 0
                ln = [aa.plot(ii,dataln[ai])[0] for ai,aa in enumerate(ax)]
                # generate objects for super plot if flag is set
                if gen_super:
                    fs,axs = plt.subplots()
                    # create line objects to update
                    lnsuper = [axs.plot(ii,dataln[jj])[0] for jj in range(num_cols)]
                    # set legend
                    axs.legend([f"{ylabels[i]}" for i in range(num_cols)])
                    # set axes labels
                    # y labels 
                    axs.set(xlabel='Index',ylabel=f"{','.join(ylabels)}",title=f"Output of All Data in\n{fname}")
                # set flag to indicate that the axes have been created
                ax_made=True
                ii+=1
                # jump to next loop
                continue
            # if the axes have been created update them
            # update line obj li by appending a new x axis value and getting the li-th value from the data ln
            # ln is a list of lists hence ln obj
            for li,ll in enumerate(ln):
                ll.set_xdata(np.append(ll.get_xdata(),ii))
                ll.set_ydata(np.append(ll.get_ydata(),dataln[li]))
                # update super plot if necessary
                if gen_super:
                    lnsuper[li].set_xdata(np.append(lnsuper[li].get_xdata(),ii))
                    lnsuper[li].set_ydata(np.append(lnsuper[li].get_ydata(),dataln[li]))
            ii+=1
        print("Saving plots!")
        # save each figure to the set plot directory
        for ff,fig in enumerate(f):
            # recalculate limits
            ax[ff].relim()
            # auto scale
            ax[ff].autoscale_view(True,True,True)
            # redraw resu;ts
            fig.canvas.draw()
            # save fig
            fig.savefig(os.path.join(plot_path,f"col-{ff}-{fname}.png"))
        # save super plot
        if gen_super:
            print("Saving super plot!")
            axs.relim()
            axs.autoscale_view(True,True,True)
            fs.canvas.draw()
            fs.savefig(os.path.join(plot_path,f"super-plot-{fname}.png"))
    print(f"Finished plotting {num_cols} columns of data from {path} to {plot_path}")
    
                
