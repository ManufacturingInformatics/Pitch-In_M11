import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

## setup argument parser
parser = argparse.ArgumentParser(description="Plot each column of the received CSV file. Assumes comma delimiter")
parser.add_argument("-f",help="File path",type=str,required=True)
parser.add_argument("--o",help="Where to put the plots. Default <filename \wo ext>-plots",type=str,action='store')
parser.add_argument("--no_buffer",help="Do not load file contents into memory. Default False",type=bool,nargs='?',const=True,default=True,required=False)
parser.add_argument("--superfig",help="Generate plot of all the data. Default False",type=bool,nargs='?',const=True,default=False,required=False)

# parse arguments
args = parser.parse_args()
path = args.f
plot_path = args.o
no_buff = args.no_buffer
gen_super = args.superfig

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
    # if flag to not read into memory is not set
    if not no_buff:
        # read data into memory
        data = np.loadtxt(path,delimiter=',')
        # get shape
        num_cols = data.shape[1]
        # generate single axes
        f,ax = plt.subplots()
        # reuse axes to plot each column in turn
        print("Plotting data!")
        for i in range(num_cols):
            ax.clear()
            ax.plot(data[:,i])
            ax.set(xlabel='Index',ylabel='Output Units',title=f"Plot of Column {i} from\n{fname}")
            f.savefig(os.path.join(plot_path,f"col-{i}-{fname}.png"))
        print("Plotting super plot!")
        if gen_super:
            print("Generating super plot!")
            for i in range(num_cols):
                ax.clear()
                ax.plot(data[:,i],label=f"Col {i}")
            # add legend based on label
            ax.legend()
            # set axes labels
            ax.set(xlabel='Index',ylabel='Output Units',title=f"Output of All Data in\n{fname}")
            # save figure
            f.savefig(os.path.join(plot_path,f"super-plot-{fname}.png"))
    # if flag to not buffer data is set
    else:
        # create flag to make axes only in first loop
        ax_made = False
        ii=0
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
                        aa.set(xlabel='Index',ylabel=f"Column {ai}",title=f"Plot of Column {ai} from\n{fname}")
                    # create line objects to update
                    # initialize with first data point at Index 0
                    ln = [aa.plot(0,dataln[ai])[0] for ai,aa in enumerate(ax)]
                    # generate objects for super plot if flag is set
                    if gen_super:
                        fs,axs = plt.subplots()
                        # create line objects to update
                        lnsuper = [axs.plot(0,dataln[jj])[0] for jj in range(num_cols)]
                        # set legend
                        axs.legend([f"Col {i}" for i in range(num_cols)])
                        # set axes labels
                        axs.set(xlabel='Index',ylabel='Output Units',title=f"Output of All Data in\n{fname}")
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
    
                
