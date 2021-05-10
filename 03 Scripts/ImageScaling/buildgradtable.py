import h5py
import numpy as np
from numpy.polynomial.polynomial import polyfit
import os
from pathlib import Path
import csv
import matplotlib.pyplot as plt

def buildVarPlotLabel(vname):
    if vname=="max":
        return "Difference in Maximum Temperature (C)"
    if vname=="min":
        return "Difference in Minimum Temperature (C)"
    if vname=="var":
        return "Difference in Temperature Variance (C)"
    if vname=="range":
        return "Difference in Temperature Range (C)"
        
def plotFormatModelName(mname):
    # split by underscore
    parts = mname.split('_')
    # uppercase first character
    parts = [p.capitalize() for p in parts]
    #join results back together
    return ' '.join(parts)
    
# vnameorder
vname_order = None
dtype_order=None
# path to hdf5 file
path = r"D:\BEAM\Scripts\CallWaifu2x\arrowshape-temperature-HDF5-denoise-metrics.hdf5"
# table of results
grad_table=None
## factor matrix for dfferent ops
# denoise
factor = np.arange(0,4,1)
# scale
## factor = np.arange(2,11,1)
# generate plots flag
gen_plots = True
if gen_plots:
    f,ax = plt.subplots()
    if factor.max()==3:
        factor_label='Denoising Factor'
    elif factor.max()==10:
        factor_label='Scaling Factor'
    
# open file in read mode
print("Opening file...")
with h5py.File(path,'r') as file:
    print("File structure")
    print(list(file.keys()))
    # get order of data types
    dtype_order = [d for d in file["dtype-order"][...]]
    # the model names are the names of the high level groups
    # used for order in table
    models = list(file.keys())
    models.remove("dtype-order")
    # iterate through each model in the pack
    print("Iterating through models...")
    for mi,m in enumerate(models):
        # get name of the datasets i.e. variable names
        if vname_order is None:
            vname_order = list(file[m].keys())
        # iterate through each dataset
        for vi,v in enumerate(vname_order):
            # if gradient table has not been created yet, make it
            # can't build without knowing size of dataset
            if grad_table is None:
                grad_table = np.zeros((len(models),4*len(dtype_order)))
            # get data
            data = file[m][v]
            # iterate through data slices i.e. dtype
            for di in range(data.shape[2]):
                #take the average of the metric for each factor
                avg = data[:,:,di].mean(axis=1)
                # fit 1d poly against it and get gradient coeff
                pp = polyfit(factor,avg,deg=1)
                grad = pp[-1]
                # update table with gradient value
                grad_table[mi,(4*di)+vi]=grad
                if gen_plots:
                    ax.clear()
                    # plot original data and line plot
                    ax.plot(factor,data[:,:,di],'rx')
                            # form poly using coefficients
                    ax.plot(factor,np.polyval(pp[::-1],avg),'bo-')
                    # assign axes labels
                    ax.set(xlabel=factor_label,ylabel=buildVarPlotLabel(v))
                    # assign legends
                    ax.legend(['Data','Fitted 1D'])
                    # assign title
                    f.suptitle('{0} vs {1} \nData and Fitted 1D Polynomial for {2}'.format(factor_label,buildVarPlotLabel(v),plotFormatModelName(m)))
                    # save plot locally
                    # labels are formatted to be used in filename
                    f.savefig('fitted-1d-poly-{0}-{1}-{2}.png'.format(m.replace('_','-'),factor_label.lower().replace(' ','-'),
                                                                    buildVarPlotLabel(v.lower()).lower().replace(' ','-')))
                    
print("size of gradient table : {}".format(str(grad_table.shape)))
# create file name for csv
print("Writing table to csv file...")
fname = os.path.splitext(os.path.basename(path))[0]
with open("{}-grad-table.csv".format(fname),mode='w') as file:
    # write dtype header to file
    file.write(",,{}\n".format(',,,,'.join(dtype_order)))
    # write variable header to file
    file.write(",{}\n".format(','.join(vname_order*len(dtype_order))))
    writer = csv.writer(file)
    # write data to file
    for r in range(grad_table.shape[0]):
        # write model name
        file.write("{},".format(models[r]))
        # write model data row
        writer.writerow(grad_table[r,:])
