import h5py
import numpy as np
import os
from pathlib import Path
import fnmatch
# path to hdf5 file
path = ""
# flag to mirror HDF5 structure as folder structure
mirror_tree = False
# open file in read mode
with h5py.File(path,'r') as file:
    # create folder for results
    fname = os.path.basename(path)
    name = os.path.splitext(fname)[0]
    os.makedirs(name,exist_ok=True)
    # get order of data types
    dtype_order = file["dtype-order"]
    # iterate through file
    for key,val in file.items():
        # if its a group
        if isinstance(val,h5py.Group):
            # if flag is set to mimic tree structure in folder structure
            # create folder
            if mirror_tree:
                os.makedirs(os.path.join(name,key),exist_ok=True)
            continue
        # if it's a dataset, unpack it
        elif isinstance(val,h5py.Dataset):
            # iterate through data
            for i in range(val.shape[2]):
                # if tree is being mirrored
                if mirror_tree:
                    # make directory for data type
                    fpath = os.path.join(name,key,dtype_order[i])
                    os.makedirs(fpath,exist_ok=True)
                    # break h5py file path into parts to get name and location
                    pparts = Path(val.name).parts
                    # reconstruct fiename
                    fname = "diff-{0}-{1}-{2}.csv".format(*pparts[:-3])
                else:
                    # if not mirroring tree, save inside results folder
                    fpath = name
                    pparts = Path(val.name).parts
                    fname = "diff-{0}-{1}-{2}.csv".format(*pparts[:-3])
                # save data slice
                np.savetxt(os.path.join(fpath,fname),val[:,:,i],delimiter=',')
            
