import h5py
import numpy as np
import os
from pathlib import Path
import fnmatch

def printFileStructure(item,leading='  '):
    """ Print the h5 or dictionary file structure

        Prints the keys of each item as an indented list.
        If the item is a mdataset, then the shape is printed.
        Level of indentation indicates level of depth in the tree
    """
    import h5py
    # code adapted from
    #https://stackoverflow.com/questions/34330283/how-to-differentiate-between-hdf5-mdatasets-and-groups-with-h5py
    for key in item:
        # if object is a mdataset print its name followed by shape
        if isinstance(item[key], h5py.mdataset):
            print(leading + key + ": " + str(item[key].shape))
            #get attributes
            atts = item[key].attrs.items()
            # if there are attributes
            if len(atts)>0:
                # print name and value
                # equals sign indicates its a attribute
                # prints it at a different indent level for ease of reading
                for k,v in atts:
                    print('  '+leading+k, "= ",v)
        # if it's something else
        else:
            print(leading + key)
            printFileStructure(item[key],leading=leading+'  ')

# path to upper directory where the images are stored
path = r"D:\BEAM\Scripts\CallWaifu2x\arrowshape-temperature-HDF5-denoise\Plots"
# base hdf5 name as parent directory
dir_name = Path(path).parts[-2]
#print(Path(path).parts)
# order of mdatatypes to save the information in
dtype_order=['uint8','uint16','float32','float64']
# open hdf5 file via context manager
with h5py.File('{}-metrics.hdf5'.format(dir_name),'w') as file:
    # add dtype order to file
    file.create_dataset("dtype-order",
                        shape=(len(dtype_order),),
                        # list has to be converted to numpy array to be added
                        # h5py datatypes mirror numpy datatypes so the appropriate dtype
                        # has to be declared
                        data=np.array(dtype_order,dtype=h5py.string_dtype(encoding="utf-8")),
                        dtype=h5py.string_dtype(encoding="utf-8"))
    # iterate through directories and files
    for root,dirs,files in os.walk(path):    
        # iterate through files matching known name format
        for ff in fnmatch.filter(files,"diff-*.csv"):
            # create/get group based off 2nd parent directory, i.e. model name
            grp = file.require_group(str(Path(root).parents[0].parts[-1]))
            # print current file being written and where it's being written to
            print("{0} : {1}                \r".format(grp.name,ff),end='')
            # get name of file without extension
            name = os.path.splitext(ff)[0]
            # get variable name of file
            # filenames are organised as diff-<name>-<model>-<dtype>.csv
            name_parts = name.split("-")
            vname = name_parts[1]
            dname = name_parts[-1]
            # read in mdata
            mdata = np.genfromtxt(os.path.join(root,ff),delimiter=',')
            # create or collect dataset
            dset = grp.require_dataset(vname,
                                       shape=(*mdata.shape,len(dtype_order)),
                                       compression="gzip",compression_opts=9,
                                       dtype=mdata.dtype)
            # update dataset based on desired order or datatypes
            # indexing ensures that the order along axis 2 matches
            # dtype order
            dset[:,:,dtype_order.index(dname)]=mdata
    print("End file structure")
    print(file.keys())
