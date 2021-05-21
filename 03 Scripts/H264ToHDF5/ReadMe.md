# Convert target H264 File to HDF5

This program is a wrapped call of readH264 that writes the data to a single dataset HDF5 file. The dataset name is based off the input path name. Users can specify the name of the output file but if they don't, the output filename is the input file name appended with HDF5 on the end.

The purpose of this script is for instances where the user wants to write the data to a separate HDF5 file.

## Requirements
 - Numpy
 - H5py
 - Scipy
 
## Scripts

**h264toh5py.py**
 - Function to convert a target h264 file to a compressed HDF5 file and example of use.
 - H264ToHDF5(inpath,flag='mask',**kwargs) => None
      - inpath : Path of the H264 file
      - outpath : Path the HDF5 file will be written to
      - dsetname : Name of the dataset.
      - compress : Method to compress the file.
      - copts : Compression options. See H5Py doc.
      - flag : Flag passed to readH264. Read doc. Default: 'mask'
      - toC : Convert temperature data to Celcius. Requires toTemp to be set.
      - toTemp : Convert the data to temperature Kelvin using Temp2IApprox. Requires e and T0 values
      - e : Emissivity.
      - T0 : Room temperature in Kelvin
 - The function is argument passing following by calls to readH264, Qr2Temp and finally a context managed write to a HDF5 file.

The function has been built to be general purpose and can be applied to any h264 file produced by a thermal camera.
## Notes 
 - If the user wants the values to be converted to temperature, the flag toTemp must be set and the appropriate emissivity and room temperature values must be provided.

 - If outpath is None then the file is written to current cwd and is based off the H264 file name.

 - If dsetname is None, it is set to the name of the input file.

 - By default the data is not compressed (False). If the flag is set to True then it is compressed using gzip. The user can specify other compression filters as well. The only other compressor installed by default with H5py is 'lzf'.

 - The compression level for gzip can be set by providing a number from 0-9 to the copts argument. Default level used is 4.

 - Before setting custom compression settings, it is advised that the user reads the [documentation](http://docs.h5py.org/en/stable/high/dataset.html) on compressing datasets.
 
## Examples
### Basic example
```Python
H264ToHDF5(path)
```
### In depth example
```Python
H264ToHDF5(path,                 # path to h264 values
           toTemp=True,toC=True, # flags to convert values to temperature and to celcius
           e=e_ss,T0=T0,         # req values to convert to temperature
           outpath="arrowshape-temperature-HDF5.hdf5", # name of the output file
           compress=True)        # flag to compress the data using gzip

```
