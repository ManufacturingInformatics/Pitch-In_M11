# Pack Images into HDF5

The attached script iterates from a hard coded path, searches for all the PNGs and TIFs and copies them to a HDF5 file. The structure of the HDF5 file mirrors the directory tree from the starting point. It is assumed that all the images in a folder are the same size and data type. It reads them all in and stacks them together into a single dataset. Each dataset is compressed using gzip option.

To unpack the HDF5, users will need to iterate through and unpack the image stacks.
