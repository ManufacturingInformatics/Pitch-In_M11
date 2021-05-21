# LBAM Modelling API

This folder is for the modelling API bringing together functions and results into a easier to use library.

The API is broken into three different modules each of which can be imported separately.

## Requirements
 + [Python >=v3.6.2](https://www.python.org/downloads/)
 + [Numpy](https://www.numpy.org/)
 + [SciPy](https://www.scipy.org/)
 + [skimage](https://scikit-image.org/)
 + [OpenCV v4.x.x](https://opencv.org/) (LBAMExperimental Only)

## Modules

There are three modules that make up the API. Each module is designed so that it can be imported separately and has their own setup script.

## Bugs/Missing Features
 - Messy import statement
 - Predict power function currently does nothing

### LBAMMaterial

This module contains functions to construct the material paramter matricies and function objects used in LBAMModel.

** **Currently only has data for Stainless Steel 316L** **

All temperature values are in Kelvin and everything else is in appropriate SI units.

#### Functions

- CelciusToK(C) => float
    * Convert the given temperature from Celcius to Kelvin
    * It's just a simple addition.
    * Arguments
        * C : Temperature in Celcius
- KelvinToC(K) => float
    * Convert the given temperature from Kelvin to Celcius
    * It's just a simple addition
    * Arguments
        + K : Temperature in Kelvin
- genKdata(material) => numpy float array, numpy float array
    * Construct a matrix of thermal conductivity values and associated temperature values from raw values for a specific material.
    * If an unsupported material is specified, then None is returned.
    * The raw values were collected from several datasheets.
    * There are too few values to make it worth saving to a file and read it back in.
    * Arguments
        + material : String indicating which material the data should be generated for
- Kp(Ks,T,Tm,scale=0.01)
    * Convert the given thermal conductivity values for a solid material so they represent the the thermal conductivity of the powdered material.
    * All the values with a temperature value less than the given melting point are scaled by the factor scale.
    * Due to the powder being separate pieces of material rather than a single piece of material it is harder for heat to move through the material.
    * Arguments:
        + Ks : Numpy array of thermal conductivity values.
        + T : Numpy array of temperature values for Ks.
        + Tm : Melting point of the material the values.
        + scale : Factor to scale Ks values by for temperature values less than Tm.
- genCData(material) => numpy float array, numpy float array
    * Construct a matrix of specific heat capacity values and associated temperature values from raw values for a specific material.
    * If an unsupported material is specified, then None is returned.
    * The raw values were collected from several datasheets.
    * There are too few values to make it worth saving to a file and read it back in.
    * Arguments
        + material : String indicating which material the data should be generated for
- genEvData(material) => numpy float array, numpy float array
    * Construct a matrix of volumetric expansion factor values and associated temperature values from raw values for a specific material.
    * If an unsupported material is specified, then None is returned.
    * The raw values were collected from several datasheets.
    * Units are mm^3/mm^3.C
    * There are too few values to make it worth saving to a file and read it back in.
    * Arguments
        + material : String indicating which material the data should be generated for.
- genDensityData(Ev,material) => numpy float array, numpy float array
    * Construct a matrix of density values and associated temperature values from raw values for a specific material and information about the volumetric expansion factor.
    * The temperature range is based off the material and known data points about density.
    * If an unsupported material is specified, then None is returned.
    * The raw values were collected from several datasheets.
    * Units are mm^3/mm^3.C
    * There are too few values to make it worth saving to a file and read it back in.
    * Arguments
        + material : String indicating which material the data should be generated for.
- genThermalDiff(K,p,C,T) => numpy float array
    * Generate thermal diffusivity data based off thermal conductivity, specific heat capacity and density data.
    * The thermal diffusivity data is for solid material.
    * Arguments
        + K : Thermal conductivity spline object for solid material
        + p : Solid material density spline.
        + C : Specific heat capacity spline
        + T : Temperature data
- genThermalDiff_powder(K,p,C,T) => numpy float array
    * Generate thermal diffusivity data based off thermal conductivity, specific heat capacity and density data.
    * The thermal diffusivity data is for powdered material.
    * Calls Kp to convert the thermal conductivity data from solid material to powdern.
    * Arguments
        + K : Thermal conductivity spline object for solid material
        + p : Solid material density spline.
        + C : Specific heat capacity spline
        + T : Temperature data
- buildMaterialData(material) => float, scipy.spline, scipy.spline
    * Construct the material objects and data used in the modelling functions.
    * Calls a combination of the previous functions to generate the required information
    * Returns the emissivity value for the material, the spline for the thermal conductivity and the spline function for thermal diffusivity.
    * Spline functions are fitted cubic functions that can be used to create more values. They are used for generating thermal conductivity and diffusivity values for predicting temperature and power density.
    * If an unsupported material is specified, then None is returned.
    * Arguments
        + material : String indicating which material the data should be generated for.

### LBAMModel

This module contains functions t* read in process data and use material parameters to predict temperature, 
power density and power profile.

** **predictPower function returns an empty matrix of the same size as the given power density matrix. The ideal method has yet to be identified and will be left blank until one is found.** **

### LBAMExperimental

This module contains functions for other simulation or other processing methods investiagted during the project. These methods
are likely not as effective as the power profile function in LBAMModel.

### WriteDataToHDF5

This module contains script for writing data produced by this API to a HDF5 file with a predetermined structure. The in-progress version and complete documentation can be found [here](../WriteDataToHDF5/ReadMe.md). The supplied [setup](./WriteDataToHPF5/setup.py) file collects the necessary dependencies.

DEV NOTE: I'm aware that the directory for it is WriteDataToHP5. I'm aware of the typo. If it becomes a particular problem, I'll change it.

#### Compression

Each dataset is compressed using GZIP at level 9 compression unshuffled. This was found by reading in the arrow shape dataset, converting the values to temperature and power density then writing them as separate datasets to a file. Different compression methods tried. Uncompressed the file is around 4 GB in size. Below is a table showing the different compression methods and the resulting file size.

| Method | Size|
|:------:|:----:|
|GZIP Level 9 (Unshuffled)|1.4 GB|
|LSF (Unshuffled)|1.7 GB|
|SZIP (Unshuffled)|1.9 GB|
|GZIP Level 9 (Shuffled)|1.6 GB|
|LSF (Shuffled)|1.8 GB
|SZIP (Shuffled)|2.1 GB|

More information about the supported compression methods can be found [here](http://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline). The compression options are currently hardcoded for the purposes of convinience but can be repacked using the [h5repack](https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack) utility supplied by H5PY when installed.

### Examples of Use
#### Read in camera footage and convert to temperature and power density
```Python
import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model

## Setup variables and material parameters
# temperature of the space where the data was recorded
# typically set to 23 degrees C
# has to be converted to Kelvin to be used in the modelling functions
T0 = mat.CelciusToK(23.0)

# Create the data and objects for stainless steel 316L
# e : emissivity of the chosen material
# K : Spline function for thermal conductivity
# D : Spline function for thermal diffusivity
e,K,D = mat.buildMaterialData("SS-316L")

# Read in and process the thermal camera footage stored in the file video.h264 file
# it returns a numpy array representing the radiative heat values recorded by the 
# thermal camera
# inside the BEAM machine
Qr = model.readH264("video.h264")

## Run model to predict data
# Predict the temperature based off the thermal camera data
# Temperature is in Kelvin
# The vectorize flag decides whether to use Numpy's ability to vectorize functions
# The option was added as weak machines can easily freeze/crash when there's a large 
# volume of data to process
# When False, it processes the data in a for loop instead
T = model.predictTemperature(Qr,e,T0,vectorize=True)

# Predict the power density based off the predicted temperature
# This function does not have a vectorize flag as it cannot be vectorized at the moment
I = model.predictPowerDensity(T,K,D,T0)

# Predict laser power
# The function is not very well optimized so will take a while to run
# pixel_pitch is the physical height and width of each pixel in the thermal camera footage
P = model.predictPower(I)
```

It is up to the user to decide how to write the data to file or plot it.

#### Read and convert camera footage to temperature and power density then write to HDF5 file along with XML data
```Python
import LBAMMaterials.LBAMMaterials as mat
import LBAMModel.LBAMModel as model
import WriteDataToHPF5.beamwritedatatohdf5 as hd

## Setup variables and material parameters
T0 = mat.CelciusToK(23.0)

# Create the data and objects for stainless steel 316Lvity
e,K,D = mat.buildMaterialData("SS-316L")

# Read in camera footage
Qr = model.readH264("video.h264")

## Run model to predict temperature, power density and laser power
T = model.predictTemperature(Qr,e,T0,vectorize=True)
I = model.predictPowerDensity(T,K,D,T0)
P = model.predictPower(I)

# read in XML data as a tuple of datasets
xml_data = hd.readXMLData("laser_head.xml")

# create the file by specifying the FULL path of where it will be created
# initializes high level groups in the file according to the file structure
with hd.initialize("run-data.hdf5") as data_f:
    ## write the data to the file using updateData method
    # createNew flag indicates that a new run group should be created to hold this data
    # Specific runs can also be stated as well using run_num keyword
    # returns a flag indicating if the update was successful
    # and the group object to the newly created run group
    hd.updateData(data_f,createNew=True,Qr=Qr,T=T,I=I,xml=xml_data)
```
