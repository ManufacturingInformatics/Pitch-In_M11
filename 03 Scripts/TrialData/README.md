# Parse Trial Data

The BEAM machine contains a range of sensors that record environmental information about the build space, a log of the communication with the laser module and information about the laser head.

The environmental sensors record the following:
 - Oxygen Concentration
 - Hydrogen Concentration
 - Pressure in Box
 - Pressure in Antechamber
 - Mode
 - Process Loops since Start
 - Light Tower (?)
 - Logging Time
 - Input oxygen
 - Inert gas controls signals
 - Fan speed
 
The laser sensors record the following:
  - Laser start/stop signals
  - Laser power set point
  - Actual laser power
  - Mean Temperature of the laser
  - Laser head position (X,Y,Z)
  - Laser head velocity (X,Y,Z)
  
As there is no documentation as present to describe the exact meaning of some of these parameters, the information is processed and plotted based on their name in the file.

The data is stored is a mixture of XML, CSV and MPF files. They have been processed and separated into separate CSV files for easier post processing.

The original data files can be found in the top level [OriginalData](OriginalData) folder.

## Software

The file [parse_beam_data_mixed](parse_beam_data_mixed.py) is a Python script for processing the formatted CSV files produced by the machine.

The MPF files can be read or imported as text files. To process the XML files see [BEAMXMLApi](../BEAMXMLApi) on how to process them.

## Software Requirements
  - Python 3
  - Numpy
  - Matplotlib
  - XML
  - MPL Toolkits (for Axes3D)

## Data

This folder contains the logged data for two builds; a series of parallel tracks and an arrow shape

 - [tree4_2_CSV-Plots](tree4_2_CSV-Plots) contains the processed data for the arrow shape. It's mostly the laser head position data and was extracted from the XML file.
 - [BeAMON-Contains-Tree-and-9welds-Plots](BeAMON-Contains-Tree-and-9welds-Plots) contains the environmental data for the tree and tracks builds.
 - [9welds_2_csv-Plots](9welds_2_csv-Plots) contains the laser head position data for the parallel tracks build.
 - [LaserNet-Log-Plots](LaserNet-Log-Plots) contains the plots and parsed laser log data. Not sure for which build.
