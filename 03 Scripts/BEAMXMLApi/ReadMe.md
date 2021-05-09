# BEAMXMLApi

The BEAM machine's PLC produces XML trace files during operations logging the laser head's position, torque and velocity. While the machine's software has an option to export this data as a CSV, there are no functions for dealing with the XML directly for in program testing or processing.

This library will be a set of utility functions to simplify processing, handling and searching the XML files for additonal meta data.

The library currently does **NOT** have a .whl file.

## Requirements
 + Python v3+
 + xml
 
## Functions

 + readXMLData : Search the target XML file for time,position,torque and velocity data and return them as lists
 + readMXLDataDict : Search the target XML file for time,position,torque and velocity data and return them in a dictionary
