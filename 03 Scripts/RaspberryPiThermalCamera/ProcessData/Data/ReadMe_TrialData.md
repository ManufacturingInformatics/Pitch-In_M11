# Metadata and Logs from Trials

The [Trial1](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19) folder contains a collection of meta data and logs recorded by the BEAM machine during a run. The files have been parsed and plotted using the [parse_beam_data_mixed.py](../parse_beam_data_mixed.py) script and the plots placed in the [Plots](../Plots) folder. The script is designed to be used with other Trial datasets.

## Data
The files contain the following data:
 * ![9welds_2_csv.csv](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/9welds_2_csv.csv) and ![9welds_2_csv.xml](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/9welds_2_csv.xml)
   - Log file produced by the BEAM file
   - Contains the torque, velocity and laser head position (X,Y,Z).
   - The XML is the original log file and CSV is the exported version of the log file.
   - The CSV has a header describing what the data is based off the XML. It then has the data along with a time vector saved as separate columns.
   
 * ![BeAMON-Contains Tree and 9welds.csv](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/BeAMON-Contains%20Tree%20and%209welds.csv)
   - Contains information recorded inside the BEAM machine during a run.
   - Don't fully know what the information is
   - It contains information on:
     + Alarm status
     + Fan Speed
     + Inert Gas measurements (?)
     + Status of inert gas pump
     + Oxygen level PPM
     + Estimated laser power
     + Laser status
     + Pressure
     + Hydrogen levels
     + Laser head position
     + Log time
 * ![LaserNet Log.txt](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/LaserNet%20Log.txt)
   - Log of events that occured in the BEAM machine.
   - File is a formatted text file with the header file contained in rows of stars
   - The actual data is afterwards in columns with their own headers
   - Contains:
     + Date of recording
     + Time of recording in terms of HH:MM:SS.mmm
     + Laser ON flag
     + Laser emission flag
     + Target laser power
     + Estimated actual power
     + Mean temperature
 * ![9welds_2_csv.xml](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/9welds_2_csv.xml)
   - Log file produced by the BEAM file
   - Contains the torque, velocity and laser head position (X,Y,Z).
   - The XML is the original log file and CSV is the exported version of the log file.
   - The CSV has a header describing what the data is based off the XML. It then has the data along with a time vector saved as separate columns.
   
 * ![tree4_2_CSV.csv](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/tree4_2_CSV.csv) and ![tree4_2_CSV.xml](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/tree4_2_CSV.xml)
   - Log file produced by the BEAM file
   - Contains the torque, velocity and laser head position (X,Y,Z).
   - The XML is the original log file and CSV is the exported version of the log file.
   - The CSV has a header describing what the data is based off the XML. It then has the data along with a time vector saved as separate columns.
   
 * ![Tree_Steel_20layers.mpf](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/Tree_Steel_20layers.mpf) file 
   - Some type of script file describing how the part is being built.
   - Includes target values for build parameters such as layer thickness, power used on layers and number layers.
   
 * ![ACSE9WeldTracks.mpf](Trial1_14.10.19-20191101T160322Z-001/Trial1_14.10.19/ACSE9WeldTracks.mpf) file
   - Some type of script file describing how the part is being built.
   - Includes target values for build parameters such as layer thickness, power used on layers and number layers.
     
   
   
   
