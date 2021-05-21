import numpy as np
def readXMLData(path):
    import xml.etree.ElementTree as ET
    """ Reads data from the given laser machine XML file

        =====ONLY READS FIRST DATA FRAME=====
        Typically only one frame in the file anyway

        path : file path to xml file

        Returns information as a list in the order
        header,time,torque,vel,x,y,z

        header : header information from the Data Frame in XML file
            time   : Sampling time vector for the values
            torque : Current applied to the motor to generate torque (A)
            vel    : motor velocity (mm/s)
            x      : motor position along x-axis (mm)
            y      : motor position along y-axis (mm)
            z      : motor position along z=axis (mm)
            
        time,torque,vel,x,y,z are lists of data

        header is a dictionary with the following entries:
            - "Date-Time" : Datetime stamp of the first Data Frame
            
            - "data-signal-0" : Dictionary of information about torque
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-1" : Dictionary of information about velocity
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-2" : Dictionary of information about x
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-3" : Dictionary of information about y
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-4" : Dictionary of information about z
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
    """
    # parse xml file to a tree structure
    tree = ET.parse(path)
    # get the root/ beginning of the tree
    #root  = tree.getroot()
    # get all the trace data
    log = tree.findall("traceData")
    # get the number of data frames/sets of recordings
    log_length = len(log)

    # data sets to record
    x = []
    y = []
    z = []
    torque = []
    vel = []
    time = []
    header = {}
    
    ## read in log data
    # for each traceData in log
    for f in log:
        # only getting first data frame as it is known that there is only one frame
        # get the header data for data frame
        head = f[0].findall("frameHeader")

        # get date-timestamp for file
        ### SYNTAX NEEDS TO BE MODIFIED ###
        temp = head[0].find("startTime")
        if temp == None:
            header["Date-Time"] = "Unknown"
        else:
            header["Date-Time"] = temp
            
        # get information about data signals
        head = f[0].findall("dataSignal")
        # iterate through each one
        for hi,h in enumerate(head):
            # create entry for each data signal recorded
            # order arranged to line up with non-time data signals
            header["data-signal-{:d}".format(hi)] = {"interval": h.get("interval"),
                                                     "description": h.get("description"),
                                                     "count":h.get("datapointCount"),
                                                     "unitType":h.get("unitsType")}
            # update unitType to something meaningful                            
            if header["data-signal-{:d}".format(hi)]["unitType"] == 'posn':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'millimetres'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'velo':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'mm/s'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'current':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'Amps'
                                
        # get all recordings in data frame
        rec = f[0].findall("rec")
        # parse data and separate it into respective lists
        for ri,r in enumerate(rec):
            # get timestamp value
            time.append(float(r.get("time")))
            
            # get torque value
            f1 = r.get("f1")
            # if there is no torque value, then it hasn't changed
            if f1==None:
                # if there is a previous value, use it
                if ri>0:
                    torque.append(float(torque[ri-1]))
            else:
                torque.append(float(f1))

            # get vel value
            f2 = r.get("f2")
            # if there is no torque value, then it hasn't changed
            if f2==None:
                # if there is a previous value, use it
                if ri>0:
                    vel.append(float(vel[ri-1]))
            else:
                vel.append(float(f2))

            # get pos1 value
            val = r.get("f3")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    x.append(float(x[ri-1]))
            else:
                x.append(float(val))

            # get pos2 value
            val = r.get("f4")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    y.append(float(y[ri-1]))
            else:
                y.append(float(val))

            # get pos3 value
            val = r.get("f5")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    z.append(float(z[ri-1]))
            else:
                z.append(float(val))

    return header,time,torque,vel,x,y,z

    

def readXMLDataDict(path):
    import xml.etree.ElementTree as ET
    """ Reads data from the given laser machine XML file

        =====ONLY READS FIRST DATA FRAME=====
        Typically only one frame in the file anyway

        path : file path to xml file

        Returns the information as a dictionary with the keywords:
            header : header information from the Data Frame in XML file
            time   : Sampling time vector for the values
            torque : Current applied to the motor to generate torque (A)
            vel    : motor velocity (mm/s)
            x      : motor position along x-axis (mm)
            y      : motor position along y-axis (mm)
            z      : motor position along z=axis (mm)

        time,torque,vel,x,y,z are lists of data

        header is a dictionary with the following entries:
            - "Date-Time" : Datetime stamp of the first Data Frame
            
            - "data-signal-0" : Dictionary of information about torque
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-1" : Dictionary of information about velocity
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-2" : Dictionary of information about x
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-3" : Dictionary of information about y
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-0" : Dictionary of information about z
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
    """
    # parse xml file to a tree structure
    tree = ET.parse(path)
    # get the root/ beginning of the tree
    #root  = tree.getroot()
    # get all the trace data
    log = tree.findall("traceData")
    # get the number of data frames/sets of recordings
    log_length = len(log)

    # data sets to record
    x = []
    y = []
    z = []
    torque = []
    vel = []
    time = []
    header = {}
    
    ## read in log data
    # for each traceData in log
    for f in log:
        # only getting first data frame as it is known that there is only one frame
        # get the header data for data frame
        head = f[0].findall("frameHeader")

        # get date-timestamp for file
        ### SYNTAX NEEDS TO BE MODIFIED ###
        temp = head[0].find("startTime")
        if temp == None:
            header["Date-Time"] = "Unknown"
        else:
            header["Date-Time"] = temp
            
        # get information about data signals
        head = f[0].findall("dataSignal")
        # iterate through each one
        for hi,h in enumerate(head):
            # create entry for each data signal recorded
            # order arranged to line up with non-time data signals
            header["data-signal-{:d}".format(hi)] = {"interval": h.get("interval"),
                                                     "description": h.get("description"),
                                                     "count":h.get("datapointCount"),
                                                     "unitType":h.get("unitsType")}
            # update unitType to something meaningful                            
            if header["data-signal-{:d}".format(hi)]["unitType"] == 'posn':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'millimetres'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'velo':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'mm/s'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'current':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'Amps'
                                
        # get all recordings in data frame
        rec = f[0].findall("rec")
        # parse data and separate it into respective lists
        for ri,r in enumerate(rec):
            # get timestamp value
            time.append(float(r.get("time")))
            
            # get torque value
            f1 = r.get("f1")
            # if there is no torque value, then it hasn't changed
            if f1==None:
                # if there is a previous value, use it
                if ri>0:
                    torque.append(float(torque[ri-1]))
            else:
                torque.append(float(f1))

            # get vel value
            f2 = r.get("f2")
            # if there is no torque value, then it hasn't changed
            if f2==None:
                # if there is a previous value, use it
                if ri>0:
                    vel.append(float(vel[ri-1]))
            else:
                vel.append(float(f2))

            # get pos1 value
            val = r.get("f3")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    x.append(float(x[ri-1]))
            else:
                x.append(float(val))

            # get pos2 value
            val = r.get("f4")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    y.append(float(y[ri-1]))
            else:
                y.append(float(val))

            # get pos3 value
            val = r.get("f5")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    z.append(float(z[ri-1]))
            else:
                z.append(float(val))

    # construct and return items as a dictionaru
    return {"header":header,"time":time,"torque":torque,"vel":vel,"x":x,"y":y,"z":z}
