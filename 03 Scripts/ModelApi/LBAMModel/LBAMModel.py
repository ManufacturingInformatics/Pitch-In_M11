import numpy as np

def readH264(path,flag='mask'):
    """ Read in image data from target H264 video file as float16
    
     path : target file path
     flag : Flag indicating how to process the data, default='mask' means the data is masked according to the way
                the values are stored in H264 by BEAM's thermal camera.

        Size of the returned array is currently hard-coded to 128x128.
        
        Returns a 128x128 array of radiative heat values in terms of Watts.
    """
    # known size of the images
    rows = 128
    cols = 128

    # read in raw bytes as a 1D array
    arr = np.fromfile(path,dtype='uint16')

    if flag=='mask':
        ## update values based on code
        # get code
        code_array = np.bitwise_and(arr,0xF000)
        # CODE_VAL_SEUIL2
        arr[code_array==0xD000] = 0xF800
        # CODE_VAL_CONTOUR
        arr[code_array==0xB000] = 0xF81F
        # CODE_VAL_MAX
        arr[code_array==0xC000] = 0x0000
        # CODE_VAL_SEUIL1
        arr[code_array==0xE000] = 0x001F

    ## just lower 12-bits
    arr = np.bitwise_and(arr,0x0FFF)

    ## convert data to frames
    # break the data into chunks that are 1d frames
    frames_set = np.split(arr,int(arr.shape[0]/(rows*cols)))
    # combined frames together into a 3d array and interpret as float16 data type
    return np.dstack([np.reshape(f,(rows,cols)) for f in frames_set]).astype('float16')

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


# function for converting radiative heat to temperature
def Qr2Temp(qr,e,T0):
    """ Function for converting radiative heat to surface temperature

        qr : radiative heat W/m2
        e  : emissivity of the material [0.0 - 1.0]
        T0 : starting temperature, K

        Returns numpy array surface temperature in Kelvin using Stefan-Boltzman theory

        qr = e*sigma*(T^4 - T0^4)
        => T = ((qr/(e*sigma))**4.0 + T0**4.0)**0.25
    """
    from scipy.constants import Stefan_Boltzmann as sigma
    from numpy import asarray, nan_to_num,full
    # if emissivity is 0.0, return T0
    # result of setting qr/(e*sigma)
    # avoids attempt to divide by 0
    if e==0.0:
        #print("emissivity is 0")
        # if T0 is a numpy array and then return T0 as is
        if type(T0)==numpy.ndarray:
            return T0
        # if T0 is a single value float
        elif type(T0)==float:
            # construct and return a matrix of that value
            return full(qr.shape,T0,qr.dtype)
    else:
        # calculate left hand portion of addition
        # due to e_ss*sigma being so small (~10^-8)
        div = (e*sigma)**-1.0
        #qr/(e*sigma)
        a = (qr*div)
        # T0^4
        #print("T0:",T0)
        b = asarray(T0,dtype=np.float32)**4.0
        #print("To^4:",b)
        # (qr/(e*sigma)) + T0^4
        c = a+b
        return (c)**0.25

def Temp2IApprox(T,T0,K,D,t=1.0/64.0):
    """ Function for converting temperature to power density

        K : thermal conductivity, spline fn
        D : thermal diffusivity, spline fn
        t : interaction time between T0 and T
        T : temperature K, numpy matrix
        T0: starter temperature K , constant

        Returns power density approximation ndarray and laser power estimate

        Approximation based on the assumption of uniform energy for the given area
    """
    # get numpy fn to interpret temperature as matrix and get pi constant
    from numpy import asarray, pi, sqrt, abs
    # temperature difference
    Tdiff = (T-T0)
    # thermal conductivity matrix
    K = K(Tdiff)
    # thermal diffusivity matrix
    D = D(Tdiff)
    # 2*sqrt(Dt/pi)
    a = ((D*t)/np.pi)
    # result of sqrt can be +/-
    # power density cannot be negative 
    b = (2.0*np.sqrt(a))
    temp = K*Tdiff
    # K*(T-T0)/(2*sqrt(Dt/pi))
    return abs(temp/b)

def predictTemperature(Qr,e,T0,vectorize=False):
    """ Predict temperature using the radiative heat data stored in Qr

        Qr : Radiative heat data recorded by thermal camera. If it is raw data (i.e. bytes),
            please re-read the file in using the readH264 function.
        e : Emissivity of the material being recorded.
        T0 : Temperature of the space at the start of the run.
        vectorize : Flag indicating whether or not to use Numpy vectorization to process data.
                    Underpowered machines may freeze or stall if flag is set. Default False

        Returns 64-bit float matrix of predicted temperature values of the same size as Qr
    """
    # if vectorize flag not set, process information in for loop
    if not vectorize:
        # create empty matrix values
        T = np.zeros(Qr.shape,dtype='float64')
        # iterate through each frame calling Qr2Temp to process each frame
        for ff in range(0,Qr.shape[2],1):
            T[:,:,ff] = Qr2Temp(Qr[:,:,ff],e,T0)
        # return processed matrix
        return T
    # if vectorize flag is set, pass entire matrix to Qr2Temp
    # Numpy's vectorization support will process this using matrix operations rather than
    # iterative for loop
    else:
        return Qr2Temp(Qr,e,T0)

def predictPowerDensity(T,K,D,T0,tc=1.0/64.0):
    """ Predict power density using the temperature matrix and built material spline objects.

        T : Temperature matrix, Kelvin
        K : Thermal conductivity spline
        D : Thermal diffusivity spline
        T0 : 'Room' temperature. Kelvin
        tc : Time between frames, secs

        Return 64-bit float matrix of power density values. Units: Watts per metre squared
    """
    # create empty matrix for results
    I = np.zeros(T.shape,dtype='float64')
    # process each frame of temperature 
    for ff in range(T.shape[2]):
        # if it's the first frame then the previous temperature is assumed to be the 'room' temperature
        # for every other frame, the previous temperature frame is used 
        if ff>0:
            I[:,:,ff] = Temp2IApprox(T[:,:,ff],T[:,:,ff-1],K,D,tc)
        else:
            T[:,:,ff] = Temp2IApprox(T[:,:,ff],T0,K,D,tc)
    return I

def predictPower(I,PP=20e-6):
    """ Predict laser power using the laser power density matrix

        I : Power density matrix, W/m2
        PP : Pixel pitch, m

        CURRENTLY RETURNS EMPTY MATRIX OF THE SAME TYPE AND SIZE OF POWER DENSITY MATRIX
    """
    return np.zeros(I.shape[2],I.dtype)
