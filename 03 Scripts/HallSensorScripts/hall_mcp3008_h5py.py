import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import time
import h5py
from datetime import datetime

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the chip select
cs = digitalio.DigitalInOut(board.D5)

# create the mcp object
mcp = MCP.MCP3008(spi, cs)

# create an analog input channel on pin 0
chan = AnalogIn(mcp, MCP.P0)

# sampling rate for hall sensor
f = 60.0 # hz

def VtoG(V0,V,S):
    """ Convert the voltage received by the Hall Sensor to field strength in Gauss
        
        V0 : Voltage when there are no magnets nearby, mV
        V : Voltage recorded, mV
        S : Sensitivity of the sensor, mV/Gauss
        
        Uses the sensor's sensitivity (from datasheet) to convert recorded voltage to field strength.
        
        Returns field strength in Gauss 
    """
    return (V0-V)/S

# voltage of sensor when there are no magnets nearby
V0 = 3.0 # mV
# sensitivity of the sensor
S = 1.5 # mv/G

## create HDF5 file to write to
filename = "hall-sensor-{}.hdf5".format(datetime.now().isoformat().replace(':','-').replace('.','-',1))
# loop until keyboard interrupt
with h5py.File(filename,'w') as file:
    try:
        print("Creating dataset")
        ## create resizeable datasets
        # voltage
        vset = file.create_dataset("Voltage",(1,2),maxshape=(None,2),dtype=type(chan.voltage),compression='gzip',compression_opt=9)
        # field strength
        gset = file.create_dataset("Gauss",(1,2),maxshape=(None,2),dtype=type(chan.voltage),compression='gzip',compression_opt=9)
        # record start time
        # time.clock records CPU time spent on currrent process
        startt = time.clock()
        while True: 
            # read in value
            v = chan.voltage
            print(str(v))
            # log value
            t = time.clock()-startt
            vset[-1,:] = [v,t]
            gset[-1,:] = [VtoG(V0,v,S),t]
            # increase datasets by 1 row
            vset.resize((vset.shape[0]+1,2))
            gset.resize((gset.shape[0]+1,2))
            # sleep for 1 second
            time.sleep(1.0/f)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    print("End shape: {}".format(dset.shape))
print("Program runtime: {} secs".format(startt-time.clock()))