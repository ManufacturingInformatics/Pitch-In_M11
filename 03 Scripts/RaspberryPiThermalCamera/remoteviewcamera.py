import paramiko
import os
import traceback
import select
import numpy as np
import cv2

ips = ["192.168.1.46"]
password = ["dTtrV3L3R"]

print("Attempting to copy files across")
# if flag is set, start logging in local directory
paramiko.util.log_to_file('./ssg_log.txt')

# get current directory
cwd = os.getcwd()

# objects created for connection
# created outside of the loop so it can be closed in the event of an exception
t = None
sftp = None
client = None

# how long to run the program for
time = 60

# timeout for waiting for data
timeout=60000

# scale factor for image
scale = 10

# temperature limits on camera, used for normalizing
temp_min = -40
temp_max = 300

# flags for indicating status
gotData = False
gotError = False

# iterate through each ip address
try:
    # setup client
    client = paramiko.SSHClient()
    # load host keys
    client.load_system_host_keys()
    # authorize SSH connection using login information
    client.connect(ips[0],port=22,username="pi",password=passwords[0])
    # send command to run rawval program
    stdin,stdout,stderr = client.exec_command("sudo ./home/pi/mlx90640-library-master/examples/rawval")
    # get channel object for stdout
    channel = stdout.channel
    # as we're just reading stdout, we can close stdin in this case
    stdin.close()
    # while the channel is not closed and there's data to be read
    while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
        # clear flags
        gotData = False
        gotError = False
        # poll stdout to see if there is any data to read
        readq, _, _ = select.select([stdout.channel],[],[],timeout)
        # iterate through each of the objects returned
        for c in readq:
            # if there's data, read it into a numpy array
            if c.recv_ready():
                # read in one frame of raw data
                if len(c.in_buffer)<3072:
                    print("Not enough data in buffer, only {0} bytes".format(len(c.in_buffer)))
                    gotData=False
                else:
                    data = stdout.channel.recv(3072)
                    # convert to frame of float values
                    frame = np.resize(np.frombuffer(data,dtype='float16'),(24,32))
                    ## display image
                    # normalize data, [-40,300] => [0,1]
                    frame_norm = ((frame-temp_min)/(temp_max-temp_min))
                    # scale to 8-bit image
                    frame_norm (frame_norm * 255).astype('uint8')
                    # scale image size
                    frame_norm = cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                    # apply colormap
                    frame_col = cv2.applyColorMap(frame_norm,cv2.COLORMAP_HOT)
                    # update displayed image
                    cv2.imshow("remote view",frame_col)
                    gotData = True
            # if there's an error message to be read
            if c.recv_stderr_ready():
                #print contents of stderr buffer to screen
                print(str(stderr.channel.recv_stderr(len(c.in_stderr_buffer))))
                gotError=True

        # if failed to get data OR recieved an error
        if (not gotData or gotError) and \
            # if the remote process has exited and returned an exit status
            stdout.channel.exit_status_ready() and \
            # if there's no error message to read from the buffer 
            not stderr.channel.recv_stderr_ready() and \
            # if there's no data to read from the file
            not stdout.channel.recv_ready():

            # print error code
            print("Got error code {0}".format(stdout.channel.recv_exit_status()))

            # send ctrl_c to stop program in case it is still running
            # the program has support for handling interrupt signal
            #client.exec_command(chr(3))
            # indicate that we are going to stop reading from stdout
            stdout.channel.shutdown_read()
            # close channel
            stdout.channel.close()
            break
                
    # close connections
    stdout.close()
    stderr.close()
    t.close()
    ssh.close()
# in the event of any exception, ensure that the connections are closed
except Exception as e:
    # print traceback of exception
    traceback.print_exc()
    # if transport object was created, close it
    if t != None:
        t.close()
    # if client
    
