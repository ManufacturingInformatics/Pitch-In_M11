import fnmatch
import paramiko
import os
import traceback

def getHDF5FilesFromPi(ips,passwords,target_dir='/home/pi/mlx90640-library-master/',logging=False):
    """ Retrieve HDF5 files from Raspberry Pis via SSH connection

        ips : List of IPs for Rasperry Pis
        passwords : Login passwords for Raspberry Pis
        target_dir : Directory where the files are located. Default /home/pi/mlx90640-library-master
        logging : Flag for whether to start log file called ssg_log.txt. Default : False.

        Returns the number of files copied.

        Connects to each Raspberry in turn and copies any HDF5 files found in the
        target directory to the current working directory.

        Assumes the same target directory for all Raspberry Pis.

        Does not recursively search the target directory.
    """

    # if flag is set, start logging in local directory
    if logging:
        paramiko.util.log_to_file('./ssg_log.txt')

    # get current directory
    cwd = os.getcwd()

    # objects created for connection
    # created outside of the loop so it can be closed in the event of an exception
    t = None
    sftp = None

    ## extra variables for keeping track of performance
    # number of files copied form host
    files_copied = 0

    # iterate through each ip address
    for c,ip in enumerate(ips):
        try:
            # create transport object to move files
            t = paramiko.Transport((ip,22))
            # authorize SSH connection using login information
            t.connect(username="pi",password=passwords[c])
            # create sftp client to move files
            sftp = paramiko.SFTPClient.from_transport(t)
            files_copied = 0
            # get the filenames in the target directory
            for filename in sftp.listdir(target_dir):
                # if the filename contains hdf5, copy to current local working directory
                if fnmatch.fnmatch(filename,"*.hdf5"):
                    sftp.get(os.path.join(target_dir,filename),
                             os.path.join(cwd,filename))
                    files_copied +=1
            # close connection
            sftp.close()
            t.close()
        # in the event of any exception, ensure that the connections are closed
        except Exception as e:
            traceback.print_exc()
            if sftp != None:
                sftp.close()
            if sftp != None:
                t.close()
            # returns the number of files copied
            return files_copied

    # return the number of files copied
    return files_copied

ips = ["1.1.1.1.1.1"]
password = ["password"]

print("Attempting to copy files across")
print("Copied ",getHDF5FilesFromPi(ips,password))
