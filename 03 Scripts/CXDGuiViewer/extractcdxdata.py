import javabridge
import bioformats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import xml.etree.ElementTree as ET
import os
import datetime

def search_node(node,target):
    # iterate through children yielding item if tag contains target
    for item in list(node):
        if target in item.tag:
            yield item
        # perform same action on next level of children
        # use function to recursively reach next levels
        # next key word required as search_node is a generator
        yield from search_node(item,target)

def getURLs(node):
    yield node.tag.split('}')[0][1:]
    for item in node.getchildren():
        yield item.tag.split('}')[0][1:]
        yield from getURLs(item)

def getCXDTData(path):
    """ Get the Time data, T, from the target bioformats file specified by path

        path : Target file

        Requires the javabridge virtual machine to be started.

        Returns the DeltaT attributes of Plane objects in XML metadata as floats
    """
    import javabridge
    import bioformats
    # flag indicating of the java VM has been started by this function
    vm_started = False
    # if a Java VM has not been started elsewhere
    if javabridge.get_env() is None:
        # start VM
        javabridge.start_vm(class_path=bioformats.JARS)
        # set flag indicating that the VM has been started
        vm_started = True
    # create tree for path
    tree = ET.ElementTree(ET.fromstring(bioformats.get_omexml_metadata(path)))
    # if java VM was started by function, kill it as we don't need it anymore
    if vm_started:
        javabridge.kill_vm()
    # get root
    root = tree.getroot()
    # create list of all elements containing the name Plane in their tag
    # for each entry return the DeltaT attribute as float
    return [float(p.attrib['DeltaT']) for p in search_node(root,'Plane')]

def formatdeltaTToDT(deltaT,DT):
    """ Takes the deltaT data and formats it as a date time stamp based off date

        deltaT : List of deltaT values extracted from bioformat file
        DT : ISO style date time stamp e.g. 2019-10-15T12:22:53

        Iterates over deltaT and adds the value to the seconds component of the datetime
        stamp DT effectively converting deltaT from a set of absolute values to datetimem stamps

        e.g. 2019-10-15T12:22:53 + 2.6 = 2019-10-15T12:22:55.600000

        Returns a generator which produces the modified date time stamps
    """
    # date is in ISO format so date and time are separated by T
    # time is separated by colons
    # get time portion
    for tt in deltaT:
        # get date time up to first semi colon (just before minute component)
        stamp = DT[:DT.find(':')+1]
        # calculate new seconds component
        new_s = int(DT[DT.rfind(':')+1:])+int(tt)
        # get minutes component
        new_m = int(DT[DT.find(':')+1:DT.rfind(':')])
        # update minutes if necessary and seconds
        if (new_s//60) > 0:
            # increase minute component by 1
            new_m += 1
            # decrement seconds
            new_s -= 60
        # get decimal portion of the number
        msecs = str(round(tt%1,6)).split('.')[-1]
        # convert new seconds component to string and add padding 0 if necessary
        new_s = ('0' if new_s<1 else '') + str(new_s)
        # reform timestamp padded milliseconds portion to match strftime format
        yield stamp+str(new_m)+':'+new_s+'.'+''.join(['0']*(6-len(msecs)))

def findSaveMaxValue(ipath,opath):
    """ Find and plot the maximum value of each frame of each series. Plots and data are saved to set output path

        ipath : Input path for CXD file
        opath : Output path for data as CSV file and plots
    """
    # start virtual machine
    javabridge.start_vm(class_path=bioformats.JARS)
    # convert 
    tree = ET.ElementTree(ET.fromstring(bioformats.get_omexml_metadata(ipath)))
    root = tree.getroot()
    #deltaT = getCXDTData(path)
    deltaT = [float(p.attrib['DeltaT']) for p in search_node(root,'Plane')]
    aq_data = list(search_node(root,'Date'))[0].text
    # create reader to access series data
    reader = bioformats.formatreader.make_image_reader_class()()
    reader.setId(ipath)
    ## get data about the images stored insize
    img_ct = reader.getImageCount()
    series_ct = reader.getSeriesCount()
    # create list to store max vals for each series
    series_max_vals = []
    f,ax = plt.subplots()
    with bioformats.ImageReader(ipath) as reader:
        # iterate through series
        for ss in range(series_ct):
            # create entry for max values for specified series
            series_max_vals.append(np.zeros(img_ct,reader.read(index=0,series=ss).dtype))
            # iterate through image data collecting max values
            for ff in range(img_ct):
                series_max_vals[ss][ff] = reader.read(index=ff,series=ss).max()
            # generate plots from collected data
            np.savetxt(os.path.join(opath.os.path.splitext(os.path.basename(path))[0]+f"-series-{ss}-max-vals.csv"),series_max_vals[ss],delimiter=',')
            ax.clear()
            ax.plot(series_max_vals[ss])
            ax.set(xlabel='Frame Index',ylabel='Max Raw Value',title=f'Maximum Raw Value of Each Frame In Series {ss} in Target File')
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}-series-{ss}-max-vals.png"))

            ax.clear()
            ax.plot(deltaT,series_max_vals[ss])
            ax.xaxis.set_major_formatter(FormatStrFoormatter('%.2f'))
            ax.set(xlabel='Time (s)',ylabel='Max Raw Value',title=f'Maximum Raw Value of Each Frame In Series {ss} in Target File')
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}-series-{ss}-max-vals-time.png"))
    # stop the virtual machine
    javabridge.kill_vm()

def findOpenCVColormap(cmap):
    """ Function for finding the code for the target colormap OpenCV

        cmap : String used as wildcard search term when searching for colormap

        Imports OpenCV (cv2) and extracts the dictionary of attributes key-values. The dictionary is then filtered for the COLORMAP attributes
        and the presence of the search term. If a matching attribute is not found, None is returned

        Returns the integer corresponding to the target colormap if found. Returns None if not found
    """
    import cv2
    attrs = cv2.__dict__
    # filter for colormap
    cmap= {k:v for k,v in attrs.items() if ('COLORMAP' in k) and (cmap in k.lower())}
    if not cmap:
        return None
    # get value
    return list(cmap.values())[0]
    
def saveImageData(path,opath,useCV=False,cmap='hsv'):
    """ Find and save the image data stored in the CXD file to the set output path

        path : path to CXD file
        opath : Output path for folders containing series data
        useCV : Flag to use OpenCV to save the data to an image. Images are normalized to local limits. Faster than Matplotlib.
    """
    # start virtual machine
    javabridge.start_vm(class_path=bioformats.JARS)
    # convert 
    tree = ET.ElementTree(ET.fromstring(bioformats.get_omexml_metadata(path)))
    root = tree.getroot()
    #deltaT = getCXDTData(path)
    deltaT = [float(p.attrib['DeltaT']) for p in search_node(root,'Plane')]
    aq_data = list(search_node(root,'Date'))[0].text
    # create reader to access series data
    reader = bioformats.formatreader.make_image_reader_class()()
    reader.setId(path)
    ## get data about the images stored insize
    img_ct = reader.getImageCount()
    series_ct = reader.getSeriesCount()
    # open image reader
    with bioformats.ImageReader(path) as reader:
        if not useCV:
            f,ax = plt.subplots()
            ax.set_axis_off()
            implot = ax.imshow(reader.read(index=0,series=0),cmap=cmap)
            div = make_axes_locatable(ax)
            cax = div.append_axes('right','5%','3%')
            cb = f.colorbar(implot,cax=cax)
            f.tight_layout()
        else:
            import cv2
            # find code for colormap
            cmap = findOpenCVColormap(cmap)
            if cmap is None:
                raise ValueError("Target colormap does not exist in OpenCV")
        # iterate over series
        for ss in range(series_ct):
            img_dir = os.path.join(opath,f"series_{ss}")
            # create folder for results
            os.makedirs(img_dir,exist_ok=True)
            # iterate over images
            for ff in range(img_ct):
                # get image
                img = reader.read(index=ff,series=ss)
                # print stats
                print(f"{ff}, ({img.min()},{img.max()},{img.shape}")
                # if using Opencv
                if useCV:
                    # normalize locally
                    img -= img.min()
                    img /= (img.max()-img.min())
                    img *= 255
                    img = img.astype("uint8")
                    # apply colormap
                    img = cv2.applyColorMap(img,cmap)
                    # save file
                    cv2.imwrite(os.path.join(img_dir,f"img_{ff:06d}.png"),img)
                # else matplotlib to plot and save the results
                else:
                    implot.set_data(img)
                    implot.set_clim(img.min(),img.max())
                    f.savefig(os.path.join(img_dir,f"img_{ff:06d}.png"))
    javabridge.kill_vm()  

def getImage(path,ff,ss=0):
    """ Function to get a specific frame from a specific series in a CXD file

        path : Path to target CXD file
        ff : Requested frame index. Supports iterable containers of requested frame indicies. Requesting -1 returns all frames in target series
        ss : Series index

        Returns list of requested frames
    """
    # check if the target frames are iterable or not
    try:
        iter(ff)
    except Exception:
        many_frames = False
    else:
        many_frames = True
    # start virtual machine
    javabridge.start_vm(class_path=bioformats.JARS)
    # convert 
    tree = ET.ElementTree(ET.fromstring(bioformats.get_omexml_metadata(path)))
    root = tree.getroot()
    #deltaT = getCXDTData(path)
    deltaT = [float(p.attrib['DeltaT']) for p in search_node(root,'Plane')]
    aq_data = list(search_node(root,'Date'))[0].text
    # create reader to access series data
    reader = bioformats.formatreader.make_image_reader_class()()
    reader.setId(path)
    # open image reader
    with bioformats.ImageReader(path) as reader:
        if not many_frames:
            # if -1 return all
            if ff != -1:
                # get image
                return reader.read(index=ff,series=ss)
            else:
                return [reader.read(index(ii),series=ss) for ii in range(reader.getImageCount())]
        else:
            return [reader.read(index(ii),series=ss) for ii in ff]

if __name__ == "__main__":
    # path to target file
    path = "..\Data\BeAM_Trials1-011.cxd"

    javabridge.kill_vm()
