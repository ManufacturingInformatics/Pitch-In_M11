import javabridge
import bioformats
import xml.etree.ElementTree as ET
from skimage.io import imsave

def printXMLFileStructure(xx,leading=' '):
    for obj in xx:
        print("{} {} = {}".format([leading]*2,obj.tag,obj.attrib))
        if len(obj)>0:
            printXMLFileStructure(obj,leading+leading)

def getXMLNamespacesDict(xx):
    np_dict = {}
    for jj in xx.iter():
        url,tag = jj.tag[1:].split("}")
        np_dict[url] = tag
    return np_dict

# path to target file
path = "D:\BEAM\BeAM_Trials1-011.cxd"

# start virtual machine
print("Starting Java VM")
javabridge.start_vm(class_path=bioformats.JARS)
# get XMl metadata as string
print("Getting XML data as string")
xx = bioformats.get_omexml_metadata(path)
# parse string into element tree
print("Generating ET from string")
tree = ET.ElementTree(ET.fromstring(xx))
root = tree.getroot()
## investigate metadata
print("Found {} objects".format(len(root.keys())))
with bioformats.ImageReader(path) as reader:
    frame = reader.read()
print("Frame data type: {}".format(frame.dtype))
print("Frame size : {}".format(frame.shape))
print("Frame : {}".format(frame))
# save image as float
imsave("test-img.tiff",frame,"tifffile")

## create alternative reader
# initialize reader
reader = bioformats.formatreader.make_image_reader_class()()
# set up readers to read contents of file if necessary
reader.setId(path)

stats = {}
# number of channels
stats["Channels"] = reader.getSizeC()
# number of time points
stats["Time Pts"] = reader.getSizeT()
# get number of images
stats["Images"] = reader.getImageCount()
# get number of series
stats["Series"] = reader.getSeriesCount()
# get nummber of slices in current series
stats["Series Slices"] = reader.getSizeZ()
# check if image is RGB
stats["isRGB"] = reader.isRGB()

for k,v in stats.items():
    print("{} : {}".format(k,v))
# make image reader class based off format readeer
wrapper = bioformats.formatreader.ImageReader(path=path,perform_init=False)
wrapper.rdr = reader
# get first image

# stop vm
javabridge.kill_vm()
