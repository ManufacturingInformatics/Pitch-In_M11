import os
import cv2
import zipfile

os.makedirs('ProcessedGray',exist_ok=True)
# get all directories in path
for entry in os.scandir('D:\BEAM'):
    # if directory is a zipfile
    if zipfile.is_zipfile(entry.path):
        print(entry.path)
        # create zip file object
        with zipfile.ZipFile(entry.path) as zz:
            # get list of processed image names
            proc = [l for l in zz.namelist() if ('Grey' in l) and ('.tif' in l)]
            if len(proc)>0:
                for pf in proc:
                    with open(os.path.join(os.getcwd(),'ProcessedGray',os.path.basename(pf)),'wb') as f:
                        f.write(zz.read(pf))
                print("Extracted {} files".format(len(proc)))
            
