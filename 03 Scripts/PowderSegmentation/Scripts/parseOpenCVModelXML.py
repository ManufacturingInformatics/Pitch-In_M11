<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import glob
import os

def parseModelXMLFile(path,justElems=False):
    tree = ET.parse(path)
    # for some reason the find and find all doesn't get the bits we want
    target = ["weights","means","covs"]
    # search for each of the target parameters
    data= [searchTag(tree,t) for t in target]
    # data is list of list
    # unpack into list
    data = [attr for ll in data for attr in ll]
    if justElems:
        return data
    else:
        # dictionary to update
        dd = {}
        # track rows and cols
        rows = 0
        cols = 0
        # iterate over target terms and collected data 
        for t,d in zip(target,data):
            # if not the covariance matrix
            if t is not "covs":
                # if the element has a data element whose text attribute is not None
                if d.find('data').text is not None:
                    # get "data" and convert to number array
                    dd[t] = np.asarray(d.find('data').text.split(),'float').reshape((int(d.find("rows").text),int(d.find("cols").text)))
            # if it's the covaraince element
            # pass onto parseCovs function
            else:
                mm = parseCovs(d)
                if mm is not None:
                    dd[t] = mm
        return dd

# parses covariances element in XML tree and returns matrices
def parseCovs(elem):
    # data is wrapped under an element with a _ tag
    try:
        d0,d1 = elem.findall("_")
    except ValueError:
        return
    ## convert "data" attribute into numpy matix
    # 'data' as text is returned as list of characters
    m0 = np.asarray(''.join(d0.find("data").text).split(),'float')
    m0 = m0.reshape((int(d0.find("rows").text),int(d0.find("rows").text)))
    m1 = np.asarray(''.join(d1.find("data").text).split(),'float')
    m1 = m1.reshape((int(d1.find("rows").text),int(d1.find("rows").text)))
    return m0,m1

# function for searching for surface children whose tag matches target
def searchTag(tree,target):
    return [e for e in tree.iter() if target in e.tag]

if __name__ == "__main__":
    print("collecting XML file paths")
    # get all XML files in folder
    fpath = glob.glob(r"C:\Users\david\Documents\CoronaWork\Data\*.xml",recursive=True)
    print(f"found {len(fpath)}")
    # sort paths to ensure they're in order
    # files have a number suffix
    fpath.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split("thermalModel_")[1]))
    # create nested dictionary of file paths
    print("creating data dictionary")
    data = {p:parseModelXMLFile(p) for p in fpath}
    # iterate over to file paths to get frames from filenames
    ff = []
    # create arrays for data
    weights = []
    means = []
    covs = []  
    # iterate over created dictionary
    print("collecting data")
    for path,v in data.items():
        # if the associated file has data
        # update frames list
        if len(v)>0:
            ff.append(int(os.path.splitext(os.path.basename(path))[0].split("thermalModel_")[1]))
        for wi,(dname,w) in enumerate(v.items()):
            # update associated list
            if dname is "weights":
                weights.append(w)
            elif dname is "means":
                means.append(w)
            elif dname is "covs":
                covs.append(w)
    print(f"{len(ff)} files had data")
    # replace empty matricies caused by empty 
    ## concatenate the arrays together to form a giant array
    # two elements
    weights = np.concatenate(weights)
    # 2 x number of features per matrix
    marr = np.concatenate(means)
    # number of features x number of features
    carr = np.concatenate(covs)
    # plot directory
    pdir = r"C:\Users\david\Documents\CoronaWork\Plots"
    # create axes
    f,ax = plt.subplots()
    ## plot weights
    print("plotting weights")
    # plot first weight
    ax.plot(ff,weights[:,0])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 0 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-0-EMSLICO.png"))
    # plot second weight
    ax.clear()
    ax.plot(ff,weights[:,1])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 1 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-1-EMSLICO.png"))
    # plot both weights
    ax.clear()
    ax.plot(ff,weights)
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Weights of GMM")
    f.legend(["class 0","class 1"],loc="upper left")
    f.savefig(os.path.join(pdir,"hist-all-weight-EMSLICO.png"))

    ## plot means
    print("plotting means")
    # as contours
    print("...plotting contours")
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-EMSLICO.png"))
    plt.close('f2')
    
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 GMM Mean")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-EMSLICO.png"))
    plt.close('f2')

    # logscale
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean (Logscale)")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-logscale-EMSLICO.png"))
    plt.close('f2')

    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 of GMM Mean (Logscale)")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-logscale-EMSLICO.png"))
    plt.close('f2')
    
    # individually
    print("...plotting collective history")
    # create folder
    os.makedirs(os.path.join(pdir,"means"),exist_ok=True)
    # get classZero rows
    classZero = marr[::2]
    plt.close('f')
    f,ax = plt.subplots()
    ax.plot(ff,classZero)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 0 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-0-EMSLICO.png"))

    classOne = marr[1::2]
    ax.clear()
    ax.plot(ff,classOne)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 1 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-1-EMSLICO.png"))
    plt.close('f')
    
    # plot individual means
    fig,ax = plt.subplots()
    print("...plotting individual histories")
    for c in range(classZero.shape[1]):
        ax.clear()
        ax.plot(ff,classZero[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 0 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-0-f-{c}-EMSLICO.png"))

    for c in range(classOne.shape[1]):
        ax.clear()
        ax.plot(ff,classOne[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 1 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-1-f-{c}-EMSLICO.png"))

    ## plot covariances
    print("plotting covariances")
    # plot as contours
    classZero = carr[::2] 
    classOne = carr[1::2]
    print("...plotting contours")
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classZero[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classOne[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    # logscale
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classZero[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classOne[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)
        
    print("...plotting individual histories")
    # collect diagonal elements
    # pretty sure the covariances are diagonal matrices
    zeroCovDiag = np.concatenate([np.diag(classZero[c])[None,:] for c in range(classZero.shape[0])])
    oneCovDiag = np.concatenate([np.diag(classOne[c])[None,:] for c in range(classOne.shape[0])])
    
    ax.clear()
    ax.plot(ff,zeroCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 0 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-0-EMSLICO.png"))

    ax.clear()
    ax.plot(ff,oneCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 1 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-1-EMSLICO.png"))

    os.makedirs(os.path.join(pdir,"covs"),exist_ok=True)
    for cc in range(zeroCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,zeroCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 0 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-0-f-{cc}-EMSLICO.png"))

    for cc in range(oneCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,oneCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 1 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-1-f-{cc}-EMSLICO.png"))
    
=======
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import glob
import os

def parseModelXMLFile(path,justElems=False):
    tree = ET.parse(path)
    # for some reason the find and find all doesn't get the bits we want
    target = ["weights","means","covs"]
    # search for each of the target parameters
    data= [searchTag(tree,t) for t in target]
    # data is list of list
    # unpack into list
    data = [attr for ll in data for attr in ll]
    if justElems:
        return data
    else:
        # dictionary to update
        dd = {}
        # track rows and cols
        rows = 0
        cols = 0
        # iterate over target terms and collected data 
        for t,d in zip(target,data):
            # if not the covariance matrix
            if t is not "covs":
                # if the element has a data element whose text attribute is not None
                if d.find('data').text is not None:
                    # get "data" and convert to number array
                    dd[t] = np.asarray(d.find('data').text.split(),'float').reshape((int(d.find("rows").text),int(d.find("cols").text)))
            # if it's the covaraince element
            # pass onto parseCovs function
            else:
                mm = parseCovs(d)
                if mm is not None:
                    dd[t] = mm
        return dd

# parses covariances element in XML tree and returns matrices
def parseCovs(elem):
    # data is wrapped under an element with a _ tag
    try:
        d0,d1 = elem.findall("_")
    except ValueError:
        return
    ## convert "data" attribute into numpy matix
    # 'data' as text is returned as list of characters
    m0 = np.asarray(''.join(d0.find("data").text).split(),'float')
    m0 = m0.reshape((int(d0.find("rows").text),int(d0.find("rows").text)))
    m1 = np.asarray(''.join(d1.find("data").text).split(),'float')
    m1 = m1.reshape((int(d1.find("rows").text),int(d1.find("rows").text)))
    return m0,m1

# function for searching for surface children whose tag matches target
def searchTag(tree,target):
    return [e for e in tree.iter() if target in e.tag]

if __name__ == "__main__":
    print("collecting XML file paths")
    # get all XML files in folder
    fpath = glob.glob(r"C:\Users\david\Documents\CoronaWork\Data\*.xml",recursive=True)
    print(f"found {len(fpath)}")
    # sort paths to ensure they're in order
    # files have a number suffix
    fpath.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split("thermalModel_")[1]))
    # create nested dictionary of file paths
    print("creating data dictionary")
    data = {p:parseModelXMLFile(p) for p in fpath}
    # iterate over to file paths to get frames from filenames
    ff = []
    # create arrays for data
    weights = []
    means = []
    covs = []  
    # iterate over created dictionary
    print("collecting data")
    for path,v in data.items():
        # if the associated file has data
        # update frames list
        if len(v)>0:
            ff.append(int(os.path.splitext(os.path.basename(path))[0].split("thermalModel_")[1]))
        for wi,(dname,w) in enumerate(v.items()):
            # update associated list
            if dname is "weights":
                weights.append(w)
            elif dname is "means":
                means.append(w)
            elif dname is "covs":
                covs.append(w)
    print(f"{len(ff)} files had data")
    # replace empty matricies caused by empty 
    ## concatenate the arrays together to form a giant array
    # two elements
    weights = np.concatenate(weights)
    # 2 x number of features per matrix
    marr = np.concatenate(means)
    # number of features x number of features
    carr = np.concatenate(covs)
    # plot directory
    pdir = r"C:\Users\david\Documents\CoronaWork\Plots"
    # create axes
    f,ax = plt.subplots()
    ## plot weights
    print("plotting weights")
    # plot first weight
    ax.plot(ff,weights[:,0])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 0 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-0-EMSLICO.png"))
    # plot second weight
    ax.clear()
    ax.plot(ff,weights[:,1])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 1 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-1-EMSLICO.png"))
    # plot both weights
    ax.clear()
    ax.plot(ff,weights)
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Weights of GMM")
    f.legend(["class 0","class 1"],loc="upper left")
    f.savefig(os.path.join(pdir,"hist-all-weight-EMSLICO.png"))

    ## plot means
    print("plotting means")
    # as contours
    print("...plotting contours")
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-EMSLICO.png"))
    plt.close('f2')
    
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 GMM Mean")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-EMSLICO.png"))
    plt.close('f2')

    # logscale
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean (Logscale)")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-logscale-EMSLICO.png"))
    plt.close('f2')

    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 of GMM Mean (Logscale)")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-logscale-EMSLICO.png"))
    plt.close('f2')
    
    # individually
    print("...plotting collective history")
    # create folder
    os.makedirs(os.path.join(pdir,"means"),exist_ok=True)
    # get classZero rows
    classZero = marr[::2]
    plt.close('f')
    f,ax = plt.subplots()
    ax.plot(ff,classZero)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 0 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-0-EMSLICO.png"))

    classOne = marr[1::2]
    ax.clear()
    ax.plot(ff,classOne)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 1 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-1-EMSLICO.png"))
    plt.close('f')
    
    # plot individual means
    fig,ax = plt.subplots()
    print("...plotting individual histories")
    for c in range(classZero.shape[1]):
        ax.clear()
        ax.plot(ff,classZero[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 0 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-0-f-{c}-EMSLICO.png"))

    for c in range(classOne.shape[1]):
        ax.clear()
        ax.plot(ff,classOne[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 1 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-1-f-{c}-EMSLICO.png"))

    ## plot covariances
    print("plotting covariances")
    # plot as contours
    classZero = carr[::2] 
    classOne = carr[1::2]
    print("...plotting contours")
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classZero[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classOne[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    # logscale
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classZero[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classOne[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)
        
    print("...plotting individual histories")
    # collect diagonal elements
    # pretty sure the covariances are diagonal matrices
    zeroCovDiag = np.concatenate([np.diag(classZero[c])[None,:] for c in range(classZero.shape[0])])
    oneCovDiag = np.concatenate([np.diag(classOne[c])[None,:] for c in range(classOne.shape[0])])
    
    ax.clear()
    ax.plot(ff,zeroCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 0 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-0-EMSLICO.png"))

    ax.clear()
    ax.plot(ff,oneCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 1 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-1-EMSLICO.png"))

    os.makedirs(os.path.join(pdir,"covs"),exist_ok=True)
    for cc in range(zeroCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,zeroCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 0 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-0-f-{cc}-EMSLICO.png"))

    for cc in range(oneCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,oneCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 1 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-1-f-{cc}-EMSLICO.png"))
    
>>>>>>> origin/master
=======
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import glob
import os

def parseModelXMLFile(path,justElems=False):
    tree = ET.parse(path)
    # for some reason the find and find all doesn't get the bits we want
    target = ["weights","means","covs"]
    # search for each of the target parameters
    data= [searchTag(tree,t) for t in target]
    # data is list of list
    # unpack into list
    data = [attr for ll in data for attr in ll]
    if justElems:
        return data
    else:
        # dictionary to update
        dd = {}
        # track rows and cols
        rows = 0
        cols = 0
        # iterate over target terms and collected data 
        for t,d in zip(target,data):
            # if not the covariance matrix
            if t is not "covs":
                # if the element has a data element whose text attribute is not None
                if d.find('data').text is not None:
                    # get "data" and convert to number array
                    dd[t] = np.asarray(d.find('data').text.split(),'float').reshape((int(d.find("rows").text),int(d.find("cols").text)))
            # if it's the covaraince element
            # pass onto parseCovs function
            else:
                mm = parseCovs(d)
                if mm is not None:
                    dd[t] = mm
        return dd

# parses covariances element in XML tree and returns matrices
def parseCovs(elem):
    # data is wrapped under an element with a _ tag
    try:
        d0,d1 = elem.findall("_")
    except ValueError:
        return
    ## convert "data" attribute into numpy matix
    # 'data' as text is returned as list of characters
    m0 = np.asarray(''.join(d0.find("data").text).split(),'float')
    m0 = m0.reshape((int(d0.find("rows").text),int(d0.find("rows").text)))
    m1 = np.asarray(''.join(d1.find("data").text).split(),'float')
    m1 = m1.reshape((int(d1.find("rows").text),int(d1.find("rows").text)))
    return m0,m1

# function for searching for surface children whose tag matches target
def searchTag(tree,target):
    return [e for e in tree.iter() if target in e.tag]

if __name__ == "__main__":
    print("collecting XML file paths")
    # get all XML files in folder
    fpath = glob.glob(r"C:\Users\david\Documents\CoronaWork\Data\*.xml",recursive=True)
    print(f"found {len(fpath)}")
    # sort paths to ensure they're in order
    # files have a number suffix
    fpath.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split("thermalModel_")[1]))
    # create nested dictionary of file paths
    print("creating data dictionary")
    data = {p:parseModelXMLFile(p) for p in fpath}
    # iterate over to file paths to get frames from filenames
    ff = []
    # create arrays for data
    weights = []
    means = []
    covs = []  
    # iterate over created dictionary
    print("collecting data")
    for path,v in data.items():
        # if the associated file has data
        # update frames list
        if len(v)>0:
            ff.append(int(os.path.splitext(os.path.basename(path))[0].split("thermalModel_")[1]))
        for wi,(dname,w) in enumerate(v.items()):
            # update associated list
            if dname is "weights":
                weights.append(w)
            elif dname is "means":
                means.append(w)
            elif dname is "covs":
                covs.append(w)
    print(f"{len(ff)} files had data")
    # replace empty matricies caused by empty 
    ## concatenate the arrays together to form a giant array
    # two elements
    weights = np.concatenate(weights)
    # 2 x number of features per matrix
    marr = np.concatenate(means)
    # number of features x number of features
    carr = np.concatenate(covs)
    # plot directory
    pdir = r"C:\Users\david\Documents\CoronaWork\Plots"
    # create axes
    f,ax = plt.subplots()
    ## plot weights
    print("plotting weights")
    # plot first weight
    ax.plot(ff,weights[:,0])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 0 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-0-EMSLICO.png"))
    # plot second weight
    ax.clear()
    ax.plot(ff,weights[:,1])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 1 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-1-EMSLICO.png"))
    # plot both weights
    ax.clear()
    ax.plot(ff,weights)
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Weights of GMM")
    f.legend(["class 0","class 1"],loc="upper left")
    f.savefig(os.path.join(pdir,"hist-all-weight-EMSLICO.png"))

    ## plot means
    print("plotting means")
    # as contours
    print("...plotting contours")
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-EMSLICO.png"))
    plt.close('f2')
    
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 GMM Mean")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-EMSLICO.png"))
    plt.close('f2')

    # logscale
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean (Logscale)")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-logscale-EMSLICO.png"))
    plt.close('f2')

    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 of GMM Mean (Logscale)")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-logscale-EMSLICO.png"))
    plt.close('f2')
    
    # individually
    print("...plotting collective history")
    # create folder
    os.makedirs(os.path.join(pdir,"means"),exist_ok=True)
    # get classZero rows
    classZero = marr[::2]
    plt.close('f')
    f,ax = plt.subplots()
    ax.plot(ff,classZero)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 0 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-0-EMSLICO.png"))

    classOne = marr[1::2]
    ax.clear()
    ax.plot(ff,classOne)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 1 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-1-EMSLICO.png"))
    plt.close('f')
    
    # plot individual means
    fig,ax = plt.subplots()
    print("...plotting individual histories")
    for c in range(classZero.shape[1]):
        ax.clear()
        ax.plot(ff,classZero[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 0 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-0-f-{c}-EMSLICO.png"))

    for c in range(classOne.shape[1]):
        ax.clear()
        ax.plot(ff,classOne[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 1 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-1-f-{c}-EMSLICO.png"))

    ## plot covariances
    print("plotting covariances")
    # plot as contours
    classZero = carr[::2] 
    classOne = carr[1::2]
    print("...plotting contours")
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classZero[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classOne[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    # logscale
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classZero[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classOne[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)
        
    print("...plotting individual histories")
    # collect diagonal elements
    # pretty sure the covariances are diagonal matrices
    zeroCovDiag = np.concatenate([np.diag(classZero[c])[None,:] for c in range(classZero.shape[0])])
    oneCovDiag = np.concatenate([np.diag(classOne[c])[None,:] for c in range(classOne.shape[0])])
    
    ax.clear()
    ax.plot(ff,zeroCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 0 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-0-EMSLICO.png"))

    ax.clear()
    ax.plot(ff,oneCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 1 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-1-EMSLICO.png"))

    os.makedirs(os.path.join(pdir,"covs"),exist_ok=True)
    for cc in range(zeroCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,zeroCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 0 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-0-f-{cc}-EMSLICO.png"))

    for cc in range(oneCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,oneCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 1 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-1-f-{cc}-EMSLICO.png"))
    
>>>>>>> origin/master
=======
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import glob
import os

def parseModelXMLFile(path,justElems=False):
    tree = ET.parse(path)
    # for some reason the find and find all doesn't get the bits we want
    target = ["weights","means","covs"]
    # search for each of the target parameters
    data= [searchTag(tree,t) for t in target]
    # data is list of list
    # unpack into list
    data = [attr for ll in data for attr in ll]
    if justElems:
        return data
    else:
        # dictionary to update
        dd = {}
        # track rows and cols
        rows = 0
        cols = 0
        # iterate over target terms and collected data 
        for t,d in zip(target,data):
            # if not the covariance matrix
            if t is not "covs":
                # if the element has a data element whose text attribute is not None
                if d.find('data').text is not None:
                    # get "data" and convert to number array
                    dd[t] = np.asarray(d.find('data').text.split(),'float').reshape((int(d.find("rows").text),int(d.find("cols").text)))
            # if it's the covaraince element
            # pass onto parseCovs function
            else:
                mm = parseCovs(d)
                if mm is not None:
                    dd[t] = mm
        return dd

# parses covariances element in XML tree and returns matrices
def parseCovs(elem):
    # data is wrapped under an element with a _ tag
    try:
        d0,d1 = elem.findall("_")
    except ValueError:
        return
    ## convert "data" attribute into numpy matix
    # 'data' as text is returned as list of characters
    m0 = np.asarray(''.join(d0.find("data").text).split(),'float')
    m0 = m0.reshape((int(d0.find("rows").text),int(d0.find("rows").text)))
    m1 = np.asarray(''.join(d1.find("data").text).split(),'float')
    m1 = m1.reshape((int(d1.find("rows").text),int(d1.find("rows").text)))
    return m0,m1

# function for searching for surface children whose tag matches target
def searchTag(tree,target):
    return [e for e in tree.iter() if target in e.tag]

if __name__ == "__main__":
    print("collecting XML file paths")
    # get all XML files in folder
    fpath = glob.glob(r"C:\Users\david\Documents\CoronaWork\Data\*.xml",recursive=True)
    print(f"found {len(fpath)}")
    # sort paths to ensure they're in order
    # files have a number suffix
    fpath.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split("thermalModel_")[1]))
    # create nested dictionary of file paths
    print("creating data dictionary")
    data = {p:parseModelXMLFile(p) for p in fpath}
    # iterate over to file paths to get frames from filenames
    ff = []
    # create arrays for data
    weights = []
    means = []
    covs = []  
    # iterate over created dictionary
    print("collecting data")
    for path,v in data.items():
        # if the associated file has data
        # update frames list
        if len(v)>0:
            ff.append(int(os.path.splitext(os.path.basename(path))[0].split("thermalModel_")[1]))
        for wi,(dname,w) in enumerate(v.items()):
            # update associated list
            if dname is "weights":
                weights.append(w)
            elif dname is "means":
                means.append(w)
            elif dname is "covs":
                covs.append(w)
    print(f"{len(ff)} files had data")
    # replace empty matricies caused by empty 
    ## concatenate the arrays together to form a giant array
    # two elements
    weights = np.concatenate(weights)
    # 2 x number of features per matrix
    marr = np.concatenate(means)
    # number of features x number of features
    carr = np.concatenate(covs)
    # plot directory
    pdir = r"C:\Users\david\Documents\CoronaWork\Plots"
    # create axes
    f,ax = plt.subplots()
    ## plot weights
    print("plotting weights")
    # plot first weight
    ax.plot(ff,weights[:,0])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 0 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-0-EMSLICO.png"))
    # plot second weight
    ax.clear()
    ax.plot(ff,weights[:,1])
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Class 1 Weight of GMM")
    f.savefig(os.path.join(pdir,"hist-weight-1-EMSLICO.png"))
    # plot both weights
    ax.clear()
    ax.plot(ff,weights)
    ax.set(xlabel="Frames of Training",ylabel="Weight",title="Evolution of Weights of GMM")
    f.legend(["class 0","class 1"],loc="upper left")
    f.savefig(os.path.join(pdir,"hist-all-weight-EMSLICO.png"))

    ## plot means
    print("plotting means")
    # as contours
    print("...plotting contours")
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-EMSLICO.png"))
    plt.close('f2')
    
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv')
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 GMM Mean")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-EMSLICO.png"))
    plt.close('f2')

    # logscale
    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[0::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 0 GMM Mean (Logscale)")
    cb = f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-0-logscale-EMSLICO.png"))
    plt.close('f2')

    f2,ax2 = plt.subplots()
    mct = ax2.contourf(marr.T[1::2,:],cmap='hsv',locator=ticker.LogLocator())
    ax2.set_xticklabels(ff)
    ax2.set(xlabel="Frames of Training",ylabel="Feature Index",title=f"Contour of Class 1 of GMM Mean (Logscale)")
    f2.colorbar(mct)
    f2.savefig(os.path.join(pdir,f"mean-1-logscale-EMSLICO.png"))
    plt.close('f2')
    
    # individually
    print("...plotting collective history")
    # create folder
    os.makedirs(os.path.join(pdir,"means"),exist_ok=True)
    # get classZero rows
    classZero = marr[::2]
    plt.close('f')
    f,ax = plt.subplots()
    ax.plot(ff,classZero)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 0 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-0-EMSLICO.png"))

    classOne = marr[1::2]
    ax.clear()
    ax.plot(ff,classOne)
    ax.set(xlabel="Frames of Training",ylabel="Feature Means",title="Training History of Class 1 GMM Mean")
    f.savefig(os.path.join(pdir,"hist-mean-1-EMSLICO.png"))
    plt.close('f')
    
    # plot individual means
    fig,ax = plt.subplots()
    print("...plotting individual histories")
    for c in range(classZero.shape[1]):
        ax.clear()
        ax.plot(ff,classZero[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 0 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-0-f-{c}-EMSLICO.png"))

    for c in range(classOne.shape[1]):
        ax.clear()
        ax.plot(ff,classOne[:,c])
        ax.set(xlabel="Frames of Training",ylabel="Feature Mean",title=f"Training History of Class 1 GMM Mean for Feature {c}")
        fig.savefig(os.path.join(pdir,"means",f"hist-mean-1-f-{c}-EMSLICO.png"))

    ## plot covariances
    print("plotting covariances")
    # plot as contours
    classZero = carr[::2] 
    classOne = carr[1::2]
    print("...plotting contours")
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classZero[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(classOne[cc,:,:],cmap='hsv')
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-EMSLICO.png"))
        plt.close(f3)

    # logscale
    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classZero[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 0\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-0-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)

    for cc,f in enumerate(ff):
        f3,ax3 = plt.subplots()
        ax3.contourf(np.abs(classOne[cc,:,:]),cmap='hsv',locator=ticker.LogLocator())
        ax3.set(xlabel="Feature Index",ylabel="Feature Index",title=f"Contour of Feature GMM Covariances for Class 1\n After {f} frames of Training (Abs. Logscale)")
        f3.savefig(os.path.join(pdir,f"hist-covs-1-frames-{f}-logscale-EMSLICO.png"))
        plt.close(f3)
        
    print("...plotting individual histories")
    # collect diagonal elements
    # pretty sure the covariances are diagonal matrices
    zeroCovDiag = np.concatenate([np.diag(classZero[c])[None,:] for c in range(classZero.shape[0])])
    oneCovDiag = np.concatenate([np.diag(classOne[c])[None,:] for c in range(classOne.shape[0])])
    
    ax.clear()
    ax.plot(ff,zeroCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 0 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-0-EMSLICO.png"))

    ax.clear()
    ax.plot(ff,oneCovDiag)
    ax.set(xlabel="Frames of Training",ylabel="Diagonal Feature Covariances",title="Training History of Class 1 Diagonal GMM Covariances")
    fig.savefig(os.path.join(pdir,"hist-cov-1-EMSLICO.png"))

    os.makedirs(os.path.join(pdir,"covs"),exist_ok=True)
    for cc in range(zeroCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,zeroCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 0 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-0-f-{cc}-EMSLICO.png"))

    for cc in range(oneCovDiag.shape[1]):
        ax.clear()
        ax.plot(ff,oneCovDiag[:,cc])
        ax.set(xlabel="Frames of Training",ylabel="Feature Covariance",title=f"Training History of Diagonal GMM Covariances for Class 1 for Feature {cc}")
        fig.savefig(os.path.join(pdir,"covs",f"hist-covs-1-f-{cc}-EMSLICO.png"))
    
>>>>>>> origin/master
