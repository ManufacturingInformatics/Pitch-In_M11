import numpy as np
import networkx as nx
import lm
import os
import cv2
import matplotlib.pyplot as plt

import descriptors
import spgraph
import utilities

# make RAG for display
from skimage.future import graph

import logging

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# parameters for SLICO algorithm
region_size = 100
ruler = 15.0
ratio = 0.075
min_element_size = 25
num_iterations = 10

# take a screenshot, find the centres of each superpixel, draw on a copy of the screenshot and return it
# showBoundary controls whether the cell boundaries are overlayed. Generated based off SLIC contour mask. Default True
# bwargs is a dictionary of paramters passed to drawBoundaries. See cv2.drawContours for supported paramters
def screenshotCentres(showBoundary=True,**bwargs):
    log.info("starting")
    # get the webcam/default camera
    log.debug("Getting camera 0")
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        log.critical("failed to open camera")
        cam.release()
        return
    log.debug("successfully opened camera 0")
    # get a frame
    ret,frame = cam.read()
    log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
    if not ret:
        log.critical("failed to get frame")
        cam.release()
        return
    log.info("performing superpixel segmentation")
    log.debug(f"performing SLICO ({cv2.ximgproc.SLICO}) cvsuperpixel segmentation, region_size : {region_size}, ruler : {ruler}")
    # perform SLICO algorithm on the frame with the target settings
    superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
    # perform a set number of iterations on the current frames
    log.debug(f"iterating superpix {num_iterations} iters")
    superpix.iterate(num_iterations)
    # force a minimum size of superpixels
    if min_element_size>0:
        log.debug(f"forcing minimum superpixel size to {min_element_size}")
        superpix.enforceLabelConnectivity(min_element_size)
    # get the labels matrix
    labels = superpix.getLabels()
    log.info(f"finished superpixels {np.unique(labels).shape}")
    log.debug("releasing camera")
    # release camera
    cam.release()
    # find the centres
    log.info("finding centres of the superpixels")
    centres = spgraph.findSuperpixelCentres(labels)
    # get a copy of frame
    cframe = frame.copy()
    # iterate over centres and draw
    log.info("drawing centres")
    for c in centres:
        log.debug(f"drawing superpixel centre at {c}")
        cframe = cv2.drawMarker(frame,c,(255,0,0),cv2.MARKER_DIAMOND,10,2)
    # if the show boundary flag is set
    # draw superpixel cell boundaries using drawBoundaries method
    if showBoundary:
        log.debug("getting boundary mask")
        bmask = superpix.getLabelContourMask()
        log.info("drawing boundaries")
        bdraw = spgraph.drawBoundaries(cframe,bmask,**bwargs)
    return cframe,centres

# take a frame using default camera, find the superpixels and centres and build a graph with the centres
# edges are based off euclidean distance between centres
def buildScreenshotGraph(wtype='potts'):
    log.info("starting")
    # get the webcam/default camera
    log.debug("Getting camera 0")
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        log.critical("failed to open camera")
        cam.release()
        return
    log.debug("successfully opened camera 0")
    # get a frame
    ret,frame = cam.read()
    log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
    if not ret:
        log.critical("failed to get frame")
        cam.release()
        return
    log.info("performing superpixel segmentation")
    log.debug(f"performing SLICO ({cv2.ximgproc.SLICO}) cvsuperpixel segmentation, region_size : {region_size}, ruler : {ruler}")
    # perform SLICO algorithm on the frame with the target settings
    superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
    # perform a set number of iterations on the current frames
    log.debug(f"iterating superpix {num_iterations} iters")
    superpix.iterate(num_iterations)
    # force a minimum size of superpixels
    if min_element_size>0:
        log.debug(f"forcing minimum superpixel size to {min_element_size}")
        superpix.enforceLabelConnectivity(min_element_size)
    # get number of superpixels
    nspixels = superpix.getNumberOfSuperpixels()
    log.info(f"finished superpixels {nspixels}")
    # get the labels matrix
    labels = superpix.getLabels()
    log.debug("releasing camera")
    # release camera
    cam.release()
    # find the centres of the superpixels
    # index corresponds to superpixel label
    centres = spgraph.findSuperpixelCentres(labels)
    # find the euclidean distance between centres
    # returns a num points x num points array where dist[a,b] is dist(pt[a],pt[b])
    log.info("finding distance between centres")
    log.debug("building euclidean cdist matrix from centres")
    dist = spatial.distance.cdist(centres,centres,'euclidean')
    # find the mean distance
    mdist = np.mean(np.unique(dist))
    log.debug(f"mean euclidean distance between centres is {mdist}")
    # build a KDTree from centres
    log.info("building kdtree")
    tree = spatial.KDTree(centres)
    # create empty graph
    log.info("building graph")
    G = nx.Graph()
    # add nodes to graph
    # node names based off superpixel index
    log.debug(f"adding {nspixels} to graph")
    G.add_nodes_from(range(nspixels))
    log.debug(f"added {G.n} nodes to graph")
    # dictionary of methods currently built and supported
    weight_dict = {'potts' : lambda c=centres : spgraph.edgeWeightPotts(c),
                   'color' : lambda f=frame,l=labels,c=centres: spgraph.edgeWeightsColor(f,l,c)}
    # if the edge weight type has been specified by a string
    if type(wtype) == str:
        # check if the string is related to a supported function
        # if not log error message and return
        if wtype not in weight_dict:
            log.error(f"Unsupported weight type {wtype} specified!. Currently supported types are {weight_dict.keys()}")
            return
        # else extract function from dictionary
        else:
            weight_fn = weight_dict[wtype]
    # if the edge weight type is something else
    else:
        # check if it's a callable function
        # if not print error message and exit
        if not callable(wtype):
            log.error(f"Specified weight class of type {type(wtype)} is not callable or a string. Supply either a supported string or a callable functionn that accepts the image, labels matrix and centres")
            return
        # if it's callable setup lambda to pass image, labels frame and centres
        else:
            weight_fn = lambda f=frame,l=labels,c=centres: wtype(f,l,c)
        
    log.debug(f"adding edges using {wtype} method")
    # use method to add weighted edges to graph
    G.add_weighted_edges_from(weight_fn())
    log.debug(f"added {G.number_of_edges()} to graph")
    # return the graph
    return G,pairs,tree,centres

def build_rag(frame,labels):
    # build RAG from mean color
    rag = graph.rag_mean_color(frame,labels)
    # get properties
    regions = regionprops(labels)
    # add centroid to regions
    for r in regions:
        rag.node[r['label']]['centroid'] = r['centroid']
    # draw edges
    frame = frame.copy()
    for edge in rag.edges_iter():
        n1,n2 = edge
        r1,c1 = map(int,rag.node[n1]['centroid'])
        r2,c2 = map(int,rag.node[n2]['centroid'])
        print(r1,c1)
        print(r2,c2)
        frame = cv2.line(frame,r1,r2,(255,0,0))
        frame = cv2.circle(frame,r1,2,(255,0,0))
    return frame

if __name__ == "__main__":
    # use default camera, get frame and perform superpixel segmentation
    # returns the image taken, the superpixel labels matrix and superpixel class
    log.info("getting data")
    frame,labels,sp = utilities.useScreenshot()
    # save test frame
    if not cv2.imwrite("test_frame.png",frame):
        print("failed to save test frame to file!")
    # save test frame labels
    if not cv2.imwrite("test_frame_labels.png",labels):
        print("failed to save test frame labels image to file!")
    # draw boundaries on frame and save
    if not cv2.imwrite("test_boundaries_frame.png",utilities.drawBoundaries(frame,sp.getLabelContourMask(),thickness=2)):
        print("failed to save boundaries image to file!")
    # make RAG
    rag = graph.rag_mean_color(frame,np.dstack((labels,)*3))
    # get centres
    log.info("getting centres")
    centres = spgraph.findSuperpixelCentres(labels)
    nc = len(centres)
    ## calculate features
    log.info("getting features")
    # get color features
    colF = descriptors.calculateAllColorFeatures(frame,labels)
    colF = descriptors.normalizeFeatures(colF)
    # get texture features
    textF = descriptors.calculateTextureFeatureVector(frame,labels,lm.makeLMfilters())
    textF = descriptors.normalizeFeatures(textF)
    # create EM class
    em = cv2.ml.EM_create()
    # set the number of classes
    em.setClustersNumber(2)
    # combine feature vectors together with labels vector
    fvec = np.concatenate((np.unique(labels).reshape(-1,1),colF,textF),axis=1)
    # train EM
    log.info("training EM")
    retval,logLH,tlabels,probs = em.trainEM(fvec)
    # save model
    em.save("test_image_model.xml")
    # segment the superpixels according to the results and update the labels mask
    resLabels = labels.copy()
    for v,t in zip(np.unique(resLabels),tlabels):
        resLabels[resLabels==v] = t[0]
    if not cv2.imwrite("test_labels_seg.png",255*resLabels.astype("uint8")):
        print("Failed to write segmented labels image for first frame to file")
        
    # build the graphs
    log.info("building graphs")
    wPotts = spgraph.edgeWeightPottsRAG(frame,labels,centres)
    wColor = spgraph.edgeWeightsColorRAG(frame,labels,centres)
    wFeat = spgraph.edgeWeightsFeaturesRAG(frame,labels,colF,textF,centres)
    # find the source and sink node
    # currently being set as the nodes with the max probability for each class
    # node that "best represents" each class
    sourceN = np.argmax(probs[:,0])
    sinkN = np.argmax(probs[:,1])
    # what do I set capacity to?

    ## draw graphs
    # get positions dictionary for drawing
    # same for all graphs
    pos = spgraph.formPosDict(centres)

    ## draw using matplotlib)
    # iterate over graphs and associated attributes
    for name,G in zip(["potts","color","features"],[wPotts,wColor,wFeat]):
        # create axes
        f,ax = plt.subplots()
        fn,axn = plt.subplots()
        # get weights
        wLabels = nx.get_edge_attributes(G,'weight')
        # convert weights to 2 decimal place labels
        wLabels = {k : f"{v:.2f}" for k,v in wLabels.items()}
        # draw nodes
        nx.draw_networkx(G,pos,ax=ax)
        nx.draw_networkx(G,pos,ax=axn)
        # draw edge labels
        nx.draw_networkx_edge_labels(G,pos,ax=ax,edge_labels=wLabels)
        # force rescale of axes to ensure everything is within range
        ax.autoscale()
        axn.autoscale()
        # save figure
        f.savefig(f"superpixel_{nc}_{name}_test.png")
        fn.savefig(f"superpixel_{nc}_{name}_justnodes_test.png")
        
    ## draw nodes on image
    for name,G in zip(["potts","color","features"],[wPotts,wColor,wFeat]):
        f,ax = plt.subplots()
        fn,axn = plt.subplots()
        # show image
        ax.imshow(frame[...,::-1])
        axn.imshow(frame[...,::-1])
        # draw nodes
        nx.draw_networkx(G,pos,ax=ax)
        nx.draw_networkx(G,pos,ax=axn)
        # get weights
        wLabels = nx.get_edge_attributes(G,'weight')
        # convert weights to 2 decimal place labels
        wLabels = {k : f"{v:.2f}" for k,v in wLabels.items()}
        # draw labels
        nx.draw_networkx_edge_labels(G,pos,ax=ax,edge_labels=wLabels)
        # force rescale of axes to ensure everything is within range
        ax.autoscale()
        axn.autoscale()
        # save
        f.savefig(f"superpixel_image_{nc}_{name}_test.png")
        fn.savefig(f"superpixel_image_{nc}_{name}_justnodes_test.png")

    ## read in new image and test
    # get new image and segment
    f2,l2,_ = utilities.useScreenshot()
    # 
    colF2 = descriptors.calculateAllColorFeatures(f2,l2)
    colF2 = descriptors.normalizeFeatures(colF)
    # get texture features
    textF2 = descriptors.calculateTextureFeatureVector(f2,l2,lm.makeLMfilters())
    textF2 = descriptors.normalizeFeatures(textF)
    fvec2 = np.concatenate((np.unique(l2).reshape(-1,1),colF2,textF2),axis=1)
    # feed into trained EM and get results
    # it returns a value reval (unknown) and a 2 column array of which class each supervector belongs to
    retval,res = em.predict(fvec2)
    # save the new image
    # save test frame
    if not cv2.imwrite("test_frame_2.png",f2):
        print("failed to save test frame to file!")
    # save test frame labels
    if not cv2.imwrite("test_frame_labels_2.png",l2):
        print("failed to save test frame labels image to file!")
    # draw boundaries on frame and save
    if not cv2.imwrite("test_boundaries_frame_2.png",utilities.drawBoundariesLabels(f2,l2,thickness=2)):
        print("failed to save boundaries image to file!")
    # segment the superpixels according to the results and update the labels mask
    resLabels = descriptors.segmentLabels(l2,res)
    if not cv2.imwrite("test_labels_seg_2.png",cv2.normalize(resLabels,0,255,cv2.NORM_MINMAX)):
        print("failed to save segmented labels for frame 2 to file!")
