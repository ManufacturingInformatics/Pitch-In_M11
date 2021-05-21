import numpy as np
import networkx as nx
import cv2
from skimage.future import graph
from skimage.measure import regionprops
from scipy import spatial
from itertools import combinations,product
import logging

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# find the centroid of all superpixel centres 
def findSuperpixelCentres(labels):
    log.info(f"finding centres in label matrix {labels.shape}")
    # anonymous function for finding the centre of a target variable
    def findCentre(v,labels=labels):
        # mask the labels matrix for target value
        log.debug(f"label {v}, masking labels")
        mask = (labels==v).astype("uint8")
        # calculate moments
        log.debug(f"label {v}, finding moments")
        M = cv2.moments(mask)
        # calculate x and y of centre
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        # return centre coordinates
        log.debug(f"label {v}, found centre at ({cX},{cY})")
        return (cX,cY)
    # use list comprehension to find the centre of each superpixel
    # list comprehension is faster than for loop
    centres = [findCentre(v) for v in np.unique(labels)]
    log.info(f"found {len(centres)} centres")
    # return the centres
    return centres

# create the dictionary of positions used in NetworkX draw method
# creates a dictionary where the keys are the node index and the value is the centres position
def formPosDict(c):
    return {n:cn for n,cn in enumerate(c)}

# function for building a list of neighbouring connections to each centre
# iterates over centres and queries the KDTree for the N+1 nearest other centres.
# the closest point is itself so is ignored. 
def getNeighbours(centres,tree=None,N=4):
    # if no tree is given, build it
    if tree is None:
        log.debug("building KDTree")
        tree = spatial.KDTree(centres)
    # list to hold neighbours
    neigh = []
    # enumerate centres
    for ci,c in enumerate(centres):
        log.debug(f"querying for centre {ci} @ {c}")
        # query the tree for the N+1 points closest to the centre
        # ignore the first point (itself) and form a list of connections between ci and each other neighbour
        # extend the neigh list with this sublist
        neigh.extend([(ci,cx) for cx in tree.query(c,N+1)[1][1:]])
    # convert neighbour list to a set to avoid duplicate connections and return it
    return set(neigh)

# function for building list of neighbouring connections to each centre
# iterates over centres and queries the KDTree for all connections within a certain distance
# the closest point is itself and is ignored
# if distance range is not given, the result of the function findMeanEucDist is used
# as spatial regularization is applied to the superpixels, the distances between their centres is very similar
# to each other, so using the mean as a default distance is a good place to start
def getPointsWithinRange(centres,dist,tree=None):
    # if the tree is not passed build it from centres and default settings
    if tree is None:
        log.debug("building KDTree")
        tree = spatial.KDTree(centres)
    # list to hold neighbours
    neigh = []
    # enumerate centres
    for ci,c in enumerate(centres):
        log.debug(f"querying for centre {ci} @ {c}")
        # search for all points within range of the target distance
        # first neighbour is ignored as it is itself
        # extend neigh list with sublist
        neigh.extend([(ci,cx) for cx in tree.query_pairs(c,dist,p=2.0)])
    # convert list to set to avoid duplicate connections
    return set(neigh)

# function for forming the list of all connection pairs between centres in KDTree
# for each centre location, the KDTree is queried for the 2 nearest points, first of which is itself
# the pair of the centre index and the nearest connecting point is added to the list
# a list of pairs is formed describing the connection between centres
def formNearestPairs(centres,tree=None):
    # if the KDTree is not given build it
    if tree is None:
        log.debug("building KDTree")
        tree = spatial.KDTree(centres)
    # for each centre in centre query the KDTree for the two nearest points
    # form a pair taking the current centre index and the index of the second nearest point
    # build list of pairs and convert to set to remove duplicates
    return set([(ci,tree.query(c,k=2)[1][1]) for ci,c in enumerate(centres)])

# find mean euclidean distance between points
def findMeanEucDist(centres):
    # form matrix of euclidean distances between each point and every other point
    dist = spatial.distance.pdist(centres)
    log.debug(f"constructed euclidean distance matrix of size {dist.shape}")
    # find the mean value of the distances
    mdist = np.mean(dist)
    log.debug(f"mean euclidean distance is {mdist}")
    return mdist

# construct a region adjacency matrix using OpenCV and Numpy
def cvRAGProx(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    log.debug(f"total {labels.max()+1} labels")
    # create graph object
    G = nx.Graph()
    # add nodes
    # number of unique labels
    G.add_nodes_from(np.unique(labels))
    # list of masks
    maskList = {v:255*(labels==v).astype("uint8") for v in np.unique(labels)}
    # edges list
    # will be converted to set afterwards to avoid duplicates
    edges = []
    # images for drawing contours on
    im1 = np.zeros(labels.shape,np.uint8)
    im2 = np.zeros(labels.shape,np.uint8)
    # iterate over all combinations of superpixel masks
    for (n0,m0),(n1,m1) in combinations(maskList.items(),2):
        # combine masks together
        superMask = m0+m1
        # find contours
        ct,_ = cv2.findContours(superMask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        # if there are two contours, then they are not adjacent
        # if there's one contour then they are combined
        if len(ct)==1:
            edges.append((n0,n1))
    # add edges to graph
    G.add_edges_from(set(edges))
    # return graph
    return G

## edge weighting algorithms
# spatial weighting, based off a distance weighted Potts model
def edgeWeightPotts(centres):
    log.debug(f"finding weights for {len(centres)}")
    # returns a num points x num points array where dist[a,b] is dist(pt[a],pt[b])
    # diagonals are zero as it is distance between the same points
    mdist = findMeanEucDist(centres)
    # iterate over matrix to build up list of edges and associated weights
    # arranged as list of (c0,c1,weight)
    log.debug("constructing weighted edge list")
    # construct set to avoid duplicate edges between nodes
    return {(c0,c1,np.linalg.norm(np.asarray(c0)-np.asarray(c1))/mdist) for c0,c1 in combinations(range(len(centres)),2)}

# spatial weighting using a region adjacency matrix from skimage
def edgeWeightPottsRAG(frame,labels,centres=None):
    # if centres is not passed
    if centres is None:
        centres = findSuperpixelCentres(labels)
    # build region adjacency graph
    rag = graph.RAG(labels,2)
    # find mean distance between centres
    mdist = findMeanEucDist(centres)
    # iterate over centres
    for n0,n1 in rag.edges():
        rag.edges[n0,n1]['weight'] = np.linalg.norm(np.asarray(centres[n0])-np.asarray(centres[n1]))*(mdist**-1.0)
    return rag
    
# normalize color distance
#              __                                            __
#             |      distance(mean_color(i),mean_color(j))     |    mean distance betwen centres
# w(i,j)= exp | - -------------------------------------------  | x ------------------------------
#             |     2 x standard deviation of all colors **2   |     distance(centre[i],centre[j])
#              --                                            --
# based off the assumption that similarly colored superpixels should be grouped together
# works well for images containing a range of colors
def edgeWeightsColor(frame,labels,centres):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # function for finding mean euclidean distance between centres
    mdist = findMeanEucDist(centres)
    log.debug(f"mean euclidean distance between centres is {mdist}")
    # find the mean colors of the superpixels
    def findMeanColor(v,labels=labels,frame=frame):
        log.debug(f"finding mean superpixel color for label {v}")
        m = (labels!=v).astype("uint8")
        fmask = np.ma.masked_array(frame,mask=np.dstack((m,)*3))
        return fmask.mean(axis=0).mean(axis=0).data
    # use list comprehension as it is faster
    meanCols = [findMeanColor(v) for v in np.unique(labels)]
    # find the standard deviation between the colors
    meanColStd = np.std(np.asarray(meanCols))
    log.debug(f"standard deviation between colors is {meanColStd}")
    # list of node indicies and their weight, (n0,n1,w01)
    wedges = set()
    # iterate over pairs of centres getting their indicies (ci*) and centres (c*)
    for (ci0,c0),(ci1,c1) in combinations(enumerate(range(len(centres))),2):
        # find the euclidean distance between mean colors
        colDist = spatial.distance.euclidean(meanCols[ci0],meanCols[ci1])
        log.debug(f"{c0}, {c1} : distance between mean colors {colDist}")
        # find euclidean distance between centres
        dist = spatial.distance.euclidean(c0,c1)
        log.debug(f"{c0}, {c1} : distance between centres {dist}")
        wedges.add((c0,c1,np.exp(-(colDist/(2.0*meanColStd**2.0)))*(mdist/dist)))
    return wedges

# build a graph from superpixel centres using RAG and features weighting using a region adjacency matrix from skimage
def edgeWeightsColorRAG(frame,labels,centres=None):
    # if centres is not passed
    if centres is None:
        centres = findSuperpixelCentres(labels)
    # find the mean colors of the superpixels
    def findMeanColor(v,labels=labels,frame=frame):
        log.debug(f"finding mean superpixel color for label {v}")
        m = (labels!=v).astype("uint8")
        fmask = np.ma.masked_array(frame,mask=np.dstack((m,)*3))
        return fmask.mean(axis=0).mean(axis=0).data
    # use list comprehension as it is faster
    meanCols = [findMeanColor(v) for v in np.unique(labels)]
    # find the standard deviation between the colors
    meanColStd = np.std(np.asarray(meanCols))
    # build region adjacency graph
    # builds off the labels matrix with connectivity set to 2 to include diagonal neighbours
    rag = graph.RAG(labels,2)
    # find mean distance between centres
    mdist = findMeanEucDist(centres)
    # iterate over centres
    # weights are currently distance between mean colors
    for n0,n1 in rag.edges():
        rag.edges[n0,n1]['weight'] = (spatial.distance.euclidean(meanCols[n0],meanCols[n1])*(meanColStd**-1.0))*(np.linalg.norm(np.asarray(centres[n0])-np.asarray(centres[n1]))*(mdist**-1.0))
    return rag

# feature weighting
# Manhattan distance between complete feature vectors normalized by their standard deviation
#              __                                            __
#             |      distance(features(i),features(j))         |    mean distance betwen centres
# w(i,j)= exp | - -------------------------------------------  | x ------------------------------
#             |     2 x standard deviation of features **2     |     distance(centre[i],centre[j])
#              --                                            --
def edgeWeightsFeatures(colF,textF,centres):
    log.debug(f"for color features matrix {colF.shape} and texture features matrix {textF.shape}")
    # finding mean distance
    mdist = findMeanEucDist(centres)
    log.debug(f"mean euclidean distance between centres is {mdist}")
    # combine features into supermatrix
    fvec = np.concatenate((colF,textF),axis=1)
    log.debug(f"formed feature matrix of shape {fvec.shape}")
    # find standard deviation
    vecStd = fvec.std(axis=(0,1))
    log.debug(f"standard deviation between features is {vecStd}")
    # features matrix are number of samples x number of features
    # samples = the superpixel centres
    # iterating over combinations of centres
    wedges = set()
    for (cix,cx),(ciy,cy) in combinations(enumerate(centres),2):
        # find distance between centres
        dist = spatial.distance.euclidean(cx,cy)
        log.debug(f"{cx}, {cy} : distance between centres {dist}")
        # compute the manhattan distance (cityblock) between the feature vectors
        # for each sample (superpixel)
        fdist = spatial.distance.cityblock(fvec[cix,:],fvec[ciy,:])
        log.debug(f"{cx}, {cy} : Manhattan distance between features is {fdist}")
        # update list with indicies of nodes the edge is between 
        wedges.add((cix,ciy,np.exp(fdist/(2.0*vecStd**2.0))*(mdist/dist)))
    return wedges

# build a graph from superpixel centres using RAG and features weighting using a region adjacency matrix from skimage
def edgeWeightsFeaturesRAG(frame,labels,colF,textF,centres=None):
    # if centres is not passed
    if centres is None:
        centres = findSuperpixelCentres(labels)
    # build region adjacency graph from labels image
    rag = graph.RAG(labels,2)
    # combine features
    fvec = np.concatenate((colF,textF),axis=1)
    log.debug(f"formed feature matrix of shape {fvec.shape}")
    # find standard deviation
    vecStd = fvec.std(axis=(0,1))
    log.debug(f"standard deviation between features is {vecStd}")
    # find mean distance between centres
    mdist = findMeanEucDist(centres)
    # iterate over the nodes in the adjacency graph and the centres
    for n0,n1 in rag.edges():
        # calculate and assign the weights for the edge between nodes n0 and n1
        rag.edges[n0,n1]['weight'] = (spatial.distance.cityblock(fvec[n0,:],fvec[n1,:])*(2.0*vecStd**-2.0))*(np.linalg.norm(np.asarray(centres[n0])-np.asarray(centres[n1]))*(mdist**-1.0))
    return rag
