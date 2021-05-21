import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

############## FROM https://gist.github.com/royshil/52bd3a0e21e9e7f6a8747bde52a3f86b #################
# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=20)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
# foreground is marked with red
# background is marked wtih blue
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
# the allocated foreground and background are based off markings set previously
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        # if the histogram index belongs to the foreground set
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        # if the histogram index belongs to the background set
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        # if the histogram index does not belong to either
        # compare the foreground histograms to the current histogram and assign to forward edge?
        # compare the background histograms to the current histogram and assign to the backward edge?
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

# perform graph cutting without marked foreground and background
def graph_cut_nofgbg(norm_hists,neighbours):
    num_nodes = norm_hists.shape[0]
    # make graph with 5 edges per node
    g = maxflow.Graph[float](num_nodes,num_nodes*5)
    # add N nodes
    nodes = g.add_nodes(num_nodes)
    # set compare algorithm
    hist_comp_alg = cv2.HISTCMP_KL_DIV

    ## smoothness term is cost between neighbours
    indptr,indicies = neighbours
    for i in range(len(indptr)-1):
        N = indicies[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        # iterate over neighbouts 
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    ## assign costs
    # add a cost based on the comparison between neightbors?
    for i,h in enumerate(norm_hists):
        for n in N:
            if (n<0) or (n>num_nodes):
                continue
            hn = norm_hists[n]
            g.add_tedge(nodes[i], cv2.compareHist(h, hn, hist_comp_alg),
                                  cv2.compareHist(hn, h, hist_comp_alg))
    g.maxflow()
    return g.get_grid_segments(nodes)

def useCamera():
    # setup camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("failed to get camera")
        cam.release()
        return

    ret,frame = cam.read()
    if not ret:
        print("failed to get test frame")
        cam.release()
        return

    # define image marking as a white image
    # meaning we don't know what is foreground or background?
    img_marking = 255*np.ones(frame.shape,frame.dtype)
    while True:
        ret,frame = cam.read()
        if not ret:
            print("failed to get frame")
            cam.release()
            return
        
        centers,colors_hists,segments,neighbors = superpixels_histograms_neighbors(frame)
        fg_segments, bg_segments = find_superpixels_under_marking(img_marking, segments)

        # get cumulative BG/FG histograms, before normalization
        fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, colors_hists)
        bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, colors_hists)
        # normalize histograms
        norm_hists = normalize_histograms(colors_hists)
        # perform graph cut
        graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                                 (fg_segments,        bg_segments),
                                 norm_hists,
                                 neighbors)

        ## plot results
        # get segmentation mask
        segmask = pixels_for_segment_selection(segments, np.nonzero(graph_cut)).astype("uint8")
        # mark bouundaries
        marks = mark_boundaries(frame, segments)
        # update windows
        cv2.imshow("frame",frame)
        cv2.imshow("mask",cv2.normalize(segmask,segmask,0,255,cv2.NORM_MINMAX).astype("uint8"))
        cv2.imshow("markings",marks)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC exit")
            break

    cam.release()

if __name__ == "__main__":
    useCamera()
    cv2.destroyAllWindows()
