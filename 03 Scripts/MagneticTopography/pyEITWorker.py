import threading
import numpy as np

from pyeit import mesh 
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.greit import GREIT as greit
from pyeit.eit.bp import BP as bp
from pyeit.eit.jac import JAC as jacobian
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.fem import Forward
from skimage.draw import line as ll
from skimage.transform import radon,iradon_sart,iradon

# queue objects are used for managing data
from queue import Queue
# the gui object is imported to use its parse line method for processing the offline data
from PathPatternSolverGUI import PathPatternSolverGUI
# time is used for simulating delays between measurements
import time

# class for radon solving
# produces weird results
class RadonReconstruction:
    def __init__(self,n_el,dist=1,h0=0.08):
        from itertools import combinations
        # problem, if I go down to say, 10, then some of the lines
        # aren't displayed.
        self.image_pixels = 100

        #self.img = np.zeros((self.image_pixels, self.image_pixels),
        #                    dtype=np.float)

        self.img = np.full((self.image_pixels, self.image_pixels),h0,
                            dtype=np.float)

        # Above should be calculated elsewhere and is only for
        # plotting purposes.
        self.x_center = self.image_pixels/2
        self.y_center = self.image_pixels/2
        self.radius = self.image_pixels/2 - self.image_pixels/10
        #
        # generate combinations of coils according to specification
        # excitation order
        #self.logfile = list(combinations(range(n_el), 2))
        self.logfile = eit_scan_lines(n_el,dist)
        # specify the angle of the electrode's positions
        self.theta_points = [np.pi + (i*np.pi/(n_el/2)) if (np.pi + (i*np.pi/(n_el/2)))%2*np.pi != 0 else 0 for i in range(n_el)]
        # order of the coils
        # used in plotting
        self.el_pos = [i for i in range(n_el)]

    def makeimages(self, data):
        #if len(data) != len(self.logfile):
        #    raise ValueError(
        #        "the datasets must match the logfile specification"
        #    )

        # calculate the positions of the electrodes in the image space
        n1 = np.add(self.x_center*np.ones(len(self.theta_points)),
                    self.radius*np.cos(self.theta_points))
        n2 = np.add(self.y_center*np.ones(len(self.theta_points)),
                    self.radius*np.sin(self.theta_points))

        x = n1.astype(np.int)
        y = n2.astype(np.int)

        d = dict()
        for i, (point1, point2) in enumerate(self.logfile):

            # get the gradient angle theta
            g1 = x[point2] - x[point1]
            g2 = y[point2] - y[point1]
            angle = np.rad2deg(np.arctan2(g2, g1))

            if angle < 0:
                angle = angle + 180
            elif angle >= 180:
                angle = 0.0

            # get the line coordinates for the connection of the two
            # considered electrodes
            l_x, l_y = ll(x[point1], y[point1], x[point2], y[point2])

            # if we are close to an existing angle reuse this
            for a in d:
                if abs(a-angle) < 5:
                    d[a][l_x, l_y] += data[i]
                    break
            else:  # create a new array in this slot
                img = np.zeros((self.image_pixels, self.image_pixels),
                               dtype=np.float)
                img[l_x, l_y] = data[i]
                d[angle] = img

        deg = list(sorted(d))

        return d, deg

    def reconstruct(self, d, deg):
        interp_projections = []

        # now interpolate each angled projection, to get an
        # approximate radon transform from the EIT data
        for i, degi in enumerate(deg):
            projections = radon(d[degi], theta=deg, circle=True)
            p = projections[:, i]

            # problem is lines at angle, or indices next to each other
            # should just be one value..
            #
            # sift through p. if
            for t in range(len(p)):
                if p[t] > 0:
                    if p[t+1] > p[t]:
                        p[t] = 0
                    if p[t] < p[t-1] or p[t] < p[t-2]:
                        p[t] = 0

            nonzeroind = np.nonzero(p)[0]
            xp = nonzeroind
            yp = p[nonzeroind]

            xp = np.append([0], nonzeroind)
            yp = np.append(yp[0], yp)

            xp = np.append(xp, [len(p)-1])
            yp = np.append(yp, yp[-1])

            # Now interpolate to acquire the preimages for the inverse
            # radon transform
            xnew = np.linspace(0, len(p), len(p))
            yinterp = np.interp(xnew, xp, yp)
            interp_projections.append(yinterp)

        interp_projections = np.array(interp_projections).transpose()

        reconstruction = iradon(interp_projections, theta=deg, circle=True)

        # SART reconstruction with two iterations for improved
        # accuracy
        reconstruction_sart = iradon_sart(interp_projections,
                                          theta=np.array(deg))
        reconstruction_sart2 = iradon_sart(interp_projections,
                                           theta=np.array(deg),
                                           image=reconstruction_sart)

        image = reconstruction_sart2
        return image

    def eit_reconstruction(self, data):
        d, deg = self.makeimages(data)
        return self.reconstruct(d, deg)

# thread manager class for reconstructing in a daemon thread
# based off the worker script in the OpenEIT reconstruction module
class ReconstructionWorkerEIT(threading.Thread):
    # build worker
    # set the number of coils, distance and step of excitation order, type of solver
    # kwargs are additional arguments passed to Thread constructor
    def __init__(self,nc,dist=None,step=1,rtype="GREIT",**kwargs):
        # initialize thread setting it as a daemon thread
        # daemon thread does not block the main thread
        super().__init__(daemon=True)
        # flag for indicating that the thread is running
        self.running = False
        # flag for pausing threads
        self.__pauseThread = False
        # reference data set
        self.f0 = None
        # flag to set reference for next reconstruction as next set of measurements
        # as read off next item in inqutQ
        self.resetRef = True
        # store type of algorithm to run
        self.__rtype = rtype
        # number of coils
        self._nc = nc
        # distance between coil pairs in terms of index
        if dist is None:
            self._dist = nc//2
        else:
            self._dist = dist
            
        self._step = step
        ## data queues
        self.inputQ = None
        self.outputQ = None
        if "inputQ" in kwargs:
            # input queue for measurement cycles
            self.inputQ = kwargs['inputQ']
        else:
            self.inputQ = Queue()

        # if an output data Q was given on creation use it
        # else build it
        if "outputQ" in kwargs:
            # output queue for results
            self.outputQ = kwargs['outputQ']
        else:
            self.outputQ = Queue()
        
        # mesh object for solver
        self.mesh,self.ex_pos = mesh.create(nc,h0=0.08)
        if "ex_mat" not in kwargs:
            # build excitation order
            # updates internal variables
            self.setEPattern(self._dist,step)
        # if a custom excitation order is supplied in kwargs
        # save it
        else:
            self.ex_mat = kwargs["ex_mat"]

        # build excitation class based off desired type
        if rtype == "GREIT":
            self._solver = greit(self.mesh,self.ex_pos,self.ex_mat,self._step,perm=self.mesh['perm'],parser='std')
            self._solver.setup(p=0.50,lamb=0.05,n=nc)
        elif rtype == "JAC":
            self._solver = jacobian(self.mesh,self.ex_pos,self.ex_mat,self._step,perm=self.mesh['perm'],parser='std')
            self._solver.setup(p=0.50,lamb=0.01,method='kotre')
        elif rtype == "BP":
            self._solver = bp(self.mesh,self.ex_pos,self.ex_mat,self._step,perm=self.mesh['perm'],parser='std')
            self._solver.setup(weight='none')
        elif rtype == "RADON":
            self._solver = RadonReconstruction(nc,dist)

    # function to return the type of solver
    def getSolverType(self):
        return self.__rtype

    # shortcut for checking if the input queue is empty
    def finished(self):
        return self.inputQ.empty()

    # get parts of the mesh data object
    def getMeshData(self):
        pts = self.mesh['node']
        return pts[:,0],pts[:,1],self.mesh['element'],self.mesh['perm'],self._solver.el_pos

    # method for explicitly setting the data reference using a matrix outside of the class
    def setRef(self,ref):
        self.f0 = ref
        self.resetRef = False

    # set reset reference flag to update reference data used in next reconstruction loop
    def resetRef(self):
        self.resetRef = True

    # update Queue objects of thread
    # only updates the ones that are given objects for
    def setQ(self,inputQ=None,outq=None):
        if inputQ is not None:
            self.inputQ = inputQ
        if outq is not None:
            self.outputQ = outq
            
    # method for building the excitation pattern
    def setEPattern(self,dist,step=1):
        self.ex_mat = []
        # generate coil pairs and add them to list
        for i in range(self._nc):
            self.ex_mat.append([i,(i+dist)%self._nc])
        # convert list to numpy array
        self.ex_mat = np.array(self.ex_mat)

    # method for changing the number of coils
    # rebuilds the reconstruction class for the new number
    def changeNumCoils(self,nc):
        # pause thread to safely update objects
        self.pause()
        # update number of coils
        self._nc = nc
        # rebuild mesh object
        self.mesh,self.ex_pos = mesh.create(self._nc,h0=0.08)
        # build excitation order
        self.setEPattern(self._dist,self._step)
        # rebuild solver
        if self.__rtype == "GREIT":
            self._solver = greit(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(p=0.50,lamb=0.05,n=nc)
        elif self.__rtype == "JAC":
            self._solver = jacobian(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(p=0.50,lamb=0.01,method='kotre')
        elif self.__rtype == "BP":
            self._solver = bp(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(weight='none')
        elif self.__rtype == "RADON":
            self._solver = RadonReconstruction(nc,dist)

        # resume running
        self.running = True

    # method for changing the reconstruction class
    # rebuilds the reconstruction class according to the desired type
    def changeRType(self,rtype):
        # pause thread so the objects can be safely updated
        self.pause()
        # update stored solver type
        self.__rtype = rtype
        # rebuild solver class
        if self.__rtype == "GREIT":
            self._solver = greit(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(p=0.50,lamb=0.05,n=nc)
        elif self.__rtype == "JAC":
            self._solver = jacobian(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(p=0.50,lamb=0.01,method='kotre')
        elif self.__rtype == "BP":
            self._solver = bp(self.mesh,self.ex_pos,self.ex_mat,parser='std')
            self._solver.setup(weight='none')
        elif self.__rtype == "RADON":
            self._solver = RadonReconstruction(self._nc,dist)

    # method for clearing the data queues, clearing histories etc
    def resetQ(self):
        # pause update
        self.pause()
        # get control of queues safely
        # clear queues
        with self.inputQ.mutex:
            self._ipq.clear()

        with self.outputQ.mutex:
            self.outputQ.clear()
        # resume update
        self.resume()

    # set run flag and start thread
    # only starts is queues have been set
    def startRecon(self):
        # check that the queues have been set
        if (self.inputQ is not None) and (self.outputQ is not None):
            # set flag to indicate that it is running
            self.running = True
            # clear pause flag
            self.__pauseThread = False
            # if the thread is not alive at the moment
            # start thread
            if not self.isAlive():
                self.start()
        else:
            raise ValueError("Input and/or output Queues not set!")

    # set pause flag
    # puts the thread in a while loop while the flag is true
    def pause(self):
        # clear flag
        self.__pauseThread = True

    # clears pause flag
    # allows main update loop to resume
    def resume(self):
        self.__pauseThread = False

    # stop thread
    # clears flag and waits for thread to stop
    def stopRecon(self):
        self.running = False
        self.__pauseThread = False
        if self.isAlive():
            self.join()
        
    # reconstruction action
    def run(self):
        # while running is set, solve and update output queue
        while self.running:
            # loop for pausing the thread
            while self.__pauseThread:
                pass
            
            # if there's data in the input queue
            if not self.inputQ.empty():
                # get input data
                idata = self.inputQ.get()
                # update dataa reference if necessary
                if self.resetRef:
                    self.f0 = idata
                    self.resetRef = False
                
                ## process data and solve
                if self.__rtype == "GREIT":
                    outdata = self._solver.solve(idata,self.f0,normalize=False)
                    # remove nans
                    _,_,outdata = self._solver.mask_value(outdata,mask_value=np.NAN)
                    # get real part
                    outdata = np.real(outdata)
                elif self.__rtype == "JAC":
                    outdata = self._solver.solve(idata,self.f0,normalize=False)
                    outdata = sim2pts(self.mesh['node'],self.mesh['element'],np.real(outdata))
                elif self.__rtype == "BP":
                    outdata = self._solver.solve(idata,self.f0)
                elif self.__rtype == "RADON":
                    outdata = self._solver.eit_reconstruction(idata)
                # place data on output
                try:
                    self.outputQ.put(outdata)
                except:
                    print("Output queue is full!")
                # indicate to the input queue that the process has finished
                self.inputQ.task_done()

def basic_example():
    # create queue objects to hold data
    # maxsize=0 means inf size
    inputQ = Queue()
    outputQ = Queue(maxsize=-1)
    # create worker
    print("Creating worker")
    worker = ReconstructionWorkerEIT(32,16,rtype="GREIT")
    # path to data file
    path = "D:\BEAM\Scripts\MagneticTopography\Scripts\OpenEIT\simdata.txt"
    print("reading in data file")
    # read in data file
    with open(path,'r') as file:
        lines = file.readlines()
    # attempt parse file
    try:
        data = np.array([PathPatternSolverGUI.parseLine(l) for l in lines],dtype='float')
    except ValueError:
        print("Failed to read in file!")
    # fill input queue
    for r in range(data.shape[0]):
        inputQ.put(data[r,:])
    # assign queues to worker
    worker.setQ(inputQ,outputQ)
    print("starting worker")
    # start worker
    worker.startRecon()
    # wait until the input queue has been processed
    print("entering loop")
    while not worker.finished():
        pass
    # kill worker
    worker.stop()
    print("worker stopped")

def delayed_feeding():
    # create queues
    inputQ = Queue()
    outputQ = Queue()
    # create worker
    print("Creating worker")
    worker = ReconstructionWorkerEIT(32,16,rtype="GREIT")
    worker.setQ(inputQ,outputQ)
    # path to data file
    path = "D:\BEAM\Scripts\MagneticTopography\Scripts\OpenEIT\simdata.txt"
    print("reading in data file")
    # read in data file
    with open(path,'r') as file:
        lines = file.readlines()
    # attempt parse file
    try:
        data = np.array([PathPatternSolverGUI.parseLine(l) for l in lines],dtype='float')
    except ValueError:
        print("Failed to read in file!")
    # start worker
    print("starting worker")
    worker.startRecon()
    # on a random delay feed in a row of data
    # delay is meant to simulate the collection of data
    print("starting feeding")
    for r in range(data.shape[0]):
        time.sleep(np.random.randint(1,3,1))
        print("feeding")
        inputQ.put(data[r,:])
    # wait for it to finish processing
    while not worker.finished():
        pass
    # stop worker
    worker.stop()
    print("finished")
    
if __name__ == "__main__":
    #delayed_feeding()
    pass
