import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cv2
import os

# LK parameters
feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

lk_params = dict( winSize  = (5,5),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

path = r"C:\Users\david\Documents\CoronaWork\Data\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

def processOpticFlowLK(path,normGlobal=False,**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
    print(f"starting index set to {istart}")

    # prev ref frame
    old_frame = None
    #print(f"refresh rate set to {ridx} frames")
    # reference time
    refTime = time.time()
    newTime = refTime
    color = np.random.randint(0,255,(100,3))
    print("opening file")
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        print(f"ending index set to {iend}")
        # get limits
        if normGlobal:
            tmin,tmax = np.nanmin(dset[()],axis=(0,1,2)),np.nanmax(dset[()],axis=(0,1,2))
            print(f"Dataset limits are {tmin},{tmax}")
        # iterate over frames
        for ff in range(istart,iend):
            print(f"{ff}/{dset.shape[2]},{ff/dset.shape[2]}")
            # get frame
            frame = dset[:,:,ff]
            # remove nans
            np.nan_to_num(frame)
            # remove negatives
            frame[frame<0.0]=0.0
            # normalize
            if normGlobal:
                frame -= tmin
                frame /= (tmax-tmin)
            else:
                frame -= frame.min()
                frame /= (frame.max()-frame.min())
            # conver to 8-bit
            frame *= 255
            frame = frame.astype('uint8')
            if old_frame is None:
                print("setting old frame")
                old_frame = frame.copy()
                p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
                mask = np.zeros(old_frame.shape,dtype=old_frame.dtype)
                continue
                
            # calcualte optic flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
            if p1 is not None:
                # get good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                img = cv2.add(frame,mask)

                cv2.imshow('frame',img)
                cv2.waitKey(1)

            p0 = good_new.reshape(-1,1,2)
            old_frame = frame.copy()

def processOpticFlowGF(path,normGlobal=True,**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
    print(f"starting index set to {istart}")

    # prev ref frame
    old_frame = None
    #print(f"refresh rate set to {ridx} frames")
    # reference time
    refTime = time.time()
    newTime = refTime
    color = np.random.randint(0,255,(100,3))

    print("opening file")
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        print(f"ending index set to {iend}")
        # get limits
        if normGlobal:
            tmin,tmax = np.nanmin(dset[()],axis=(0,1,2)),np.nanmax(dset[()],axis=(0,1,2))
            print(f"Dataset limits are {tmin},{tmax}")
        # iterate over frames
        for ff in range(istart,iend):
            print(f"{ff}/{dset.shape[2]},{ff/dset.shape[2]}")
            # get frame
            frame = dset[:,:,ff]
            # remove nans
            np.nan_to_num(frame)
            # remove negatives
            frame[frame<0.0]=0.0
            # normalize
            if normGlobal:
                frame -= tmin
                frame /= (tmax-tmin)
            else:
                frame -= frame.min()
                frame /= (frame.max()-frame.min())
            # conver to 8-bit
            frame *= 255
            frame = frame.astype('uint8')
            # check if first frame and initialize variables
            if ff==istart:
                hsv = np.zeros((*frame.shape,3),dtype=frame.dtype)
                hsv[...,1]=255
                prvs = frame.copy()
                continue
            # calculate optic flow
            flow = cv2.calcOpticalFlowFarneback(prvs,frame,None,0.5,3,15,3,5,1.2,0)
            # convert flow magnitude and angle to 
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            #hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            # update previous points
            prvs = frame.copy()
            # show coloured results
            cv2.imshow("flow",rgb)
            cv2.waitKey(1)

def processOpticFlowQuiverGF(path,normGlobal=True,**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
    print(f"starting index set to {istart}")

    # prev ref frame
    old_frame = None
    #print(f"refresh rate set to {ridx} frames")
    # reference time
    refTime = time.time()
    newTime = refTime
    color = np.random.randint(0,255,(100,3))
    f,ax = plt.subplots(1,2)
    print("opening file")
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        print(f"ending index set to {iend}")
        # get limits
        if normGlobal:
            tmin,tmax = np.nanmin(dset[()],axis=(0,1,2)),np.nanmax(dset[()],axis=(0,1,2))
            print(f"Dataset limits are {tmin},{tmax}")
        # iterate over frames
        for ff in range(istart,iend):
            print(f"{ff}/{iend},{ff/iend}")
            # get frame
            frame = dset[:,:,ff]
            ax[0].imshow(frame,cmap='gray')
            # remove nans
            np.nan_to_num(frame)
            # remove negatives
            frame[frame<0.0]=0.0
            # normalize
            if normGlobal:
                frame -= tmin
                frame /= (tmax-tmin)
            else:
                frame -= frame.min()
                frame /= (frame.max()-frame.min())
            # conver to 8-bit
            frame *= 255
            frame = frame.astype('uint8')
            # check if first frame and initialize variables
            if ff==istart:
                prvs = frame.copy()
                XX,YY = np.meshgrid(np.arange(0,frame.shape[1]),np.arange(0,frame.shape[0]))
                qv = ax[1].quiver(XX,YY,np.zeros(XX.shape,XX.dtype),np.zeros(YY.shape,YY.dtype),units='xy',scale=0.5)
                ax[1].invert_yaxis()
                plt.draw()
                plt.pause(1)
                continue
            # calculate optic flow
            flow = cv2.calcOpticalFlowFarneback(prvs,frame,None,0.5,3,15,3,5,1.2,0)
            # convert flow magnitude and angle to 
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # normalize magnitude
            mag -= mag.min()
            mag /= (mag.max()-mag.min())
            # calculate X and Y components
            U = mag*np.cos(ang)
            V = mag*np.sin(ang)
            qv.set_UVC(U,V)
            plt.draw()
            plt.pause(0.02)
        plt.show()

def processFlowQuiver(path,normGlobal=True,**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
    print(f"starting index set to {istart}")

    # prev ref frame
    old_frame = None
    #print(f"refresh rate set to {ridx} frames")
    # reference time
    refTime = time.time()
    newTime = refTime
    color = np.random.randint(0,255,(100,3))
    f,ax = plt.subplots(1,2)
    print("opening file")
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        print(f"ending index set to {iend}")
        # get limits
        if normGlobal:
            tmin,tmax = np.nanmin(dset[()],axis=(0,1,2)),np.nanmax(dset[()],axis=(0,1,2))
            print(f"Dataset limits are {tmin},{tmax}")
        # iterate over frames
        for ff in range(istart,iend):
            print(f"{ff}/{iend},{ff/iend}")
            # get frame
            frame = dset[:,:,ff]
            frame = np.rot90(frame,k=1,axes=(1,0))
            ax[0].imshow(frame,cmap='gray')
            # find differences
            U = np.diff(frame,axis=0)
            V = np.diff(frame,axis=1)
            # norm
            U -= U.min()
            U /= (U.max()-U.min())
            V -= V.min()
            V /= (V.max()-V.min())
            # check if first frame and initialize variables
            if ff==istart:
                prvs = frame.copy()
                XX,YY = np.meshgrid(np.arange(0,U.shape[1]),np.arange(0,V.shape[0]))
                qv = ax[1].quiver(XX,YY,np.zeros(XX.shape,XX.dtype),np.zeros(YY.shape,YY.dtype),units='xy',scale=0.5)
                ax[1].invert_yaxis()
                plt.draw()
                plt.pause(1)
                continue
            qv.set_UVC(U[:,:V.shape[1]],V[:U.shape[0],:])
            f.suptitle(f"Directional change for frame {ff}")
            plt.draw()
            plt.pause(0.02)
        plt.show()

def processFlowQuiverAnim(path,animpath="thermalQuiverAnim.mp4",**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
    print(f"starting index set to {istart}")
    
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        ## setup axes
        f,ax = plt.subplots(ncols=2)
        # meshgrid for quiver
        XX,YY = np.meshgrid(np.arange(0,dset.shape[1]-1),np.arange(0,dset.shape[0]-1))
        # initialize quiver object
        qv = ax[1].quiver(XX,YY,np.zeros(XX.shape,XX.dtype),np.zeros(YY.shape,YY.dtype),units='xy',scale=0.5)
        # invert y axis so the 0,0 pixel is in the top left corner
        #ax[1].invert_yaxis()
        # function to update plot
        def set_quiver(ff):
            print(f"Animating frame {ff}")
            # update plot
            frame = dset[:,:,ff]
            # rotate image so laser is coming down from the top
            frame = np.rot90(frame,k=1,axes=(1,0))
            # update image data
            ax[0].imshow(frame,cmap='gray')
            # find differences in the respective direction
            U = np.diff(frame,axis=0)
            V = np.diff(frame,axis=1)
            # normalize
            U -= U.min()
            U /= (U.max()-U.min())
            V -= V.min()
            V /= (V.max()-V.min())
            # update quiver
            qv.set_UVC(V[:U.shape[0],:],U[:,:V.shape[1]])
            # update title
            f.suptitle(f"Directional change for frame {ff}")
            return qv,
        print("starting animation")
        anim = animation.FuncAnimation(f,set_quiver,frames=np.arange(istart,iend+1),interval=20,blit=True)
        print(f"saving animation to {animpath}")
        anim.save(animpath,writer='ffmpeg')

def processFlowContourAnim(path,animpath="thermalContourfAnim.mp4",**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
        print(f"starting index set to {istart}")
        if 'picsOnly' in kwargs:
            if bool(kwargs['picsOnly']):
                picsOnly=True
                os.makedirs("contourPicsOnly",exist_ok=True)
        else:
            picsOnly=False
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        print(f"dataset size {dset.shape}")
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        ## setup axes
        f,ax = plt.subplots(nrows=2,ncols=2)
        ax[0,0].set_axis_off()
        ax[0,1].set_axis_off()
        ax[1,0].set_axis_off()
        ax[0,1].set_axis_off()
        # meshgrids for contours
        zX,zY = np.meshgrid(np.arange(0,dset.shape[0]-1),np.arange(0,dset.shape[1]-1))
        uX,uY = np.meshgrid(np.arange(0,dset.shape[0]),np.arange(0,dset.shape[1]-1))
        vX,vY = np.meshgrid(np.arange(0,dset.shape[0]-1),np.arange(0,dset.shape[1]))
        # initialize quiver object
        uct = ax[0,1].contourf(uX,uY,np.zeros(uX.shape,uX.dtype),cmap='hsv')
        vct = ax[1,0].contourf(vX,vY,np.zeros(vX.shape,vX.dtype),cmap='hsv')
        zct = ax[1,1].contourf(zX,zY,np.zeros(zX.shape,zX.dtype),cmap='hsv')
        # setup colorbar axes
        udiv = make_axes_locatable(ax[0,1])
        vdiv = make_axes_locatable(ax[1,0])
        zdiv = make_axes_locatable(ax[1,1])
        uax = udiv.append_axes('right','5%','3%')
        vax = vdiv.append_axes('right','5%','3%')
        zax = zdiv.append_axes('right','5%','3%')
        # setup colorbars
        ucb = f.colorbar(uct,cax=uax)
        vcb = f.colorbar(vct,cax=vax)
        zcb = f.colorbar(zct,cax=zax)
        # invert y axis so the 0,0 pixel is in the top left corner
        ax[0,1].invert_yaxis()
        ax[1,0].invert_yaxis()
        ax[1,1].invert_yaxis()
        # set axes labels
        ax[0,1].set_title("X direction change")
        ax[1,0].set_title("Y direction change")
        ax[1,1].set_title("$\sqrt{U^{2}+V^{2}}$ direction change")
        # function to update plot
        def set_quiver(ff):
            print(f"Animating frame {ff}")
            # update plot
            frame = dset[:,:,ff]
            # rotate image so laser is coming down from the top
            frame = np.rot90(frame,k=1,axes=(1,0))
            # update image data
            ax[0,0].imshow(frame,cmap='gray')
            # find differences in the respective direction
            U = np.diff(frame,axis=0)
            V = np.diff(frame,axis=1)
            # normalize
            U -= U.min()
            U /= (U.max()-U.min())
            V -= V.min()
            V /= (V.max()-V.min())
            # calculate norm
            Z = np.sqrt(U[:,:V.shape[1]]**2 + V[:U.shape[0],:]**2)
            ## update colorbars
            # clear cb axes
            uax.cla()
            vax.cla()
            zax.cla()
            # rebuild contours
            print("rebuilding U ct")
            print(f" uX {uX.shape}, uY {uY.shape}, U {U.shape}")
            uct = ax[0,1].contourf(uX,uY,U,cmap='hsv')
            print("rebuilding V ct")
            vct = ax[1,0].contourf(vX,vY,V,cmap='hsv')
            print("rebuilding Z ct")
            print(f" zX {zX.shape}, zY {zY.shape}, Z {Z.shape}")
            zct = ax[1,1].contourf(zX,zY,Z,cmap='hsv')
            # rebuild colorbars
            print("rebuilding colorbars")
            print(" U cb")
            ucb = f.colorbar(uct,cax=uax)
            print(" V cb")
            vcb = f.colorbar(vct,cax=vax)
            print(" Z cb")
            zcb = f.colorbar(zct,cax=zax)
            # update title
            print("setting plot title")
            f.suptitle(f"Directional change for frame {ff}")
            return

        if picsOnly:
            print("starint collecting pictures only")
            for ff in range(istart,iend+1):
                set_quiver(ff)
                plt.gcf().savefig(os.path.join("contourPicsOnly",f"contour_change_{ff}.png"))
        else:
            print("starting animation")
            anim = animation.FuncAnimation(f,set_quiver,frames=np.arange(istart,iend+1),interval=20)
            print(f"saving animation to {animpath}")
            anim.save(animpath,writer='ffmpeg')

def processFlowImshowAnim(path,animpath="thermalImshowAnim.mp4",**kwargs):
    istart=0
    if kwargs is not None:
        if 'istart' in kwargs:
            istart = int(kwargs['istart'])
        print(f"starting index set to {istart}")
        if 'picsOnly' in kwargs:
            if bool(kwargs['picsOnly']):
                picsOnly=True
                os.makedirs("contourPicsOnly",exist_ok=True)
        else:
            picsOnly=False
    
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        if 'iend' in kwargs:
            iend = kwargs['iend']
        else:
            iend = dset.shape[2]
        ## setup axes
        f,ax = plt.subplots(nrows=2,ncols=2)
        # meshgrid for quiver
        #XX,YY = np.meshgrid(np.arange(0,dset.shape[0]-1),np.arange(0,dset.shape[1]-1))
        # initialize imshow object
        zX,zY = np.meshgrid(np.arange(0,dset.shape[0]-1),np.arange(0,dset.shape[1]-1))
        uX,uY = np.meshgrid(np.arange(0,dset.shape[0]),np.arange(0,dset.shape[1]-1))
        vX,vY = np.meshgrid(np.arange(0,dset.shape[0]-1),np.arange(0,dset.shape[1]))
        
        uim = ax[0,1].imshow(np.zeros(uX.shape,uX.dtype),cmap='hsv')
        vim = ax[1,0].imshow(np.zeros(vX.shape,vX.dtype),cmap='hsv')
        zim = ax[1,1].imshow(np.zeros(zX.shape,zX.dtype),cmap='hsv')
        # setup colorbar axes
        udiv = make_axes_locatable(ax[0,1])
        vdiv = make_axes_locatable(ax[1,0])
        zdiv = make_axes_locatable(ax[1,1])
        uax = udiv.append_axes('right','5%','3%')
        vax = vdiv.append_axes('right','5%','3%')
        zax = zdiv.append_axes('right','5%','3%')
        # setup colorbars
        ucb = f.colorbar(uim,cax=uax)
        vcb = f.colorbar(vim,cax=vax)
        zcb = f.colorbar(zim,cax=zax)
        # invert y axis so the 0,0 pixel is in the top left corner
        ax[0,1].invert_yaxis()
        ax[1,0].invert_yaxis()
        ax[1,1].invert_yaxis()
        # set axes labels
        ax[0,1].set_title("X direction change")
        ax[1,0].set_title("Y direction change")
        ax[1,1].set_title("Z direction change")
        # function to update plot
        def set_quiver(ff):
            print(f"Animating frame {ff}")
            # update plot
            frame = dset[:,:,ff]
            # rotate image so laser is coming down from the top
            frame = np.rot90(frame,k=1,axes=(1,0))
            # update image data
            ax[0,0].imshow(frame,cmap='gray')
            # find differences in the respective direction
            U = np.diff(frame,axis=0)
            V = np.diff(frame,axis=1)
            # normalize
            U -= U.min()
            U /= (U.max()-U.min())
            V -= V.min()
            V /= (V.max()-V.min())
            # calculate norm
            Z = np.sqrt(U[:,:V.shape[1]]**2 + V[:U.shape[0],:]**2)
            # update images
            uim.set_data(U)
            uim.set_clim(U.min(),U.max())
            vim.set_data(V)
            vim.set_clim(V.min(),V.max())
            zim.set_data(Z)
            zim.set_clim(Z.min(),Z.max())
            # update title
            f.suptitle(f"Directional change for frame {ff}")
            return uim,vim,zim
        if picsOnly:
            print("starint collecting pictures only")
            for ff in range(istart,iend+1):
                set_quiver(ff)
                plt.gcf().savefig(os.path.join("imshowPicsOnly",f"contour_change_{ff}.png"))
        else:
            print("starting animation")
            anim = animation.FuncAnimation(f,set_quiver,frames=np.arange(istart,iend+1),interval=20)
            print(f"saving animation to {animpath}")
            anim.save(animpath,writer='ffmpeg')
            
if __name__ == "__main__":
    path = r"C:\Users\david\Documents\CoronaWork\Data\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
    print("calling function")
    #processFlowQuiver(path,istart=91399,iend=150000)
    processFlowContourAnim(path,istart=91399,iend=150000,picsOnly=True)
    cv2.destroyAllWindows()
