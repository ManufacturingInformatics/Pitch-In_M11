# coding: utf-8
""" demo on forward 2D """
# Based off demo_forward2d example
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
import pyeit.eit.bp as bp

""" 0. build mesh """
def buildMesh(num_coils,h0):
    mesh_obj, el_pos = mesh.create(num_coils, h0=h0)

    # extract node, element, alpha
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    quality.stats(pts, tri)
    return mesh_obj,el_pos

""" 1. FEM forward simulations """
def forwardSim(num_coils,mesh_obj,el_pos):
    # setup EIT scan conditions
    ex_dist, step = (num_coils//2)-1, 1
    #ex_dist,step = 1,1
    ex_mat = eit_scan_lines(num_coils, ex_dist)
    ex_line = ex_mat[0].ravel()

    # calculate simulated data using FEM
    fwd = Forward(mesh_obj, el_pos)
    # run the solver using only the permeability in the given mesh object
    # only one mesh object is used as with live data we'll only have the bk
    # permeability matrix ??
    f0 = fwd.solve_eit(ex_mat,step=step, perm=mesh_obj['perm'])
    return f0,ex_mat,step


if __name__ == "__main__":
    # set number of coils
    num_coils = 16
    # set reverse solver method
    solverMethod = "greit"
    # build mesh data objects and coil reference matrix
    # set background permeability
    mesh_obj,el_pos = buildMesh(num_coils,h0 = 0.08)
    # forward simulatiion
    f,ex_mat,step = forwardSim(num_coils,mesh_obj,el_pos)

    # set variable references
    pts = mesh_obj['node']
    x,y = pts[:,0],pts[:,1]
    tri = mesh_obj['element']

    if solverMethod=="greit":
        eit = greit.GREIT(mesh_obj,el_pos,ex_mat=ex_mat,step=step,parser='std')
        eit.setup(p=0.50,lamb=0.001)

        # setting up the solver function
        def solve(f1,f0,obj=eit):
            ds = obj.solve(f1.v,f0.v)
            _,_,ds = obj.mask_value(ds,mask_value=np.NAN)
            return ds

        # setting up the reverse solving plotting function
        def plotReverse(ds):
            f,ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(np.real(ds),interpolation='none',cmap=plt.cm.viridis)
            f.colorbar(im)
            ax.axis('equal')
            return [f]
        
    elif solverMethod=="bp":
        eit = bp.BP(mesh_obj,el_pos,ex_mat=ex_mat,step=step,parser='std')
        eit.setup(weight='none')

        def solve(f1,f0,obj=eit):
            return 192.0 * eit.solve(f1.v,f0.v)

        def plotReverse(ds,pts=pts,tri=tri):
            f,ax = plt.subplots(figsize=(6,4))
            im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds, cmap=plt.cm.viridis)
            ax.set_title(r'$\Delta$ Conductivities')
            ax.axis('equal')
            f.colorbar(im)
            return [f]
        
    elif solverMethod=="jac":
        eit = jac.JAC(mesh_obj,el_pos,ex_mat=ex_mat,step=step,perm=1.,parser='std')
        eit.setup(p=0.5,lamb=0.01,method='kotre')

        def solve(f1,f0,obj=eit,tri=tri):
            ds = obj.solve(f1.v,f0.v,normalize=False)
            ds_n = sim2pts(pts,tri,np.real(ds))
            return ds_n

        def plotReverse(ds_n,x=x,y=y,tri=tri,perm=mesh_obj['perm'],el_pos=el_pos):
            # plot ground truth of JAC method
            fgt,axgt = plt.subplots(figsize=(6,4))
            im = axgt.tripcolor(x, y, tri, np.real(perm), shading='flat')
            axgt.set_title("JAC Ground Truth")
            fgt.colorbar(im)
            axgt.set_aspect('equal')

            # plot reconstruction
            frec,axrec = plt.subplots(figsize=(6,4))
            im = axrec.tripcolor(x, y, tri, ds_n, shading='flat')
            for i,e in enumerate(el_pos):
                axrec.annotate(str(i+1), xy=(x[e], y[e]), color='r')
            frec.colorbar(im)
            axrec.set_aspect('equal')
            return fgt,frec

    # plot the forward results
    """ 2. plot """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # draw equi-potential lines
    # draw lines to demonstraate the predicted lines of magnetic potential
    # twice the resolution of the number of coils??
    vf = np.linspace(min(f), max(f), 32)
    ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
    # draw mesh structure, grid
    ax1.tripcolor(x, y, tri, np.real(mesh_obj['perm']),
                  edgecolors='k', shading='flat', alpha=0.5,
                  cmap=plt.cm.Greys)
    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], 'ro')
    for i, e in enumerate(el_pos):
        ax1.text(x[e], y[e], str(i+1), size=12)
    ax1.set_title('equi-potential lines')
    # clean up
    ax1.set_aspect('equal')
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
    # fig.savefig('demo_bp.png', dpi=96)

    # solve and plot the reverse results
    ds = solve(f0=f,f1=f)
    plotReverse(ds)
    
    plt.show()
