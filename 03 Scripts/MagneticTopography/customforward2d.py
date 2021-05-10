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

""" 0. build mesh """
def buildMesh(num_coils,h0):
    mesh_obj, el_pos = mesh.create(num_coils, h0=h0)

    # extract node, element, alpha
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = pts[:, 0], pts[:, 1]
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
    f, _ = fwd.solve(ex_line, perm=mesh_obj['perm'])
    f = np.real(f)
    return f


if __name__ == "__main__":
    # set number of coils
    num_coils = 16
    # build mesh data objects and coil reference matrix
    # set background permeability
    mesh_obj,el_pos = buildMesh(num_coils,h0 = 0.08)
    # forward simulatiion
    f = forwardSim(num_coils,mesh_obj,el_pos)

    # set variable references
    pts = mesh_obj['node']
    x,y = pts[:,0],pts[:,1]
    tri = mesh_obj['element']
    
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
    plt.show()
