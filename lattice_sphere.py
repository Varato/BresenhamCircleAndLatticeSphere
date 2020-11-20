## implementation of Ranita Biswas, Partha Bhowmick, 2016
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

import LatticeSphere_kernel

def LatticeSphere3D(r):
    """
    input:
        r: int, radius of target sphere
    output:
        Q1: indices of Q1 of lattice sphere index S
        arc_length: length of each logitudinal arc in Q1
    """
    if r==0:
        return [(0,0,0)], [1]
    i = j = 0
    # sub 0 for starting points of a new arc
    k = k0 = r
    # sum of squares of i and j
    s = s0 = 0
    # initial interval in Theorem 3
    v = v0 = r - 1
    l = l0 = 2*v0

    Q1 = []
    arc_length = []

    while i <= k: # loop over arcs Ai
        al = 0
        while j <= k: # loop over voxels on Ai
            if s > v: 
                # if s = i^2+j^2 exceeds the interval [u, v], 
                # then k should decrement by 1, and l, v update.
                k = k - 1
                v = v + l
                l = l - 2
            if j <= k and (s != v or j !=k):
                # Theorem 1. Condition of non-simple voxel
                Q1.append((i,j,k))
                al += 1
            # j increases by 1
            s = s + 2*j + 1 
            j = j + 1

        if al > 0:
            arc_length.append(al)
        # start a new arc, where i = j. 
        # Thus increment of i by 1 causes s0 increase by 4i + 2
        s0 = s0 + 4*i + 2
        i = i + 1

        while s0 > v0 and i <= k0:
            k0 = k0 - 1
            v0 = v0 + l0
            l0 = l0 - 2
        j = i
        k = k0
        v = v0
        l = l0
        s = s0

    return Q1, arc_length

def apply_xyz_sym(Q1, cat = True):
    """
    construct Q2 - Q6 by Q1.
    carefully remove repeated points by the rule of edge-containing.
    input:
        the first q-octant Q1 of anything that has permutational symmetry, (num_points, 3)
    output:
        all q-octants, without concatenating.

    To remove repeated points by symmetry, define the folowing:
        edge1 = {p(x,y,z) | p in Q1 and x=y & y!=z}
        edge2 = {p(x,y,z) | p in Q1 and y=z & x!=y}
        edge3 = {p(x,y,z) | p in Q1 and x=y=z}
        Q1_body is Q1 - Union(edge1, edge2, edge3)
    """
    Q1 = np.array(Q1, dtype=np.int) # Q1 contains every edge of it.
    edge1 = Q1[ np.logical_and(Q1[:,0]==Q1[:,1], Q1[:,1]!=Q1[:,2]) ]
    edge2 = Q1[ np.logical_and(Q1[:,1]==Q1[:,2], Q1[:,0]!=Q1[:,1]) ]
    Q1_body = Q1[ reduce(np.logical_and, (Q1[:,0]!=Q1[:,1], Q1[:,1]!=Q1[:,2], Q1[:,2]!=Q1[:,0])) ]

    Q2 = Q1_body[:, [1,0,2]] # Q2 contains edge2
    Q3 = Q1_body[:, [2,0,1]] # Q3 contains edge1
    Q4 = Q1_body[:, [2,1,0]] # Q4 contains edge2
    Q5 = Q1_body[:, [1,2,0]] # Q5 contains edge1
    Q6 = Q1_body[:, [0,2,1]] # Q6 contains none
    if len(edge1) > 0:
        Q3 = np.concatenate((Q3, edge1[:, [2,0,1]]), axis = 0)
        Q5 = np.concatenate((Q5, edge1[:, [1,2,0]]), axis = 0)
    if len(edge2) > 0:
        Q2 = np.concatenate((Q2, edge2[:, [1,0,2]]), axis = 0)
        Q4 = np.concatenate((Q4, edge2[:, [2,1,0]]), axis = 0)

    if cat:
        return np.concatenate((Q1, Q2, Q3, Q4, Q5, Q6), axis = 0)
    else:
        return (Q1, Q2, Q3, Q4, Q5, Q6)

def apply_sign_sym(O1, return_half = False):
    """
    construct other 7 O2-O8 octants given O1
    input:
        O1: points in first octant.
    """
    n = 4 if return_half else 8
    O1 = np.array(O1, dtype = np.int)
    face_xy0 = O1[ reduce(np.logical_and, (O1[:,2]==0, O1[:,0]>0, O1[:,1]>0)) ]
    face_yz0 = O1[ reduce(np.logical_and, (O1[:,0]==0, O1[:,1]>0, O1[:,2]>0)) ]
    face_xz0 = O1[ reduce(np.logical_and, (O1[:,1]==0, O1[:,0]>0, O1[:,2]>0)) ]
    axis_x0 = O1[ reduce(np.logical_and, (O1[:,0]>0, O1[:,1]==0, O1[:,2]==0))]
    axis_y0 = O1[ reduce(np.logical_and, (O1[:,0]==0, O1[:,1]>0, O1[:,2]==0))]
    axis_z0 = O1[ reduce(np.logical_and, (O1[:,0]==0, O1[:,1]==0, O1[:,2]>0))]
    origin = O1[ reduce(np.logical_and, (O1[:,0]==0, O1[:,1]==0, O1[:,2]==0))]

    O1_body = O1[reduce(np.logical_and, (O1[:,0]>0, O1[:,1]>0, O1[:,2]>0))]

    # if return half, then x > 0
    octants = [O1_body * np.array([i,j,k]) for i in (1,-1) for j in (1,-1) for k in (1,-1)][:n]
    face_xy = [face_xy0 * np.array([i,j,1]) for i in (1,-1) for j in (1,-1)][:n//2]
    face_yz = [face_yz0 * np.array([1,j,k]) for j in (1,-1) for k in (1,-1)][:n//2]
    face_xz = [face_xz0 * np.array([i,1,k]) for i in (1,-1) for k in (1,-1)][:n//2]
    axis_x = [axis_x0 * np.array([i,1,1]) for i in (1,-1)][:n//4]
    axis_y = [axis_y0 * np.array([1,j,1]) for j in (1,-1)][:n//4]
    axis_z = [axis_z0 * np.array([1,1,k]) for k in (1,-1)][:n//4]


    objs = octants + face_xy + face_yz + face_xz + axis_x + axis_y + axis_z
    if len(origin) > 0:
        objs += [origin]
    return np.concatenate(objs, axis = 0)



def LatticeShell3D(r_in, r_out, return_half = False):
    assert(r_in < r_out)

    def Arc(i, Q, arc_len):
        low = sum(arc_len[0:i])
        high = low + arc_len[i]
        return Q[low:high]

    Q_in, al_in = LatticeSphere3D(r_in)
    Q_out, al_out = LatticeSphere3D(r_out)

    # num of arcs
    n_in = len(al_in)
    n_out = len(al_out)

    A_in = [Arc(i, Q_in, al_in) for i in range(n_in)]
    A_out = [Arc(i, Q_out, al_out) for i in range(n_out)]

    fill = []
    for i in range(0, A_in[-1][0][0] + 1):
        # notice j is not directly index, but j - i, due to j starts from i in each arc Ai
        fill += [(i,j,k) for j in range(A_in[i][0][1], A_in[i][-1][1] + 1) \
                         for k in range(A_in[i][j-i][2] + 1, A_out[i][j-i][2])]

        fill += [(i,j,k) for j in range(A_in[i][-1][1] + 1, A_out[i][-1][1] + 1) \
                         for k in range(j, A_out[i][j-i][2])]

    fill += [(i,j,k) for i in range(A_in[-1][0][0] + 1, A_out[-1][0][0] + 1) 
                     for j in range(i, A_out[i][-1][1] + 1) \
                     for k in range(j, A_out[i][j-i][2])]

    Q_in = apply_xyz_sym(Q_in)
    Q_out = apply_xyz_sym(Q_out)

    Q_in = apply_sign_sym(Q_in, return_half)
    Q_out = apply_sign_sym(Q_out, return_half)
    if len(fill) > 0:
        fill = apply_xyz_sym(fill)
        fill = apply_sign_sym(fill, return_half)
        return np.array(Q_in), np.array(Q_out), np.array(fill)
    else:
        return np.array(Q_in), np.array(Q_out)



if __name__ == "__main__":
    # test0: draw 1/48 sphere

    import time

    R = 12
    tic = time.time()
    Q1, arc_len = LatticeSphere_kernel.LatticeSphere3D(r = R)
    toc = time.time()
    print("C ext: {:.3f}s".format(toc - tic))

    tic = time.time()
    Q1, arc_len = LatticeSphere3D(r = R)
    toc = time.time()
    print("python code: {:.3f}s".format(toc - tic))


    Q1 = np.array(Q1)
    print(Q1.shape)

    n = 64
    c = n//2
    arr = np.zeros((n,)*3, dtype = np.int)
    arr[c+Q1.T[0], c+Q1.T[1], c+Q1.T[2]] = 1

    fig = plt.figure(figsize = [10,10])
    ax = fig.gca(projection='3d')
    ax.voxels(arr, facecolors = "cyan", edgecolor="k")
    plt.axis("equal")
    plt.show()


    # test1: draw 1/8 sphere
   
    # Q1, arc_length = LatticeSphere3D(r=18)
    # Qs = apply_xyz_sym(Q1, cat = False)
       
    # S = np.concatenate(Qs)

    # u = np.unique(S, axis = 0)
    # print(len(S), len(u))

    # n = 64
    # c = n//2
    # arr = np.zeros((n,)*3, dtype = np.int)
    # arr[c+S.T[0], c+S.T[1], c+S.T[2]] = 1

    # # colors = ['red', 'green', 'blue', 'cyan', 'm', 'y']
    # # color_arr = np.empty(arr.shape, dtype=object)
    # # for i, Q in enumerate(Qs):
    # #     color_arr[c+Q.T[0],c+Q.T[1],c+Q.T[2]] = colors[i]
    
    # fig = plt.figure(figsize = [10,10])
    # ax = fig.gca(projection='3d')
    # ax.voxels(arr, facecolors = "cyan", edgecolor="k")
    # plt.axis("equal")
    # plt.show()

    # 
    # Q1, arc_length = LatticeSphere3D(r=17)
    # Qs = apply_xyz_sym(Q1)
    # S = np.concatenate(Qs)

    ### ======= test for LatticeShell3D =======
    # r1 = 5
    # r2 = 10
    # objs = LatticeShell3D(r1, r2, return_half = True)

    # objs = [o[:,[1,0,2]] for o in objs]
    # obj = np.concatenate(objs)
    # u = np.unique(obj, axis = 0)
    # print(len(obj), len(u))
    # n = 2*r2 + 3
    # c = n//2
    # arr = np.zeros((n,)*3, dtype=np.int)
    # arr[c+obj.T[0], c+obj.T[1], c+obj.T[2]] = 1

    # colors = ['red', 'red', 'cyan']
    # color_arr = np.empty(arr.shape, dtype=object)
    # for i, o in enumerate(objs):
    #     color_arr[c+o.T[0], c+o.T[1], c+o.T[2]] = colors[i]
        
    # fig = plt.figure(figsize = [6,6])
    # ax = fig.gca(projection='3d')
    # ax.voxels(arr, facecolors = color_arr, edgecolor="k")
    # plt.axis("equal")
    # plt.axis("off")
    # plt.show()























