import numpy as np
import Octree
import BoundingSphereClass as BSC

import pandas as pd
import CloudRegistration_ as cloudregistration
import pa345DataImport as dataimport
import findtipdk as findtipdk
import BruteSearch as BruteSearch
import BoundingSphereSearch as BSSearch
import FindCloestPoint as FCP
# import IterativeClosestPoint as ICP

# def OcTreeSearch(s_k, d_k, vertices, triangles):
def OcTreeSearch(d_k, vertices, triangles):
    d_kall = []
    for d_k4 in d_k[0]:
        # print(d_k4)
        d_k = np.reshape(d_k4[0:3], (1,3))[0]
        d_kall.append(d_k)
    
    boundspheres, numspheres = BSC.Closet_bounding_points(vertices, triangles)
        
    tree = Octree.BoundingOcTree(boundspheres, numspheres)

    new_closest = []
    first_closest = vertices[int(triangles[0][0])]
    first_bound = np.linalg.norm(first_closest - d_kall[0])
    all_closest = np.zeros((len(d_kall), 3))
    all_bound = np.zeros((len(d_kall), 1))
    for i in range(len(d_kall)):
        for j in range(3):
            all_closest[i][j] = first_closest[j]
        all_bound[i][0] = first_bound
    closest = [0]
    bound = [0]
    adjust = 1
    for i in range(len(d_kall)):
        closest[0] = all_closest[i]
        bound[0] = all_bound[i]
        # print(d, 'd_k at Octree search for loop')
        tree.FindClosestPoint(d_kall[i], bound, closest)
        new = closest[0]
        all_closest[i] = closest[0] + adjust
        all_bound[i] = bound[0] + adjust
        new_closest.append(new)
    # print(newcloest, ' new c_k at OctreeSearch')
    return new_closest