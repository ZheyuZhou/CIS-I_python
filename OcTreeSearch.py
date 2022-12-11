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

    newcloest = []
    for d in d_kall:
        cloest = [vertices[int(triangles[0][0])]]
        bound = [np.linalg.norm(cloest - d)]
        # print(d, 'd_k at Octree search for loop')
        tree.FindClosestPoint(d, bound, cloest)
        new = cloest[0]
        newcloest.append(new)
    # print(newcloest, ' new c_k at OctreeSearch')
    return newcloest
