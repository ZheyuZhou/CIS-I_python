import numpy as np
import BruteSearch as BruteSearch
import BoundingSphereSearch as BSSearch
import Octree
import FindCloestPoint as FCP
import CloudRegistration_ as CloudReg
import BoundingSphereClass as BSC
import OcTreeSearch

def IterativePointFrameSearch(d_k, Brute, Bounding, Tree, vertices, triangles):
    # initial Frame guess
    F_reg = np.eye(4)

    niter = 0
    # c_k_last = vertices[int(triangles[0][0])]

    last_error = 0.0

    boundspheres, numspheres = BSC.Closet_bounding_points(vertices, triangles)
    
    tree = Octree.BoundingOcTree(boundspheres, numspheres)
    while niter < 200:

        s_k = np.array([F_reg@d_k])

        # if niter == 0:
        #     s_k = s_k + 10
        # print(np.shape(s_k), 's_k shape start ICP')
        
        # select search method
        if Brute == True:
            c_k = BruteSearch.BruteSearch(s_k, vertices, triangles)
            c_k = c_k[0]
        if Bounding == True:
            c_k = BSSearch.BoundingSphereSearch(s_k, vertices, triangles) # s_k 3d [frame, X, 4, 1]
            c_k = c_k[0]
            # print(c_k, 'check c_k')
        if Tree == True:
            c_k = OcTreeSearch.OcTreeSearch(s_k, vertices, triangles)

        # c_k_last = c_k

        # calc delta F_reg
        # print(np.shape(s_k), 's_k shape')
        # print(np.shape(c_k), 'c_k shape')
        
        delta_F_reg = CloudReg.Cloudregistration(s_k, c_k) # s_k 

        # itered F_reg
        F_reg_new = delta_F_reg@F_reg

        tolerance = 0.000001
        check, last_error_new = CheckCloseness(F_reg,F_reg_new,tolerance,last_error)

        # check = check_last_error_new[0]
        last_error = last_error_new

        if check == [1]:
            print('Found close enough c_k and F_reg', niter)
            return c_k, F_reg_new
        
        F_reg = F_reg_new
        niter = niter + 1
    print('Out of iteration loop c_k and F_reg', niter)
    return c_k, F_reg

def CheckCloseness(F_reg,F_reg_new,tolerance,last_error):
    err = np.linalg.norm(F_reg - F_reg_new)
    if abs(err-last_error)<= tolerance or err <= tolerance:
        check = [1]
        return check, err
    last_error = err
    check = [0]
    return check, last_error

def IterativeClosestPoint(d_k, Brute, Bounding, Tree, vertices, triangles):
    c_k, F_reg = IterativePointFrameSearch(d_k, Brute, Bounding, Tree, vertices, triangles)

    s_k = F_reg@d_k
    # print(np.shape(c_k), 'c_k shape at ICP end ')
    s_k3_n = s_k
    s_k3 = []
    for row in s_k3_n:
        s_k3.append(np.reshape(row[0:3], (1,3))[0])
    s_k = np.array(s_k3)
    # print(np.shape(s_k), 's_k 3d shape at ICP end')
    # dist = np.linalg.norm(s_k - c_k)
    # print(dist, 'dist at ICP end')
    return s_k, c_k