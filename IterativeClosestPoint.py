import numpy as np
import BruteSearch as BruteSearch
import BoundingSphereSearch as BoundingSphereSearch
import FindCloestPoint as FCP
import CloudRegistration_ as CloudReg

def IterativePointFrameSearch(d_k, Brute, Bounding, Tree, vertices, triangles):
    # initial Frame guess
    F_reg = np.eye(4)

    niter = 0
    c_k_last_iter = np.zeros((4))

    last_error = 0.0

    s_k = F_reg@d_k

    while niter < 200:
        
        # select search method
        if Brute == True:
            c_k = BruteSearch.BruteSearch(s_k, vertices, triangles)
        if Bounding == True:
            c_k = BoundingSphereSearch.BoundingSphereSearch(s_k, vertices, triangles)
        # if Tree == True:
            # c_k = _TreeSearch._TreeSearch(_, s_k, vertices, triangles)

        # save c_k
        c_k_last_iter = c_k

        # calc delta F_reg
        delta_F_reg = CloudReg.Cloudregistration(s_k, c_k)

        # itered F_reg
        F_reg_new = delta_F_reg@F_reg

        tolerance = 0.000001
        check, last_error = CheckCloseness(F_reg,F_reg_new,tolerance,last_error)
        if check == True:
            print('Found close enough c_k and F_reg')
            return c_k, F_reg_new
        
        F_reg = F_reg_new
        niter = niter + 1
    print('Out of iteration loop c_k and F_reg')
    return c_k, F_reg

def CheckCloseness(F_reg,F_reg_new,tolerance,last_error):
    err = np.linalg.norm(F_reg - F_reg_new)
    if abs(err-last_error)<= tolerance or err <= tolerance:
        return True
    last_error = err
    return False, last_error

def IterativeClosestPoint(d_k, Brute, Bounding, Tree, vertices, triangles):
    c_k, F_reg = IterativePointFrameSearch(d_k, Brute, Bounding, Tree, vertices, triangles)

    s_k = F_reg@d_k

    dist = np.linalg.norm(s_k - c_k)
    return s_k, c_k, dist