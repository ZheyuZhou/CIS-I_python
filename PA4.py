#######################################################################################
#######################################################################################
############### Imports  ##############################################################
#######################################################################################
#######################################################################################

import numpy as np
# D. Cournapeau, P. Virtanen, A. R. Terrel, NumPy, GitHub. (n.d.). https://github.com/numpy (accessed October 12, 2022). 
import pandas as pd
# W. McKinney, Pandas, Pandas. (2022). https://pandas.pydata.org/ (accessed October 12, 2022). 
# from tqdm import tqdm_gui


#######################################################################################
#######################################################################################
############### Functions    ##########################################################
#######################################################################################
#######################################################################################
import CloudRegistration_ as cloudregistration
import pa345DataImport as dataimport
import pa345_sk_ck as sk_ck
import findtipdk as findtipdk
import BruteSearch as BruteSearch
import BoundingSphereSearch as BSSearch
import FindCloestPoint as FCP
import IterativeClosestPoint as ICP



#######################################################################################
#######################################################################################
############### Main         ##########################################################
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    # pa4 file names run A-K one by one since long run time
    # pa4_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown=', 'J-Unknown', 'K-Unknown'])
    pa4_address_name = np.array(['A-Debug'])

    # read s_k d_k vertices triangles
    s_k_frame, d_k_frame, pa4_vertices, pa4_triangles = sk_ck.sk_ck(pa4_address_name)
    
    # find c_closest point
    c_closest_frame = BSSearch.BoundingSphereSearch(s_k_frame, pa4_vertices,pa4_triangles)

    # choose match/search method
    useBrute = False
    useBounding = False
    useTree = True

    # return s_k c_k
    s_icp_frame = []
    c_icp_frame = np.zeros(3)
    dist_icp_frame = []
    for d_k in d_k_frame:
        s_k, c_k = ICP.IterativeClosestPoint(d_k, useBrute, useBounding, useTree, pa4_vertices, pa4_triangles)
        s_icp_frame.append(s_k)
        c_icp_frame = np.vstack((c_icp_frame,c_k))
    c_icp_frame = np.array(c_icp_frame[1:])
    # print(c_icp_frame, 'c_icp_frame')
    s_icp_frame = s_icp_frame[0]
    # print(s_icp_frame, 's_icp_frame')

    # calc dist
    dist = []
    for i in range(len(s_icp_frame)):
        dist.append(np.array([np.linalg.norm(s_icp_frame[i] - c_icp_frame[i])]))
    dist_icp_fram = np.array(dist)

    # print(dist_icp_fram, 'dist_icp_fram')

    #######################################################################################
    #######################################################################################
    ############### Output       ##########################################################
    #######################################################################################
    #######################################################################################
    # output_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K]) run A-K one by one
    output_name = np.array(['A'])
    
    # combine all the output to a single array
    out_put_array_frame = np.hstack((s_icp_frame, c_icp_frame, dist_icp_frame))
        
    output_num_frame = len(out_put_array_frame)
    output_text_frame = "pa4-"+output_name[0]+"-Output.txt"
    output_row = np.array([[output_num_frame, output_text_frame,"NaN","NaN","NaN","NaN","NaN"]])
        
    out_put_frame = np.vstack((output_row, out_put_array_frame))
    print(out_put_frame)

    # export to txt
    pd.DataFrame(out_put_frame).to_csv('2022_pa345_student_data\Output\PA4_'+output_name[0]+'_output.txt')