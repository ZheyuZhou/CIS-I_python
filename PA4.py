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
import math
# N. Samuel, Math - mathematical functions, Math - Mathematical Functions - Python 3.10.8 Documentation. (2022). https://docs.python.org/3/library/math.html (accessed October 26, 2022). 
from itertools import product


#######################################################################################
#######################################################################################
############### Functions    ##########################################################
#######################################################################################
#######################################################################################
import CloudRegistration_ as cloudregistration
import pa345DataImport as dataimport
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
    # pa3 file names
    # pa3_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown=', 'J-Unknown'])
    pa4_address_name = np.array(['A-Debug'])
    len_pa4_address_name = len(pa4_address_name)
    # get PA3 Marker data
    pa4_BodyA = pd.read_csv('2022_pa345_student_data\Problem4-BodyA.txt', header=None)

    pa4_BodyB = pd.read_csv('2022_pa345_student_data\Problem4-BodyB.txt', header=None)

    pa4_BodyA = pa4_BodyA.to_numpy()
    pa4_BodyB = pa4_BodyB.to_numpy()

    pa4_Num_A_Marker = dataimport.Num_Marker(pa4_BodyA)
    pa4_Num_B_Marker = dataimport.Num_Marker(pa4_BodyB)

    pa4_Marker_A_XYZ = dataimport.Marker_XYZ(pa4_BodyA)
    pa4_Marker_B_XYZ = dataimport.Marker_XYZ(pa4_BodyB)

    tip_A_XYZ = dataimport.tip_XYZ(pa4_BodyA)
    tip_B_XYZ = dataimport.tip_XYZ(pa4_BodyB)

    # get PA3 mesh data
    pa4_Mesh = pd.read_csv('2022_pa345_student_data\Problem4MeshFile.sur', header=None)

    pa4_Mesh = pa4_Mesh.to_numpy()

    pa4_vertices = dataimport.Vertices(pa4_Mesh)

    pa4_triangles = dataimport.Triangles(pa4_Mesh)

    # print(pa3_vertices, 'ver')
    # print(pa3_triangles, 'tri')
    # get PA3 A B LED marker data
    pa3_frameA2J_A_B_record_dict, pa4_frameA2J_A_B_record_dict_A_key, pa4_frameA2J_A_B_record_dict_B_key = dataimport.SampleReading_A_B_Record_Dictionary(pa4_address_name, pa4_Num_A_Marker, pa4_Num_B_Marker)

    # frame AB marker cloudregistration

    # Cloud Registration
    F_A_frame = []
    F_B_frame = []
    for frame in range(len_pa4_address_name):
        A_key = pa4_frameA2J_A_B_record_dict_A_key[frame]
        B_key = pa4_frameA2J_A_B_record_dict_B_key[frame]
        frame_A_record_XYZ = pa3_frameA2J_A_B_record_dict[A_key]
        frame_B_record_XYZ = pa3_frameA2J_A_B_record_dict[B_key]

        F_A = []
        F_B = []
        for sample in range(len(frame_A_record_XYZ)):
            F_A.append(cloudregistration.Cloudregistration(pa4_Marker_A_XYZ, frame_A_record_XYZ[sample]))
            F_B.append(cloudregistration.Cloudregistration(pa4_Marker_B_XYZ, frame_B_record_XYZ[sample]))
            # F_A.append(cloudregistration.Cloudregistration(frame_A_record_XYZ[sample], pa3_Marker_A_XYZ))
            # F_B.append(cloudregistration.Cloudregistration(frame_B_record_XYZ[sample], pa3_Marker_B_XYZ))
        F_A = np.array(F_A)
        F_B = np.array(F_B)
        F_A_frame.append(F_A)
        F_B_frame.append(F_B)

    F_A_frame = np.array(F_A_frame)
    F_B_frame = np.array(F_B_frame)
    

    # Calculate d_k tip in B coordinate frame
    d_k_frame = []
    
    for frame in range(len(F_A_frame)):
        d_k = findtipdk.findTip(F_A_frame[frame], F_B_frame[frame], tip_A_XYZ)
        d_k_frame.append(d_k)
    d_k_frame = np.array(d_k_frame)

    d_k3_frame = []
    for d_k_sample in d_k_frame:
        d_k3_sample = []
        for d_k in d_k_sample:
            d_k3 = np.reshape(d_k[0:3], (1,3))[0]
            d_k3_sample.append(d_k3)
        d_k3_sample = np.array(d_k3_sample)
        d_k3_frame.append(d_k3_sample)
    d_k3_frame = np.array(d_k3_frame)

    # s_k sample points
    s_k_frame = []
    F_reg = np.eye(4) # Identity for PA 3
    
    for frame in range(len(d_k_frame)):
        s_k = F_reg@d_k_frame[frame]
        s_k_frame.append(s_k)
    

    # c_closest_frame = BruteSearch.BruteSearch(s_k_frame, pa3_vertices,pa3_triangles)

    c_closest_frame = BSSearch.BoundingSphereSearch(s_k_frame, pa4_vertices,pa4_triangles)

    useBrute = False
    useBounding = False
    useTree = True

    s_icp_frame = []
    c_icp_frame = []
    dist_icp_frame = []
    for d_k in d_k_frame:
        s_k, c_k, dist = ICP.IterativeClosestPoint(d_k, useBrute, useBounding, useTree, pa4_vertices, pa4_triangles)
        
        s_icp_frame.append(s_k)
        c_icp_frame.append(c_k)
        dist_icp_frame.append(dist)
    # print(c_closest_frame, 'c_closest_frame')
    # print(d_k3_frame)
    print(c_icp_frame, 'c_icp_frame')
    print(np.shape(c_icp_frame))

    # print(s_k_frame, 's_k_frame')
    print(s_icp_frame, 's_icp_frame')

    print(dist_icp_frame, 'dist_icp_frame')



    #######################################################################################
    #######################################################################################
    ############### Output       ##########################################################
    #######################################################################################
    #######################################################################################

    output_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'])

    output_frame = []
    for i in range (len_pa4_address_name):
        d_k3_sample = d_k3_frame[i]
        c_closest_sample = c_closest_frame[i]
        
        output_sample = []
        output_num = len(d_k3_sample)
        output_text = "pa4-"+output_name[i]+"-Output.txt"
        output_row = np.array([[output_num, output_text,"NaN","NaN","NaN","NaN","NaN"]])
        
        for j in range(len(d_k3_sample)):
            d_k3 = d_k3_sample[j]
            c_closest = c_closest_sample[j]
            diff = np.array([np.linalg.norm(d_k3 - c_closest)])

            output = np.hstack((d_k3, c_closest, diff))
            output_sample.append(output)
        output_sample = np.array(output_sample)

        output = np.concatenate((output_row, output_sample),axis=0)
        print(np.shape(output))
        print(output)
        pd.DataFrame(output).to_csv('2022_pa345_student_data\Output\PA4_'+output_name[i]+'_output.txt')
        pd.DataFrame(output).to_csv('2022_pa345_student_data\Output\PA4_'+output_name[i]+'_output.csv')

        output_frame.append(output_sample)
    output_frame = np.array(output_frame)

    #######################################################################################
    #######################################################################################
    ############### Final Test   ##########################################################
    #######################################################################################
    #######################################################################################
    # pa3_test_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug'])
    # for i in range(len(pa4_test_name)):
    #     check = pd.read_csv('2022_pa345_student_data\PA4-'+pa4_test_name[i]+'-Output.txt',header=None, skiprows = 1)
    #     check = check.to_numpy()
    #     check_array = []
    #     for row in check:
    #         row_array = np.fromstring(row[0],dtype=float,sep=' ')
    #         check_array.append(row_array)
    #     check_array = np.array(check_array)    

    #     diff = np.linalg.norm(check_array - output_frame[i])
    #     if diff < 0.1:
    #         print("Test Passed \(o^ ^o)/", diff)
    #     else:
    #         print("Test Fail!!!!!!", diff)
                
        
