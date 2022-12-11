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

# # find the tip related to B
# def findTip(F_Ak, F_Bk, tipA):
#     tipA4 = np.append(tipA, np.array([1]))
#     # D. Gupta, Numpy.append#, Numpy.append - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.mean.html (accessed November 12, 2022). 
#     tipA4 = np.reshape(tipA4, (4,1))
#     # D. Gupta, Numpy.reshape#, Numpy.reshape - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.reshape.html (accessed October 13, 2022). 
#     d_k = []
#     for i in range(len(F_Ak)):
#         F_aki = F_Ak[i]
#         F_bki = F_Bk[i]
#         d_k.append(np.linalg.inv(F_bki)@F_aki@tipA4)
#         # I. Polat, Numpy.linalg.inv#, Numpy.linalg.inv - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/gene
#     d_k = np.array(d_k)
#     return d_k

# # Find closet triangle point
# def Find_closet_triangle_point(a,p,q,r):
#     k = np.array([[q[0]-p[0],r[0]-p[0]],[q[1]-p[1],r[1]-p[1]],[q[2]-p[2],r[2]-p[2]]])
#     a = np.reshape(a[0:3], (1,3))[0]
#     # D. Gupta, Numpy.reshape#, Numpy.reshape - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.reshape.html (accessed October 13, 2022). 
#     ld = np.linalg.lstsq(k,a - p, rcond=None)[0][0]
#     # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
#     u = np.linalg.lstsq(k,a - p, rcond=None)[0][1]
#     c = p + ld * (q-p) + u * (r-p)
#     # Discussion of the point out of the triangle
#     if ld>=0 and u>=0 and ld+u<=1:
#         cnew = c
#     elif ld<0:
#         cnew = ProjectOnSegment(c,r,p)
#     elif u<0:
#         cnew = ProjectOnSegment(c,p,q)
#     elif ld + u >1:
#         cnew = ProjectOnSegment(c,q,r)
#     return cnew

# # Find the projection of the vector to the plane
# def ProjectOnSegment(c,p,q):
#     ld = (c-p).dot(q-p)/(q-p).dot(q-p)
#     lds = max(0,min(ld,1))
#     cs = p+lds*(q-p)
#     return cs

# # linear search through all the triangles (brute)
# def Find_closet_points(Meshpoints,Triangle_record,s_kn):
#     # Meshpoints is N vertices,Triangle_record is Vertex indices
#     c_total = []
#     d_total = []
#     # Doing the loop for the all points of triangle 
#     for i in range(len(Triangle_record)):
#         n1 = int(Triangle_record[i][0])
#         n2 = int(Triangle_record[i][1])
#         n3 = int(Triangle_record[i][2])
#         p = Meshpoints[n1]
#         q = Meshpoints[n2]
#         r = Meshpoints[n3]
#         # Find the closet point of the triangle 
#         c_k = Find_closet_triangle_point(s_kn,p,q,r)
#         c_total.append(c_k)
#         s_kn = np.reshape(s_kn[0:3], (1,3))[0]
#         dis = np.linalg.norm(c_k - s_kn)
#         # D. Gupta, Numpy.linalg.norm - NumPy v1.23 Manual. (2022).https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
#         d_total.append(dis)
#         # storage of the data
#     c_total = np.array(c_total)
#     d_total = np.array(d_total)
#     return c_total,d_total

# # Select the real cloest points from all the points we . 
# def Find_closet_mesh_point(c_total,s_kn):
#     dist = []
#     s_kn = np.reshape(s_kn[0:3], (1,3))[0]
#     for i in range(len(c_total)):
#         dist.append(np.linalg.norm(c_total[i] - s_kn))
#     dist = np.array(dist)
#     closet_dist_index = np.argmin(dist)
#      # D. Gupta, Numpy.argmax and argmin #, Numpy.argmin - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.arg
#     return c_total[closet_dist_index]

# # Bounding sphere of three points triangle
# def Finding_centralpoint_radius(a,b,c):
#     f = (a+b)/2
#     u = a - f
#     v = c - f
#     d = np.cross(np.cross(u,v),u)
#     ld = max(0, (v.dot(v) - u.dot(u))/(2*d.dot(v-u)))
#     q = f + ld*d
#     # q is the central point of sphere
#     p = np.linalg.norm(a-q)
#     # p is the radius
#     return q,p


# # Find the c_k with the bounding sphere judgement
# def Closet_bounding_points(Meshpoints,Triangle_record,s_kn):
#     s_kn3 = np.reshape(s_kn[0:3], (1,3))[0]
#     bound = np.linalg.norm(s_kn3)
#     for i in range(len(Triangle_record)):
#         n1 = int(Triangle_record[i][0])
#         n2 = int(Triangle_record[i][1])
#         n3 = int(Triangle_record[i][2])
#         p = Meshpoints[n1]
#         q = Meshpoints[n2]
#         r = Meshpoints[n3]
#         ph = Finding_centralpoint_radius(p,q,r)[1]
#         ct = Finding_centralpoint_radius(p,q,r)[0]
#         # calculate the radius of the sphere 
#         d1 = np.linalg.norm(s_kn3 - ct) - ph
#         if d1 < bound:
#             c_k1 = Find_closet_triangle_point(s_kn,p,q,r)
#             d2 = np.linalg.norm(s_kn3 - c_k1)
#             if d2 < bound:
#                 c_k = c_k1
#                 bound = d2
        
#     return c_k



#######################################################################################
#######################################################################################
############### Main         ##########################################################
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    # pa3 file names
    pa3_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown=', 'J-Unknown'])
    # pa3_address_name = np.array(['A-Debug'])
    len_pa3_address_name = len(pa3_address_name)
    # get PA3 Marker data
    pa3_BodyA = pd.read_csv('2022_pa345_student_data\Problem3-BodyA.txt', header=None)

    pa3_BodyB = pd.read_csv('2022_pa345_student_data\Problem3-BodyB.txt', header=None)

    pa3_BodyA = pa3_BodyA.to_numpy()
    pa3_BodyB = pa3_BodyB.to_numpy()

    pa3_Num_A_Marker = dataimport.Num_Marker(pa3_BodyA)
    pa3_Num_B_Marker = dataimport.Num_Marker(pa3_BodyB)

    pa3_Marker_A_XYZ = dataimport.Marker_XYZ(pa3_BodyA)
    pa3_Marker_B_XYZ = dataimport.Marker_XYZ(pa3_BodyB)

    tip_A_XYZ = dataimport.tip_XYZ(pa3_BodyA)
    tip_B_XYZ = dataimport.tip_XYZ(pa3_BodyB)

    # get PA3 mesh data
    pa3_Mesh = pd.read_csv('2022_pa345_student_data\Problem3Mesh.sur', header=None)

    pa3_Mesh = pa3_Mesh.to_numpy()

    pa3_vertices = dataimport.Vertices(pa3_Mesh)

    pa3_triangles = dataimport.Triangles(pa3_Mesh)

    # print(pa3_vertices, 'ver')
    print(pa3_triangles, 'tri')
    # get PA3 A B LED marker data
    pa3_frameA2J_A_B_record_dict, pa3_frameA2J_A_B_record_dict_A_key, pa3_frameA2J_A_B_record_dict_B_key = dataimport.SampleReading_A_B_Record_Dictionary(pa3_address_name, pa3_Num_A_Marker, pa3_Num_B_Marker)

    # frame AB marker cloudregistration

    # Cloud Registration
    F_A_frame = []
    F_B_frame = []
    for frame in range(len_pa3_address_name):
        A_key = pa3_frameA2J_A_B_record_dict_A_key[frame]
        B_key = pa3_frameA2J_A_B_record_dict_B_key[frame]
        frame_A_record_XYZ = pa3_frameA2J_A_B_record_dict[A_key]
        frame_B_record_XYZ = pa3_frameA2J_A_B_record_dict[B_key]

        F_A = []
        F_B = []
        for sample in range(len(frame_A_record_XYZ)):
            F_A.append(cloudregistration.Cloudregistration(pa3_Marker_A_XYZ, frame_A_record_XYZ[sample]))
            F_B.append(cloudregistration.Cloudregistration(pa3_Marker_B_XYZ, frame_B_record_XYZ[sample]))
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
    

    # Brute force
    # c_closest_frame = []
    # for s_k in s_k_frame:
    #     c_closest_sample = []
    #     for s in s_k:
    #         c_total,_ = Find_closet_points(pa3_vertices,pa3_triangles,s)
    #         c_closest = Find_closet_mesh_point(c_total,s)
    #         c_closest_sample.append(c_closest)
    #     c_closest_sample = np.array(c_closest_sample)
    #     c_closest_frame.append(c_closest_sample)
    # c_closest_frame = np.array(c_closest_frame)

    # c_closest_frame = BruteSearch.BruteSearch(s_k_frame, pa3_vertices,pa3_triangles)
    
    

    # # Bounding
    # c_closest_frame = []
    # for s_k in s_k_frame:
    #     c_closest_sample = []
    #     for s in s_k:
    #         c_closest = Closet_bounding_points(pa3_vertices,pa3_triangles,s)
    #         c_closest_sample.append(c_closest)
    #     c_closest_sample = np.array(c_closest_sample)
    #     c_closest_frame.append(c_closest_sample)
    # c_closest_frame = np.array(c_closest_frame)

    c_closest_frame = BSSearch.BoundingSphereSearch(s_k_frame, pa3_vertices,pa3_triangles)

    # print(c_closest_frame)
    # print(d_k3_frame)

    #######################################################################################
    #######################################################################################
    ############### Output       ##########################################################
    #######################################################################################
    #######################################################################################

    # output_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'])

    # output_frame = []
    # for i in range (len_pa3_address_name):
    #     d_k3_sample = d_k3_frame[i]
    #     c_closest_sample = c_closest_frame[i]
        
    #     output_sample = []
    #     output_num = len(d_k3_sample)
    #     output_text = "pa3-"+output_name[i]+"-Output.txt"
    #     output_row = np.array([[output_num, output_text,"NaN","NaN","NaN","NaN","NaN"]])
        
    #     for j in range(len(d_k3_sample)):
    #         d_k3 = d_k3_sample[j]
    #         c_closest = c_closest_sample[j]
    #         diff = np.array([np.linalg.norm(d_k3 - c_closest)])

    #         output = np.hstack((d_k3, c_closest, diff))
    #         output_sample.append(output)
    #     output_sample = np.array(output_sample)

    #     output = np.concatenate((output_row, output_sample),axis=0)
    #     print(np.shape(output))
    #     print(output)
    #     pd.DataFrame(output).to_csv('2022_pa345_student_data\Output\PA3_'+output_name[i]+'_output.txt')
    #     pd.DataFrame(output).to_csv('2022_pa345_student_data\Output\PA3_'+output_name[i]+'_output.csv')

    #     output_frame.append(output_sample)
    # output_frame = np.array(output_frame)

    #######################################################################################
    #######################################################################################
    ############### Final Test   ##########################################################
    #######################################################################################
    #######################################################################################
    # pa3_test_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug'])
    # for i in range(len(pa3_test_name)):
    #     check = pd.read_csv('2022_pa345_student_data\PA3-'+pa3_test_name[i]+'-Output.txt',header=None, skiprows = 1)
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
                
        
