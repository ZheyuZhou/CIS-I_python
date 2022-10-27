#  .............................................
#                     _ooOoo_  
#                    o8888888o  
#                    88" . "88  
#                    (| -_- |)  
#                     O\ = /O  
#                 ____/`---'\____  
#               .   ' \\| |// `.  
#                / \\||| : |||// \  
#              / _||||| -:- |||||- \  
#                | | \\\ - /// | |  
#              | \_| ''\---/'' | |  
#               \ .-\__ `-` ___/-. /  
#            ___`. .' /--.--\ `. . __  
#         ."" '< `.___\_<|>_/___.' >'"".  
#        | | : `- \`.;`\ _ /`;.`/ - ` : | |  
#          \ \ `-. \_ __\ /__ _/ .-` / /  
#  ======`-.____`-.___\_____/___.-`____.-'======  
#                     `=---='  
#
#           佛祖保佑             永无BUG 
#  .............................................  

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

from torch import q_zero_point


#######################################################################################
#######################################################################################
############### Functions  ############################################################
#######################################################################################
#######################################################################################
def Cloudregistration(a,A):

    # Calc average
    a_bar = np.mean(a, axis=0)

    A_bar = np.mean(A, axis=0)
    # D. Gupta, Numpy.mean#, Numpy.mean - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.mean.html (accessed October 12, 2022). 
    # Calc difference
    a_tilde = []
    for a_ in a:
        a_tilde.append(a_bar - a_)
    
    A_tilde = []
    for A_ in A:
        A_tilde.append(A_bar - A_)

    # Get H matrix
    H = np.zeros((3,3))

    for num_markers in range(len(A)): # len(A) = len(a) becaus the len is num of marker
        d_tx_D_tx = a_tilde[num_markers][0]*A_tilde[num_markers][0]
        d_tx_D_ty = a_tilde[num_markers][0]*A_tilde[num_markers][1]
        d_tx_D_tz = a_tilde[num_markers][0]*A_tilde[num_markers][2]

        d_ty_D_tx = a_tilde[num_markers][1]*A_tilde[num_markers][0]
        d_ty_D_ty = a_tilde[num_markers][1]*A_tilde[num_markers][1]
        d_ty_D_tz = a_tilde[num_markers][1]*A_tilde[num_markers][2]

        d_tz_D_tx = a_tilde[num_markers][2]*A_tilde[num_markers][0]
        d_tz_D_ty = a_tilde[num_markers][2]*A_tilde[num_markers][1]
        d_tz_D_tz = a_tilde[num_markers][2]*A_tilde[num_markers][2]

        H += np.array([
            [d_tx_D_tx, d_tx_D_ty, d_tx_D_tz],
            [d_ty_D_tx, d_ty_D_ty, d_ty_D_tz],
            [d_tz_D_tx, d_tz_D_ty, d_tz_D_tz]
        ])
        # C. Harris, Numpy.array#, Numpy.array - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.array.html (accessed October 12, 2022). 

    # Treat H and Get G
    Tr_H = np.array([[np.trace(H)]])
    
    H23 = H[1][2]
    H32 = H[2][1]
    H31 = H[2][0]
    H13 = H[0][2]
    H12 = H[0][1]
    H21 = H[1][0]

    
    Delta = np.array([
        [H23-H32],
        [H31-H13],
        [H12-H21]
    ])
    # transpose the matrix
    Delta_T = Delta.T
    
    H_HT_TrH_I = H + H.T - np.eye(3)*np.trace(H)
    # C. Harris, Numpy.trace#, Numpy.trace - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.trace.html (accessed October 12, 2022). 
    # Combine two matrix by horizonal
    G1 = np.hstack((Tr_H, Delta_T))
    G2 = np.hstack((Delta, H_HT_TrH_I))
    # C. Harris, Numpy.hstack#, Numpy.hstack - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.hstack.html (accessed October 12, 2022). 
    # Combine two matrix by vertical
    G = np.vstack((G1,G2))
    # M. Bussonnier, Numpy.vstack#, Numpy.vstack - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.vstack.html (accessed October 12, 2022). 

    # Eig-val Decomp to get Eig-vec corresponds to largest(first) Eig-val
    eig_val, eig_vec = np.linalg.eig(G)
    # I. Polat, Numpy.linalg.eig#, Numpy.linalg.eig - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html (accessed October 12, 2022). 
    # Choose the biggest eigenvalue and corresponding eigenvector
    max_eig_val_index = np.argmax(eig_val)
    # D. Gupta, Numpy.argmax#, Numpy.argmax - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.argmax.html (accessed October 13, 2022). 
    Qk = eig_vec[:,max_eig_val_index]
    # print(eig_vec, 'eig_vec')
    # print(Qk, 'Qk')

    q0 = Qk[0]
    q1 = Qk[1]
    q2 = Qk[2]
    q3 = Qk[3]



    # Plug in Unit Quaternion(Eig-vec above) to get Rotation Matrix
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])

    # Calc translation
    p = A_bar - R@a_bar
    
    p = np.reshape(p, (3,1))
    # D. Gupta, Numpy.reshape#, Numpy.reshape - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.reshape.html (accessed October 13, 2022). 

    # Organize to get Transformation
    F1 = np.hstack((R,p))
    F2 = np.array([0,0,0,1])
    F = np.vstack((F1,F2))

    return F 
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
def Find_min_max(em_all):
    em_all_x = em_all.T[0]
    em_all_y = em_all.T[1]
    em_all_z = em_all.T[2]

    qmin_x = np.min(em_all_x)
    qmin_y = np.min(em_all_y)
    qmin_z = np.min(em_all_z)

    qmax_x = np.max(em_all_x)
    qmax_y = np.max(em_all_y)
    qmax_z = np.max(em_all_z)

    return qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z

def Scale_To_Box(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q):
    
    q_x = em_q[0]
    q_y = em_q[1]
    q_z = em_q[2]

    u_x = (q_x - qmin_x) / (qmax_x - qmin_x)
    u_y = (q_y - qmin_y) / (qmax_y - qmin_y)
    u_z = (q_z - qmin_z) / (qmax_z - qmin_z)

    u = np.array([u_x, u_y, u_z])
    return(u)

def B_5_Poly(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q, k):
    u = Scale_To_Box(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q)
    v = 1 - u
    N = 5

    bionomial_coef = math.comb(N,k)
    
    B_5_k = bionomial_coef * u**(N-k) * v**(k)

    # print(np.shape(B_5_k), ' shape B_5_k')
    return B_5_k

def Tensor_Form(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q):
    P_total = np.zeros((1,3))
    for df_rd_P in em_q:
        P_total = np.vstack((P_total,df_rd_P))
    P_total = P_total[1: , :]

    B_5_k_Poly = []
    for P in em_q:
        for i in range(6):
            B_5_k_Poly.append(B_5_Poly(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, P, i))
    B_5_k_Poly = np.array(B_5_k_Poly)
    B_5_k_Poly = B_5_k_Poly.reshape((len(P_total), 6, 3))

    F_ijk = np.zeros((216))
    for B_5_k in B_5_k_Poly:
        F_row = []
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    F_row.append(B_5_k[i][0]*B_5_k[j][1]*B_5_k[k][2])
        F_row = np.array(F_row)
        # print(np.shape(F_row), ' shape F_row')
        F_ijk = np.vstack((F_ijk,F_row))
    F_ijk = F_ijk[1:, :]
    return F_ijk

def c_ijk_lstsq(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q, C_vec_expected_2D):
    # print(np.shape(em_q), ' shape em_q in c_ijk_lstsq')
    P_F_ijk = Tensor_Form(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q)
    # print(np.shape(em_q), ' shape em_q in c_ijk_lstsq after P_F_ijk')
    P_c_ijk = np.linalg.lstsq(P_F_ijk,C_vec_expected_2D, rcond=None)[0]

    # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
    return P_c_ijk

def Correct_Distortion(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q, C_vec_expected_2D):
    # print(np.shape(em_q), ' shape em_q in Correct_Distortion')
    c_ijk = c_ijk_lstsq(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, em_q, C_vec_expected_2D)
    # print(np.shape(em_q), ' shape em_q in Correct_Distortion after c_ijk')
    P_total = np.zeros((1,3))
    for df_rd_P in em_q:
        P_total = np.vstack((P_total,df_rd_P))
    P_total = P_total[1: , :]

    B_5_k_Poly = []
    for P in em_q:
        for i in range(6):
            B_5_k_Poly.append(B_5_Poly(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, P, i))
    B_5_k_Poly = np.array(B_5_k_Poly)
    B_5_k_Poly = B_5_k_Poly.reshape((len(P_total), 6, 3))

    corrected_P = []
    for B_5_k in B_5_k_Poly:
        corrected_P_row = np.zeros((3))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    order = 36*i+6*j+k
                    # print(np.shape(c_ijk[order]), ' shape cijk order')
                    corrected_P_row += c_ijk[order]*B_5_k[i][0]*B_5_k[j][1]*B_5_k[k][2]
        corrected_P_row = np.array(corrected_P_row)
        # print(np.shape(corrected_P_row), 'shape corrected_P_row')
        corrected_P.append(corrected_P_row)
    corrected_P = np.array(corrected_P)

    return corrected_P

# def Scale_To_Box(q_total, q_c):
#     q_total_T = q_total.T
#     q_total_x = q_total_T[0]
#     q_total_y = q_total_T[1]
#     q_total_z = q_total_T[2]
#     q_total_x_min = np.min(q_total_x)
#     q_total_y_min = np.min(q_total_y)
#     q_total_z_min = np.min(q_total_z)
    
#     q_total_x_max = np.max(q_total_x)
#     q_total_y_max = np.max(q_total_y)
#     q_total_z_max = np.max(q_total_z)
    
#     q_c_T = q_c.T
#     q_c_x = q_c_T[0]
#     q_c_y = q_c_T[1]
#     q_c_z = q_c_T[2]

#     ux = (q_c_x-q_total_x_min) / (q_total_x_max-q_total_x_min)
#     uy = (q_c_y-q_total_y_min) / (q_total_y_max-q_total_y_min)
#     uz = (q_c_z-q_total_z_min) / (q_total_z_max-q_total_z_min)
#     u = np.array([ux, uy, uz])
#     return u

# def B_5_Poly(q_total, q_c, k):
#     u = Scale_To_Box(q_total, q_c)
#     v = 1 - u
#     N = 5

#     bionomial_coef = math.comb(N,k)
#     # ihritik, Python - math.comb() method, GeeksforGeeks. (2020). https://www.geeksforgeeks.org/python-math-comb-method/ (accessed October 26, 2022). 
    
#     B_5_k = bionomial_coef * u**(N-k) * v**(k)

#     # print(np.shape(B_5_k), ' shape B_5_k')
#     return B_5_k

# def Tensor_Form(rd_P, rd_C):
#     P_total = np.zeros((1,3))
#     for df_rd_P in rd_P:
#         P_total = np.vstack((P_total,df_rd_P))
#     P_total = P_total[1: , :]

#     C_total = np.zeros((1,3))
#     for df_rd_C in rd_C:
#         C_total = np.vstack((C_total,df_rd_C))
#     C_total = C_total[1: , :]

#     B_5_k_Poly = []
#     for df_rd_C in rd_C:
#         for C in df_rd_C:
#             for i in range(6):
#                 B_5_k_Poly.append(B_5_Poly(P_total, C, i))
#     B_5_k_Poly = np.array(B_5_k_Poly)
#     B_5_k_Poly = B_5_k_Poly.reshape((len(C_total), 6, 3))

#     F_ijk = np.zeros((216))
#     for B_5_k in B_5_k_Poly:
#         F_row = []
#         for i in range(6):
#             for j in range(6):
#                 for k in range(6):
#                     F_row.append(B_5_k[i][0]*B_5_k[j][1]*B_5_k[k][2])
#         F_row = np.array(F_row)
#         # print(np.shape(F_row), ' shape F_row')
#         F_ijk = np.vstack((F_ijk,F_row))
#     F_ijk = F_ijk[1:, :]
#     return F_ijk

# def c_ijk_lstsq(rd_P, rd_C, C_vec_expected):
#     P_F_ijk = Tensor_Form(rd_P, rd_C)
#     P_c_ijk = np.linalg.lstsq(P_F_ijk,C_vec_expected, rcond=None)[0]
#     # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
#     return P_c_ijk

# def Correct_Distortion(rd_P,rd_C, C_vec_expected):
#     c_ijk = c_ijk_lstsq(rd_P, rd_C, C_vec_expected)
#     P_total = np.zeros((1,3))
#     for df_rd_P in rd_P:
#         P_total = np.vstack((P_total,df_rd_P))
#     P_total = P_total[1: , :]

#     C_total = np.zeros((1,3))
#     for df_rd_C in rd_C:
#         C_total = np.vstack((C_total,df_rd_C))
#     C_total = C_total[1: , :]

#     B_5_k_Poly = []
#     for df_rd_C in rd_C:
#         for C in df_rd_C:
#             for i in range(6):
#                 B_5_k_Poly.append(B_5_Poly(P_total, C, i))
#     B_5_k_Poly = np.array(B_5_k_Poly)
#     B_5_k_Poly = B_5_k_Poly.reshape((len(C_total), 6, 3))

#     corrected_P = []
#     for B_5_k in B_5_k_Poly:
#         corrected_P_row = np.zeros((3))
#         for i in range(6):
#             for j in range(6):
#                 for k in range(6):
#                     order = 36*i+6*j+k
#                     # print(np.shape(c_ijk[order]), ' shape cijk order')
#                     corrected_P_row += c_ijk[order]*B_5_k[i][0]*B_5_k[j][1]*B_5_k[k][2]
#         corrected_P_row = np.array(corrected_P_row)
#         # print(np.shape(corrected_P_row), 'shape corrected_P_row')
#         corrected_P.append(corrected_P_row)
#     corrected_P = np.array(corrected_P)

#     return corrected_P

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

def LeastSquare(F):
        R = np.array([[0,0,0]])
        p = np.array([[0]])
        neg_I = np.array([[0,0,0]])
        for F_j in F:
            # Slice R and p out of F
            R_j = F_j[0:3 , 0:3]
            p_j = F_j[0:3, 3:4]

            neg_I_j = -np.eye(3)

            R = np.concatenate((R,R_j),axis=0)
            # I. Polat, Numpy.linalg.eig#, Numpy.linalg.eig - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html (accessed October 12, 2022). 
            p = np.concatenate((p,p_j),axis=0)
            neg_I = np.concatenate((neg_I,neg_I_j),axis=0)

        # Slice out the 0,0,0 on first row for R, p, neg_I
        R = R[1:, :]
        p = p[1:, :]
        neg_I = neg_I[1:, :]
        
        Coeff_Matrix = np.concatenate((R,neg_I),axis=1)

        # print(Coeff_Matrix)
        # print(np.shape(Coeff_Matrix))
        # Calc t_G and P_dimple with least square
        LS_sol = np.linalg.lstsq(Coeff_Matrix, -p)[0]
        # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
        t_G = LS_sol[0:3]
        P_dimple = LS_sol[3:6]

        return P_dimple, t_G

def Point_Calibration(calpivot, num_calpivot_Frame):
        # points in first data frame
        P_1 = calpivot[0]

        # points' average in first data frame
        P_1_0 = np.mean(P_1, axis=0)

        # Calc p_j differences between points and points average
        p_j = []
        for P_j in P_1:
            p_j.append(P_j - P_1_0)
        p_j = np.array(p_j)

        # Calc F_P Transformation between EM coordinate to pointer
        F_P = []
        for j in range (num_calpivot_Frame):
            # EM points from each data frame
            P_EM = calpivot[j]
        
            # calculate EM marker Point Cloud Transformation 
            # Store all the matrices
            F_P.append(Cloudregistration(p_j, P_EM))
            
        F_P = np.array(F_P)

        P_dimple = LeastSquare(F_P)[0]
        t_p = LeastSquare(F_P)[1]

        return P_dimple, t_p, F_P, p_j

def Framecalcuation(calpivot, num_calpivot_Frame):
        # points in first data frame
        P_1 = calpivot[0]

        # points' average in first data frame
        P_1_0 = np.mean(P_1, axis=0)

        # Calc p_j differences between points and points average
        p_j = []
        for P_j in P_1:
            p_j.append(P_j - P_1_0)
        p_j = np.array(p_j)

        # Calc F_P Transformation between EM coordinate to pointer
        F_P = []
        for j in range (num_calpivot_Frame):
            # EM points from each data frame
            P_EM = calpivot[j]
        
            # calculate EM marker Point Cloud Transformation 
            # Store all the matrices
            F_P.append(Cloudregistration(p_j, P_EM))

        F_P = np.array(F_P)

        return F_P


#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################
# Name = np.array(['debug-a', 'debug-b', 'debug-c', 'debug-d', 'debug-e', 'debug-f', 'unknown-g', 'unknown-h' , 'unknown-i', 'unknown-j', 'unknown-k'])
Name = np.array(['debug-a'])
name = ''
for nm in Name:

    # calbody
    # Read from TXT
    df_pa2_calbody = pd.read_csv('pa2_student_data\pa2-'+nm+'-calbody.txt',
    header=None, names=['N_d','N_a','N_c','Name_CALBODY'])
    # W. McKinney, PANDAS.READ_CSV#, Pandas.read_csv - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html (accessed October 12, 2022). 

    calbody = df_pa2_calbody[['N_d','N_a','N_c']].to_numpy()    
    # W. McKinney, Pandas.dataframe.to_numpy#, Pandas.DataFrame.to_numpy - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html (accessed October 12, 2022). 

    # Get num
    num_calbody = calbody[0]
    num_calbody_d = int(num_calbody[0])
    num_calbody_a = int(num_calbody[1])
    num_calbody_c = int(num_calbody[2])

    # Trim to get d, a, c
    calbody_d = calbody[1:(1+num_calbody_d),:]
    calbody_a = calbody[(1+num_calbody_d):(1+num_calbody_d+num_calbody_a),:]
    calbody_c = calbody[(1+num_calbody_d+num_calbody_a):(1+num_calbody_d+num_calbody_a+num_calbody_c),:]


    # calreadings
    # Read from TXT
    df_pa2_calreadings = pd.read_csv('pa2_student_data\pa2-'+nm+'-calreadings.txt', 
    header=None, names=['N_D','N_A','N_C','N_Frame','Name_CALREADING'])

    calreadings = df_pa2_calreadings[['N_D','N_A','N_C','N_Frame']].to_numpy()
    
    num_calreadings = calreadings[0]

    # get num
    num_calreadings_D = int(num_calreadings[0])
    num_calreadings_A = int(num_calreadings[1])
    num_calreadings_C = int(num_calreadings[2])
    num_calreadings_Frame = int(num_calreadings[3])

    num_calreadings_len = num_calreadings_D + num_calreadings_A + num_calreadings_C

    # Trim to get D, A, C
    calreadings_D = np.zeros((num_calreadings_Frame,num_calreadings_D,3))
    calreadings_A = np.zeros((num_calreadings_Frame,num_calreadings_A,3))
    calreadings_C = np.zeros((num_calreadings_Frame,num_calreadings_C,3))
    # C. Harris, Numpy.zeros#, Numpy.zeros - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.zeros.html (accessed October 12, 2022). 

    for i in range (num_calreadings_Frame):
        I = i*num_calreadings_len
        calreadings_D[i] = calreadings[(1) + I : (1+num_calreadings_D) + I , :-1]
        calreadings_A[i] = calreadings[(1+num_calreadings_D) + I : (1+num_calreadings_D+num_calreadings_A) + I,:-1]
        calreadings_C[i] = calreadings[(1+num_calreadings_D+num_calreadings_A) + I : (1+num_calreadings_D+num_calreadings_A+num_calreadings_C) + I,:-1]

    

    # empivot
    # Read from TXT
    df_pa2_empivot = pd.read_csv('pa2_student_data\pa2-'+nm+'-empivot.txt', 
    header=None, names=['N_G','N_Frame','Name_EMPIVOT'])
    calempivot = df_pa2_empivot[['N_G','N_Frame','Name_EMPIVOT']].to_numpy()

    # get num 
    num_calempivot = calempivot[0]

    num_calempivot_G = int(num_calempivot[0])
    num_calempivot_Frame = int(num_calempivot[1])

    num_calempivot_len = num_calempivot_G

    # Trim to get em G
    calempivot_G = np.zeros((num_calempivot_Frame,num_calempivot_G,3))

    for j in range (num_calempivot_Frame):
        J = j*num_calempivot_len
        calempivot_G[j] = calempivot[1 + J : (1+num_calempivot_G) + J, : ]


    
    # optpivot
    # Read from TXT
    df_pa2_optpivot = pd.read_csv('pa2_student_data\pa2-'+nm+'-optpivot.txt', 
    header=None, names=['N_D','N_H','N_Frames','Name_OPTPIVOT'])
    caloptpivot = df_pa2_optpivot[['N_D','N_H','N_Frames']].to_numpy()

    # get num
    num_caloptpivot = caloptpivot[0]

    num_caloptpivot_D = int(num_caloptpivot[0])
    num_caloptpivot_H = int(num_caloptpivot[1])
    num_caloptpivot_Frame = int(num_caloptpivot[2])

    num_caloptpivot_len = num_caloptpivot_D + num_caloptpivot_H

    # Trim to get opt D, opt H
    caloptpivot_D = np.zeros((num_caloptpivot_Frame,num_caloptpivot_D ,3))
    caloptpivot_H = np.zeros((num_caloptpivot_Frame,num_caloptpivot_H,3))

    for k in range (num_caloptpivot_Frame):
        K = k*num_caloptpivot_len
        caloptpivot_D[k] = caloptpivot[(1) + K : (1+num_caloptpivot_D) + K , : ]
        caloptpivot_H[k] = caloptpivot[(1+num_caloptpivot_D) + K : (1+num_caloptpivot_D+num_caloptpivot_H) + K,: ]

    
    
    # ct-fiducials
    # Read from TXT
    df_pa2_ct_fiducials = pd.read_csv('pa2_student_data\pa2-'+nm+'-ct-fiducials.txt', 
    header=None, names=['N_B','NAME-CT-FIDUCIALS.TXT'])
    ct_fiducials = df_pa2_ct_fiducials[['N_B','NAME-CT-FIDUCIALS.TXT']].to_numpy()

    # get num
    num_ct_fiducials = ct_fiducials[0]

    num_ct_fiducials_B = int(num_ct_fiducials[0])
    # Trim to get b coordinates
    ct_fiducials_B = pd.read_csv('pa2_student_data\pa2-'+nm+'-ct-fiducials.txt', 
    header=None, skiprows = 1).to_numpy()



    # em-fiducials
    # Read from TXT
    df_pa2_em_fiducials = pd.read_csv('pa2_student_data\pa2-'+nm+'-em-fiducialss.txt', 
    header=None, names=['N_G','N_B','NAME-EM-FIDUCIALS.TXT'])
    em_fiducials = df_pa2_em_fiducials[['N_G','N_B','NAME-EM-FIDUCIALS.TXT']].to_numpy()

    # get num
    num_em_fiducials = em_fiducials[0]

    num_em_fiducials_G = int(num_em_fiducials[0])
    num_em_fiducials_B = int(num_em_fiducials[1])

    # Trim to get G
    em_fiducials_G = np.zeros((num_em_fiducials_B, num_em_fiducials_G, 3))
    num_em_fiducials_len = num_em_fiducials_G

    for k in range (num_em_fiducials_B):
        K = k*num_em_fiducials_len
        em_fiducials_G[k] = em_fiducials[(1) + K : (1+num_em_fiducials_G) + K , : ]

    

    # em-nav
    df_pa2_em_nav= pd.read_csv('pa2_student_data\pa2-'+nm+'-EM-nav.txt', 
    header=None, names=['N_G','N_Frames','NAME-EMNAV.TXT'])
    em_nav = df_pa2_em_nav[['N_G','N_Frames','NAME-EMNAV.TXT']].to_numpy()

    # get num
    num_em_nav = em_nav[0]
    
    num_em_nav_G = int(num_em_nav[0])
    num_em_nav_Frame = int(num_em_nav[1])

    # Trim to get G
    em_nav_G = np.zeros((num_em_nav_Frame, num_em_nav_G, 3))
    num_em_nav_len = num_em_nav_G

    for k in range (num_em_nav_Frame):
        K = k*num_em_nav_len
        em_nav_G[k] = em_nav[(1) + K : (1+num_em_nav_G) + K , : ]

#######################################################################################
#######################################################################################
############### Calculation  ##########################################################
#######################################################################################
#######################################################################################

# F_D Transformation between optical tracker and EM tracker coordinates through all data frames
F_D = []
for i in range (len(calreadings_D)):
    F_D.append(Cloudregistration(calbody_d,calreadings_D[i]))
    # print('F_D')
# print(len(calreadings_D), len(calbody_d))
F_D = np.array([F_D])[0]
# print(F_D)
# print(np.shape(F_D),' shape F_D')

# F_A transformation between calibration object and optical trackercoordinates through all data frames
F_A = []
for j in range (len(calreadings_A)):
    F_A.append(Cloudregistration(calbody_a,calreadings_A[j]))
    # print('F_A')
F_A = np.array([F_A])[0]
# print(F_A)
# print(np.shape(F_A),' shape F_A')

# Calc C_vector expected for each data frame
C_vec_expected = []
C_vec_expected_d = []
# print(calbody_c)
# print(np.shape(calbody_c))

# Make c into 4x1 matrix
c_I = np.ones((len(calbody_c), 1))
# K. Lieret, Numpy.ones#, Numpy.ones - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.ones.html (accessed October 13, 2022). 
c = np.hstack((calbody_c,c_I))
# C. Harris, Numpy.hstack#, Numpy.hstack - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.hstack.html (accessed October 12, 2022). 
c_T = c.T
# print(c)
# print(np.shape(c))
# print(c.T)
# print(np.shape(c.T))
# print(c_T[:,0])
for d in range(len(F_D)):
    F_D_d = F_D[d]
    F_A_d = F_A[d]
    for k in range(len(calbody_c)):
        C = np.linalg.inv(F_D_d) @ F_A_d @ c_T[:,k]
        # I. Polat, Numpy.linalg.inv#, Numpy.linalg.inv - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html (accessed October 13, 2022). 
        C = C[0:3]
        C_vec_expected.append(C)


# Get qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z for all EM readings
C_vec_expected = np.array([C_vec_expected])
# print(C_vec_expected, 'C_expected')
print(np.shape(C_vec_expected), 'C_expected shape')
print(np.shape(calreadings_C), ' shape calreadings_C')
print(np.shape(calempivot_G), 'shape calempivot_G')
print(np.shape(em_fiducials_G), ' shape em_fiducials_G')
print(np.shape(em_nav_G), 'shape em_nav_G')

C_vec_expected_flat = np.ndarray.flatten(C_vec_expected)
calreadings_C_flat = np.ndarray.flatten(calreadings_C)
calempivot_G_flat = np.ndarray.flatten(calempivot_G)
em_fiducials_G_flat = np.ndarray.flatten(em_fiducials_G)
em_nav_G_flat = np.ndarray.flatten(em_nav_G)

em_all = np.hstack((C_vec_expected_flat, calreadings_C_flat, calempivot_G_flat, em_fiducials_G_flat, em_nav_G_flat))
# print(np.shape(em_all), ' shape em_all')
em_all = em_all.reshape(int(len(em_all)/3), 3)
# print(np.shape(em_all), ' shape em_all')

qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z = Find_min_max(em_all)

# Convert all dataframe to 2D
C_vec_expected_2D = C_vec_expected_flat.reshape(int(len(C_vec_expected_flat)/3), 3)
print(np.shape(C_vec_expected_2D), ' shape C_vec_expected_2D')
calreadings_C_2D = calreadings_C_flat.reshape(int(len(calreadings_C_flat)/3), 3)
print(np.shape(calreadings_C_2D), ' shape calreadings_C_2D')


# Calculate the corrected of C
corrected_C = Correct_Distortion(qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z, calreadings_C_2D, C_vec_expected_2D)
print(corrected_C[3370], 'corrected_C')
print(C_vec_expected_2D[3370], 'C_expected_2D')
print(np.shape(corrected_C), 'shape corrected_C')
print(np.shape(C_vec_expected_2D), 'C_vec_expected_2D')

#EM Caliberation with distortion correct data
# corrected_G = Correct_Distortion(calempivot_G, calreadings_C, C_vec_expected)
# print(np.shape(corrected_G), 'shape corrected_G')
# P_G_dimple, t_G, F_G, g_j = Point_Calibration(corrected_G, num_calempivot_Frame)

#Correct distortion of the em fiducials.
#correct_J = Correct_Distortion(em_fiducials_G, calreadings_C, C_vec_expected)
#Fj = Framecalcuation(correct_J,num_em_fiducials_B)
#Compute the locations b_j of the fiducials points:
#b_j = []
#for i in num_em_fiducials_B:
#    b_j.append(Fj[i]*P_G_dimple)
#b_j = np.array(b_j)

#Compute the registration frame F_reg:
#F_reg = Cloudregistration(b_j,num_ct_fiducials)

#Correct distortion of the em nav.
#correct_nav = Correct_Distortion(num_em_nav_G, calreadings_C, C_vec_expected)
#F_nav = Framecalcuation(correct_nav,num_em_nav_Frame)
#B_N = []
#for i in num_em_nav_Frame:
#    B_N.append(F_nav[i]*P_G_dimple)
#B_N = np.array(B_N)
#b_N = []
#for i in num_em_nav_Frame:
#    b_N.append(F_reg*B_N[i])
#b_N = np.array(b_N)