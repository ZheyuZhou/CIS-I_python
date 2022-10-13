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


#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################
Name = np.array(['debug-a', 'debug-b', 'debug-c', 'debug-d', 'debug-e', 'debug-f', 'debug-g', 'unknown-h' , 'unknown-i', 'unknown-j', 'unknown-k'])
name = ''
for nm in Name:
    print(nm)
    name = nm

    # h_calbody
    # Read from TXT
    df_pa1_h_calbody = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-"+nm+"-calbody.txt", 
    header=None, names=['N_d','N_a','N_c','Name_CALBODY'])
    # W. McKinney, PANDAS.READ_CSV#, Pandas.read_csv - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html (accessed October 12, 2022). 
    h_calbody = df_pa1_h_calbody[['N_d','N_a','N_c']].to_numpy()
    # W. McKinney, Pandas.dataframe.to_numpy#, Pandas.DataFrame.to_numpy - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html (accessed October 12, 2022). 
    # get num
    h_num_calbody = h_calbody[0]

    h_num_calbody_d = int(h_num_calbody[0])
    h_num_calbody_a = int(h_num_calbody[1])
    h_num_calbody_c = int(h_num_calbody[2])

    # Trim to get d, a, c
    h_calbody_d = h_calbody[1:(1+h_num_calbody_d),:]
    h_calbody_a = h_calbody[(1+h_num_calbody_d):(1+h_num_calbody_d+h_num_calbody_a),:]
    h_calbody_c = h_calbody[(1+h_num_calbody_d+h_num_calbody_a):(1+h_num_calbody_d+h_num_calbody_a+h_num_calbody_c),:]
    # print(h_calbody_d)
    # print(h_calbody_a)
    # print(h_calbody_c)

    # h_calreadings
    # Read from TXT
    df_pa1_h_calreadings = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-"+nm+"-calreadings.txt", 
    header=None, names=['N_D','N_A','N_C','N_Frame','Name_CALREADING'])
    h_calreadings = df_pa1_h_calreadings[['N_D','N_A','N_C','N_Frame']].to_numpy()
    # print(h_calreadings)
    h_num_calreadings = h_calreadings[0]

    # get num
    h_num_calreadings_D = int(h_num_calreadings[0])
    h_num_calreadings_A = int(h_num_calreadings[1])
    h_num_calreadings_C = int(h_num_calreadings[2])
    h_num_calreadings_Frame = int(h_num_calreadings[3])

    h_num_calreadings_len = h_num_calreadings_D + h_num_calreadings_A + h_num_calreadings_C

    # Trim to get D, A, C
    h_calreadings_D = np.zeros((h_num_calreadings_Frame,h_num_calreadings_D,3))
    # C. Harris, Numpy.zeros#, Numpy.zeros - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.zeros.html (accessed October 12, 2022). 
    h_calreadings_A = np.zeros((h_num_calreadings_Frame,h_num_calreadings_A,3))
    h_calreadings_C = np.zeros((h_num_calreadings_Frame,h_num_calreadings_C,3))

    for i in range (h_num_calreadings_Frame):
        I = i*h_num_calreadings_len
        h_calreadings_D[i] = h_calreadings[(1) + I : (1+h_num_calreadings_D) + I , :-1]
        h_calreadings_A[i] = h_calreadings[(1+h_num_calreadings_D) + I : (1+h_num_calreadings_D+h_num_calreadings_A) + I,:-1]
        h_calreadings_C[i] = h_calreadings[(1+h_num_calreadings_D+h_num_calreadings_A) + I : (1+h_num_calreadings_D+h_num_calreadings_A+h_num_calreadings_C) + I,:-1]
        # print(h_calreadings_D[i])
        # print(h_calreadings_A[i])
        # print(h_calreadings_C[i])

    # h_empivot
    # Read from TXT
    df_pa1_h_empivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-"+nm+"-empivot.txt", 
    header=None, names=['N_G','N_Frame','Name_EMPIVOT'])
    h_calempivot = df_pa1_h_empivot[['N_G','N_Frame','Name_EMPIVOT']].to_numpy()

    # get num 
    h_num_calempivot = h_calempivot[0]

    h_num_calempivot_G = int(h_num_calempivot[0])
    h_num_calempivot_Frame = int(h_num_calempivot[1])

    h_num_calempivot_len = h_num_calempivot_G

    # Trim to get em G
    h_calempivot_G = np.zeros((h_num_calempivot_Frame,h_num_calempivot_G,3))

    for j in range (h_num_calempivot_Frame):
        J = j*h_num_calempivot_len
        h_calempivot_G[j] = h_calempivot[1 + J : (1+h_num_calempivot_G) + J, : ]
        # print(h_calempivot_G[j])

    # h_optpivot
    # Read from TXT
    df_pa1_h_optpivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-"+nm +"-optpivot.txt", 
    header=None, names=['N_D','N_H','N_Frames','Name_OPTPIVOT'])
    h_caloptpivot = df_pa1_h_optpivot[['N_D','N_H','N_Frames']].to_numpy()

    # get num
    h_num_caloptpivot = h_caloptpivot[0]

    h_num_caloptpivot_D = int(h_num_caloptpivot[0])
    h_num_caloptpivot_H = int(h_num_caloptpivot[1])
    h_num_caloptpivot_Frame = int(h_num_caloptpivot[2])

    h_num_caloptpivot_len = h_num_caloptpivot_D + h_num_caloptpivot_H

    # Trim to get opt D, opt H
    h_caloptpivot_D = np.zeros((h_num_caloptpivot_Frame,h_num_caloptpivot_D ,3))
    h_caloptpivot_H = np.zeros((h_num_caloptpivot_Frame,h_num_caloptpivot_H,3))

    for k in range (h_num_caloptpivot_Frame):
        K = k*h_num_caloptpivot_len
        h_caloptpivot_D[k] = h_caloptpivot[(1) + K : (1+h_num_caloptpivot_D) + K , : ]
        h_caloptpivot_H[k] = h_caloptpivot[(1+h_num_caloptpivot_D) + K : (1+h_num_caloptpivot_D+h_num_caloptpivot_H) + K,: ]
        # print(h_caloptpivot_D[k])
        # print(h_caloptpivot_H[k])

    #######################################################################################
    #######################################################################################
    ############### Functions  ############################################################
    #######################################################################################
    #######################################################################################


    # Point Cloud to Point Cloud 
    # B. K. P. Horn, Closed-form solution of absolute orientation using unit quaternions, Optica Publishing Group. (1987). https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-4-4-629&amp;id=2711 (accessed October 12, 2022). 
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

    #######################################################################################
    #######################################################################################
    ############### Calculation  ##########################################################
    #######################################################################################
    #######################################################################################

    # F_D Transformation between optical tracker and EM tracker coordinates through all data frames
    F_D = []
    for i in range (len(h_calbody_d)):
        F_D.append(Cloudregistration(h_calbody_d,h_calreadings_D[i]))
        # print('F_D')

    F_D = np.array([F_D])[0]
    # print(F_D)
    # print(np.shape(F_D),' shape F_D')

    # F_A transformation between calibration object and optical trackercoordinates through all data frames
    F_A = []
    for j in range (len(h_calbody_a)):
        F_A.append(Cloudregistration(h_calbody_a,h_calreadings_A[j]))
        # print('F_A')
    F_A = np.array([F_A])[0]
    # print(F_A)
    # print(np.shape(F_A))

    # Calc C_vector expected for each data frame
    C_vec_expected = []
    C_vec_expected_d = []
    # print(h_calbody_c)
    # print(np.shape(h_calbody_c))

    # Make c into 4x1 matrix
    c_I = np.ones((len(h_calbody_c), 1))
    # K. Lieret, Numpy.ones#, Numpy.ones - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.ones.html (accessed October 13, 2022). 
    c = np.hstack((h_calbody_c,c_I))

    c_T = c.T
    # print(c)
    # print(np.shape(c))
    # print(c.T)
    # print(np.shape(c.T))
    # print(c_T[:,0])
    for d in range(len(F_D)):
        F_D_d = F_D[d]
        F_A_d = F_A[d]
        for k in range(len(h_calbody_c)):
            C = np.linalg.inv(F_D_d) @ F_A_d @ c_T[:,k]
            # I. Polat, Numpy.linalg.inv#, Numpy.linalg.inv - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html (accessed October 13, 2022). 
            C = C[0:3]
            C_vec_expected.append(C)
    # print(C_vec_expected, 'C_expected')
    # print(np.shape(C_vec_expected), 'C_expected shape')


    # Calibration of EM
    P_G_dimple, t_G, F_G, g_j = Point_Calibration(h_calempivot_G, h_num_calempivot_Frame)
    # print(P_G_dimple, 'P')
    # print(np.shape(P_G_dimple))
    # print(t_G, 't')

    # Calibration of Opt
    F_D_opt = []
    for i in range (h_num_caloptpivot_Frame):
        F_D_opt.append(Cloudregistration(h_calbody_d,h_caloptpivot_D[i]))
    F_D_opt = np.array([F_D_opt])[0]

    P_H_dimple, t_H, F_H, h_j = Point_Calibration(h_caloptpivot_H, h_num_caloptpivot_Frame)

    F_D_opt_inv = np.linalg.inv(F_D_opt)
    F_D_opt_inv_H = F_D_opt_inv@F_H
    P_H_dimple = LeastSquare(F_D_opt_inv_H)[0]
    # print(P_H_dimple, 'H')
    # print(np.shape(P_H_dimple))


    #######################################################################################
    #######################################################################################
    ############### Output ################################################################
    #######################################################################################
    #######################################################################################
    # Row 1: NC ,Nframes, NAME-OUTPUT1.TXT
    PA_1_output_row_1 = np.array([[h_num_calbody_c, h_num_calreadings_Frame, 'PA1_' + nm + '-OUTPUT1.TXT']])

    # Row 2: Estimated post position with EM probe pivot calibration
    PA_1_output_row_2 = P_G_dimple.reshape(1,3)

    # Row 2: Estimated post position with optical probe pivot calibration
    PA_1_output_row_3 = P_H_dimple.reshape(1,3)

    # Rest of Rows: Coordinates of C_j expected in all dataframes
    PA_1_output_row_C_exp = C_vec_expected

    PA_1_output = np.concatenate((PA_1_output_row_1, PA_1_output_row_2, PA_1_output_row_3, C_vec_expected), axis = 0)

    # print(PA_1_output)
    # print(np.shape(PA_1_output))

    pd.DataFrame(PA_1_output).to_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Output\PA1_"+nm+"_output.csv")



#######################################################################################
#######################################################################################
############### Test   ################################################################
#######################################################################################
#######################################################################################
# Read from TXT
df_pa1_a_output = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-debug-a-output1.txt", 
header=None, skiprows = 1)
debug_a_output = df_pa1_a_output.to_numpy()

# print(debug_a_output[0])
# print(np.shape(debug_a_output))


# Read from a output
a_output = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Output\PA1_debug_a_output.csv", 
header=None, skiprows = 2)
# print(a_output)
result_a_output = a_output.to_numpy()
result_a_output = result_a_output[: , 1:]
# print(result_a_output[0])
# print(np.shape(result_a_output))


err_2d = debug_a_output - result_a_output

err_abs_1d = np.ndarray.flatten(np.abs(err_2d))

err = 0.01

if np.max(err_abs_1d) > err:
    print("Test Fail!!!!!!")
else:
    print("Test Passed \(o^ ^o)/")