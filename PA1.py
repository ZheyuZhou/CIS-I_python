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
import pandas as pd


#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################

# h_calbody
# Read from TXT
df_pa1_h_calbody = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calbody.txt", 
header=None, names=['N_d','N_a','N_c','Name_CALBODY'])
h_calbody = df_pa1_h_calbody[['N_d','N_a','N_c']].to_numpy()

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
df_pa1_h_calreadings = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calreadings.txt", 
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
df_pa1_h_empivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-empivot.txt", 
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
df_pa1_h_optpivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-optpivot.txt", 
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
def Cloudregistration(a,A):

    # Calc average
    a_bar = np.mean(a, axis=0)

    A_bar = np.mean(A, axis=0)

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

    Delta_T = Delta.T
    
    H_HT_TrH_I = H + H.T - np.eye(3)*np.trace(H)

    G1 = np.hstack((Tr_H, Delta_T))
    G2 = np.hstack((Delta, H_HT_TrH_I))

    G = np.vstack((G1,G2))

    # Eig-val Decomp to get Eig-vec corresponds to largest(first) Eig-val
    eig_val, eig_vec = np.linalg.eig(G)

    Qk = eig_vec[:,0]

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

    # Organize to get Transformation
    F1 = np.hstack((R,p))
    F2 = np.array([0,0,0,1])
    F = np.vstack((F1,F2))

    return F 



#######################################################################################
#######################################################################################
############### Test  #################################################################
#######################################################################################
#######################################################################################

# F_D Transformation between optical tracker and EM tracker coordinates through all data frames
F_D = []
for i in range (len(h_calbody_d)):
    F_D.append(Cloudregistration(h_calbody_d,h_calreadings_D[i]))

F_D = np.array([F_D])[0]
# print(F_D)
# print(np.shape(F_D))

# F_A transformation between calibration object and optical trackercoordinates through all data frames
F_A = []
for j in range (len(h_calbody_a)):
    F_A.append(Cloudregistration(h_calbody_a,h_calreadings_A[i]))

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

c = np.hstack((h_calbody_c,c_I))

c_T = c.T
# print(c)
# print(np.shape(c))
# print(c.T)
# print(np.shape(c.T))
for d in range(len(F_D)):
    F_D_d = F_D[d]
    F_A_d = F_A[d]
    for k in range(len(h_calbody_c)):
        C = np.linalg.inv(F_D_d) @ F_A_d @ c_T[:,k]
        C_vec_expected.append(C)
# print(C_vec_expected)
# print(np.shape(C_vec_expected))

# Testing of part 2 
a = np.array([[1,2,3,1],[2,3,4,1],[3,6,4,1]])
b = np.array([[1,2,3,0],[2,3,4,2],[3,1,5,2],[0,0,0,1]])
c = []
for i in range (len(a)):
    c.append(b.dot(a[i]))
c = np.array(c)
# print(c)

T = np.array([[1/np.sqrt(2),1/np.sqrt(2),0],[-1/np.sqrt(2),-1/np.sqrt(2),0],[0,0,1]])
t = np.array([[1,0,0],[-1,0,0],[0,0,1]])

T1 = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,1],[-1/np.sqrt(2),-1/np.sqrt(2),0,1],[0,0,1,1]])
t1 = np.array([[1,0,0,1],[-1,0,0,1],[0,0,1,1]])

# da an
# Fk = np.array([[1,2,3,0],[2,3,4,2],[3,1,5,2],[0,0,0,1]])
# T = Fk*t
print(Cloudregistration(t,T))
f1 = Cloudregistration(t,T)
print(T1[2] == f1@t1[2])


# Calibration of EM
F_G = []
for j in range (h_num_calempivot_Frame):
    # EM points
    G_EM = h_calempivot_G[j]

    # Calc average
    G0 = np.mean(G_EM, axis=0)

    # Calc diff between each vec and average
    gj = []
    for i in range (len(G_EM)):
        gj.append(G_EM[i]-G0)
    
    gj = np.array(gj)
    # print(gj)
    # print(np.shape(gj))
    
    # calculate EM marker Point Cloud Transformation 
    
    F_G.append(Cloudregistration(gj, G_EM))
    #Store all the matrix
F_G = np.array(F_G)

# print(F_G)
# print(np.shape(F_G))


    

    
        


    




    












# def Rotation(psi, theta, phi):
#     # rotate about x -> psi
#     # rotate about y -> theta
#     # rotate about z -> phi
#     R = np.zeros((3,3))
#     R = np.array([
#         [np.cos(phi)*np.cos(theta),   -np.sin(phi)*np.cos(psi)+np.cos(phi)*np.sin(theta)*np.sin(psi),   np.sin(phi)*np.sin(psi)+np.cos(phi)*np.sin(theta)*np.cos(psi)],
#         [np.sin(phi)*np.cos(theta),   np.cos(phi)*np.cos(psi)+np.sin(phi)*np.sin(theta)*np.sin(psi),   -np.cos(phi)*np.sin(psi)+np.sin(phi)*np.sin(theta)*np.cos(psi)],
#         [-np.sin(theta),   np.cos(theta)*np.sin(psi),   np.cos(theta)*np.cos(psi)]
#     ])
    
#     return R

# def Frame_Transform(R,p):
#     F = np.zeros((4,4))
#     F= np.array([
#         [R[0,0],R[0,1],R[0,2],p[0]],
#         [R[1,0],R[1,1],R[1,2],p[1]],
#         [R[2,0],R[2,1],R[2,2],p[2]],
#         [0,0,0,1]])

#     return F

# # PA_1 Q3
# def Pivot_Calibration():
    
#     return 0
