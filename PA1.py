# Imports

import numpy as np
import pandas as pd

# Import Data

# read text file into pandas DataFrame

# h_calbody
df_pa1_h_calbody = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calbody.txt", 
header=None, names=['N_d','N_a','N_c','Name_CALBODY'])
h_calbody = df_pa1_h_calbody[['N_d','N_a','N_c']].to_numpy()

h_num_calbody = h_calbody[0]

h_num_calbody_d = int(h_num_calbody[0])
h_num_calbody_a = int(h_num_calbody[1])
h_num_calbody_c = int(h_num_calbody[2])


h_calbody_d = h_calbody[1:(1+h_num_calbody_d),:]
h_calbody_a = h_calbody[(1+h_num_calbody_d):(1+h_num_calbody_d+h_num_calbody_a),:]
h_calbody_c = h_calbody[(1+h_num_calbody_d+h_num_calbody_a):(1+h_num_calbody_d+h_num_calbody_a+h_num_calbody_c),:]

# h_calreadings
df_pa1_h_calreadings = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calreadings.txt", 
header=None, names=['N_D','N_A','N_C','N_Frame','Name_CALREADING'])
h_calreadings = df_pa1_h_calreadings[['N_D','N_A','N_C','N_Frame']].to_numpy()
print(np.shape(h_calreadings))
h_num_calreadings = h_calreadings[0]

h_num_calreadings_D = int(h_num_calreadings[0])
h_num_calreadings_A = int(h_num_calreadings[1])
h_num_calreadings_C = int(h_num_calreadings[2])
h_num_calreadings_Frame = int(h_num_calreadings[3])

h_num_calreadings_len = h_num_calreadings_D + h_num_calreadings_A + h_num_calreadings_C

h_calreadings_D = np.zeros((h_num_calreadings_Frame,h_num_calreadings_D,3))
h_calreadings_A = np.zeros((h_num_calreadings_Frame,h_num_calreadings_A,3))
h_calreadings_C = np.zeros((h_num_calreadings_Frame,h_num_calreadings_C,3))

for i in range (h_num_calreadings_Frame):
    I = i*h_num_calreadings_len
    h_calreadings_D[i] = h_calreadings[(1) + I : (1+h_num_calreadings_D) + I , :-1]
    h_calreadings_A[i] = h_calreadings[(1+h_num_calreadings_D) + I : (1+h_num_calreadings_D+h_num_calreadings_A) + I,:-1]
    h_calreadings_C[i] = h_calreadings[(1+h_num_calreadings_D+h_num_calreadings_A) + I : (1+h_num_calreadings_D+h_num_calreadings_A+h_num_calreadings_C) + I,:-1]

# h_empivot
df_pa1_h_empivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-empivot.txt", 
header=None, names=['N_G','N_Frame','Name_EMPIVOT'])
h_calempivot = df_pa1_h_empivot[['N_G','N_Frame']].to_numpy()

h_num_calempivot = h_calempivot[0]

h_num_calempivot_G = int(h_num_calempivot[0])
h_num_calempivot_Frame = int(h_num_calempivot[1])

h_num_calempivot_len = h_num_calempivot_G

h_calempivot_G = np.zeros((h_num_calempivot_Frame,h_num_calempivot_G,3))

for j in range (h_num_calempivot_Frame):
    J = j*h_num_calempivot_len
    h_calempivot_G[j] = h_calempivot[1 + J : (1+h_num_calempivot_G) + J, : -1]

# h_optpivot
df_pa1_h_optpivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-optpivot.txt", 
header=None, names=['N_D','N_H','N_Frames','Name_OPTPIVOT'])
h_caloptpivot = df_pa1_h_optpivot[['N_D','N_H','N_Frames']].to_numpy()

h_num_caloptpivot = h_caloptpivot[0]

h_num_caloptpivot_D = int(h_num_caloptpivot[0])
h_num_caloptpivot_H = int(h_num_caloptpivot[1])
h_num_caloptpivot_Frame = int(h_num_caloptpivot[2])

h_num_caloptpivot_len = h_num_caloptpivot_D + h_num_caloptpivot_H

h_caloptpivot_D = np.zeros((h_num_caloptpivot_Frame,h_num_caloptpivot_D ,3))
h_caloptpivot_H = np.zeros((h_num_caloptpivot_Frame,h_num_caloptpivot_H,3))

for k in range (h_num_caloptpivot_Frame):
    K = k*h_num_caloptpivot_len
    h_caloptpivot_D[i] = h_caloptpivot[(1) + K : (1+h_num_caloptpivot_D) + K , :-1]
    h_caloptpivot_H[i] = h_caloptpivot[(1+h_num_caloptpivot_D) + K : (1+h_num_caloptpivot_D+h_num_caloptpivot_H) + K,:-1]


# 
def Cloudregistration(A,B):
    sh = np.shape
    for i in range (sh(A)[0]):
        for j in range (sh(A)[1]):
            a_bar=
            
            
            



# h_num_caloptpivot = df_pa1_h_optpivot[0]

# h_num_caloptpivot_D = h_num_caloptpivot[0]
# h_num_caloptpivot_H = h_num_caloptpivot[1]
# h_num_caloptpivot_Frame = h_num_caloptpivot[2]

# h_caloptpivot_D = h_caloptpivot[1:(1+h_num_caloptpivot_D),:]
# h_caloptpivot_H = h_caloptpivot[(1+h_num_caloptpivot_D):(1+h_num_caloptpivot_D+h_num_caloptpivot_H),:]
# h_caloptpivot_Frame = h_caloptpivot[(1+h_num_caloptpivot_D+h_num_caloptpivot_H):(1+h_num_caloptpivot_D+h_num_caloptpivot_H+h_num_caloptpivot_Frame),:]


# Functions

# 


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
