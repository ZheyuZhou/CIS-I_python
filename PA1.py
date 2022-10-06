# Imports

import numpy as np
import pandas as pd

# Import Data

# read text file into pandas DataFrame

# h_calbody
df_pa1_h_calbody = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calbody.txt", 
header=None, names=['N_D','N_A','N_C','Name_CALBODY'])
h_calbody = df_pa1_h_calbody[['N_D','N_A','N_C']].to_numpy()

h_num_calbody = h_calbody[0]

h_num_calbody_D = int(h_num_calbody[0])
h_num_calbody_A = int(h_num_calbody[1])
h_num_calbody_C = int(h_num_calbody[2])

h_calbody_D = h_calbody[1:(1+h_num_calbody_D),:]
h_calbody_A = h_calbody[(1+h_num_calbody_D):(1+h_num_calbody_D+h_num_calbody_A),:]
h_calbody_C = h_calbody[(1+h_num_calbody_D+h_num_calbody_A):(1+h_num_calbody_D+h_num_calbody_A+h_num_calbody_C),:]

# h_calreadings
df_pa1_h_calreadings = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calreadings.txt", 
header=None, names=['N_D','N_A','N_C','N_Frame','Name_CALREADING'])
h_calreadings = df_pa1_h_calreadings[['N_D','N_A','N_C','N_Frame']].to_numpy()

h_num_calreadings = h_calreadings[0]

h_num_calreadings_D = int(h_num_calreadings[0])
h_num_calreadings_A = int(h_num_calreadings[1])
h_num_calreadings_C = int(h_num_calreadings[2])
h_num_calreadings_Frame = int(h_num_calreadings[3])

h_calreadings_D = h_calreadings[1:(1+h_num_calreadings_D),:]
h_calreadings_A = h_calreadings[(1+h_num_calreadings_D):(1+h_num_calreadings_D+h_num_calreadings_A),:]
h_calreadings_C = h_calreadings[(1+h_num_calreadings_D+h_num_calreadings_A):(1+h_num_calreadings_D+h_num_calreadings_A+h_num_calreadings_C),:]
h_calreadings_Frame = h_calreadings[(1+h_num_calreadings_D+h_num_calreadings_A+h_num_calreadings_C):(1+h_num_calreadings_D+h_num_calreadings_A+h_num_calreadings_C+h_num_calreadings_Frame),:]

# h_empivot
df_pa1_h_empivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-empivot.txt", 
header=None, names=['N_G','N_Frame','Name_EMPIVOT'])
h_calempivot = df_pa1_h_empivot[['N_G','N_Frame']].to_numpy()

h_num_calempivot = h_calempivot[0]

h_num_calempivot_G = int(h_num_calempivot[0])
h_num_calempivot_Frame = int(h_num_calempivot[1])

h_calempivot_G = h_calempivot[1:(1+h_num_calempivot_G),:]
h_calempivot_Frame = h_calempivot[(1+h_num_calempivot_G):(1+h_num_calempivot_G+h_num_calempivot_Frame),:]


# h_optpivot
df_pa1_h_optpivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-optpivot.txt", 
header=None, names=['N_D','N_H','N_Frames','Name_OPTPIVOT'])
h_caloptpivot = df_pa1_h_optpivot[['N_D','N_H','N_Frames']].to_numpy()

h_num_caloptpivot = df_pa1_h_optpivot[0]

h_num_caloptpivot_D = h_num_caloptpivot[0]
h_num_caloptpivot_H = h_num_caloptpivot[1]
h_num_caloptpivot_Frame = h_num_caloptpivot[2]

h_caloptpivot_D = h_caloptpivot[1:(1+h_num_caloptpivot_D),:]
h_caloptpivot_H = h_caloptpivot[(1+h_num_caloptpivot_D):(1+h_num_caloptpivot_D+h_num_caloptpivot_H),:]
h_caloptpivot_Frame = h_caloptpivot[(1+h_num_caloptpivot_D+h_num_caloptpivot_H):(1+h_num_caloptpivot_D+h_num_caloptpivot_H+h_num_caloptpivot_Frame),:]


# Functions

# PA_1 Q1

def Rotation(psi, theta, phi):
    # rotate about x -> psi
    # rotate about y -> theta
    # rotate about z -> phi
    R = np.zeros((3,3))
    R = np.array([
        [np.cos(phi)*np.cos(theta),   -np.sin(phi)*np.cos(psi)+np.cos(phi)*np.sin(theta)*np.sin(psi),   np.sin(phi)*np.sin(psi)+np.cos(phi)*np.sin(theta)*np.cos(psi)],
        [np.sin(phi)*np.cos(theta),   np.cos(phi)*np.cos(psi)+np.sin(phi)*np.sin(theta)*np.sin(psi),   -np.cos(phi)*np.sin(psi)+np.sin(phi)*np.sin(theta)*np.cos(psi)],
        [-np.sin(theta),   np.cos(theta)*np.sin(psi),   np.cos(theta)*np.cos(psi)]
    ])
    
    return R

def Frame_Transform(R,p):
    F = np.zeros((4,4))
    F= np.array([
        [R[0,0],R[0,1],R[0,2],p[0]],
        [R[1,0],R[1,1],R[1,2],p[1]],
        [R[2,0],R[2,1],R[2,2],p[2]],
        [0,0,0,1]])

    return F

# PA_1 Q3
def Pivot_Calibration():
    
    return 0
