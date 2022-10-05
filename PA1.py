# Imports

import numpy as np
import pandas as pd

# Import Data

# read text file into pandas DataFrame

# h_calbody
df_pa1_h_calbody = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calbody.txt", 
header=None, names=['a','b','c','d'])
h_calbody = df_pa1_h_calbody[['a','b','c']].to_numpy()
h_calbody = h_calbody[1:,:]

# h_calreadings
df_pa1_h_calreadings = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-calreadings.txt", 
header=None, names=['a','b','c','d'])
h_calreadings = df_pa1_h_calreadings[['a','b','c']].to_numpy()
h_calreadings = h_calreadings[1:,:]

# h_empivot
df_pa1_h_empivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-empivot.txt", 
header=None, names=['a','b','c','d'])
h_calempivot = df_pa1_h_empivot[['a','b','c']].to_numpy()
h_calempivot = h_calempivot[1:,:]

# h_optpivot
df_pa1_h_optpivot = pd.read_csv(r"C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\pa1_student_data\PA1 Student Data\pa1-unknown-h-optpivot.txt", 
header=None, names=['a','b','c','d'])
h_caloptpivot = df_pa1_h_optpivot[['a','b','c']].to_numpy()
h_caloptpivot = h_caloptpivot[1:,:]


# Functions

# PA_1 Q1
# def Three_D_point(Origin):
#     p= [0,0,0]
#     p= [Origin[0],Origin[1],Origin[2]]

#     return p
def Three_D_point():
    return 0

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
