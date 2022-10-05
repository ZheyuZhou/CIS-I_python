# Imports

import numpy as np
import pandas as pd


# Functions

# PA_1 Q1
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

def Frame_Transform():
    return 0

# PA_1 Q3
def Pivot_Calibration():
    return 0
