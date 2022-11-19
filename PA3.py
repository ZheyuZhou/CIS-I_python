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

import CloudRegistration_ as cr

#######################################################################################
#######################################################################################
############### Functions    ##########################################################
#######################################################################################
#######################################################################################
# For Body Reading
def Num_Marker(Body): # Get Num of Marker
    Num = np.fromstring(Body[0][0],dtype=int,sep=' ')[0]
    return Num

def Body_Array(Body): # Get Body Array
    Body_list = []
    for i in range (1,len(Body)):
        row = np.fromstring(Body[i][0],dtype=float,sep=' ')
        Body_list.append(row)

    Body_array = np.array(Body_list)
    return Body_array

def Marker_XYZ(Body): # Get Marker Coordnate
    Body_array = Body_Array(Body)
    Num_marker = Num_Marker(Body)
    Marker_xyz = Body_array[:Num_marker]
    return Marker_xyz

def tip_XYZ(Body): # Get tip Coordnate
    Body_array = Body_Array(Body)
    tip_xyz = Body_array[-1]
    return tip_xyz



# For SampleReadingsTest reading
def SampleReadingsTest(address): # Get Body Array
    SampleReadingsTest_ = pd.read_csv(address, header=None,  dtype=float, skiprows=1)
    SampleReadingsTest_Array = SampleReadingsTest_.to_numpy()
    return SampleReadingsTest_Array

def SampleReadingsTest_head(address): # Get Head Array
    SampleReadingsTest_head_ = pd.read_csv(address, header=None,  dtype=float, nrows=1)
    SampleReadingsTest_head_Array = SampleReadingsTest_head_.to_numpy()
    return SampleReadingsTest_head_Array

def SampleReadingsTest_ABDS(address, Num_A, Num_B):
    SampleReadingsTest_Array = SampleReadingsTest(address)
    SampleReadingsTest_head_Array = SampleReadingsTest_head(address)
    N_S = SampleReadingsTest_head_Array[0]
    N_samp = SampleReadingsTest_head_Array[1]
    N_A = Num_A
    N_B = Num_B
    N_D = N_S - Num_A - Num_B

    N_A_Record = SampleReadingsTest_Array[:N_A]
    N_B_Record = SampleReadingsTest_Array[N_A:(N_A+N_B)]
    N_D_record = SampleReadingsTest_Array[(N_A+N_B):(N_A+N_B+N_D)]
    N_S_record = SampleReadingsTest_Array[(N_A+N_B+N_D):]

    return N_A_Record, N_B_Record, N_D_record, N_S_record

def pa3_SampleReadingDictionary(address_name, Num_A, Num_B):
    frameA2J = {}
    for name in address_name:
        pa3_frame_SampleReadingsTest = pd.read_csv(r'C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\2022_pa345_student_data\2022 PA345 Student Data\PA3-'+name'-SampleReadingsTest.txt',
header=None, nrows = 1)
    return pa3_Dictionary

#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################
pa3_BodyA = pd.read_csv('2022_pa345_student_data\Problem3-BodyA.txt', header=None)

pa3_BodyB = pd.read_csv('2022_pa345_student_data\Problem3-BodyB.txt', header=None)

pa3_BodyA = pa3_BodyA.to_numpy()
pa3_BodyB = pa3_BodyB.to_numpy()

# pa3_Name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'D-Unknown', 'D-Unknown'])
pa3_Name = np.array(['A-Debug'])

pa3_A_SampleReadingsTest = pd.read_csv(r'C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\2022_pa345_student_data\2022 PA345 Student Data\PA3-A-Debug-SampleReadingsTest.txt',
header=None,  dtype=float ,skiprows=1)

pa3_A_SampleReadingsTest_Array =pa3_A_SampleReadingsTest.to_numpy()


pa3_A_SampleReadingsTest_head = pd.read_csv(r'C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\2022_pa345_student_data\2022 PA345 Student Data\PA3-A-Debug-SampleReadingsTest.txt',
header=None, nrows = 1)

pa3_A_SampleReadingsTest_head_Array = pa3_A_SampleReadingsTest_head.to_numpy()

pa3_A_N_S = pa3_A_SampleReadingsTest_head_Array[0]
pa3_A_N_samps = pa3_A_SampleReadingsTest_head_Array[1]


#######################################################################################
#######################################################################################
############### Main         ##########################################################
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    Num_A_Marker = Num_Marker(pa3_BodyA)
    Num_B_Marker = Num_Marker(pa3_BodyB)

    Marker_A_XYZ = Marker_XYZ(pa3_BodyA)
    Marker_B_XYZ = Marker_XYZ(pa3_BodyB)

    tip_A_XYZ = tip_XYZ(pa3_BodyA)
    tip_B_XYZ = tip_XYZ(pa3_BodyB)

    