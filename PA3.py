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
    # print(Num)
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
def pa3_Frame_SampleReadingsTest_Body(address_name): # Get Body Array
    pa3_frame_SampleReadingsTest_body = pd.read_csv('2022_pa345_student_data\PA3-'+address_name+'-SampleReadingsTest.txt', header=None, skiprows = 1)
    pa3_frame_SampleReadingsTest_body_Array = pa3_frame_SampleReadingsTest_body.to_numpy()
    return pa3_frame_SampleReadingsTest_body_Array

def pa3_Frame_SampleReadingsTest_Head(address_name): # Get Head Array
    pa3_frame_SampleReadingsTest_head = pd.read_csv('2022_pa345_student_data\PA3-'+address_name+'-SampleReadingsTest.txt', header=None, nrows = 1)
    pa3_frame_SampleReadingsTest_head_Array = pa3_frame_SampleReadingsTest_head.to_numpy()
    return pa3_frame_SampleReadingsTest_head_Array

def pa3_Frame_SampleReadingsTest_ABDS(address_name, Num_A, Num_B):
    pa3_frame_SampleReadingsTest_body_Array = pa3_Frame_SampleReadingsTest_Body(address_name)
    pa3_frame_SampleReadingsTest_head_Array = pa3_Frame_SampleReadingsTest_Head(address_name)
    pa3_Frame_N_S = pa3_frame_SampleReadingsTest_head_Array[0][0]
    pa3_Frame_N_samp = pa3_frame_SampleReadingsTest_head_Array[0][1]
    pa3_Frame_N_A = Num_A
    pa3_Frame_N_B = Num_B
    pa3_Frame_N_D = pa3_Frame_N_S - Num_A - Num_B

    pa3_Frame_N_A_Record = pa3_frame_SampleReadingsTest_body_Array[:pa3_Frame_N_A]
    pa3_Frame_N_B_Record = pa3_frame_SampleReadingsTest_body_Array[pa3_Frame_N_A:(pa3_Frame_N_A+pa3_Frame_N_B)]
    pa3_Frame_N_D_record = pa3_frame_SampleReadingsTest_body_Array[(pa3_Frame_N_A+pa3_Frame_N_B):(pa3_Frame_N_A+pa3_Frame_N_B+pa3_Frame_N_D)]
    pa3_Frame_N_S_record = pa3_frame_SampleReadingsTest_body_Array[(pa3_Frame_N_A+pa3_Frame_N_B+pa3_Frame_N_D):]

    return pa3_Frame_N_A_Record, pa3_Frame_N_B_Record, pa3_Frame_N_D_record, pa3_Frame_N_S_record

def pa3_SampleReading_A_B_Record_Dictionary(address_name, Num_A, Num_B):
    pa3_frameA2J_A_B_record_dict = {}
    pa3_frameA2J_A_B_record_dict_A_key = []
    pa3_frameA2J_A_B_record_dict_B_key = []
    for name in address_name:
        # pa3_Frame_N_A_Record, pa3_Frame_N_B_Record, pa3_Frame_N_D_record, pa3_Frame_N_S_record = pa3_Frame_SampleReadingsTest_ABDS(address_name, Num_A, Num_B)
        pa3_Frame_N_A_Record, pa3_Frame_N_B_Record,_,_= pa3_Frame_SampleReadingsTest_ABDS(name, Num_A, Num_B)
        
        frame_name = name
        frame_name_A_record = name + '_A_record'
        frame_name_B_record = name + '_B_record'

        pa3_frameA2J_A_B_record_dict_A_key.append(frame_name_A_record)
        pa3_frameA2J_A_B_record_dict_B_key.append(frame_name_B_record)

        pa3_frameA2J_A_B_record_dict[frame_name_A_record] = pa3_Frame_N_A_Record
        pa3_frameA2J_A_B_record_dict[frame_name_B_record] = pa3_Frame_N_B_Record

    pa3_frameA2J_A_B_record_dict_A_key = np.array(pa3_frameA2J_A_B_record_dict_A_key)
    pa3_frameA2J_A_B_record_dict_B_key = np.array(pa3_frameA2J_A_B_record_dict_B_key)
    return pa3_frameA2J_A_B_record_dict, pa3_frameA2J_A_B_record_dict_A_key, pa3_frameA2J_A_B_record_dict_B_key

# pa3_Name = np.array(['A-Debug'])

# pa3_A_SampleReadingsTest = pd.read_csv('2022_pa345_student_data\PA3-'+pa3_Name[0]+'-SampleReadingsTest.txt',
# header=None,  dtype=float ,skiprows=1)

# pa3_A_SampleReadingsTest_Array =pa3_A_SampleReadingsTest.to_numpy()

# print(pa3_A_SampleReadingsTest_Array)
# pa3_A_SampleReadingsTest_head = pd.read_csv('2022_pa345_student_data\PA3-'+pa3_Name[0]+'-SampleReadingsTest.txt',
# header=None, nrows = 1)

# pa3_A_SampleReadingsTest_head_Array = pa3_A_SampleReadingsTest_head.to_numpy()
# print(pa3_A_SampleReadingsTest_head_Array)
# pa3_A_N_S = pa3_A_SampleReadingsTest_head_Array[0][0]
# pa3_A_N_samps = pa3_A_SampleReadingsTest_head_Array[0][1]
# print(pa3_A_N_S, pa3_A_N_samps)

#######################################################################################
#######################################################################################
############### Main         ##########################################################
#######################################################################################
#######################################################################################

if __name__ == '__main__':
    #######################################################################################
    #######################################################################################
    ############### Data Import  ##########################################################
    #######################################################################################
    #######################################################################################
    # get PA3 Marker data
    pa3_BodyA = pd.read_csv('2022_pa345_student_data\Problem3-BodyA.txt', header=None)

    pa3_BodyB = pd.read_csv('2022_pa345_student_data\Problem3-BodyB.txt', header=None)

    pa3_BodyA = pa3_BodyA.to_numpy()
    pa3_BodyB = pa3_BodyB.to_numpy()

    Num_A_Marker = Num_Marker(pa3_BodyA)
    Num_B_Marker = Num_Marker(pa3_BodyB)

    Marker_A_XYZ = Marker_XYZ(pa3_BodyA)
    Marker_B_XYZ = Marker_XYZ(pa3_BodyB)

    tip_A_XYZ = tip_XYZ(pa3_BodyA)
    tip_B_XYZ = tip_XYZ(pa3_BodyB)

    # get PA3 A B LED marker data
    # pa3_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown', 'I-Unknown', 'J-Unknown', 'K-Unknown'])
    pa3_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown=', 'J-Unknown'])

    pa3_frameA2J_A_B_record_dict, pa3_frameA2J_A_B_record_dict_A_key, pa3_frameA2J_A_B_record_dict_B_key = pa3_SampleReading_A_B_Record_Dictionary(pa3_address_name, Num_A_Marker, Num_B_Marker)

    
    