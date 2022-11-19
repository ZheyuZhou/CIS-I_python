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

import CloudRegistration_ as cloudregistration
import pa345DataImport as dataimport
#######################################################################################
#######################################################################################
############### Functions    ##########################################################
#######################################################################################
#######################################################################################


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

    # get PA3 A B LED marker data
    pa3_address_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug', 'G-Unknown', 'H-Unknown=', 'J-Unknown'])

    pa3_frameA2J_A_B_record_dict, pa3_frameA2J_A_B_record_dict_A_key, pa3_frameA2J_A_B_record_dict_B_key = dataimport.SampleReading_A_B_Record_Dictionary(pa3_address_name, pa3_Num_A_Marker, pa3_Num_B_Marker)

    # get PA3 mesh data
    pa3_Mesh = pd.read_csv('2022_pa345_student_data\Problem3Mesh.sur', header=None)

    pa3_Mesh = pa3_Mesh.to_numpy()

    pa3_vertices = dataimport.Vertices(pa3_Mesh)

    pa3_triangles = dataimport.Triangles(pa3_Mesh)
    
