import numpy as np
import pandas as pd
#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
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
def Frame_SampleReadingsTest_Body(address_name): # Get Body Array
    frame_SampleReadingsTest_body = pd.read_csv('2022_pa345_student_data\PA4-'+address_name+'-SampleReadingsTest.txt', header=None, skiprows = 1)
    frame_SampleReadingsTest_body_Array = frame_SampleReadingsTest_body.to_numpy()
    return frame_SampleReadingsTest_body_Array

def Frame_SampleReadingsTest_Head(address_name): # Get Head Array
    frame_SampleReadingsTest_head = pd.read_csv('2022_pa345_student_data\PA4-'+address_name+'-SampleReadingsTest.txt', header=None, nrows = 1)
    frame_SampleReadingsTest_head_Array = frame_SampleReadingsTest_head.to_numpy()
    return frame_SampleReadingsTest_head_Array

def Frame_SampleReadingsTest_ABDS(address_name, Num_A, Num_B):
    frame_SampleReadingsTest_body_Array = Frame_SampleReadingsTest_Body(address_name)
    frame_SampleReadingsTest_head_Array = Frame_SampleReadingsTest_Head(address_name)
    Frame_N_S = frame_SampleReadingsTest_head_Array[0][0]
    Frame_N_samp = frame_SampleReadingsTest_head_Array[0][1]

    Frame_N_A = Num_A
    Frame_N_B = Num_B
    Frame_N_D = Frame_N_S - Num_A - Num_B

    Frame_N_A_Record = []
    Frame_N_B_Record = []
    Frame_N_D_Record = []
    for i in range(Frame_N_samp):
        r = i*(Frame_N_A + Frame_N_B + Frame_N_D)
        Frame_N_A_Record_Sample = frame_SampleReadingsTest_body_Array[r:(r+Frame_N_A)]
        Frame_N_B_Record_Sample = frame_SampleReadingsTest_body_Array[(r+Frame_N_A):(r+Frame_N_A+Frame_N_B)]
        Frame_N_D_Record_Sample = frame_SampleReadingsTest_body_Array[(r+Frame_N_A+Frame_N_B):(r+Frame_N_A+Frame_N_B+Frame_N_D)]

        Frame_N_A_Record.append(Frame_N_A_Record_Sample)
        Frame_N_B_Record.append(Frame_N_B_Record_Sample)
        Frame_N_D_Record.append(Frame_N_D_Record_Sample)
    return Frame_N_A_Record, Frame_N_B_Record, Frame_N_D_Record

def SampleReading_A_B_Record_Dictionary(address_name, Num_A, Num_B):
    frameA2J_A_B_record_dict = {}
    frameA2J_A_B_record_dict_A_key = []
    frameA2J_A_B_record_dict_B_key = []
    for name in address_name:
        Frame_N_A_Record, Frame_N_B_Record,_= Frame_SampleReadingsTest_ABDS(name, Num_A, Num_B)
        
        frame_name_A_record = name + '_A_record'
        frame_name_B_record = name + '_B_record'

        frameA2J_A_B_record_dict_A_key.append(frame_name_A_record)
        frameA2J_A_B_record_dict_B_key.append(frame_name_B_record)

        frameA2J_A_B_record_dict[frame_name_A_record] = Frame_N_A_Record
        frameA2J_A_B_record_dict[frame_name_B_record] = Frame_N_B_Record

    frameA2J_A_B_record_dict_A_key = np.array(frameA2J_A_B_record_dict_A_key)
    frameA2J_A_B_record_dict_B_key = np.array(frameA2J_A_B_record_dict_B_key)
    return frameA2J_A_B_record_dict, frameA2J_A_B_record_dict_A_key, frameA2J_A_B_record_dict_B_key



# read from mesh data
def Num_Vertices(Mesh):
    num_vertices = int(Mesh[0][0])
    return num_vertices

def Vertices(Mesh):
    num_vertices = Num_Vertices(Mesh)
    vertices = Mesh[1:(1+num_vertices)]

    vertices_list = []
    for row in vertices:
        row = np.fromstring(row[0],dtype=float,sep=' ')
        vertices_list.append(row)
    vertices_array = np.array(vertices_list)
    return vertices_array

def Num_Triangles(Mesh):
    num_vertices = Num_Vertices(Mesh)
    num_traingles = int(Mesh[(1+num_vertices)][0])
    return num_traingles

def Triangles(Mesh):
    num_vertices = Num_Vertices(Mesh)
    traingles = Mesh[(2+num_vertices):]

    traingles_list = []
    for row in traingles:
        row = np.fromstring(row[0],dtype=float,sep=' ')
        traingles_list.append(row)
    traingles_array = np.array(traingles_list)
    return traingles_array