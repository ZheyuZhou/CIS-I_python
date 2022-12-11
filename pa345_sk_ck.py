import numpy as np
import pandas as pd
import CloudRegistration_ as cloudregistration
import pa345DataImport as dataimport
import findtipdk as findtipdk

def sk_ck(address_name):
    len_pa4_address_name = len(address_name)
    # get PA3 Marker data
    pa4_BodyA = pd.read_csv('2022_pa345_student_data\Problem4-BodyA.txt', header=None)

    pa4_BodyB = pd.read_csv('2022_pa345_student_data\Problem4-BodyB.txt', header=None)

    pa4_BodyA = pa4_BodyA.to_numpy()
    pa4_BodyB = pa4_BodyB.to_numpy()

    pa4_Num_A_Marker = dataimport.Num_Marker(pa4_BodyA)
    pa4_Num_B_Marker = dataimport.Num_Marker(pa4_BodyB)

    pa4_Marker_A_XYZ = dataimport.Marker_XYZ(pa4_BodyA)
    pa4_Marker_B_XYZ = dataimport.Marker_XYZ(pa4_BodyB)

    tip_A_XYZ = dataimport.tip_XYZ(pa4_BodyA)
    tip_B_XYZ = dataimport.tip_XYZ(pa4_BodyB)

    # get PA3 mesh data
    pa4_Mesh = pd.read_csv('2022_pa345_student_data\Problem4MeshFile.sur', header=None)

    pa4_Mesh = pa4_Mesh.to_numpy()

    pa4_vertices = dataimport.Vertices(pa4_Mesh)

    pa4_triangles = dataimport.Triangles(pa4_Mesh)

    # print(pa3_vertices, 'ver')
    # print(pa3_triangles, 'tri')
    # get PA3 A B LED marker data
    pa4_frameA2J_A_B_record_dict, pa4_frameA2J_A_B_record_dict_A_key, pa4_frameA2J_A_B_record_dict_B_key = dataimport.SampleReading_A_B_Record_Dictionary(address_name, pa4_Num_A_Marker, pa4_Num_B_Marker)

    # frame AB marker cloudregistration

    # Cloud Registration
    F_A_frame = []
    F_B_frame = []
    for frame in range(len_pa4_address_name):
        A_key = pa4_frameA2J_A_B_record_dict_A_key[frame]
        B_key = pa4_frameA2J_A_B_record_dict_B_key[frame]
        frame_A_record_XYZ = pa4_frameA2J_A_B_record_dict[A_key]
        frame_B_record_XYZ = pa4_frameA2J_A_B_record_dict[B_key]

        F_A = []
        F_B = []
        for sample in range(len(frame_A_record_XYZ)):
            F_A.append(cloudregistration.Cloudregistration(pa4_Marker_A_XYZ, frame_A_record_XYZ[sample]))
            F_B.append(cloudregistration.Cloudregistration(pa4_Marker_B_XYZ, frame_B_record_XYZ[sample]))
            # F_A.append(cloudregistration.Cloudregistration(frame_A_record_XYZ[sample], pa3_Marker_A_XYZ))
            # F_B.append(cloudregistration.Cloudregistration(frame_B_record_XYZ[sample], pa3_Marker_B_XYZ))
        F_A = np.array(F_A)
        F_B = np.array(F_B)
        F_A_frame.append(F_A)
        F_B_frame.append(F_B)

    F_A_frame = np.array(F_A_frame)
    F_B_frame = np.array(F_B_frame)


    # Calculate d_k tip in B coordinate frame
    d_k_frame = []

    for frame in range(len(F_A_frame)):
        d_k = findtipdk.findTip(F_A_frame[frame], F_B_frame[frame], tip_A_XYZ)
        d_k_frame.append(d_k)
    d_k_frame = np.array(d_k_frame)

    # d_k3_frame = []
    # for d_k_sample in d_k_frame:
    #     d_k3_sample = []
    #     for d_k in d_k_sample:
    #         d_k3 = np.reshape(d_k[0:3], (1,3))[0]
    #         d_k3_sample.append(d_k3)
    #     d_k3_sample = np.array(d_k3_sample)
    #     d_k3_frame.append(d_k3_sample)
    # d_k3_frame = np.array(d_k3_frame)

    # s_k sample points
    s_k_frame = []
    F_reg = np.eye(4) # Identity for PA 3

    for frame in range(len(d_k_frame)):
        s_k = F_reg@d_k_frame[frame]
        s_k_frame.append(s_k)

    return s_k_frame, d_k_frame, pa4_vertices, pa4_triangles