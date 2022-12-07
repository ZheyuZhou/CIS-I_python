import numpy as np
import FindCloestPoint as FCP
# linear search through all the triangles (brute)
def Find_closet_points(Meshpoints,Triangle_record,s_kn):
    # Meshpoints is N vertices,Triangle_record is Vertex indices
    c_total = []
    d_total = []
    # Doing the loop for the all points of triangle 
    for i in range(len(Triangle_record)):
        n1 = int(Triangle_record[i][0])
        n2 = int(Triangle_record[i][1])
        n3 = int(Triangle_record[i][2])
        p = Meshpoints[n1]
        q = Meshpoints[n2]
        r = Meshpoints[n3]
        # Find the closet point of the triangle 
        c_k = FCP.Find_closet_triangle_point(s_kn,p,q,r)
        c_total.append(c_k)
        s_kn = np.reshape(s_kn[0:3], (1,3))[0]
        dis = np.linalg.norm(c_k - s_kn)
        # D. Gupta, Numpy.linalg.norm - NumPy v1.23 Manual. (2022).https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        d_total.append(dis)
        # storage of the data
    c_total = np.array(c_total)
    d_total = np.array(d_total)
    return c_total,d_total

# Select the real cloest points from all the points we . 
def Find_closet_mesh_point(c_total,s_kn):
    dist = []
    s_kn = np.reshape(s_kn[0:3], (1,3))[0]
    for i in range(len(c_total)):
        dist.append(np.linalg.norm(c_total[i] - s_kn))
    dist = np.array(dist)
    closet_dist_index = np.argmin(dist)
     # D. Gupta, Numpy.argmax and argmin #, Numpy.argmin - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.arg
    return c_total[closet_dist_index]

def BruteSearch(s_k_frame, vertices,triangles):
    c_closest_frame = []
    for s_k in s_k_frame:
        c_closest_sample = []
        for s in s_k:
            c_total,_ = Find_closet_points(vertices,triangles,s)
            c_closest = Find_closet_mesh_point(c_total,s)
            c_closest_sample.append(c_closest)
        c_closest_sample = np.array(c_closest_sample)
        c_closest_frame.append(c_closest_sample)
    c_closest_frame = np.array(c_closest_frame)
    return c_closest_frame