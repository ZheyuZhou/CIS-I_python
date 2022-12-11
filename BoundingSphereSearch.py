import numpy as np
import FindCloestPoint as FCP

def Finding_centralpoint_radius(a,b,c):
    f = (a+b)/2
    u = a - f
    v = c - f
    d = np.cross(np.cross(u,v),u)
    ld = max(0, (v.dot(v) - u.dot(u))/(2*d.dot(v-u)))
    q = f + ld*d
    # q is the central point of sphere
    p = np.linalg.norm(a-q)
    # p is the radius
    return q,p


# Find the c_k with the bounding sphere judgement
def Closet_bounding_points(Meshpoints,Triangle_record,s_kn):
    # print(s_kn, 's_kn')
    # print(np.shape(s_kn), 's_kn shape')
    
    s_kn3 = np.reshape(s_kn[0:3], (1,3))[0]
    bound = np.linalg.norm(s_kn3)
    for i in range(len(Triangle_record)):
        n1 = int(Triangle_record[i][0])
        n2 = int(Triangle_record[i][1])
        n3 = int(Triangle_record[i][2])
        p = Meshpoints[n1]
        q = Meshpoints[n2]
        r = Meshpoints[n3]
        ph = Finding_centralpoint_radius(p,q,r)[1]
        ct = Finding_centralpoint_radius(p,q,r)[0]
        # calculate the radius of the sphere 
        d1 = np.linalg.norm(s_kn3 - ct) - ph
        if d1 < bound:
            c_k1 = FCP.Find_closet_triangle_point(s_kn,p,q,r)
            d2 = np.linalg.norm(s_kn3 - c_k1)
            if d2 < bound:
                c_k = c_k1
                bound = d2
        
    return c_k

def BoundingSphereSearch(s_k_frame, vertices,triangles):
    c_closest_frame = []
    # print(np.shape(s_k_frame), 's_k_frame shape')
    for s_k in s_k_frame:
        c_closest_sample = []
        for s in s_k:
            c_closest = Closet_bounding_points(vertices,triangles,s)
            c_closest_sample.append(c_closest)
        c_closest_sample = np.array(c_closest_sample)
        c_closest_frame.append(c_closest_sample)
    c_closest_frame = np.array(c_closest_frame)
    return c_closest_frame