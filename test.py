import numpy as np
import math

a = np.array([0,0,0])
p = np.array([2,2,3])
q = np.array([2,0,3])
r = np.array([-2,2,3])
# k = np.array([[q[0]-p[0],r[0]-p[0]],[q[1]-p[1],r[1]-p[1]],[q[2]-p[2],r[2]-p[2]]])
# ld = np.linalg.lstsq(k,a - p, rcond=None)[0][0]
# u = np.linalg.lstsq(k,a - p, rcond=None)[0][1]
# c = p + ld * (q-p) + u * (r-p)
# print(ld)
# print(u)
# print(c)
# k = np.array([[p[0]-q[0],r[0]-q[0]],[p[1]-q[1],r[1]-q[1]],[p[2]-q[2],r[2]-q[2]]])
# print(k)
# print(a-p)
# ld = np.linalg.lstsq(k,a - p, rcond=None)[0][0]
# u = np.linalg.lstsq(k,a - p, rcond=None)[0][1]
# c = p + ld * (q-p) + u * (r-p)
# ld1 = (c-p).dot(q-p)/((q-p).dot(q-p))
# lds = max(0,min(ld1,1))
# cs = p+lds*(q-p)
# print (cs)
# print(lds)

# triangle point finding
def Find_closet_triangle_point(a,p,q,r):
    k = np.array([[q[0]-p[0],r[0]-p[0]],[q[1]-p[1],r[1]-p[1]],[q[2]-p[2],r[2]-p[2]]])
    ld = np.linalg.lstsq(k,a - p, rcond=None)[0][0]
    u = np.linalg.lstsq(k,a - p, rcond=None)[0][1]
    c = p + ld * (q-p) + u * (r-p)
    if ld>=0 and u>=0 and ld+u<=1:
        cnew = c
    elif ld<0:
        cnew = ProjectOnSegment(c,r,p)
    elif u<0:
        cnew = ProjectOnSegment(c,p,q)
    elif ld + u >1:
        cnew = ProjectOnSegment(c,q,r)

    return cnew

def ProjectOnSegment(c,p,q):
    ld = (c-p).dot(q-p)/(q-p).dot(q-p)
    lds = max(0,min(ld,1))
    cs = p+lds*(q-p)
    return cs


def findTip(F_Ak, F_Bk, tipA):
    d_k = []
    for i in range(F_Ak):
        F_aki = F_Ak[i]
        F_bki = F_Bk[i]
        d_k.append(np.linalg.inv(F_aki)@F_bki@tipA)
    d_k = np.array(d_k)
    return d_k

# Brute
def Find_closet_points(Meshpoints,Triangle_record,s_kn):
    # Meshpoints is N vertices,Triangle_record is Vertex indices
    c_total = []
    for i in range(len(Triangle_record)):
        n1 = Triangle_record[i][0]
        n2 = Triangle_record[i][1]
        n3 = Triangle_record[i][2]
        p = Meshpoints[n1]
        q = Meshpoints[n2]
        r = Meshpoints[n3]
        c_k = Find_closet_triangle_point(s_kn,p,q,r)
        c_total.append(c_k)
    c_total = np.array(c_total)
    return c_total

def Find_closet_mesh_point(c_total,s_kn):
    dist = []
    for i in range(len(c_total)):
        dist.append(np.linalg.norm(c_total[i] - s_kn))
    dist = np.array(dist)
    closet_dist_index = np.argmin(dist)
    return c_total[closet_dist_index]

# Bounding of three points
def Finding_centralpoint_radius(a,b,c):
    f = (a+b)/2
    u = a - f
    v = c - f
    d = np.cross(np.cross(u,v),u)
    ld = max(0, (v.dot(v) - u.dot(u))/(2*d.dot(v-u)))
    q = f + ld*d
    # q is the centralpoint
    p = np.linalg.norm(a-q)
    # p is the radius
    return q,p



def Closet_bounding_points(Meshpoints,Triangle_record,s_kn):
    bound = np.linalg.norm(s_kn)
    for i in range(len(Triangle_record)):
        n1 = Triangle_record[i][0]
        n2 = Triangle_record[i][1]
        n3 = Triangle_record[i][2]
        p = Meshpoints[n1]
        q = Meshpoints[n2]
        r = Meshpoints[n3]
        ph = Finding_centralpoint_radius(p,q,r)[1]
        ct = Finding_centralpoint_radius(p,q,r)[0]
        # calculate the radius of the sphere
        # ppt 17 
        d1 = np.linalg.norm(s_kn - ct) - ph
        if d1 < bound:
            c_k1 = Find_closet_triangle_point(s_kn,p,q,r)
            d2 = np.linalg.norm(s_kn - c_k1)
            if d2 < bound:
                c_k = c_k1
                bound = d2
        
    return c_k
















# print(np.cross(p,q), 'test')
# print(Find_closet_triangle_point(a,p,q,r))



