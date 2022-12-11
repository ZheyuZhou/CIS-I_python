import numpy as np
import FindCloestPoint as FCP

class BoundingSphereClass:
    def __init__(self, radius, center, corner):

        # self.s_k_frame = s_k_frame
        # self.vertices = vertices
        # self.triangles = triangles
    
        # self.Finding_centralpoint_radius = self.Finding_centralpoint_radius()
        # self.Closet_bounding_points = self.Closet_bounding_points()
        # self.BoundingSphereSearch = self.BoundingSphereSearch()
        self.radius = radius
        self.center = center
        self.corner = corner
        

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
def Closet_bounding_points(Meshpoints,Triangle_record):

    boundspheres = []
    for i in range(len(Triangle_record)):
        n1 = int(Triangle_record[i][0])
        n2 = int(Triangle_record[i][1])
        n3 = int(Triangle_record[i][2])
        p = Meshpoints[n1]
        q = Meshpoints[n2]
        r = Meshpoints[n3]

        corner = np.array([p,q,r])

        radius = Finding_centralpoint_radius(p,q,r)[1]
        center = Finding_centralpoint_radius(p,q,r)[0]
        boundspheres.append(BoundingSphereClass(radius, center, corner))
    
    numspheres = len(boundspheres)
    return boundspheres, numspheres