import numpy as np

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

class octree:
# construct octree class
    def __init__(self, BS, nS):

        self.Spheres = BS
        self.nSpheres = nS
        self.Center = self.Centroid()
        self.MaxRadius = self.FindMaxRadius()
        self.UB = self.FindMaxCoordinates()
        self.LB = self.FindMinCoordinates()
        self.Subtrees = [[[[]for i in range(2)]for j in range(2)]for k in range(2)]
        self.ConstructSubtrees()




    def Centroid(self):
        if len(self.Spheres) == 0:
            return 
        cen = np.array([0,0,0])
        for s in self.Spheres:
            cen = cen + Finding_centralpoint_radius(s)[0]
        centroid = cen/self.nSpheres
        return centroid

    def FindMaxCoordinates(self):
        if len(self.Spheres) == 0:
            return

        UB = [] 
        for s in self.Spheres:
            UB_1 = Finding_centralpoint_radius(s)[0]+ Finding_centralpoint_radius(s)[1]
            UB.append(UB_1)
        Final_UB = np.max(UB)