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
    

# Get the Max radius of the sphere with the sphere list
    def FindMaxRadius(self):
        if len(self.Spheres) == 0:
            return
        MaxR = 0
        R_total = []
        for s in self.Spheres:
            R_total.append(Finding_centralpoint_radius(s)[1])
        MaxR = np.max(R_total)
        return MaxR


# Get centroid of all the sphere center point.
    def Centroid(self):
        if len(self.Spheres) == 0:
            return    
        cen = np.array([0,0,0])
        for s in self.Spheres:
            cen = cen + Finding_centralpoint_radius(s)[0]
        centroid = cen/self.nSpheres
        return centroid

# Get the upper bound of the coordinate.    
    def FindMaxCoordinates(self):
        if len(self.Spheres) == 0:
            return
        UB = [] 
        for s in self.Spheres:
            UB_1 = Finding_centralpoint_radius(s)[0]+ Finding_centralpoint_radius(s)[1]
            UB.append(UB_1)
            UB = np.vstack(UB)
        Final_UB = np.max(UB, axis=0)
        return Final_UB

# Get the lower bound of the coordinate.
    def FindMinCoordinates(self):
        if len(self.Spheres) == 0:
            return
        LB = []
        for s in self.Spheres:
            LB_1 = Finding_centralpoint_radius(s)[0]+ Finding_centralpoint_radius(s)[1]
            LB.append(LB_1)
            LB = np.vstack(LB)
        Final_LB = np.min(LB, axis=0)
        return Final_LB


# Construct a subtree.
    def ConstructSubtrees(self):
        if self.nSpheres <= 1 or self.UB == self.LB:
            self.havesubtrees = False
            return
        self.havesubtrees = True
        Sphere = self.SplitSort()
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    s = Sphere[a][b][c]
                    if len(self.Subtrees[a][b][c])==0:
                        self.Subtrees[a][b][c] = octree(s,len(s))


# spilt the sphere 
    def SplitSort(self,centerpoint):
        if len(self.Spheres) <= 1:
            return
        Reorder_Sphere = [[[[]for i in range(2)]for j in range(2)]for k in range(2)]
        if len(self.Spheres) == 2:
            Reorder_Sphere[0][0][0] = self.Spheres[0]
            Reorder_Sphere[1][1][1] = self.Spheres[1]
            return Reorder_Sphere

        for s in self.Spheres:
            if Finding_centralpoint_radius(s)[0][0] >= centerpoint[0]:
                a = 1
            else:
                a = 0
            if Finding_centralpoint_radius(s)[0][1] >= centerpoint[1]:
                b = 1
            else:
                b = 0
            if Finding_centralpoint_radius(s)[0][2] >= centerpoint[2]:
                c = 1
            else:
                c = 0
            Reorder_Sphere[a][b][c] = s
            return Reorder_Sphere




# update the closest 
    def FindClosestPoint(self, v, bound, closest):
        if len(self.Spheres) == 0:
            return
        
        double_dist = bound[0] + self.MaxRadius
        if v[0] > self.upperbound[0] + double_dist:
            return
        if v[1] > self.upperbound[1] + double_dist:
            return
        if v[2] > self.upperbound[2] + double_dist:
            return
        if v[0] < self.lowerbound[0] - double_dist:
            return
        if v[1] < self.lowerbound[1] - double_dist:
            return
        if v[2] < self.lowerbound[2] - double_dist:
            return
        if self.havesubtrees:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.Subtrees[i][j][k][0].FindClosestPoint(v, bound, closest)
        else:
            for i in range(self.nSpheres):
                self.UpdateClosest(self.Spheres[i],v,bound,closest)
        
    def UpdateClosest(self,sphere,v,bound,closest):
        return