import numpy as np
import Matching_Speedup as ms
import BoundingSphere

class BoundingBoxTreeNode:
    def __init__(self, boundingSpheres, numSpheres):
        self.Spheres = boundingSpheres
        self.nSpheres = numSpheres
        self.Center = self.Centroid()
        self.MaxRadius = self.FindMaxRadius()
        self.upperbound = self.FindMaxCoordinates()
        self.lowerbound = self.FindMinCoordinates()
        # create a 3D array for the 8 subtrees of each node
        self.Subtrees = [[[[] for i in range(2)] for j in range(2)] for k in range(2)] 
        self.ConstructSubtrees()
    

    def ConstructSubtrees(self):
        if len(self.Spheres) <= 1:
            self.HaveSubtrees = False
            return
        self.HaveSubtrees = True
        splitsphere = self.SplitSort()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    subspheres = splitsphere[i][j][k]
                    self.Subtrees[i][j][k].append(BoundingBoxTreeNode(subspheres, len(subspheres)))
            
    
    def SplitSort(self):
        if len(self.Spheres) == 0 or len(self.Spheres) == 1:
            return
        splitsphere = [[[[] for i in range(2)] for j in range(2)] for k in range(2)]
        if len(self.Spheres) == 2:
            splitsphere[0][0][0].append(self.Spheres[0])
            splitsphere[1][1][1].append(self.Spheres[1])
            return splitsphere
        for sphere in self.Spheres:
            if sphere.center[0] <= self.Center[0]:
                if sphere.center[1] <= self.Center[1]:
                    if sphere.center[2] <= self.Center[2]:
                        splitsphere[0][0][0].append(sphere)
                    else:
                        splitsphere[0][0][1].append(sphere)
                else:
                    if sphere.center[2] <= self.Center[2]:
                        splitsphere[0][1][0].append(sphere)
                    else:
                        splitsphere[0][1][1].append(sphere)
            else:
                if sphere.center[1] <= self.Center[1]:
                    if sphere.center[2] <= self.Center[2]:
                        splitsphere[1][0][0].append(sphere)
                    else:
                        splitsphere[1][0][1].append(sphere)
                else:
                    if sphere.center[2] <= self.Center[2]:
                        splitsphere[1][1][0].append(sphere)
                    else:
                        splitsphere[1][1][1].append(sphere)
        return splitsphere

    def Centroid(self):
        if len(self.Spheres) == 0:
            return
        centroid = np.zeros(3)
        for sphere in self.Spheres:
            centroid += sphere.center
        cent = centroid/self.nSpheres
        return cent
    
    def FindMaxRadius(self):
        if len(self.Spheres) == 0:
            return
        MaxRadius = self.Spheres[0].radius
        for sphere in self.Spheres:
            if sphere.radius > MaxRadius:
                MaxRadius = sphere.radius
        return MaxRadius
    
    def FindMaxCoordinates(self):
        if len(self.Spheres) == 0:
            return
        maxcoord = np.vstack([sphere.center + sphere.radius for sphere in self.Spheres])
        MaxCoordinates = np.max(maxcoord, axis=0)
        # MaxCoordinates = self.Spheres[0].center + self.Spheres[0].radius
        # for sphere in self.Spheres:
        #     if (sphere.center + sphere.radius > MaxCoordinates).all():
        #         MaxCoordinates = sphere.center + sphere.radius
        return MaxCoordinates

    def FindMinCoordinates(self):
        if len(self.Spheres) == 0:
            return
        mincoord = np.vstack([sphere.center - sphere.radius for sphere in self.Spheres])
        MinCoordinates = np.min(mincoord, axis=0)
        # MinCoordinates = self.Spheres[0].center - self.Spheres[0].radius
        # for sphere in self.Spheres:
        #     if (sphere.center - sphere.radius < MinCoordinates).all():
        #         MinCoordinates = sphere.center - sphere.radius
        return MinCoordinates

    def FindClosestPoint(self, v_coord, bound, closest):
        if len(self.Spheres) == 0:
            return
        distance = self.MaxRadius + bound[0]
        if v_coord[0]> self.upperbound[0] + distance:
            return
        if v_coord[0] < self.lowerbound[0] - distance:
            return
        if v_coord[1] > self.upperbound[1] + distance:
            return
        if v_coord[1] < self.lowerbound[1] - distance:
            return
        if v_coord[2] > self.upperbound[2] + distance:
            return
        if v_coord[2] < self.lowerbound[2] - distance:
            return
        if self.HaveSubtrees:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.Subtrees[i][j][k][0].FindClosestPoint(v_coord, bound, closest)
        else:
            for i in range(self.nSpheres):
                self.UpdateClosestPoint(self.Spheres[i], v_coord, bound, closest)

    def UpdateClosestPoint(self, sphere: BoundingSphere, v_coord, bound, closest_point):
        closest =  ms.ClosestPointTo(v_coord, sphere.corner[0], sphere.corner[1], sphere.corner[2])
        distance = np.linalg.norm(v_coord - sphere.center)
        currDistance = np.linalg.norm(v_coord - closest)
        if distance - sphere.radius > bound[0]:
            return
        if currDistance < bound[0]:
            bound[0] = currDistance
            closest_point[0] = closest