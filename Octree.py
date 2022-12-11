import numpy as np
import BoundingSphereClass
import FindCloestPoint as FCP

class BoundingOcTree:
    def __init__(self, boundingSpheres, nSpheres):
        self.Spheres = boundingSpheres
        self.nSpheres = nSpheres
        self.Center = self.Centroid()
        self.MaxRadius = self.FindMaxRadius()
        self.MaxCoord = self.FindMaxCoordinates()
        self.MinCoord = self.FindMinCoordinates()
        self.Subtrees = [[[[] for i in range(2)] for j in range(2)] for k in range(2)]
        self.ConstructSubtrees()

    # Get centroid of all the sphere center point.
    def Centroid(self):
        if self.nSpheres == 0:
            return
        centroid = np.zeros(3)
        for sphere in self.Spheres:
            centroid += sphere.center
        cent = centroid/self.nSpheres
        return cent

    # Get the Max radius of the sphere with the sphere list
    def FindMaxRadius(self):
        if self.nSpheres == 0:
            return
        MaxRadius = self.Spheres[0].radius
        for sphere in self.Spheres:

            if sphere.radius > MaxRadius:
                MaxRadius = sphere.radius
        return MaxRadius
        
    # Get the upper bound of the coordinate.    
    def FindMaxCoordinates(self):
        if self.nSpheres == 0:
            return
        maxcoord_stack = np.vstack([S.center + S.radius for S in self.Spheres])
        maxcoord = np.max(maxcoord_stack, axis=0)
        return maxcoord
    
    # Get the lower bound of the coordinate.
    def FindMinCoordinates(self):
        if self.nSpheres == 0:
            return
        mincoord_stack = np.vstack([S.center - S.radius for S in self.Spheres])
        mincoord = np.min(mincoord_stack, axis=0)
        return mincoord


    # Construct a subtree.
    def ConstructSubtrees(self):
        # print(self.nSpheres, self.MaxCoord, self.MinCoord, 'XXXXXXXXXXXXXXXXXXXXXXX')
        if self.nSpheres <= 1 or all(self.MaxCoord) == all(self.MinCoord):
            self.havesubtrees = False
            return
        self.havesubtrees = True
        Sphere = self.SplitSort()
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    s = Sphere[a][b][c]
                    if len(self.Subtrees[a][b][c])==0:
                        self.Subtrees[a][b][c] = BoundingOcTree(s,len(s))
    
    # spilt the sphere
    def Splitsort(self):
        if self.nSpheres <= 1:
            return

        Reorder_Sphere = [[[[]for i in range(2)]for j in range(2)]for k in range(2)]

        if self.nSpheres == 2:
            Reorder_Sphere[0][0][0] = self.Spheres[0]
            Reorder_Sphere[1][1][1] = self.Spheres[1]
            return Reorder_Sphere

        for S in self.Spheres:
            if S.center[0] < self.Center[0]:
                i = 1
            elif S.center[0] >= self.Center[0]:
                i = 0

            if S.center[1] < self.Center[1]:
                j = 1
            elif S.center[1] >= self.Center[1]:
                j = 0

            if S.center[2] < self.Center[2]:
                k = 1
            elif S.center[2] >= self.Center[2]:
                k = 0

            Reorder_Sphere[i][j][k] = S
        return Reorder_Sphere
            
    
    # update the closest
    def FindClosestPoint(self, v_coord, bound, closest):
        if self.nSpheres == 0:
            return
        # print(bound[0], 'bound at octree fcp')
        distance = self.MaxRadius + bound[0]

        # print(v_coord, 'v_coord at octree fcp')
        # print(self.MaxCoord, 'self.MaxCoord at octree fcp')
        if (v_coord[0] > self.MaxCoord[0] + distance).all():
            return
        if (v_coord[0] < self.MinCoord[0] - distance).all():
            return
        if (v_coord[1] > self.MaxCoord[1] + distance).all():
            return
        if (v_coord[1] < self.MinCoord[1] - distance).all():
            return
        if (v_coord[2] > self.MaxCoord[2] + distance).all():
            return
        if (v_coord[2] < self.MinCoord[2] - distance).all():
            return

        if self.havesubtrees:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.Subtrees[i][j][k][0].FindClosestPoint(v_coord, bound, closest)
        else:
            for i in range(self.nSpheres):
                self.UpdateClosest(self.Spheres[i], v_coord, bound, closest)


    def UpdateClosest(self, sphere: BoundingSphereClass, v_coord, bound, closest_point):
        closest = FCP.Find_closet_triangle_point(v_coord, sphere.corner[0], sphere.corner[1], sphere.corner[2])
        
        distance = np.linalg.norm(v_coord - sphere.center)

        currDistance = np.linalg.norm(v_coord - closest)

        if distance - sphere.radius > bound[0]:
            return
        if currDistance < bound[0]:
            bound[0] = currDistance
            # print(closest_point, 'cloest_point at Octree')
            # print(closest, 'cloest at Octree')
            closest_point[0] = closest