import numpy as np
import BoundingSphereClass

class BoundingBoxTreeNode:
    def __init__(self, boundingSpheres, numSpheres):
        self.Spheres = boundingSpheres
        self.nSpheres = numSpheres
        self.Center = self.Centroid()
        self.MaxRadius = self.FindMaxRadius()
        self.upperbound = self.FindMaxCoordinates()
        self.lowerbound = self.FindMinCoordinates()
        self.Subtrees = [[[[] for i in range(2)] for j in range(2)] for k in range(2)]
        self.ConstructSubtrees()