import numpy as np

class BoundingBoxTreeNode:
    def __init__(self, BoundingSpheres, numSpheres):
        self.Spheres = BoundingSpheres
        self.nSpheres = numSpheres
        self.Center = self.Centroid()
        self.MaxRadius = self.FindMaxRadius()
        self.upperbound = self.FindMaxCoordinates()
        self.lowerbound = self.FindMinCoordinates()