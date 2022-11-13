import numpy as np
# D. Cournapeau, P. Virtanen, A. R. Terrel, NumPy, GitHub. (n.d.). https://github.com/numpy (accessed October 12, 2022). 
import pandas as pd
# W. McKinney, Pandas, Pandas. (2022). https://pandas.pydata.org/ (accessed October 12, 2022). 
# from tqdm import tqdm_gui
import math
# N. Samuel, Math - mathematical functions, Math - Mathematical Functions - Python 3.10.8 Documentation. (2022). https://docs.python.org/3/library/math.html (accessed October 26, 2022). 
from itertools import product

def LeastSquare(F):
        R = np.array([[0,0,0]])
        p = np.array([[0]])
        neg_I = np.array([[0,0,0]])
        for F_j in F:
            # Slice R and p out of F
            R_j = F_j[0:3 , 0:3]
            p_j = F_j[0:3, 3:4]

            neg_I_j = -np.eye(3)

            R = np.concatenate((R,R_j),axis=0)
            # I. Polat, Numpy.linalg.eig#, Numpy.linalg.eig - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html (accessed October 12, 2022). 
            p = np.concatenate((p,p_j),axis=0)
            neg_I = np.concatenate((neg_I,neg_I_j),axis=0)

        # Slice out the 0,0,0 on first row for R, p, neg_I
        R = R[1:, :]
        p = p[1:, :]
        neg_I = neg_I[1:, :]
        
        Coeff_Matrix = np.concatenate((R,neg_I),axis=1)

        # print(Coeff_Matrix)
        # print(np.shape(Coeff_Matrix))
        # Calc t_G and P_dimple with least square
        LS_sol = np.linalg.lstsq(Coeff_Matrix, -p, rcond=None)[0]
        # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
        t_G = LS_sol[0:3]
        P_dimple = LS_sol[3:6]

        return P_dimple, t_G