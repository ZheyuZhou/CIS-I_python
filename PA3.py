#######################################################################################
#######################################################################################
############### Imports  ##############################################################
#######################################################################################
#######################################################################################

import numpy as np
# D. Cournapeau, P. Virtanen, A. R. Terrel, NumPy, GitHub. (n.d.). https://github.com/numpy (accessed October 12, 2022). 
import pandas as pd
# W. McKinney, Pandas, Pandas. (2022). https://pandas.pydata.org/ (accessed October 12, 2022). 
# from tqdm import tqdm_gui
import math
# N. Samuel, Math - mathematical functions, Math - Mathematical Functions - Python 3.10.8 Documentation. (2022). https://docs.python.org/3/library/math.html (accessed October 26, 2022). 
from itertools import product

import Cloudregistration as cr


#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################
pa3_BodyA = pd.read_csv(r'C:\Users\14677\Documents\GitHub\FA22-CIS-I_python\2022_pa345_student_data\2022 PA345 Student Data\Problem3-BodyA.txt')
pa3_BodyB = pd.read_csv('2022_pa345_student_data\2022 PA345 Student Data\Problem3-BodyB.txt')

print(pa3_BodyA)
