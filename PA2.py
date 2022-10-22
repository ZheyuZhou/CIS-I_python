#  .............................................
#                     _ooOoo_  
#                    o8888888o  
#                    88" . "88  
#                    (| -_- |)  
#                     O\ = /O  
#                 ____/`---'\____  
#               .   ' \\| |// `.  
#                / \\||| : |||// \  
#              / _||||| -:- |||||- \  
#                | | \\\ - /// | |  
#              | \_| ''\---/'' | |  
#               \ .-\__ `-` ___/-. /  
#            ___`. .' /--.--\ `. . __  
#         ."" '< `.___\_<|>_/___.' >'"".  
#        | | : `- \`.;`\ _ /`;.`/ - ` : | |  
#          \ \ `-. \_ __\ /__ _/ .-` / /  
#  ======`-.____`-.___\_____/___.-`____.-'======  
#                     `=---='  
#
#           佛祖保佑             永无BUG 
#  .............................................  

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

#######################################################################################
#######################################################################################
############### Data Import  ##########################################################
#######################################################################################
#######################################################################################
# Name = np.array(['debug-a', 'debug-b', 'debug-c', 'debug-d', 'debug-e', 'debug-f', 'unknown-g', 'unknown-h' , 'unknown-i', 'unknown-j', 'unknown-k'])
Name = np.array(['debug-a'])
name = ''
for nm in Name:

    # calbody
    # Read from TXT
    df_pa2_calbody = pd.read_csv('pa2_student_data\pa2-'+nm+'-calbody.txt',
    header=None, names=['N_d','N_a','N_c','Name_CALBODY'])
    # W. McKinney, PANDAS.READ_CSV#, Pandas.read_csv - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html (accessed October 12, 2022). 

    calbody = df_pa2_calbody[['N_d','N_a','N_c']].to_numpy()    
    # W. McKinney, Pandas.dataframe.to_numpy#, Pandas.DataFrame.to_numpy - Pandas 1.5.0 Documentation. (2022). https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html (accessed October 12, 2022). 

    # Get num
    num_calbody = calbody[0]
    num_calbody_d = int(num_calbody[0])
    num_calbody_a = int(num_calbody[1])
    num_calbody_c = int(num_calbody[2])

    # Trim to get d, a, c
    calbody_d = calbody[1:(1+num_calbody_d),:]
    calbody_a = calbody[(1+num_calbody_d):(1+num_calbody_d+num_calbody_a),:]
    calbody_c = calbody[(1+num_calbody_d+num_calbody_a):(1+num_calbody_d+num_calbody_a+num_calbody_c),:]


    # calreadings
    # Read from TXT
    df_pa2_calreadings = pd.read_csv('pa2_student_data\pa2-'+nm+'-calreadings.txt', 
    header=None, names=['N_D','N_A','N_C','N_Frame','Name_CALREADING'])

    calreadings = df_pa2_calreadings[['N_D','N_A','N_C','N_Frame']].to_numpy()
    
    num_calreadings = calreadings[0]

    # get num
    num_calreadings_D = int(num_calreadings[0])
    num_calreadings_A = int(num_calreadings[1])
    num_calreadings_C = int(num_calreadings[2])
    num_calreadings_Frame = int(num_calreadings[3])

    num_calreadings_len = num_calreadings_D + num_calreadings_A + num_calreadings_C

    # Trim to get D, A, C
    calreadings_D = np.zeros((num_calreadings_Frame,num_calreadings_D,3))
    calreadings_A = np.zeros((num_calreadings_Frame,num_calreadings_A,3))
    calreadings_C = np.zeros((num_calreadings_Frame,num_calreadings_C,3))
    # C. Harris, Numpy.zeros#, Numpy.zeros - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.zeros.html (accessed October 12, 2022). 

    for i in range (num_calreadings_Frame):
        I = i*num_calreadings_len
        calreadings_D[i] = calreadings[(1) + I : (1+num_calreadings_D) + I , :-1]
        calreadings_A[i] = calreadings[(1+num_calreadings_D) + I : (1+num_calreadings_D+num_calreadings_A) + I,:-1]
        calreadings_C[i] = calreadings[(1+num_calreadings_D+num_calreadings_A) + I : (1+num_calreadings_D+num_calreadings_A+num_calreadings_C) + I,:-1]

    

    # empivot
    # Read from TXT
    df_pa2_empivot = pd.read_csv('pa2_student_data\pa2-'+nm+'-empivot.txt', 
    header=None, names=['N_G','N_Frame','Name_EMPIVOT'])
    calempivot = df_pa2_empivot[['N_G','N_Frame','Name_EMPIVOT']].to_numpy()

    # get num 
    num_calempivot = calempivot[0]

    num_calempivot_G = int(num_calempivot[0])
    num_calempivot_Frame = int(num_calempivot[1])

    num_calempivot_len = num_calempivot_G

    # Trim to get em G
    calempivot_G = np.zeros((num_calempivot_Frame,num_calempivot_G,3))

    for j in range (num_calempivot_Frame):
        J = j*num_calempivot_len
        calempivot_G[j] = calempivot[1 + J : (1+num_calempivot_G) + J, : ]


    
    # optpivot
    # Read from TXT
    df_pa2_optpivot = pd.read_csv('pa2_student_data\pa2-'+nm+'-optpivot.txt', 
    header=None, names=['N_D','N_H','N_Frames','Name_OPTPIVOT'])
    caloptpivot = df_pa2_optpivot[['N_D','N_H','N_Frames']].to_numpy()

    # get num
    num_caloptpivot = caloptpivot[0]

    num_caloptpivot_D = int(num_caloptpivot[0])
    num_caloptpivot_H = int(num_caloptpivot[1])
    num_caloptpivot_Frame = int(num_caloptpivot[2])

    num_caloptpivot_len = num_caloptpivot_D + num_caloptpivot_H

    # Trim to get opt D, opt H
    caloptpivot_D = np.zeros((num_caloptpivot_Frame,num_caloptpivot_D ,3))
    caloptpivot_H = np.zeros((num_caloptpivot_Frame,num_caloptpivot_H,3))

    for k in range (num_caloptpivot_Frame):
        K = k*num_caloptpivot_len
        caloptpivot_D[k] = caloptpivot[(1) + K : (1+num_caloptpivot_D) + K , : ]
        caloptpivot_H[k] = caloptpivot[(1+num_caloptpivot_D) + K : (1+num_caloptpivot_D+num_caloptpivot_H) + K,: ]

    
    
    # ct-fiducials
    # Read from TXT
    df_pa2_ct_fiducials = pd.read_csv('pa2_student_data\pa2-'+nm+'-ct-fiducials.txt', 
    header=None, names=['N_B','NAME-CT-FIDUCIALS.TXT'])
    ct_fiducials = df_pa2_ct_fiducials[['N_B','NAME-CT-FIDUCIALS.TXT']].to_numpy()

    # get num
    num_ct_fiducials = ct_fiducials[0]

    num_ct_fiducials_B = int(num_ct_fiducials[0])
    # Trim to get b coordinates
    ct_fiducials_B = pd.read_csv('pa2_student_data\pa2-'+nm+'-ct-fiducials.txt', 
    header=None, skiprows = 1).to_numpy()



    # em-fiducials
    # Read from TXT
    df_pa2_em_fiducials = pd.read_csv('pa2_student_data\pa2-'+nm+'-em-fiducialss.txt', 
    header=None, names=['N_G','N_B','NAME-EM-FIDUCIALS.TXT'])
    em_fiducials = df_pa2_em_fiducials[['N_G','N_B','NAME-EM-FIDUCIALS.TXT']].to_numpy()

    # get num
    num_em_fiducials = em_fiducials[0]

    num_em_fiducials_G = int(num_em_fiducials[0])
    num_em_fiducials_B = int(num_em_fiducials[1])

    # Trim to get G
    em_fiducials_G = np.zeros((num_em_fiducials_B, num_em_fiducials_G, 3))
    num_em_fiducials_len = num_em_fiducials_G

    for k in range (num_em_fiducials_B):
        K = k*num_em_fiducials_len
        em_fiducials_G[k] = em_fiducials[(1) + K : (1+num_em_fiducials_G) + K , : ]

    

    # em-nav
    df_pa2_em_nav= pd.read_csv('pa2_student_data\pa2-'+nm+'-EM-nav.txt', 
    header=None, names=['N_G','N_Frames','NAME-EMNAV.TXT'])
    em_nav = df_pa2_em_nav[['N_G','N_Frames','NAME-EMNAV.TXT']].to_numpy()

    # get num
    num_em_nav = em_nav[0]
    
    num_em_nav_G = int(num_em_nav[0])
    num_em_nav_Frame = int(num_em_nav[1])

    # Trim to get G
    em_nav_G = np.zeros((num_em_nav_Frame, num_em_nav_G, 3))
    num_em_nav_len = num_em_nav_G

    for k in range (num_em_nav_Frame):
        K = k*num_em_nav_len
        em_nav_G[k] = em_nav[(1) + K : (1+num_em_nav_G) + K , : ]
