import numpy as np
import pandas as pd
pa3_test_name = np.array(['A-Debug'])
for i in range(len(pa3_test_name)):
    check = pd.read_csv('2022_pa345_student_data\PA3-'+pa3_test_name[i]+'-Output.txt',header=None, skiprows = 1)
    check = check.to_numpy()
    check_array = []
    for row in check:
        row_array = np.fromstring(row[0],dtype=float,sep=' ')
        check_array.append(row_array)
    check_array = np.array(check_array)    
    print(check_array)