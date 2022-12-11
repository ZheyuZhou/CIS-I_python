import numpy as np
import pandas as pd
pa4_test_name = np.array(['A-Debug', 'B-Debug', 'C-Debug', 'D-Debug', 'E-Debug', 'F-Debug'])
pa4_output_name = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
for i in range(len(pa4_test_name)):
    check = pd.read_csv('2022_pa345_student_data\PA4-'+pa4_test_name[i]+'-Output.txt',header=None, skiprows = 1)

    check = check.to_numpy()
    check_array = []
    for row in check:
        row_array = np.fromstring(row[0],dtype=float,sep=' ')
        check_array.append(row_array)
    check_array = np.array(check_array)    

    output = pd.read_csv('2022_pa345_student_data\Output\PA4_'+pa4_output_name[i]+'_output.txt')
    output = output.to_numpy()

    diff = np.linalg.norm(check_array - output)
    if diff < 0.5:
        print(pa4_test_name[i]+" Test Passed \(o^ ^o)/", diff)
    else:
        print(pa4_test_name[i]+" Test Fail!!!!!!", diff)