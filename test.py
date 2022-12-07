import numpy as np
import FindCloestPoint as FCP
# import pandas as pd
# pa3_test_name = np.array(['A-Debug'])
# for i in range(len(pa3_test_name)):
#     check = pd.read_csv('2022_pa345_student_data\PA3-'+pa3_test_name[i]+'-Output.txt',header=None, skiprows = 1)
#     check = check.to_numpy()
#     check_array = []
#     for row in check:
#         row_array = np.fromstring(row[0],dtype=float,sep=' ')
#         check_array.append(row_array)
#     check_array = np.array(check_array)    
#     print(check_array)

a = np.array([-9.827567265024626, -13.442350123557883, -1.221254889135011])

p = np.array([-20.002115, -22.548090, -46.172775])
q = np.array([-16.744041, -23.458670, -44.089203])
r = np.array([-20.032549, -25.838306, -43.115608])

c, lac = FCP.Find_closet_triangle_point(a,p,q,r)

print(c)
print(lac)