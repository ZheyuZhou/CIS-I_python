import numpy as np
import pandas as pd

output = pd.read_csv('2022_pa345_student_data\Output\PA4_'+"H"+'_output.txt')
output = output.to_numpy()[:, 1:]
print(output)
pd.DataFrame(output).to_csv('2022_pa345_student_data\Output\PA4_'+"H"+'_output.txt', sep = ',', index = False, header= False)