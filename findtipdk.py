import numpy as np
# find the tip related to B
def findTip(F_Ak, F_Bk, tipA):
    tipA4 = np.append(tipA, np.array([1]))
    # D. Gupta, Numpy.append#, Numpy.append - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.mean.html (accessed November 12, 2022). 
    tipA4 = np.reshape(tipA4, (4,1))
    # D. Gupta, Numpy.reshape#, Numpy.reshape - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.reshape.html (accessed October 13, 2022). 
    d_k = []
    for i in range(len(F_Ak)):
        F_aki = F_Ak[i]
        F_bki = F_Bk[i]
        d_k.append(np.linalg.inv(F_bki)@F_aki@tipA4)
        # I. Polat, Numpy.linalg.inv#, Numpy.linalg.inv - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/gene
    d_k = np.array(d_k)
    return d_k