import numpy as np
# Find closet triangle point
def Find_closet_triangle_point(a,p,q,r):
    k = np.array([[q[0]-p[0],r[0]-p[0]],[q[1]-p[1],r[1]-p[1]],[q[2]-p[2],r[2]-p[2]]])
    a = np.reshape(a[0:3], (1,3))[0]
    # D. Gupta, Numpy.reshape#, Numpy.reshape - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.reshape.html (accessed October 13, 2022). 
    ld = np.linalg.lstsq(k,a - p, rcond=None)[0][0]
    # I. Polat, Numpy.linalg.lstsq#, Numpy.linalg.lstsq - NumPy v1.23 Manual. (2022). https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html (accessed October 13, 2022). 
    u = np.linalg.lstsq(k,a - p, rcond=None)[0][1]
    c = p + ld * (q-p) + u * (r-p)
    # Discussion of the point out of the triangle
    if ld>=0 and u>=0 and ld+u<=1:
        cnew = c
    elif ld<0:
        cnew = ProjectOnSegment(c,r,p)
    elif u<0:
        cnew = ProjectOnSegment(c,p,q)
    elif ld + u >1:
        cnew = ProjectOnSegment(c,q,r)

    # l_ac = np.linalg.norm(a-cnew)
    # return cnew, l_ac
    return cnew

# Find the projection of the vector to the plane
def ProjectOnSegment(c,p,q):
    ld = (c-p).dot(q-p)/(q-p).dot(q-p)
    lds = max(0,min(ld,1))
    cs = p+lds*(q-p)
    return cs