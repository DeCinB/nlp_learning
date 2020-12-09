import numpy as np 

a=np.zeros((5,3))
b=np.ones((2,3))
indices=[2,4]
a[indices]=b
print(a)