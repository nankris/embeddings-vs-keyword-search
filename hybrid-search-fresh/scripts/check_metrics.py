"""Hand-check metric conventions, RRF weighting and deterministic ties."""
import numpy as np
from run import metrics,fuse,rank
x={'docids':['a','b','c','d'],'qids':['q'],'qrels':{'q':{'a':2,'c':1}}}
z=metrics(x,np.array([[1,2,0,3]]))[0]
expected=(1/np.log2(3)+2/np.log2(4))/(2+1/np.log2(3))
assert abs(z[0]-expected)<1e-12 and z[1]==1
assert rank(np.array([[1.,1.,0.]]),2).tolist()==[[0,1]]
y=fuse(np.array([[0,1,2]]),np.array([[2,1,0]]),10,.8)
assert y[0,0]==2
print('PASS: hand-calculated graded nDCG, recall, stable ties and weighted RRF')
