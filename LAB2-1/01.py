from numpy import arange
M = 20
a = arange(1, M+1)

ScalP = 0
for j in range(M):
    ScalP += a[j]*a[j]
print('ScalP =', ScalP)
  
