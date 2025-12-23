import numpy as np


with open('C:\\Users\\Parvus\\Desktop\\щеголев\\lr\\matr.dat', 'r', encoding='utf-8') as f1:
    N = int(f1.readline())
    M = int(f1.readline())


A = np.empty((M, N), dtype=float)
x = np.empty(N, dtype=float)
b = np.empty(M, dtype=float)


with open('C:\\Users\\Parvus\\Desktop\\щеголев\\lr\\amatr.dat', 'r', encoding='utf-8') as f2:
    for j in range(M):
        for i in range(N):
            A[j, i] = float(f2.readline())

with open('C:\\Users\\Parvus\\Desktop\\щеголев\\lr\\vektor.dat', 'r', encoding='utf-8') as f3:
    for i in range(N):
        x[i] = float(f3.readline())


b = A.dot(x)

with open('res.dat', 'w', encoding='utf-8') as f4:
    for val in b:
        print(val, file=f4)