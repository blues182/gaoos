TOL = 1e-12  


def gauss(A, b):
    n = len(A)

    
    for k in range(n):

        
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i

        
        if abs(A[p][k]) < TOL:
            if k == n - 1 and abs(b[k]) > TOL:
                print("El sistema no tiene solucion (queda 0 =", b[k], ")")
            elif k == n - 1:
                print("El sistema tiene infinitas soluciones (queda 0 = 0)")
            else:
                print("El sistema no tiene solucion unica (columna", k + 1, "sin pivote)")
            return None

        
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]

       
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]

    
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma = suma + A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]

    return x



n = 0
while n <= 1:
    n = int(input("Tamano de la matriz (n, para una matriz de n x n): "))
    if n <= 1:
        print("n debe ser mayor que 1, intenta de nuevo")

print("La matriz es de", n, "x", n, "->", n, "ecuaciones con", n, "incognitas")

A = []
b = []
print("Escribe cada ecuacion:", n, "coeficientes y al final el resultado, separados por espacios")
for i in range(n):
    numeros = []
    
    while len(numeros) != n + 1:
        fila = input("Ecuacion " + str(i + 1) + ": ").split()
        numeros = [float(v) for v in fila]
        if len(numeros) != n + 1:
            print("Deben ser", n + 1, "numeros, intenta de nuevo")
    A.append(numeros[:n])   
    b.append(numeros[n])    

x = gauss(A, b)

if x is not None:
    print("Solucion:")
    for i in range(n):
        print("x" + str(i + 1), "=", round(x[i], 6))