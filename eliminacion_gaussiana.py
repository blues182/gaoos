# Universidad de las Américas Puebla | Análisis Numérico (LFA3082)
# Alumno: Alejandro Luis Cisneros Bernal | ID 182246 | Profesora: Ana Lizbeth Cortes Cortes
#
# Método de Eliminación Gaussiana con pivoteo parcial
#
# Resuelve un sistema de n ecuaciones con n incógnitas  A x = b  en dos fases:
#   1) Eliminación hacia adelante: con operaciones de renglón convertimos A en
#      una matriz triangular superior (puros ceros debajo de la diagonal).
#      Las operaciones de renglón no cambian la solución del sistema.
#   2) Sustitución hacia atrás: la última ecuación ya tiene una sola incógnita;
#      la despejamos y vamos subiendo, sustituyendo lo que ya conocemos.
#
# Solo se usan listas de Python. NumPy se usa únicamente al final para
# comprobar que nuestro resultado es correcto.

import numpy as np

# Tolerancia: en la computadora los números decimales tienen errores de redondeo,
# así que un número "cero" puede salir como 1e-17. Todo lo que sea menor que TOL
# en valor absoluto lo tratamos como cero.
TOL = 1e-12


def imprimir_sistema(A, b):
    # Muestra la matriz aumentada [A | b] renglón por renglón.
    for i in range(len(A)):
        fila = ""
        for j in range(len(A[i])):
            fila = fila + "{:10.4f}".format(A[i][j])
        print(fila, "  |", "{:10.4f}".format(b[i]))
    print()


def gauss(A, b, mostrar_pasos=True):
    # Copiamos A y b para no modificar las listas originales de quien llama
    # a la función (así después podemos comparar con NumPy usando los datos originales).
    A = [fila[:] for fila in A]
    b = b[:]
    n = len(A)

    if mostrar_pasos:
        print("Matriz aumentada inicial [A | b]:")
        imprimir_sistema(A, b)

    # ---------------- FASE 1: ELIMINACIÓN HACIA ADELANTE ----------------
    # k es la columna que vamos a "limpiar" (hacer ceros debajo de A[k][k]).
    for k in range(n):

        # Pivoteo parcial: buscamos, de la fila k hacia abajo, la fila cuyo
        # elemento en la columna k sea el más grande en valor absoluto.
        # ¿Por qué? Porque después dividimos entre el pivote A[k][k]:
        #   - si el pivote es 0, la división es imposible;
        #   - si el pivote es muy pequeño, el multiplicador m sale enorme y
        #     los errores de redondeo crecen. Con el pivote más grande, |m| <= 1
        #     y el método es más estable.
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i

        # Si hasta el mayor valor de la columna es (casi) cero, no hay pivote:
        # la matriz es singular (determinante 0) y no existe solución única.
        if abs(A[p][k]) < TOL:
            if k == n - 1 and abs(b[k]) > TOL:
                # La última fila quedó  0 = b[k]  con b[k] distinto de 0: contradicción.
                print("El sistema no tiene solución (queda 0 =", b[k], ")")
            elif k == n - 1:
                # La última fila quedó  0 = 0: sobra una ecuación.
                print("El sistema tiene infinitas soluciones (queda 0 = 0)")
            else:
                print("Matriz singular: el sistema no tiene solución única "
                      "(la columna", k + 1, "no tiene pivote)")
            return None

        # Intercambiamos la fila k con la fila del pivote. Se cambian también
        # los valores de b, porque cada b[i] pertenece a su ecuación.
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            if mostrar_pasos:
                print("Paso", k + 1, "- intercambio fila", k + 1, "<-> fila", p + 1,
                      "(pivote =", round(A[k][k], 4), ")")

        # Hacemos cero cada elemento debajo del pivote.
        for i in range(k + 1, n):
            # El multiplicador m dice cuántas veces la fila k cabe en la fila i
            # para que A[i][k] se vuelva cero:  A[i][k] - m * A[k][k] = 0.
            m = A[i][k] / A[k][k]
            # Fila i = Fila i - m * Fila k. Empezamos en j = k porque las columnas
            # anteriores a k ya son ceros en ambas filas.
            for j in range(k, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]
            if mostrar_pasos:
                print("Paso", k + 1, "- F" + str(i + 1), "= F" + str(i + 1),
                      "- (" + str(round(m, 4)) + ") * F" + str(k + 1))

        # En la última columna no hay nada debajo del pivote, así que no se imprime.
        if mostrar_pasos and k < n - 1:
            print("Matriz después de eliminar la columna", k + 1, ":")
            imprimir_sistema(A, b)

    # ---------------- FASE 2: SUSTITUCIÓN HACIA ATRÁS ----------------
    # Ahora A es triangular superior. Despejamos desde la última incógnita
    # hasta la primera:  x[i] = (b[i] - suma de A[i][j] * x[j], j > i) / A[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):          # i va de n-1 hasta 0 (de abajo hacia arriba)
        suma = 0.0
        for j in range(i + 1, n):           # incógnitas que ya calculamos
            suma = suma + A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]      # A[i][i] no es 0 porque fue un pivote
        if mostrar_pasos:
            print("x" + str(i + 1), "= (" + str(round(b[i], 4)), "- (" + str(round(suma, 4)) + ")) /",
                  "(" + str(round(A[i][i], 4)) + ") =", round(x[i], 6))

    return x


# ======================= EJEMPLO DE USO (3 x 3) =======================
#   2x +  y -  z =   8
#  -3x -  y + 2z = -11
#  -2x +  y + 2z =  -3
A = [[2.0, 1.0, -1.0],
     [-3.0, -1.0, 2.0],
     [-2.0, 1.0, 2.0]]
b = [8.0, -11.0, -3.0]

print("===== Ejemplo 1: sistema 3x3 con solución única =====")
x = gauss(A, b)

if x is not None:
    print("\nSolución por Eliminación Gaussiana:")
    for i in range(len(x)):
        print("  x" + str(i + 1), "=", round(x[i], 6))

    # Comprobación con NumPy: resuelve el mismo sistema con su propia rutina.
    x_numpy = np.linalg.solve(np.array(A), np.array(b))
    print("\nSolución con numpy.linalg.solve:")
    for i in range(len(x_numpy)):
        print("  x" + str(i + 1), "=", round(float(x_numpy[i]), 6))

    # Diferencia más grande entre ambas soluciones (debe ser prácticamente 0).
    diferencia = max(abs(x[i] - x_numpy[i]) for i in range(len(x)))
    print("\nDiferencia máxima entre ambos métodos:", diferencia)
    if diferencia < 1e-9:
        print("Coinciden: el resultado es correcto.")
    else:
        print("No coinciden: revisar el método.")

# ============== EJEMPLO 2: matriz singular (sin solución única) ==============
# La fila 2 es el doble de la fila 1, así que el determinante es 0.
print("\n===== Ejemplo 2: matriz singular =====")
A2 = [[1.0, 2.0, 3.0],
      [2.0, 4.0, 6.0],
      [1.0, 1.0, 1.0]]
b2 = [6.0, 12.0, 3.0]
gauss(A2, b2, mostrar_pasos=False)
