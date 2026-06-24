import sys
import os
import numpy as np
from itertools import permutations


##Reutilizo las funciones de Xabier del QAP-Descomposition


def read_QAP(filename):
	with open(filename) as f:
		lines = [line.strip() for line in f if line.strip()]

		n = int(lines[0])

		d_matrix = np.array([
			[float(x) for x in lines[i + 1].split()]
			for i in range(n)
		])

		h_matrix = np.array([
			[float(x) for x in lines[i + 1 + n].split()]
			for i in range(n)
		])
		return(np.array(d_matrix),np.array(h_matrix),n)

# Function that prints a QAP instance in QAPLIB format
def print_QAP(d_matrix,h_matrix,n,filename):
	with open(filename,"+w") as f:
		print(str(n),file=f)
		for i in range(n):
			for j in range(n):
				print(str(round(d_matrix[i][j],15))+" ",end="",file=f)
			print(file=f)
		for i in range(n):
			for j in range(n):
				print(str(round(h_matrix[i][j],15))+" ",end="",file=f)
			print(file=f)


def decompose(h_matrix,n):

	######## (n) component ########
	
	# 0th order (mean)

	in_diagonal = h_matrix[np.eye(n, dtype=bool)]
	off_diagonal = h_matrix[~np.eye(n, dtype=bool)]

	n_matrix = np.full((n,n),off_diagonal.mean())
	np.fill_diagonal(n_matrix, in_diagonal.mean())

	######## (n-1,1) component ########
	
	# 1st order
	
	row_sum = h_matrix.sum(axis=1, keepdims=True)  # shape: [n, 1]
	col_sum = h_matrix.sum(axis=0, keepdims=True)  # shape: [1, n]
	
	in_diagonal_sum = in_diagonal.sum()
	off_diagonal_sum = off_diagonal.sum()
	total_sum = off_diagonal_sum + in_diagonal_sum

	n_1_1_matrix = (
	(n - 1) * row_sum
	+ col_sum.T
	+ (n - 1) * col_sum
	+ row_sum.T
	- 2 * total_sum
	- n*np.diag(h_matrix)[:, None]
	- n*np.diag(h_matrix)[None, :]
	+ 2 * in_diagonal_sum
	) / (n * (n - 2))

	n_1_1_matrix[np.diag_indices(n)] += ((
	n*np.diag(h_matrix)[:, None]
	- in_diagonal_sum
	- row_sum
	- col_sum.T
	+ (2 / n) * total_sum
	) / (n - 2)).reshape(n)
	
	######## (n-2,1,1) component ########
	
	# Symmetric part (2nd order)
	
	n_2_1_1_matrix = (h_matrix-h_matrix.T+(-col_sum+row_sum.T-row_sum+col_sum.T)/n)/2
	
	######## (n-2,2) component ########
	
	# Anti-symmetric part (2nd order)
	
	n_2_2_matrix = h_matrix-(n_matrix+n_1_1_matrix+n_2_1_1_matrix) 

	return n_matrix, n_1_1_matrix, n_2_2_matrix, n_2_1_1_matrix



##  Calcular la varianza de g^λ sobre todas las permutaciones
##  Si n es pequeño (≤8) se hace por fuerza bruta (n!).
##  Si n es >8, se estima con una población elegida, 5000 por defecto.

def eval_qap(d_matrix, h_matrix, sigma):
    n = len(sigma)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += d_matrix[i][j] * h_matrix[sigma[i]][sigma[j]]
    return total


def compute_variance(d_matrix, l_matrix, n, n_samples=5000):
    """
    Calcula la varianza de g^λ(σ) sobre el espacio de búsqueda.
    
    - Si n! ≤ 40320 (n ≤ 8): fuerza bruta exacta.
    - Si n > 8: estimación por muestreo aleatorio con n_samples permutaciones.
    """
    import math
    factorial_n = math.factorial(n)

    if factorial_n <= 40320:
        base = list(range(n))
        values = []
        for perm in permutations(base):
            values.append(eval_qap(d_matrix, l_matrix, perm))
        values = np.array(values)
    else:
        rng = np.random.default_rng(seed=42)
        values = []
        for _ in range(n_samples):
            perm = rng.permutation(n)
            values.append(eval_qap(d_matrix, l_matrix, perm))
        values = np.array(values)

    return np.var(values)


# ─────────────────────────────────────────────
#  Generación de instancias con pesos β
#
#  H_GEN = Σ_λ  sqrt(α_λ) / sqrt(Var[g^λ])  *  L^λ
#
#  donde α_λ = sqrt(β_i / num_subinstancias_de_orden_i)
#
#  Los β son los pesos por orden:
#    β1 = peso de interacciones de primer orden  (sub-instancia n_1_1)
#    β2 = peso de interacciones de segundo orden (sub-instancias n_2_2 y n_2_1_1)
#
#  Restricción: β1 + β2 = 1  (y β1, β2 ≥ 0)
# ─────────────────────────────────────────────

def generate_instance(d_matrix, sub_instances, variances, betas):
    """
    Genera una nueva instancia QAP combinando las sub-instancias con los pesos β.

    Parámetros:
        d_matrix      : matriz de distancias (se mantiene fija)
        sub_instances : dict con claves 'n_1_1', 'n_2_2', 'n_2_1_1'
                        y valores las matrices L^λ correspondientes
        variances     : dict con las mismas claves y los valores Var[g^λ]
        betas         : dict con claves 'beta1', 'beta2'
                        representando el peso de cada orden

    Devuelve:
        h_gen : la nueva matriz de flujos generada
    """
    beta1 = betas['beta1']
    beta2 = betas['beta2']

    assert abs(beta1 + beta2 - 1.0) < 1e-9, "Los betas deben sumar 1"
    assert beta1 >= 0 and beta2 >= 0, "Los betas deben ser no negativos"


    alpha = {
        'n_1_1'  : np.sqrt(beta1),
        'n_2_2'  : np.sqrt(beta2 / 2),
        'n_2_1_1': np.sqrt(beta2 / 2),
    }

    h_gen = np.zeros_like(sub_instances['n_1_1'])

    for key in ['n_1_1', 'n_2_2', 'n_2_1_1']:
        var = variances[key]
        if var < 1e-12:
            print(f"Var[g^{key}] ≈ 0, se omite esta componente, no aporta.")
            continue
        coef = alpha[key] / np.sqrt(var)
        h_gen += coef * sub_instances[key]

    return h_gen




def main():
    print("=" * 70)
    print("  Generador de instancias QAP basado en descomposición de Fourier")
    print("=" * 70)
    print()

  
    while True:
        infile = input("Ruta a la instancia QAP fuente (formato QAPLIB): ").strip()
        print()
        try:
            d_matrix, h_matrix, n = read_QAP(infile)
            print(f"  Instancia leída: n={n}")
            break
        except Exception as e:
            print(f"  Error al leer la instancia: {e}. Inténtalo de nuevo.")
            print()


    print("\nDescomponiendo la instancia...")
    n_matrix, n_1_1_matrix, n_2_2_matrix, n_2_1_1_matrix = decompose(h_matrix, n)
    print("  Descomposición completada.")

    sub_instances = {
        'n_1_1'  : n_1_1_matrix,
        'n_2_2'  : n_2_2_matrix,
        'n_2_1_1': n_2_1_1_matrix,
    }

    print("\nCalculando varianzas de las sub-instancias...")
    print(f"  (n={n} → {'fuerza bruta exacta' if n <= 8 else 'estimación por muestreo'})")
    variances = {}
    for key, l_matrix in sub_instances.items():
        var = compute_variance(d_matrix, l_matrix, n)
        variances[key] = var
        print(f"  Var[g^{key}] = {var:.6f}")

    print()
    print("Define los pesos β (deben sumar 1):")
    print("  β1 = peso de interacciones de primer orden  (componente n_1_1)")
    print("  β2 = peso de interacciones de segundo orden (componentes n_2_2 y n_2_1_1)")
    print()

    while True:
        try:
            beta1 = float(input("  β1 (entre 0 y 1): ").strip())
            beta2 = float(input("  β2 (entre 0 y 1): ").strip())
            if abs(beta1 + beta2 - 1.0) > 1e-6:
                print("  Error: β1 + β2 debe ser igual a 1. Inténtalo de nuevo.")
                print()
                continue
            if beta1 < 0 or beta2 < 0:
                print("  Error: los betas deben ser no negativos.")
                print()
                continue
            break
        except ValueError:
            print("  Error: introduce un número válido.")
            print()

    betas = {'beta1': beta1, 'beta2': beta2}


    print("\nGenerando nueva instancia...")
    h_gen = generate_instance(d_matrix, sub_instances, variances, betas)
    print("  Instancia generada.")


    while True:
        outfile = input("\nRuta donde guardar la instancia generada (ej: output/gen.dat): ").strip()
        try:
            os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
            print_QAP(d_matrix, h_gen, n, outfile)
            print(f"\n  ¡Hecho! Instancia guardada en: {outfile}")
            break
        except Exception as e:
            print(f"  Error al guardar: {e}. Inténtalo de nuevo.")




if __name__ == "__main__":
    main()
