# Generador de Instancias QAP basado en Descomposición de Fourier

**Proyecto basado en el artículo:**  
*Fourier Transform-based instance decomposition for k-adic Assignment Problems* (IEEE TEVC, 2026)

---

## 📌 Descripción

Este repositorio contiene un **generador de instancias del Quadratic Assignment Problem (QAP)** que utiliza la **descomposición basada en Transformada de Fourier** propuesta en el paper de Benavides et al.

A partir de una instancia fuente, el programa:
1. Descompone la matriz de flujos (`H`) en sus componentes espectrales (órdenes 0, 1 y 2).
2. Calcula la varianza de cada componente.
3. Genera **nuevas instancias** combinando linealmente estas componentes con pesos controlados (`β₁` y `β₂`).

Esto permite crear instancias con **dificultad tunable** de forma sistemática, controlando el peso de las interacciones de primer y segundo orden.

---

## ✨ Características

- Descomposición exacta para QAP (k=2) siguiendo el marco del paper.
- Cálculo exacto de varianzas para `n ≤ 8` (fuerza bruta) y estimación precisa para `n > 8`.
- Interfaz por consola sencilla e intuitiva.
- Salida en formato **QAPLIB** (compatible con la mayoría de solvers y benchmarks).
- Reutiliza y adapta código de descomposición de Xabier Benavides.

---

## 📁 Archivos del repositorio

| Archivo                    | Descripción |
|---------------------------|-----------|
| `generator.py`            | Script principal del generador |
| `mi_instancia.dat`        | Instancia fuente de ejemplo (n=5) |
| `instancia_generada.dat`  | Ejemplo de instancia generada |

---

## 🚀 Cómo usar

### 1. Requisitos

```bash
pip install numpy
