# Reducción de un vector en CUDA — optimizaciones

Implementación y optimización de la **suma de los elementos de un arreglo de `float`** en CUDA.
El repo compara una versión base (`reduccionSuma.cu`) contra una optimizada
(`reduccionSuma_Optimizada.cu`).

Trabajo del **Taller de Programación en GPU** (UNLP), 2021 — aprobado con 10.

## Versión base
Reducción *interleaved* sobre memoria global: en cada iteración, la mitad de los hilos suma su
posición con `global_id + distancia`, y el arreglo se reduce a la mitad. Se lanza el kernel
`log2(n)` veces con `cudaDeviceSynchronize()` como barrera. Es coalescente, pero **la mitad de
los hilos queda ociosa** en cada iteración.

## Optimizaciones aplicadas
- **Memoria compartida**: carga coalescente de global a shared y reducción dentro del bloque.
- **Primera reducción en la carga**: cada hilo suma dos elementos al copiar
  (`global_data[id] + global_data[id + blockDim.x]`), reduciendo a la mitad los bloques y
  evitando hilos ociosos en la primera iteración.
- **Loop reverso** (`distancia >>= 1`): evita divergencia por warp y conflictos de banco, y
  garantiza *sequential addressing*.
- **Unrolling del último warp** (32 hilos): elimina el `for` y su overhead; dentro del warp no
  hace falta `__syncthreads()`.
- **Escritura coalescente**: cada bloque escribe en `global_data[blockIdx.x]`; el grid se reduce
  en cada iteración.
- **Cierre en CPU**: las últimas `N ≤ blockDim` posiciones se suman en CPU (la GPU rinde con N
  grande, la CPU con N chico).

## Resultados
Medido en una **NVIDIA RTX 2070 (Turing)**, hasta N = 2²⁹ (536.870.912 floats).

| Comparación | Speedup |
|---|---|
| GPU optimizado vs. GPU base (con transferencias) | ~1.13× |
| CPU vs. GPU base (con transferencias) | ~2.1× |
| CPU vs. GPU optimizado (con transferencias) | ~2.4× |
| CPU vs. GPU optimizado (**solo kernel**, sin H2D/D2H) | hasta **~119×** |

El costo dominante es la transferencia **H2D**; el informe propone mejorarla con *pinned
memory* y *streams*.

## Compilar y correr
```bash
nvcc -O2 reduccionSuma.cu -o reduccion
nvcc -O2 reduccionSuma_Optimizada.cu -o reduccion_opt
```

## Informe completo
📄 [Optimización de reducción de un vector en CUDA (2021)](GPU%20-%20optimizaciones%20en%20reduccion%20de%20un%20vector%20-%20entrega%202021%20-%20Tidball.pdf)

---

## English

Implementation and optimization of the **sum of an array of `float`** in CUDA, comparing a
baseline kernel (`reduccionSuma.cu`) against an optimized one (`reduccionSuma_Optimizada.cu`).
Final project for the **GPU Programming Workshop** (UNLP, 2021), graded 10/10.

**Baseline:** interleaved reduction in global memory, launched `log2(n)` times with a
`cudaDeviceSynchronize()` barrier; half the threads sit idle each iteration.

**Optimizations:** shared memory with coalesced loads; first reduction during the load (each
thread adds two elements, halving the grid); reverse loop with `>>= 1` to avoid warp divergence
and bank conflicts; last-warp unrolling; coalesced write-back; shrinking grid; tail reduced on
CPU.

**Results** on an **NVIDIA RTX 2070 (Turing)**, up to N = 2²⁹:
- Optimized vs. baseline GPU (incl. transfers): **~1.13×**
- CPU vs. optimized GPU (incl. transfers): **~2.4×**
- CPU vs. optimized GPU (**kernel only**): up to **~119×**

The dominant cost is the host→device transfer; the report suggests pinned memory and streams.

📄 [Full report (Spanish)](GPU%20-%20optimizaciones%20en%20reduccion%20de%20un%20vector%20-%20entrega%202021%20-%20Tidball.pdf)
