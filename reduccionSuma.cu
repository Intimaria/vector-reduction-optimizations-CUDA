/* 
 * Trabajo para la catedra:
 * Taller De Programación Sobre GPUs (General Purpose Computation on Graphics Processing Unit)
 * UNLP. Facultad de Informatica, profesores 
 * Adrián Pousa,
 * Victoria Sanz. 
 * 
 * Copyright 2021 Inti Tidball <intimaria@alu.ing.unlp.edu.ar>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>

// Tipo de los elementos en los vectores
// Compilar con -D_INT_ para vectores de tipo entero
// Compilar con -D_DOUBLE_ para vectores de tipo double
// Predeterminado vectores de tipo float

#ifdef _INT_
typedef int basetype;     // Tipo para elementos: int
#define labelelem    "ints"
#define PRINT "%c"
#elif _DOUBLE_
typedef double basetype;  // Tipo para elementos: double
#define labelelem    "doubles"
#define PRINT "%lf"
#else
typedef float basetype;   // Tipo para elementos: float     PREDETERMINADO
#define labelelem    "floats"
#define PRINT "%f"
#endif

const int N = 1048576;    // Número predeterminado de elementos en los vectores

const int CUDA_BLK = 64;  // Tamaño predeterminado de bloque de hilos CUDA


/* 
   Para medir el tiempo transcurrido (elapsed time):

   resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
   timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar

   timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido

   printtime: abstrae función usada para imprimir el tiempo transcurrido

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener 
   el tiempo transcurrido entre dos medidas
*/

#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


/*
  Función para inicializar el vector que vamos a utilizar
*/
void init_CPU_array(basetype array[], const unsigned int n)
{
  unsigned int i;
  for(i = 0; i < n; i++) {
    array[i] = (basetype)1;
  }
}


//  Función secuencial: reduccion para CPU (*r* veces)
 basetype reduccion_CPU(basetype arrayV[], const unsigned int n) {
 unsigned int i;
 basetype suma = 0;
 for(i = 0; i < n; i++) {
     suma += arrayV[i];
 }
 return suma;
}


__global__ void suma_kernel_cuda(basetype *const global_data, const int n, const int distancia){
  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id < distancia){
    global_data[global_id] = global_data[global_id] + global_data[global_id + distancia];
  }
}

//  Función para reducir vectores en la GPU
void reduccion_GPU( basetype arrayV[], const unsigned int n, const unsigned int blk_size, basetype * res){
double timetick;

  // Número de bytes de cada uno de nuestros vectores
  unsigned int numBytes = n * sizeof(basetype);

  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  basetype *cV, *cV2;


  cudaMalloc((void **) &cV, numBytes);
  cudaMalloc((void **) &cV2, numBytes);
  printf("Tiempo en GPU:\n"); 
  timetick = dwalltime();
  cudaMemcpy(cV, arrayV, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  printf("%f\n", dwalltime() - timetick);

  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);

  // Grid unidimensional (*ceil(n/blk_size)* bloques)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

  // Lanzamos ejecución del kernel en la GPU
  //timestamp(start);            // Medimos tiempo de cálculo en GPU
  timetick = dwalltime();
  int distancia = n/2;
  unsigned int i;
  unsigned int hasta = ceil(log(n)/log(2));
  for (i=0; i < hasta; i++) {
    suma_kernel_cuda<<<dimGrid, dimBlock>>>(cV, n, distancia);
    cudaDeviceSynchronize();
    distancia = ceil(distancia / 2);
  }
  printf("%f\n", dwalltime() - timetick);
  //timestamp(end);

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  basetype resultado;
  cudaMemcpy(&resultado, cV, sizeof(basetype), cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("%f\n", dwalltime() - timetick);
  printf("resultado GPU: %f\n", resultado);
  // Liberamos memoria global del device utilizada
  cudaFree (cV);


}


// Declaración de función para comprobar y ajustar los parámetros de
// ejecución del kernel a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *cb);

int main(int argc, char *argv[]){

double timetick;

  // Aceptamos algunos parámetros desde línea de comandos

  // Número de elementos del vector (predeterminado: N 1048576)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;

  // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK 64)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;


  checkparams(&n, &cb);

  // Número de bytes a reservar para nuestro vector
  unsigned int numBytes = n * sizeof(basetype);

  // Reservamos e inicializamos el vector en CPU

  basetype *vectorV = (basetype *) malloc(numBytes); // Vector con datos de entrada
  basetype *resultadoGPU = (basetype *) malloc(sizeof(basetype)); // resultado GPU
  init_CPU_array(vectorV, n);
  printf("Tiempo en CPU:\n");
  // Ejecutamos reduccion en CPU
  timetick = dwalltime();
  basetype suma = 0;
  if (n > pow(2,24)) {
    unsigned int dimL = n/32;
    for (unsigned int i = 0; i < 32; i ++) {
      suma += reduccion_CPU(vectorV+(dimL*i), dimL);
    }
  }
  else 
    suma = reduccion_CPU(vectorV,n);
  printf("%f\n", dwalltime() - timetick);
  printf("resultado CPU: %f\n", suma);

  //Inicializa nuevamente el vector para realizar la ejecucion en GPU
  init_CPU_array(vectorV, n);
  
  // Ejecutamos reduccion en GPU
  reduccion_GPU(vectorV, n,  cb, resultadoGPU);

  free(resultadoGPU);
  free(vectorV);

  return(0);
}

//  Función que ajusta el número de hilos, de bloques, y de bloques por hilo 
//  de acuerdo a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *cb){
  struct cudaDeviceProp capabilities;

  // Si menos numero total de hilos que tamaño bloque, reducimos bloque
  if (*cb > *n)
    *cb = *n;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n\n", 
	   *cb);
  }

  if (((*n + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (*n - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n", 
	     *cb);
      if (*n > (capabilities.maxGridSize[0] * *cb)) {
	*n = capabilities.maxGridSize[0] * *cb;
	printf("->Núm. total de hilos cambiado a %d (máx por grid para \
dev)\n\n", *n);
      } else {
	printf("\n");
      }
    } else {
      printf("->Núm. hilos/bloq cambiado a %d (%d máx. bloq/grid para \
dev)\n\n", 
	     *cb, capabilities.maxGridSize[0]);
    }
  }
}
