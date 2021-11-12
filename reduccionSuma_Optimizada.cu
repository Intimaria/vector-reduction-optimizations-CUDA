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
basetype reduccion_CPU(basetype arrayV[], const unsigned int n, const int gpu) {
 basetype suma = 0;
 for(unsigned int i = 0; i < n; i++) {
     suma += arrayV[i];
 }
 return suma;
}


__device__ void unrolling_warp(volatile basetype* shared_data, int id) {
  shared_data[id] += shared_data[id + 32];
  shared_data[id] += shared_data[id + 16];
  shared_data[id] += shared_data[id + 8];
  shared_data[id] += shared_data[id + 4];
  shared_data[id] += shared_data[id + 2];
  shared_data[id] += shared_data[id + 1];
  }

//  Definición de nuestro kernel para función reduccion
__global__ void reduccion_kernel_cuda(basetype *const global_data, unsigned int n){
  extern __shared__ basetype shared_data[];  
// op aritmentica para "reducir a la mitad" la cantidad de bloques
  unsigned long int global_id = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  shared_data[threadIdx.x] = 0;
 if (global_id < n){
// si global_id < la dim de n actual agarro valor en mi pos 
// sumo en shared ese valor & valor en global_id + dim bloque
// hago una 1er reduccion & reduzco cant de hilos ociosos en 1er iteracion
    shared_data[threadIdx.x] = global_data[global_id] + global_data[global_id + blockDim.x];
}
 __syncthreads();
 // usar loop reverso e indexacion con el id del hilo para sequential addressing
 // desde mitad del tam de bloque, se usa el bitwise operator >> sobre dist 
 // para correr sus bits a la derecha, reduciendo el valor de dist en potencia de 2
 // es equivalente a una funcion log2 pero mas eficiente pal kernel
 // hasta 32 para desenrollar el ultimo warp por fuera del loop
 for (unsigned int distancia = blockDim.x/2; distancia > 32; distancia >>= 1){
 // si el thread se encuentra en la mitad inferior del bloque
  if (threadIdx.x < distancia){
 // agarra lo que esta a su id + distancia y se lo suma a si mismo
    shared_data[threadIdx.x] += shared_data[threadIdx.x + distancia];
  }
  __syncthreads();
 }

 // unrolling se usa la palabra clave volatile para asegurar que no se aplican optimizaciones 
 // por complilador que podrian romper la suma. 
 // ya que todos los hilos de un warp esan sincronizados no es necesario __syncthreads
 if (threadIdx.x < 32) unrolling_warp(shared_data, threadIdx.x);

 // guardo el valor que esta en la primera posicion (suma por reduccion) de mem compartida
 // en la posicion equivalente al numero de mi bloque, pero en memoria global, 
 // asi garantizando que los valores queden contiguos en el arreglo en la memoria global
 if (threadIdx.x == 0){
  global_data[blockIdx.x] = shared_data[0];
  }
}

//  Función para reducir vectores en la GPU
void reduccion_GPU( basetype arrayV[], const unsigned int n, const unsigned int blk_size){
double timetick;

  // Número de bytes de cada uno de nuestros vectores
  unsigned int numBytes = n * sizeof(basetype);


  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  basetype *cV;
  

  cudaMalloc((void **) &cV, numBytes);
  printf("Tiempo de GPU\n");
  timetick = dwalltime();
  cudaMemcpy(cV, arrayV, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  printf("%f\n", dwalltime() - timetick);
  
  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);
  


  // Lanzamos ejecución del kernel en la GPU
  //timestamp(start);            
  // Medimos tiempo de cálculo en GPU
  timetick = dwalltime();
  unsigned int numblockBytes = blk_size * sizeof(basetype);

  // min es el punto en el cual corto la ejecucion - cuando los elementos fueron reducidos
  // hasta llenar un solo bloque o menos 
  unsigned int min = dimBlock.x; 
  unsigned int i; // i representa inicialmente n, el tamano del vector en mem global
  // se divide por tam de bloque para darnos dimGrid - numero de bloques
  // hasta el punto que ya no es mayor que el tam de un solo bloque que se puede 
  // reducir en CPU
  for (i = n; i > min; i = i/dimBlock.x) {
  // se reasigna el valor de numero de bloques en cada iteracion con un nuevo i
  // 4194304 / 512 = 8192 ... 8192 / 512 = 16 .. 16 es menor que 512, se corta la iteracion
  dim3 dimGrid((i + dimBlock.x - 1) / dimBlock.x); 
  // En i se envia la dimension actual del arreglo al kernel
  reduccion_kernel_cuda<<<dimGrid, dimBlock, numblockBytes>>>(cV, i);
  cudaDeviceSynchronize();
}
  printf("%f\n", dwalltime() - timetick);
  //timestamp(end);

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  basetype * resultado = (basetype *) malloc(i*sizeof(basetype));
  cudaMemcpy(resultado, cV, i*sizeof(basetype), cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("%f\n", dwalltime() - timetick);

  // los ultimos i elementos se reducen en CPU
  basetype suma = reduccion_CPU(resultado, i, 1);
  printf("resultado en GPU: %f\n", suma);

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
  init_CPU_array(vectorV, n);

  printf("Tiempo en CPU: \n");
  // Ejecutamos reduccion en CPU para GPU2070
  timetick = dwalltime();
  basetype suma = 0;
  if (n > pow(2,24)) {
    unsigned int dimL = n/32;
    for (unsigned int i = 0; i < 32; i ++) {
      suma += reduccion_CPU(vectorV+(dimL*i), dimL, 0);
    }
  }
  else 
    suma = reduccion_CPU(vectorV,n, 0);
  printf("%f\n", dwalltime() - timetick);
  printf("-> resultado en CPU: %f\n", suma);

  //Inicializa nuevamente el vector para realizar la ejecucion en GPU
  init_CPU_array(vectorV, n);
  
  // Ejecutamos reduccion en GPU
  reduccion_GPU(vectorV, n,  cb);

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
