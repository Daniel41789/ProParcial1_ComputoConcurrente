#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__device__ float determinanteSubmatriz(float *matriz, int fila, int columna, int N){
    return (matriz[(fila + 1) % N*N + (columna + 1) % N] * matriz[(fila + 2) % N * N + (columna + 2) % N] -
           matriz[(fila + 1) % N*N + (columna + 2) % N] * matriz[(fila + 2) % N * N + (columna + 1) % N]);
}

__global__ void determinante(float *matriz, float *det, int N){
    for(int j = 0; j < N; j++)
        det[0] += matriz[j] * determinanteSubmatriz(matriz, 0, j, N);
}

__global__ void matrizAdjunta(float *matriz, float *adjunta, int N){
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if(idx < N ){
        for(int i=0; i < N; i++){
            adjunta[idx * N + i] = determinanteSubmatriz(matriz, idx, i, N);
            if((idx + i) % 2 == 1)
                adjunta[idx * N + i] = -adjunta[idx * N + i];
        }   
    }
}

__global__ void matrizInversa(float *matriz, float *inversa, float *det, int N){
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N && columna < N) {
        int idx = fila * N + columna;
        inversa[idx] = matriz[idx] / det[0];
    }
}

int main(int argc, char *argv[]){
    int N = atoi(argv[1]);
    float * matrizE_h, *matrizInversa_h, *d_matriz, *d_matrizAdjunta, *d_determinante, *d_inversa, det;
    srand(time(NULL));


    //Reserva de memoria para la matriz de entrada y salida en el host
    matrizE_h=(float*)malloc(N*N*sizeof(float));
    matrizInversa_h=(float*)malloc(N*N*sizeof(float));

    for(int i = 0; i < N * N; i++)
        matrizE_h[i] = rand() % 100; 

    cudaMalloc(&d_matriz, N*N*sizeof(float));
    cudaMalloc(&d_matrizAdjunta, N*N*sizeof(float));
    cudaMalloc(&d_determinante, N*sizeof(float));
    cudaMalloc(&d_inversa, N*N*sizeof(float));

    cudaMemcpy(d_matriz, matrizE_h, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int tamBloque = prop.maxThreadsPerBlock;
    int numBloques = (N+tamBloque-1)/tamBloque;

    dim3 tamanoBloque(numBloques, numBloques);//
	dim3 tamanoMalla((N + numBloques - 1) / numBloques, (N + numBloques - 1) / numBloques);

    matrizAdjunta<<<numBloques, tamanoBloque>>>(d_matriz, d_matrizAdjunta, N);
    determinante<<<1,1>>>(d_matriz, d_determinante, N);
    matrizInversa<<<tamanoMalla, tamanoBloque>>>(d_matrizAdjunta, d_inversa, d_determinante, N);

    cudaMemcpy(matrizInversa_h, d_inversa, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matriz Inversa:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%.2f\t", matrizInversa_h[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }
    cudaDeviceSynchronize();

    free(matrizE_h);
    free(matrizInversa_h);
    cudaFree(d_matriz);
    cudaFree(d_matrizAdjunta);
    cudaFree(d_determinante);
    cudaFree(d_inversa);

}
