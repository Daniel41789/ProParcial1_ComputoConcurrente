#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void imprimirMatriz(float **matriz, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%.2f\t", matriz[i][j]);
        printf("\n");
    }
}

float determinanteSubmatriz(float **matriz, int fila, int columna, int N){
    return matriz[(fila + 1) % N][(columna + 1) % N] * matriz[(fila + 2) % N][(columna + 2) % N] -
           matriz[(fila + 1) % N][(columna + 2) % N] * matriz[(fila + 2) % N][(columna + 1) % N];
}

void matrizAdjunta(float **matriz, float **adjunta, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            adjunta[j][i] = determinanteSubmatriz(matriz, i, j, N);
            if((i + j) % 2 == 1)
                adjunta[j][i] = -adjunta[j][i];  
        }
    }
}

float determinante(float **matriz, int N){
    float det = 0;
    for(int j = 0; j < N; j++)
        det += matriz[0][j] * determinanteSubmatriz(matriz, 0, j, N);
    return det;
}

void matrizInversa(float **matriz, float **inversa, int N){
    float **adjunta = malloc(N * sizeof(float *));
    for(int i = 0; i < N; i++)
        adjunta[i] = malloc(N * sizeof(float));
    
    float det = determinante(matriz, N);
    if(det == 0){
        printf("La matriz no es invertible.\n");
        return;
    }

    matrizAdjunta(matriz, adjunta, N);

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            inversa[i][j] = adjunta[i][j]/det;

    for(int i = 0; i < N; i++)
        free(adjunta[i]);
    
    free(adjunta);
}

int main(int argc, char *argv[]){
    int N = atoi(argv[1]);
    srand(time(NULL));

    float **matriz = malloc(N * sizeof(float *));
    for(int i = 0; i < N; i++)
        matriz[i] = malloc(N * sizeof(float));
    

    float **inversa = malloc(N * sizeof(float *));
    for(int i = 0; i < N; i++)
        inversa[i] = malloc(N * sizeof(float));
    

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            matriz[i][j] = rand() % 100; 
        
    printf("Matriz Original:\n");
    imprimirMatriz(matriz, N);
    
    matrizInversa(matriz, inversa, N);

 
    printf("\nMatriz Inversa:\n");
    imprimirMatriz(inversa, N);

    for(int i = 0; i < N; i++){
        free(matriz[i]);
        free(inversa[i]);
    }
    free(matriz);
    free(inversa);

    return 0;
}