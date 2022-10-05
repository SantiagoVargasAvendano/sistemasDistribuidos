#include <stdio.h>   
#include <stdlib.h>  
#include <math.h>

#define SIZE 1

int determinante(int matriz[][SIZE], int orden);
int cofactor(int matriz[][SIZE], int orden, int fila, int columna);
void printMatrix(int matrix[][SIZE]);
void printMatrixf(float matrix[][SIZE]);

int main(){
    int matrix[SIZE][SIZE], adj[SIZE][SIZE], cofaux[SIZE][SIZE], aux_cof = 1, det = 0;
    float inv[SIZE][SIZE];

    printf("Ingrese los elementos de la matriz:\n");
    for (int x=0;x<SIZE;x++)
        for (int i=0;i<SIZE;i++)
            //scanf("%d",&matrix[x][i]);
            matrix[x][i] = rand()%1000;
    
    printMatrix(matrix);

    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            cofaux[i][j]= cofactor(matrix, SIZE, i, j);
        }
    }
    
    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            adj[i][j]= cofaux[j][i];
        }
    }
    printf("La matriz adjunta es :\n");
    printMatrix(adj);

    det = determinante(matrix, SIZE);
    printf("\nEl determinante es: %d \n", det);

    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            inv[i][j] = (1.0/det)*adj[i][j];
        }
    }

    printf("\nLa matriz inversa es: \n");
    printMatrixf(inv);
    
}

void printMatrix(int matrix[][SIZE]){
    for (int i=0;i<SIZE;i++){
        printf("| ");
        for (int j=0;j<SIZE;j++)
            printf("%d | ", matrix[i][j]);
        printf("\n");
    }
}

void printMatrixf(float matrix[][SIZE]){
    for (int i=0;i<SIZE;i++){
        printf("| ");
        for (int j=0;j<SIZE;j++)
            printf("%f | ", matrix[i][j]);
        printf("\n");
    }
}

int determinante(int matriz[][SIZE], int orden){
   int det = 0.0;
   
   if (orden == 1) {
      det = matriz[0][0];
   } else {
      for (int j = 0; j < orden; j++) {
         det = det + matriz[0][j] * cofactor(matriz, orden, 0, j);
      }
   }
   return det;
}

int cofactor(int matriz[][SIZE], int orden, int fila, int columna){
    int submatriz[SIZE][SIZE];
    int n = orden - 1, x = 0, y = 0;

    for (int i = 0; i < orden; i++) {
        for (int j = 0; j < orden; j++) {
            if (i != fila && j != columna) {
                submatriz[x][y] = matriz[i][j];
                y++;
                if (y >= n) {
                    x++;
                    y = 0;
                }
            }
        }
    }
    int aux = -1;
    if((fila+columna)%2==0){
        aux = 1;
    }
    return aux * determinante(submatriz, n);
}