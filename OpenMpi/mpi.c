#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define dimension 10
float aumentedMatrix[dimension][dimension * 2];

void matrix_read(char *filename){
    FILE *fp;
    int row, col;

    fp = fopen(filename, "r");
    if(fp == NULL)
        return;

    for(row = 0; row < dimension; row++){
        for(col = 0; col < 2*dimension; col++){
            if (col < dimension){
                if(fscanf(fp, "%f,", &aumentedMatrix[row][col]) == EOF) 
                    break;
            }else{
                if(row == col%dimension){
                    aumentedMatrix[row][col] = 1;
                }else{
                    aumentedMatrix[row][col] = 0;
                }
            }
        }
        if(feof(fp)) break;
    }

    fclose(fp);
}

void print_matrix(float m[dimension][2*dimension], int n, int opc)
{
  int i, j = 0;
  for (i=0; i<n; i++) {
    printf("\n\t| ");
    for (j=0; j<n; j++){
      if(opc == 0)
        printf("%2f ", m[i][j]); 
      else
        printf("%2f ", m[i][j+n]);
    }
    printf("|");
  }
  printf("\n");
}

int main (int argc, char *argv[])
{
       int i, j, k, tag=1, tasks, iam, n, a, tot;
       float aux[dimension/2][dimension * 2], pivote[dimension*2], r; // poner sobre 10 para el real
       MPI_Status status;
       clock_t start, finish;
       double totalTime;
 
      char *filename = "../matrices/matix10.txt";
      matrix_read(filename);
 
      /* Initialize the message passing system, get the number of nodes,
          and find out which node we are. */
       MPI_Init(&argc, &argv);
       MPI_Comm_size(MPI_COMM_WORLD, &tasks);
       MPI_Comm_rank(MPI_COMM_WORLD, &iam);
 
      n = (dimension*dimension*2)/(tasks);
      tot = dimension/tasks;
 
      start = clock();
 
      for(k = 0; k < dimension; k++){ // Recorrer las filas por pivote
        MPI_Scatter (aumentedMatrix, n, MPI_FLOAT, aux, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(aumentedMatrix, dimension*dimension*2, MPI_FLOAT, 0, MPI_COMM_WORLD);
          
        for(i = 0; i < tot; i++){
          r = aux[i][k];
          for(j = 0; j < 2* dimension; j++){
              if ((i+iam*tot) != k){
                  aux[i][j] -= (aumentedMatrix[k][j] / aumentedMatrix[k][k])*r;
              }else{
                  aux[i][j] /= r;
              }
          }
        }

        MPI_Gather (aux, n, MPI_FLOAT,  aumentedMatrix, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
 
      finish = clock();
 
      if(iam == 0){
          totalTime = ((double) (finish - start)) / CLOCKS_PER_SEC * 1000;
          printf("Time: %f ms\n\n", totalTime);
        }
       
       /* Shut down the message passing system. */
       MPI_Finalize();
}