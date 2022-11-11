#include<stdio.h>
#include<math.h>
#include<time.h>
#include<string.h>

#define dimension 20
float aumentedMatrix[dimension][dimension * 2];

int i, j, k, temp;

void out_txt(){
    FILE *ofile;
    ofile = fopen("inversa_omp", "w");
    for(i = 0; i < dimension; i++){
        for(j = dimension; j< 2* dimension; j++){
            fprintf(ofile,"%f\t", aumentedMatrix[i][j]);
        }
        fprintf(ofile,"\n");
    }
    fclose(ofile);
}


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

int printMatrix(){
	int x, y;
	for(y = 0; y < dimension; y++){
		printf("\n");
		for(x = 0; x < dimension * 2; x++){
		    printf("%f ",aumentedMatrix[y][x]);
		}
	}
	printf("\n");
}


void gauss_jordan(){
    float temporary, r;

    for(j = 0; j < dimension; j++){
        temp = j;
        for(i = j+1; i < dimension; i++){
            if(aumentedMatrix[i][j]>aumentedMatrix[temp][j])
                temp = i;
        }
        if(temp != j){
            for(k = 0; k < 2*dimension; k++){
                temporary = aumentedMatrix[j][k];
                aumentedMatrix[j][k] = aumentedMatrix[temp][k];
                aumentedMatrix[temp][k] = temporary;
            }
        }
        #pragma omp paralell for
        for(i = 0; i< dimension; i++){
            r = aumentedMatrix[i][j];
            for(k = 0; k < 2* dimension; k++){
                if (i != j){
                    aumentedMatrix[i][k] -= (aumentedMatrix[j][k] / aumentedMatrix[j][j])*r;
                }else{
                    aumentedMatrix[i][k] /= r;
                }
            }
        }
    }
}

int main(){

    char *filename = "matrix/matrix20.txt";
    matrix_read(filename);

    clock_t start, finish;
    double totalTime;
    start = clock();

    gauss_jordan();

    finish = clock();
    totalTime = ((double) (finish - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %f ms\n", totalTime);

    out_txt();

    return 0;

}