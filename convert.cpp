#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

int main(int argc, char** argv) {
    FILE *fi;
    FILE *fo;
    fi = fopen("input_matrix", "r");
    fo = fopen("input_matrix.bin", "wb");
    int size = atoi(argv[1]);
    double row;
    for (int i = 0; i < size*(size+1); i++) {
        fscanf(fi, "%lf", &row);
        fwrite (&row, sizeof(double), 1, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}