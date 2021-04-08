#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>

using namespace std;

class Matrix {
    private:
    Matrix() {};
    public:
    vector<vector<double>> data;
    Matrix(int rows, int cols): data(rows, vector<double>(cols, 0)) {};
    Matrix(char i, int size_i, int size_global): data(size_global, vector<double>(size_global, 0)) {
        for (int si = 0; si < size_i; si++) {
            data[si][si] = 1;
        }
    };
    Matrix(vector<vector<double>> mat): data(mat) {};
    Matrix(const Matrix &mat_in) {
        data = mat_in.data;
    };
    ~Matrix() {};

    // transpose
    Matrix transpose () {
        vector<vector<double>> temp(data[0].size(), vector<double> (data.size(), 0));
        Matrix ret(temp);
        for (int row = 0; row < ret.data.size(); row++) {
            for (int col = 0; col < ret.data[0].size(); col++) {
                ret.data[row][col] = data[col][row];
            }
        }
        return ret;
    }

    // norm (for vectors only)
    double norm () {
        double ret = 0;
        for (int row = 0; row < data.size(); row++) {
            ret += data[row][0] * data[row][0];
        }
        return sqrt(ret);
    }

    // formated output
    friend ostream &operator<< (ostream &os, const Matrix &mat) {
        for (int row = 0; row < mat.data.size(); row++) {
            for (int col = 0; col < mat.data[0].size(); col++) {
                os << mat.data[row][col] << '\t';
            }
            os << endl;
        }
        return os;
    };

    // number * matrix
    friend Matrix operator* (double mult, const Matrix &mat) {
        vector<vector<double>> temp(mat.data.size(), vector<double> (mat.data[0].size(), 0));
        Matrix ret(temp);
        for (int row = 0; row < mat.data.size(); row++) {
            for (int col = 0; col < mat.data[0].size(); col++) {
                ret.data[row][col] = mult * mat.data[row][col];
            }
        }
        return ret;
    }

    // matrix * matrix
    friend Matrix operator* (const Matrix &mat1, const Matrix &mat2) {
        vector<vector<double>> temp(mat1.data.size(), vector<double> (mat2.data[0].size(), 0));
        Matrix ret(temp);
        for (int row_1 = 0; row_1 < mat1.data.size(); row_1++) {
            for (int col_2 = 0; col_2 < mat2.data[0].size(); col_2++) {
                for (int pos = 0; pos < mat1.data[0].size(); pos++) {
                    ret.data[row_1][col_2] += mat1.data[row_1][pos] * mat2.data[pos][col_2];
                }
            }
        }
        return ret;
    }

    // matrix - matrix
    friend Matrix operator- (const Matrix &mat1, const Matrix &mat2) {
        vector<vector<double>> temp(mat1.data.size(), vector<double> (mat1.data[0].size(), 0));
        Matrix ret(temp);
        for (int row = 0; row < mat1.data.size(); row++) {
            for (int col = 0; col < mat1.data[0].size(); col++) {
                ret.data[row][col] = mat1.data[row][col] - mat2.data[row][col];
            }
        }
        return ret;
    }
};



int main (int argc, char **argv) {
    int my_rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int matrix_size = stoi(argv[1]); // input matrix size via command line
    int matrix_columns_proc = matrix_size / num_procs;
    if (matrix_size % num_procs > my_rank) matrix_columns_proc++;

    vector<vector<double>> A(matrix_size, vector<double>(matrix_columns_proc, 0)); // test matrix filled with indexes
    for (int row = 0; row < matrix_size; row++) {
        for (int col = 0; col < matrix_columns_proc; col++) {
            A[row][col] = 10*row + col;
        }
    }
    Matrix Am(A);
    
    printf("[%d / %d]: %d columns with numbers ", my_rank, num_procs, matrix_columns_proc);
    for (int i = 0; i < matrix_columns_proc; i++) {
        printf("%d, ", my_rank + i * num_procs);
    }
    printf("\n");

    cout << Am.norm() << endl;

    MPI_Finalize();
    return 0;
}