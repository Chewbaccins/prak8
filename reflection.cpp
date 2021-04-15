#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Matrix {
    public:
    Matrix() {};
    vector<vector<double>> data;
    Matrix(int rows, int cols): data(rows, vector<double>(cols, 0)) {};
    Matrix(char i, int size_i, int size_global): data(size_global, vector<double>(size_global, 0)) {
        for (int si = 0; si < size_i; si++) {
            data[si][si] = 1;
        }
    };
    Matrix(vector<vector<double>> mat): data(mat) {};
    Matrix(vector<double> vect): data(vect.size(), vector<double>(1, 0)) {
        for (int pos = 0; pos < vect.size(); pos++) {
            data[pos][0] = vect[pos];
        }
    };
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

    // get X(i)
    Matrix get_Xi (int column, int start_pos = 0) {
        vector<vector<double>> temp(data.size(), vector<double> (1, 0));
        Matrix ret(temp);
        for (int pos = start_pos; pos < data.size(); pos++) {
            ret.data[pos][0] = data[pos][column];
        }
        double ai_norm = ret.norm();
        Matrix e(temp);
        e.data[start_pos][0] = 1;
        ret = (1 / (ret - ai_norm * e).norm())*(ret - ai_norm * e); 
        return ret;
    }

    // get column
    vector<double> get_column (int column) {
        vector<double> ret(data.size(), 0);
        for (int pos = 0; pos < data.size(); pos++) {
            ret[pos] = data[pos][column];
        }
        return ret;
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

    // matrix - matrix
    friend Matrix operator+ (const Matrix &mat1, const Matrix &mat2) {
        vector<vector<double>> temp(mat1.data.size(), vector<double> (mat1.data[0].size(), 0));
        Matrix ret(temp);
        for (int row = 0; row < mat1.data.size(); row++) {
            for (int col = 0; col < mat1.data[0].size(); col++) {
                ret.data[row][col] = mat1.data[row][col] + mat2.data[row][col];
            }
        }
        return ret;
    }
};

// get U(Xi) with Xi as input
Matrix get_Uxi (Matrix mat) {
    Matrix ret('i', mat.data.size(), mat.data.size());
    ret = ret - 2 * (mat * mat.transpose());
    return ret;
}

Matrix multiply_by_element (Matrix mat1, Matrix mat2, int mat2_col) {
    Matrix ret(mat1.transpose());
    for (int pos = 0; pos < mat1.data.size(); pos++) {
        ret.data[pos][0] = mat1.data[0][pos] * mat2.data[pos][mat2_col];
    }
    return ret;
}

int main (int argc, char **argv) {
    int my_rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double time[2];
    int matrix_size = stoi(argv[1]); // input matrix size via command line
    int matrix_columns_proc = matrix_size / num_procs;
    if (matrix_size % num_procs > my_rank) matrix_columns_proc++;
    Matrix b(1, 1);
    bool proc_has_b = (my_rank == num_procs-1);
    if (proc_has_b) b = Matrix(matrix_size, 1);

    vector<vector<double>> A(matrix_size, vector<double>(matrix_columns_proc, 0)); // test matrix filled with indexes
    
    /* read matrix with formula */
    for (int row = 0; row < matrix_size; row++) {
        for (int col = 0; col < matrix_columns_proc; col++) {
            A[row][col] = 10*row + (my_rank + col * num_procs);
            if (proc_has_b) b.data[row][0] = b.data[row][0] + A[row][col];
        }
    }
    /* end of read section */

    /* read matrix from file, matrix must be transposed 
    MPI_File input_matrix;
    MPI_File_open(MPI_COMM_WORLD, "input_matrix.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &input_matrix);
    double *buf;
    buf = (double *)malloc( matrix_size * sizeof(double) );
    for (int col = 0; col < matrix_size; col++) {
        MPI_File_read(input_matrix, buf, matrix_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
        if (my_rank == col%num_procs) {
            for (int row = 0; row < matrix_size; row++) {
                A[row][col/num_procs] = buf[row];
            }
        }
    }
    MPI_File_read(input_matrix, buf, matrix_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
    if (proc_has_b) {
        for (int row = 0; row < matrix_size; row++) {
            b.data[row][0] = buf[row];
        }
    }
    free(buf);
    MPI_File_close(&input_matrix);
    /* end of read section */
    
    Matrix Am(A);

    /* first part */ 
    if (proc_has_b) time[0] = MPI_Wtime();
    for (int matrix_col = 0; matrix_col < matrix_size-1; matrix_col++) {
        Matrix my_Xi(matrix_size, 1);
        vector<double> my_Xi_vect(matrix_size, 0);
        if (my_rank == (matrix_col % num_procs)) {
            my_Xi_vect = Am.get_Xi(matrix_col / num_procs, matrix_col).get_column(0);
        }
        MPI_Bcast(my_Xi_vect.data(), my_Xi_vect.size(), MPI_DOUBLE, matrix_col%num_procs, MPI_COMM_WORLD);
        Matrix my_Uxi(my_Xi_vect);
        my_Uxi = get_Uxi(my_Uxi);
        //if (my_rank == 0) cout << Am << endl;
        Am = my_Uxi * Am;
        //if (my_rank == 0) cout << Am;
        if (proc_has_b) {
            b = my_Uxi * b;
        }
    }
    if (proc_has_b) time[0] = MPI_Wtime() - time[0];

    //cout << "rank[" << my_rank << "]:" << endl << Am << endl;
    //if (proc_has_b) cout << b << endl;
   
    /* second part */
    if (proc_has_b) time[1] = MPI_Wtime();
    vector<double> result_x (matrix_size, 0);
    for (int global_matrix_col = matrix_size-1; global_matrix_col >= 0; global_matrix_col--) {
        double b_sum = 0, b_recv = 0;
        if (proc_has_b) b_sum = b.data[global_matrix_col][0];
        for (int local_matrix_col = matrix_size-1; local_matrix_col > global_matrix_col; local_matrix_col--) {
            if (my_rank == (local_matrix_col % num_procs)) {
                b_sum -= Am.data[global_matrix_col][local_matrix_col / num_procs] * result_x[local_matrix_col];
            }
        }
        MPI_Reduce(&b_sum, &b_recv, 1, MPI_DOUBLE, MPI_SUM, (global_matrix_col%num_procs), MPI_COMM_WORLD);
        if (my_rank == (global_matrix_col % num_procs)) {
            result_x[global_matrix_col] = b_recv / 
                                        Am.data[global_matrix_col][global_matrix_col / num_procs];
        }
    }
    vector<double> final_x (matrix_size, 0);
    MPI_Allreduce(result_x.data(), final_x.data(), matrix_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Matrix x(final_x);
    if (proc_has_b) time[1] = MPI_Wtime() - time[1];

    /* get ||Ax-b|| */
    Matrix Ax(matrix_size, 1);
    for (int col = 0; col < matrix_columns_proc; col++) {
        Matrix temp(Am.get_column(col));
        Ax = Ax + x.data[my_rank + col*num_procs][0] * temp;
    }
    vector<double> Ax_send (matrix_size,0);
    Ax_send = Ax.get_column(0);
    vector<double> Ax_recv (matrix_size,0);
    MPI_Reduce(Ax_send.data(), Ax_recv.data(), matrix_size, MPI_DOUBLE, MPI_SUM, num_procs-1, MPI_COMM_WORLD);
    if (proc_has_b) {
        Matrix Axb(Ax_recv);
        double result_norm = (Axb - b).norm();
        //cout << "x = " << x.transpose();
        cout << "result norm = " << result_norm << endl;
        cout << "time[0] = " << time[0] << endl;
        cout << "time[1] = " << time[1] << endl;
        cout << "sum time= " << time[0] + time[1] << endl;
    }

    /* if x is known */
    /*if (proc_has_b) {
        Matrix x_precise(matrix_size, 0);
        double result_norm = (x_precise - x).norm();
        //cout << "x = " << x.transpose();
        cout << "result norm = " << result_norm << endl;
        cout << "time[0] = " << time[0] << endl;
        cout << "time[1] = " << time[1] << endl;
        cout << "sum time= " << time[0] + time[1] << endl;
    }*/

    /*ofstream fout("matrix.txt", ios_base::app);
    fout << Am << endl;
    fout.close();*/

    MPI_Finalize();
    return 0;
}