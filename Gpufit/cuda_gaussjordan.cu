/* CUDA implementation of Gauss-Jordan elimination algorithm.
*  
* Gauss-Jordan elimination method
* ===============================
*
* This function solves a set of linear equations using the Gauss-Jordan elimination method.
* Considering a set of N equations with N unknowns, this can be written in matrix form as
* an NxN matrix of coefficients and a Nx1 column vector of right-hand side values.
*
* For example, consider the following problem with 3 equations and 3 unknowns (N=3):
* 
*   A x + B y + C z = MM
*   D x + E y + F z = NN
*   G x + H y + J z = PP
* 
* We can write this as follows in matrix form:
* 
*   [ A B C ] [ x ] = [ MM ]
*   [ D E F ] [ y ] = [ NN ] 
*   [ G H I ] [ z ] = [ PP ]
* 
* or, [A]*[X] = [B] where [A] is the matrix of coefficients and [B] is the vector of 
* right-hand side values.
*
* The Gauss Jordan elimiation method solves the system of equations in the following
* manner.  First, we form the augmented matrix (A|B):
*
*   [ A B C | MM ] 
*   [ D E F | NN ] 
*   [ G H I | PP ] 
*
* and then the augmented matrix is manipulated until its left side has the reduced
* row-echelon form.  That is to say that any individual row may be multiplied
* by a scalar factor, and any linear combination of rows may be added to another 
* row.  Finally, two rows may be swapped without affecting the solution.
* 
* When the manipulations are complete and the left side of the matrix has the desired
* form, the right side then corresponds to the solution of the system. 
*
*
* Description of the cuda_gaussjordan function
* ============================================
* 
* This algorithm is designed to perform many solutions of the Gauss Jordan elimination
* method in parallel.  One limitation of the algorithm implemented here is that for
* each solution the number of equations and unknowns (N) must be identical.  
*
* Parameters:
* 
* alpha: Coefficients matrices.  The matrix of coefficients for a single solution is 
*        a vector of NxN, where N is the number of equations.  This array stores the 
*        coefficients for the entire set of M input problems, concatenated end to end, 
*        and hence the total size of the array is MxNxN.  
*
* beta: Vector of right hand side values, concatenated together for all input problems. 
*       For a set of M inputs, the size of the vector is MxN.  Upon completion, this 
*       vector contains the results vector X for each solution.
*
* skip_calculation: An input vector which allows the calculation to be skipped for
*                   a particular solution.  For a set of M inputs, the size of this
*                   vector is M. 
*
* singular: An output vector used to report whether a given solution is singular.  For
*           a set of M inputs, this vector has size M.  Memory needs to be allocated
*           by the calling the function.
*
* n_equations: The number of equations and unknowns for a single solution.  This is
*              equal to the size N.
*
* n_equations_pow2: The next highest power of 2 greater than n_equations.
*
*
* Calling the cuda_gaussjordan function
* =====================================
*
* When calling the function, the blocks and threads must be set up correctly, as well
* as the shared memory space, as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = n_equations + 1;
*   threads.y = n_equations;
*   blocks.x = n_solutions;
*   blocks.y = 1;
*
*   int const shared_size = sizeof(REAL) * 
*       ( (threads.x * threads.y) + n_parameters_pow2 + n_parameters_pow2 );
*
*   int * singular;
*   CUDA_CHECK_STATUS(cudaMalloc((void**)&singular, n_solutions * sizeof(int)));
*
*   cuda_gaussjordan<<< blocks, threads, shared_size >>>(
*       alpha,
*       beta,
*       skip_calculation,
*       singular,
*       n_equations,
*       n_equations_pow2);
*
*/

#include "cuda_gaussjordan.cuh"

__global__ void cuda_gaussjordan(
    REAL * delta,
    REAL const * beta,
    REAL const * alpha,
    int const * skip_calculation,
    int * singular,
    std::size_t const n_equations,
    std::size_t const n_equations_pow2)
{
    extern __shared__ REAL extern_array[];     //shared memory between threads of a single block, 
    //used for storing the calculation_matrix, the 
    //abs_row vector, and the abs_row_index vector

    // In this routine we will store the augmented matrix (A|B), referred to here
    // as the calculation matrix in a shared memory space which is visible to all
    // threads within a block.  Also stored in shared memory are two vectors which 
    // are used to find the largest element in each row (the pivot).  These vectors 
    // are called abs_row and abs_row_index.
    //
    // Sizes of data stored in shared memory:
    //
    //      calculation_matrix: n_equations * (n_equations+1)
    //      abs_row:            n_equations_pow2
    //      abs_row_index:      n_equations_pow2
    //  
    // Note that each thread represents an element of the augmented matrix, with
    // the column and row indicated by the x and y index of the thread.  Each 
    // solution is calculated within one block, and the solution index is the 
    // block index x value.

    int const col_index = threadIdx.x;                  //column index in the calculation_matrix
    int const row_index = threadIdx.y;                  //row index in the calculation_matrix
    int const solution_index = blockIdx.x;

    int const n_col = blockDim.x;                       //number of columns in calculation matrix (=threads.x)
    int const n_row = blockDim.y;                       //number of rows in calculation matrix (=threads.y)
    int const alpha_size = blockDim.y * blockDim.y;     //number of entries in alpha matrix for one solution (NxN)

    if (skip_calculation[solution_index])
        return;

    REAL p;                                            //local variable used in pivot calculation

    REAL * calculation_matrix = extern_array;                          //point to the shared memory

    REAL * abs_row = extern_array + n_equations * (n_equations + 1);     //abs_row is located after the calculation_matrix
    //within the shared memory

    int * abs_row_index = (int *)(abs_row + n_equations_pow2);            //abs_row_index is located after abs_row
    //
    //note that although the shared memory is defined as
    //REAL, we are storing data of type int in this
    //part of the shared memory

    //initialize the singular vector
    if (col_index == 0 && row_index == 0)
    {
        singular[solution_index] = 0;
    }

    //initialize abs_row and abs_row_index, using only the threads on the diagonal
    if (col_index == row_index)
    {
        abs_row[col_index + (n_equations_pow2 - n_equations)] = 0.0;
        abs_row_index[col_index + (n_equations_pow2 - n_equations)] = col_index + (n_equations_pow2 - n_equations);
    }

    //initialize the calculation_matrix (alpha and beta, concatenated, for one solution)
    if (col_index != n_equations)
        calculation_matrix[row_index*n_col + col_index] = alpha[solution_index * alpha_size + row_index * n_equations + col_index];
    else
        calculation_matrix[row_index*n_col + col_index] = beta[solution_index * n_equations + row_index];

    //wait for thread synchronization

    __syncthreads();

    //start of main outer loop over the rows of the calculation matrix

    for (int current_row = 0; current_row < n_equations; current_row++)
    {

        // work in only one row, skipping the last column
        if (row_index == current_row && col_index != n_equations)
        {

            //save the absolute values of the current row
            abs_row[col_index] = abs(calculation_matrix[row_index * n_col + col_index]);

            //save the column indices
            abs_row_index[col_index] = col_index;

            __threadfence();

            //find the largest absolute value in the current row and write its index in abs_row_index[0]
            for (int n = 2; n <= n_equations_pow2; n = n * 2)
            {
                if (col_index < (n_equations_pow2 / n))
                {
                    if (abs_row[abs_row_index[col_index]] < abs_row[abs_row_index[col_index + (n_equations_pow2 / n)]])
                    {
                        abs_row_index[col_index] = abs_row_index[col_index + (n_equations_pow2 / n)];
                    }
                }
            }
        }

        __syncthreads();

        //singularity check - if all values in the row are zero, no solution exists
        if (row_index == current_row && col_index != n_equations)
        {
            if (abs_row[abs_row_index[0]] == 0.0)
            {
                singular[solution_index] = 1;
            }
        }

        //devide the row by the biggest value in the row
        if (row_index == current_row)
        {
            calculation_matrix[row_index * n_col + col_index]
                = calculation_matrix[row_index * n_col + col_index] / calculation_matrix[row_index * n_col + abs_row_index[0]];
        }

        __syncthreads();

        //The value of the largest element of the current row was found, and then current
        //row was divided by this value such that the largest value of the current row 
        //is equal to one.  
        //
        //Next, the matrix is manipulated to reduce to zero all other entries in the column 
        //in which the largest value was found.   To do this, the values in the current row
        //are scaled appropriately and substracted from the other rows of the matrix. 
        //
        //For each element of the matrix that is not in the current row, calculate the value
        //to be subtracted and let each thread store this value in the scalar variable p.

        p = calculation_matrix[current_row * n_col + col_index] * calculation_matrix[row_index * n_col + abs_row_index[0]];
        __syncthreads();

        if (row_index != current_row)
        {
            calculation_matrix[row_index * n_col + col_index] = calculation_matrix[row_index * n_col + col_index] - p;
        }
        __syncthreads();

    }

    //At this point, if the solution exists, the calculation matrix has been reduced to the 
    //identity matrix on the left side, and the solution vector on the right side.  However
    //we have not swapped rows during the procedure, so the identity matrix is out of order.
    //
    //For example, starting with the following augmented matrix as input:
    //
    //  [  3  2 -4 |  4 ]
    //  [  2  3  3 | 15 ]
    //  [  5 -3  1 | 14 ]
    //
    //we will obtain:
    //
    //  [  0  0  1 |  2 ]
    //  [  0  1  0 |  1 ]
    //  [  1  0  0 |  3 ]
    //
    //Which needs to be re-arranged to obtain the correct solution vector.  In the final
    //step, each thread checks to see if its value equals 1, and if so it assigns the value
    //in its rightmost column to the appropriate entry in the beta vector.  The solution is
    //stored in beta upon completetion.

    if (col_index != n_equations && calculation_matrix[row_index * n_col + col_index] == 1)
        delta[n_row * solution_index + col_index] = calculation_matrix[row_index * n_col + n_equations];

    __syncthreads();
}
