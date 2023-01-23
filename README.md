# HPC_Compression_SpMV
Improving the performance of Iterative SpMV kernel via compression of I/O data.

R_mat.h has the necessary functions to create the Sparse Matrix, run the matrix vector multiplication, compress the vector and propogate it back to all the processes used in the matrix multiplication. 
