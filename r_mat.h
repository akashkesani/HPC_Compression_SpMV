#pragma once
#include<mpi.h>
#include<math.h>
#include<algorithm>
#include<omp.h>
#include<string>
#include<libpressio_ext/cpp/libpressio.h>
#include<sz.h>
#include<libpressio.h>
using namespace std;
//using string;
void compress_data(void*, size_t, void*, uint64_t*);
void decompres_data(void*, size_t, size_t, void*);
typedef struct block_probablity
{
	float a, b, c, d;
	long nnz;
}block;
typedef struct EDGE
{
	//edge from v to u with weight w
	int v;
	long u;
	float w;
}edge;
typedef struct CSR_MATRIX
{
	int *row_ptr;
	long *col_ptr;
	float *val_ptr;
	long nnz, rows;
}csr_data;
typedef struct SpMV_DATA
{
	csr_data *csr_mat;
	float *multi_vector;
	float *result_vector;
	long mat_size;
}SpMV_data;
typedef struct TIME
{
	double t, avg_t, min_t, max_t;
}time_stats;
typedef struct SpMV_STATS
{
	double comm_time, compute_time;
	float *compression_ratio;
}SpMV_stats;
long** block_allocation(int n, int m)
{
	size_t size = (size_t)n*m;
	long **pe_blocks = (long**)malloc(sizeof(long*)*n);
	pe_blocks[0] = (long*)malloc(sizeof(long)*size);
	for(int i=1;i<n;i++)
		pe_blocks[i] = pe_blocks[0]+(size_t)i*m;
	return pe_blocks;
}
long calculate_nnz(long*, int);
long* calculate_nnz_distribution(int rank, int npes, block *mat_prop)
{
	long **pe_blocks = block_allocation(npes, npes);
	int i, j, k, l, m, x, y;
	int n=log2(npes);
	for(i=0;i<npes;i++)
		for(j=0;j<npes;j++)
			pe_blocks[i][j] = mat_prop->nnz;
	int num_blocks, grid_size, block_row, block_col, grid_row, grid_col;
	float prob;
	for(i=0;i<n;i++)
	{
		num_blocks = 1<<i;
		grid_size = 1<<(n-1-i);
		for(j=0;j<num_blocks;j++)
		{
			block_row = j*grid_size*2;
			for(k=0;k<num_blocks;k++)
			{
				block_col = k*grid_size*2;
				for(l=0;l<2;l++)
				{
					grid_row = block_row+l*grid_size*l;
					for(m=0;m<2;m++)
					{
						grid_col = block_col+grid_size*m;
						prob = l?(m?mat_prop->d:mat_prop->c):(m?mat_prop->b:mat_prop->a);
						for(x=0;x<grid_size;x++)
							for(y=0;y<grid_size;y++)
								pe_blocks[grid_row+x][grid_col+y] = (long)round(pe_blocks[grid_row+x][grid_col+y]*prob);
					}
				}
			}
		}
	}
	long *nnz_dist = (long*)malloc(sizeof(long)*npes);
	memcpy(nnz_dist, pe_blocks[rank], (size_t)npes*sizeof(long));
	free(pe_blocks[0]);
	free(pe_blocks);
	return nnz_dist;
}
long calculate_nnz(long *nnz_dist, int n)
{
	int i;
	long nnz=0;
	for(i=0;i<n;i++)
		//printf("%ld ", pe_blocks[i][j]);
		nnz += nnz_dist[i];
	//printf("nnz: (%ld)\n", nnz);
	return nnz;
}
//edgelist generation based on r-mat probabilities using stochastic Kronecker algorithm
void stochastic_Kronecker_grpah(edge *edge_array, long pe_nnz, long dim, long start_idx, float *prob, float *c_prob, int *row_nnz_count)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	srand(start_idx+rank);
	long mat_dim, row, col, nnz_count=0;
	int p_idx, prow, pcol, i, k;
	float p;
	k = log2(dim);
	while(nnz_count < pe_nnz)
	{
		i=0;
		mat_dim = dim;
		row=0;
		col=0;
		while (i++<k)
		{
			p = (float)rand()/RAND_MAX;
			p_idx = p<c_prob[0]?0:(p<c_prob[1]?1:(p<c_prob[2]?2:3));
			prow = p_idx/2;
			pcol = p_idx%2;
			mat_dim/=2;
			row = row + mat_dim*prow;//optimize with +=
			col = col + mat_dim*pcol;
		}
		edge_array[nnz_count].v = row;
		edge_array[nnz_count].u = start_idx+col;
		edge_array[nnz_count++].w = p/10;
		row_nnz_count[row]++;
	}
}
//for validation
void print_csr(csr_data *csr_mat)
{
	printf("rows: %ld, nnz: %ld\n", csr_mat->rows, csr_mat->nnz);
	for(long i=0;i<csr_mat->rows;i++)
	{
		printf("%ld: ",i);
		for(long j=csr_mat->row_ptr[i];j<csr_mat->row_ptr[i+1];j++)
			printf("%ld ", csr_mat->col_ptr[j]);
		printf("\n");
	}
}
//for validation
void validate_csr(csr_data *csr_mat)
{
	for(long i=0;i<csr_mat->rows;i++)
		for(long j=csr_mat->row_ptr[i]+1;j<csr_mat->row_ptr[i+1];j++)
			if(csr_mat->col_ptr[j] < csr_mat->col_ptr[j-1])
			{
				printf("data incorrect r:%ld, c:%ld\n", i, j);
				return;
			}
	printf("data validated\n");
}
csr_data* create_csr_data(edge *edge_array, int *row_nnz_count, long pe_nnz, long rows_per_pe)
{
	size_t n = (size_t)sizeof(csr_data)+(rows_per_pe+1)*sizeof(int) + pe_nnz*(sizeof(long) + sizeof(float));
	csr_data *csr_mat = (csr_data*)malloc(n);
	csr_mat->nnz = pe_nnz;
	csr_mat->rows = rows_per_pe;
	long i, idx;
	if(csr_mat == NULL)
	{
		printf("csr memory allocation failed (%ld B)\n", n);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	n = sizeof(csr_data);
	csr_mat->row_ptr = (int*)((char*)csr_mat + n);
	n+=((rows_per_pe+1)*sizeof(int));
	csr_mat->col_ptr = (long*)((char*)csr_mat + n);
	n+=(pe_nnz*sizeof(long));
	csr_mat->val_ptr = (float*)((char*)csr_mat + n);

	//row_ptr
	csr_mat->row_ptr[0] = 0;
	for(i=1;i<=rows_per_pe;i++)
		csr_mat->row_ptr[i] = csr_mat->row_ptr[i-1] + row_nnz_count[i-1];
	for(i=0;i<pe_nnz;i++)
	{
		idx = edge_array[i].v;
		row_nnz_count[idx]--;
		idx = csr_mat->row_ptr[idx] + row_nnz_count[idx];
		csr_mat->col_ptr[idx] = edge_array[i].u;
		csr_mat->val_ptr[idx] = edge_array[i].w;
	}
	for(i=0;i<rows_per_pe;i++)
		std::sort(&csr_mat->col_ptr[csr_mat->row_ptr[i]], &csr_mat->col_ptr[csr_mat->row_ptr[i+1]]);
	return csr_mat;
}
csr_data* create_matrix_data(long *nnz_dist, long pe_nnz, long rows_per_pe, int npes, block *mat_prob)
{
	edge *edge_array = (edge*)malloc((size_t)pe_nnz*sizeof(edge));
	int i, *row_nnz_count = (int*)calloc(rows_per_pe, sizeof(int));
	csr_data *csr_mat;
	long block_nnz, start_idx, idx = 0;
	float prob[] = {mat_prob->a, mat_prob->b, mat_prob->c, mat_prob->d}, c_prob[4];
	c_prob[0] = prob[0];
	for(i=1;i<4;i++)
		c_prob[i] = prob[i] + c_prob[i-1];
	for(i=0;i<npes;i++)
	{
		block_nnz = nnz_dist[npes-1-i];
		start_idx = rows_per_pe*(npes-1-i);
		stochastic_Kronecker_grpah(&edge_array[idx], block_nnz, rows_per_pe, start_idx, prob, c_prob, row_nnz_count);
		idx += block_nnz;
	}
	csr_mat = create_csr_data(edge_array, row_nnz_count, pe_nnz, rows_per_pe);
	//free data;
	free(edge_array);
	free(row_nnz_count);
	return csr_mat;
}
void SpMV_kernel(SpMV_data *data, int rank, MPI_Win window)
{
	long i, j;
	int *row_ptr = data->csr_mat->row_ptr;
	long *col_ptr = data->csr_mat->col_ptr;
	float *val_ptr = data->csr_mat->val_ptr,
		*multi_vector = data->multi_vector,
		*result_vector = data->result_vector;
	MPI_Aint start_idx = data->csr_mat->rows*rank;
	#pragma omp parallel shared(multi_vector,row_ptr,col_ptr,val_ptr,result_vector) private(i, j)
	{
		int t_id = omp_get_thread_num(), num_threads = omp_get_num_threads();
		long start_idx = (data->csr_mat->rows*t_id)/num_threads, end_idx = (data->csr_mat->rows*(t_id+1))/num_threads;
		//if(rank == 0)printf("t_id: %d|%d, idx: %ld -> %ld\n", t_id, num_threads, start_idx, end_idx);
		for(i=start_idx;i<end_idx;i++)
		{
			result_vector[i] = 0;
			for(j=row_ptr[i];j<row_ptr[i+1];j++)
				result_vector[i] += val_ptr[j]*multi_vector[col_ptr[j]];
				//if(rank==0) printf("r:%d c:%ld v:%f mv:%f rv:%f\n", i, col_ptr[j], val_ptr[j], multi_vector[col_ptr[j]], result_vector[i]);
			if(rank) MPI_Put(&result_vector[i], 1, MPI_FLOAT, 0, start_idx+i, 1, MPI_FLOAT, window);
		}
	}
	/*#pragma omp parallel for shared(multi_vector,row_ptr,col_ptr,val_ptr,result_vector) private(i, j)
	for(i=0;i<data->csr_mat->rows;i++)
	{
		result_vector[i] = 0;
		for(j=row_ptr[i];j<row_ptr[i+1];j++)
			result_vector[i] += val_ptr[j]*multi_vector[col_ptr[j]];
			//if(rank==0) printf("r:%d c:%ld v:%f mv:%f rv:%f\n", i, col_ptr[j], val_ptr[j], multi_vector[col_ptr[j]], result_vector[i]);
		if(rank) MPI_Put(&result_vector[i], 1, MPI_FLOAT, 0, start_idx+i, 1, MPI_FLOAT, window);
	}*/
	MPI_Win_fence(0, window);
}
SpMV_stats* iterative_SpMV(SpMV_data *data, int iterations, int rank, int npes)
{
	SpMV_stats *stats = (SpMV_stats*)malloc(sizeof(SpMV_stats));
	stats->comm_time = 0;
	stats->compute_time = 0;
	stats->compression_ratio = (float*)calloc(iterations+1, sizeof(float));
	MPI_Win window[2];
	MPI_Win_create(data->result_vector, data->mat_size*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &window[0]);
	MPI_Win_create(data->multi_vector, data->mat_size*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &window[1]);
	MPI_Win_fence(0, window[0]);
	MPI_Win_fence(0, window[1]);
	double time;
	for(int i=0;i<iterations;i++)
	{
		time = MPI_Wtime();
		SpMV_kernel(data, rank, window[i%2]);
		time = MPI_Wtime() - time;
		stats->compute_time += time;
#ifdef _COMPRESS
		uint64_t compressed_size;
		compress_data(data->result_vector, data->mat_size, (void*)data->multi_vector, &compressed_size);
		MPI_Bcast(&compressed_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
		time = MPI_Wtime();
		MPI_Bcast(data->multi_vector, compressed_size, MPI_CHAR, 0, MPI_COMM_WORLD);
		time = MPI_Wtime() - time;
		stats->compression_ratio[i] = float(((double)data->mat_size*sizeof(float))/compressed_size);
		stats->compression_ratio[iterations] += stats->compression_ratio[i];
		if(!rank)printf("%d: (%f)\n", i, stats->compression_ratio[i]);
#else
		time = MPI_Wtime();
		MPI_Bcast(data->result_vector, data->mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
		time = MPI_Wtime() - time;
#endif
		stats->comm_time += time;
#ifdef _COMPRESS
		decompres_data(data->multi_vector, data->mat_size, (size_t)compressed_size, (void*)data->result_vector);
#endif
		float *temp = data->multi_vector;
		data->multi_vector = data->result_vector;
		data->result_vector = temp;
	}
#ifdef _COMPRESS
	string filename = "vector_comp_" + to_string(data->mat_size);
	stats->compression_ratio[iterations] /= iterations;
	if(!rank)printf("avg compression ratio: %f\n", stats->compression_ratio[iterations]);
#else
	string filename = "vector_" + to_string(data->mat_size);
#endif
	if(!rank)
	{
		float *val = data->multi_vector;
		printf("--%f, %f, %f, %f\n", val[0], val[1], val[2], val[3], val[4]);
		printf("-------%s\n", &filename[0]);
		FILE *fp_vector;
		fp_vector = fopen(&filename[0],"wb");
		if (fp_vector == NULL) printf("Unable to open file in write mode...\ncheck permissions of input path\n");
		else
		{
			fwrite(&data->mat_size, sizeof(long), 1, fp_vector);
			fwrite(data->multi_vector, sizeof(float), data->mat_size, fp_vector);
			fclose(fp_vector);
		}
	}
	MPI_Win_free(&window[0]);
	MPI_Win_free(&window[1]);
	MPI_Barrier(MPI_COMM_WORLD);
	return stats;
}

void compress_data(void *input_data, size_t n, void *out_data, uint64_t *size)
{
	// Creating the compressor object
	SpMV_stats *stats = (SpMV_stats*)malloc(sizeof(SpMV_stats));
	pressio library;
	auto compressor = library.get_compressor("sz");
	// Congiguring the metrics object
	const char* metrics_ids[] = {"size"};
	auto metrics = pressio_metrics(library.get_metrics(std::begin(metrics_ids), std::end(metrics_ids)));
	compressor->set_metrics(metrics);	
	// Setting the compressor options ( Mode of compression and error bound)
	pressio_options options = compressor->get_options();
	options.set("sz:error_bound_mode", ABS);
	options.set("sz:abs_err_bound", 0.005);
	//check options
	if(compressor->check_options(options))
	{
		std::cerr << compressor->error_msg() << std::endl;
		exit(compressor->error_code());
	}
	//set options
	if(compressor->set_options(options))
	{
		std::cerr << compressor->error_msg() << std::endl;
		exit(compressor->error_code());
	}
	//Get Pointer to input data
	float *temp_loc, *recv_buffer = (float*)malloc((size_t)n*sizeof(float));
	//data dimentions
 	std::vector<size_t> dims{(size_t)n};
	//Creating the pressio data object for compressed_obj data
	pressio_data input_obj = pressio_data::move(pressio_float_dtype, input_data, dims, nullptr,nullptr);
	pressio_data compressed_obj = pressio_data::empty(pressio_byte_dtype, {});
	//Compressing the data
	if(compressor->compress(&input_obj, &compressed_obj))
	{
		std::cerr << library.err_msg() << std::endl;
		exit(library.err_code());
	}
	void* compressed_data_ptr = (void*)pressio_data_ptr(&compressed_obj,NULL);
	auto metrics_results = compressor->get_metrics_results();
	*size = pressio_option_get_uinteger64(&metrics_results.get("size:compressed_size"));
	memcpy((void*)out_data, (void*)compressed_data_ptr, (size_t)(*size));
}
void decompres_data(void *input_buffer, size_t n, size_t size, void *output_buffer)
{
	pressio library;
	auto compressor = library.get_compressor("sz");
	// Congiguring the metrics object
	const char* metrics_ids[] = {"size"};
	auto metrics = pressio_metrics(library.get_metrics(std::begin(metrics_ids), std::end(metrics_ids)));
	compressor->set_metrics(metrics);	
	// Setting the compressor options ( Mode of compression and error bound)
	pressio_options options = compressor->get_options();
	options.set("sz:error_bound_mode", ABS);
	options.set("sz:abs_err_bound", 0.005);
	//check options
	if(compressor->check_options(options))
	{
		std::cerr << compressor->error_msg() << std::endl;
		exit(compressor->error_code());
	}
	//set options
	if(compressor->set_options(options))
	{
		std::cerr << compressor->error_msg() << std::endl;
		exit(compressor->error_code());
	}
	std::vector<size_t>compresseddims{(size_t)size};
	std::vector<size_t> dims{(size_t)n};
	pressio_data compressed_obj = pressio_data::move(pressio_byte_dtype, (void*)input_buffer, compresseddims, nullptr, nullptr);
	pressio_data decompressed_obj = pressio_data::empty(pressio_float_dtype, dims);
	//Decompressing the data
	if(compressor->decompress(&compressed_obj, &decompressed_obj))
	{
		std::cerr << library.err_msg() << std::endl;
		exit(library.err_code()); 
	}
	float* decompressed_ptr = (float*) pressio_data_ptr(&decompressed_obj,NULL);
	memcpy((void*)output_buffer, (void*)decompressed_ptr, (size_t)(n)*sizeof(float));
}