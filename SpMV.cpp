//#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include"r_mat.h"
void print_help(char);
void set_parameters(int, char**, int*, int*, block*, long*, int*);
int main(int argc, char* argv[])
{
	//MPI_Init(&argc, &argv);
	int provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
	if(provided < MPI_THREAD_MULTIPLE)
	{
		printf("The threading support level is lesser than that demanded.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	int scaling_factor = 15, edge_factor = 27, iterations = 5;
	long mat_size, rows_per_pe, pe_nnz, nnz=0, *nnz_dist;
	time_stats graph_time = {0, 0, 0, 0}, SpMV_time = {0, 0, 0, 0}, comm_time = {0, 0, 0, 0};
	block mat_prob = {0.25, 0.25, 0.25, 0.25, 1};
	set_parameters(argc, argv, &scaling_factor, &edge_factor, &mat_prob, &mat_size, &iterations);
	int rank, npes, mat_blocks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	rows_per_pe = mat_size/npes;
	mat_blocks = mat_size/rows_per_pe;
	//probability distribution
	nnz_dist = calculate_nnz_distribution(rank, npes, &mat_prob);
	pe_nnz = calculate_nnz(nnz_dist, npes);
	//matrix creation
	graph_time.t = MPI_Wtime();
	csr_data *csr_mat = create_matrix_data(nnz_dist, pe_nnz, rows_per_pe, npes, &mat_prob);
	graph_time.t = MPI_Wtime() - graph_time.t;
	SpMV_data data;
	data.csr_mat = csr_mat;
	data.mat_size = mat_size;
	data.multi_vector = (float*)calloc(mat_size, sizeof(float));
	data.result_vector = (float*)calloc(mat_size, sizeof(float));
	//initialise multiplication vector
	for(long i=0;i<mat_size;i++) data.multi_vector[i] = ((float)rand()/RAND_MAX);//1e3;
	//iterative SpMV
	SpMV_stats *stats = iterative_SpMV(&data, iterations, rank, npes);
	SpMV_time.t = stats->compute_time;
	comm_time.t = stats->comm_time;
	//free data
	free(nnz_dist);
	free(csr_mat);
	free(data.multi_vector);
	free(data.result_vector);
	MPI_Barrier(MPI_COMM_WORLD);
	//stats
	MPI_Reduce(&pe_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&graph_time.t, &graph_time.min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&graph_time.t, &graph_time.max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&graph_time.t, &graph_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&SpMV_time.t, &SpMV_time.min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&SpMV_time.t, &SpMV_time.max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&SpMV_time.t, &SpMV_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&comm_time.t, &comm_time.min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&comm_time.t, &comm_time.max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&comm_time.t, &comm_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	graph_time.avg_t /= npes;
	SpMV_time.avg_t /= npes;
	comm_time.avg_t /= npes;
	MPI_Barrier(MPI_COMM_WORLD);
	if(!rank)
	{
		printf("scaling factor: %d\n", scaling_factor);
		printf("edge factor: %d (%0.3lf)  iterations: %d\n", edge_factor, ((double)nnz/mat_size), iterations);
		printf("probabilities: %0.3f, %0.3f, %0.3f, %0.3f\n", mat_prob.a, mat_prob.b, mat_prob.c, mat_prob.d);
		printf("matrix size: %ld, nnz: %ld (%ld)\n", mat_size, nnz, mat_prob.nnz);
		printf("rows per PE: %ld, number of PEs: %d\n", rows_per_pe, npes);
#ifdef _COMPRESS
		printf("avg compression ratio: %f\n", stats->compression_ratio[iterations]);
#endif
		//printf("mat blocks: %d\n",mat_blocks);
		printf("Broadcast Size: %lf GB\n", (double)(mat_size*sizeof(float))/(1024*1024*1024));
		printf("graph time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n", graph_time.avg_t, graph_time.min_t, graph_time.max_t);
		printf("SpMV time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n", SpMV_time.avg_t, SpMV_time.min_t, SpMV_time.max_t);
		printf("comm time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n", comm_time.avg_t, comm_time.min_t, comm_time.max_t);
	}
	if(!rank)
	{
		FILE * fp_stats;
		fp_stats = fopen("stats.txt","a+");
		if (fp_stats != NULL)
		{
			//fprintf(fp_stats,"rows per pe: %ld\tpe_elems: %ld\n", rows_per_pe, pe_elems);
			fprintf(fp_stats,"scaling factor: %d\n", scaling_factor);
			fprintf(fp_stats,"edge factor: %d (%0.3lf)  iterations: %d\n", edge_factor, ((double)nnz/mat_size), iterations);
			fprintf(fp_stats,"probabilities: %0.3f, %0.3f, %0.3f, %0.3f\n", mat_prob.a, mat_prob.b, mat_prob.c, mat_prob.d);
			fprintf(fp_stats,"matrix size: %ld, nnz: %ld (%ld)\n", mat_size, nnz, mat_prob.nnz);
			fprintf(fp_stats,"rows per PE: %ld, number of PEs: %d\n", rows_per_pe, npes);
#ifdef _COMPRESS
			for(int i=0;i<iterations;i++)
				fprintf(fp_stats,"iter %d: %f   ", i, stats->compression_ratio[i]);	
			fprintf(fp_stats,"\navg compression ratio: %f\n", stats->compression_ratio[iterations]);
#endif
			//fprintf(fp_stats,"mat blocks: %d\n",mat_blocks);
			fprintf(fp_stats,"Broadcast Size: %lf GB\n", (double)(mat_size*sizeof(float))/(1024*1024*1024));
			fprintf(fp_stats,"graph time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n", graph_time.avg_t, graph_time.min_t, graph_time.max_t);
			fprintf(fp_stats,"SpMV time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n", SpMV_time.avg_t, SpMV_time.min_t, SpMV_time.max_t);
			fprintf(fp_stats,"comm time: avg: %0.5lf s,  min: %0.5lf s,  max: %0.5lf s\n\n\n", comm_time.avg_t, comm_time.min_t, comm_time.max_t);
		}
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}

void print_help(char ch)
{
	if(ch == 's') printf("invalid value for -s (it requires positive integer)\n");
	if(ch == 'e') printf("invalid value for -e (it requires positive integer)\n");
	if(ch == 'a') printf("invalid value for -a (it requires positive value [0,1])\n");
	if(ch == 'b') printf("invalid value for -b (it requires positive value [0,1])\n");
	if(ch == 'c') printf("invalid value for -c (it requires positive value [0,1])\n");
	if(ch == 'i') printf("invalid value for -i (range 5 - 20)\n");
	printf("scaling factor -s (15)\n");
	printf("edge factor -e (20)\n");
	printf("probability values: -a(0.57) -b(0.19) -c(0.19)\n");
	printf("number of iterations -i (10)\n");
	printf("help -h\n");
	exit(1);
}
void set_parameters(int argc, char **argv, int *scaling_factor, int *edge_factor, block *mat_prob, long *mat_size, int *iterations)
{
	int opt;
	while((opt = getopt(argc, argv, ":s:e:a:b:c:h:i:")) != -1)
	{
		switch(opt)
		{
			case 's':
				*scaling_factor=atoi(optarg);
				if(*scaling_factor < 1) print_help(opt);
				break;
			case 'e':
				*edge_factor=atoi(optarg);
				if(*edge_factor < 1) print_help(opt);
				break;
			case 'a':
				mat_prob->a = atof(optarg);
				if(mat_prob->a > 1 || mat_prob->a < 0) print_help(opt);
				break;
			case 'b':
				mat_prob->b = atof(optarg);
				if(mat_prob->b > 1 || mat_prob->b < 0) print_help(opt);
				break;
			case 'c':
				mat_prob->c = atof(optarg);
				if(mat_prob->c > 1 || mat_prob->c < 0) print_help(opt);
				break;
			case 'i':
				*iterations=atoi(optarg);
				if(*iterations < 5 || *iterations > 50) print_help(opt);
				break;
			case 'h':
				print_help(opt);
				break;
			default:
				if(optopt == 's') *scaling_factor = 15;
				else if(optopt == 'e') *edge_factor = 14;
				else if(optopt == 'a') mat_prob->a = 0.25;
				else if(optopt == 'b') mat_prob->b = 0.25;
				else if(optopt == 'c') mat_prob->c = 0.25;
				else if(optopt == 'i') *iterations = 10;
				else print_help('h');
		}
	}
	mat_prob->d = 1 - (mat_prob->a + mat_prob->b + mat_prob->c);
	if(mat_prob->d > 1 || mat_prob->d < 0)
	{
		printf("sum of probabilities (a: %0.3f, b: %0.3f, c: %0.3f, d: %0.3f) != 1\n", mat_prob->a, mat_prob->b, mat_prob->c, mat_prob->d);
		print_help('h');
	}
	*mat_size = (long)1<<(*scaling_factor);
	mat_prob->nnz = (*mat_size)*(*edge_factor);
}