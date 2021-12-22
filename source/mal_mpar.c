#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

void prop(int *x, float *w);
float alpha_zero(int R, float* w);
double uni_dist();
int find_r(float *w, int R, float a0, float u2);
// int cmpfunc (const void * a, const void * b);
static double get_wall_seconds();
void write_output(int* data, int len, double par_runtime, double comm_runtime, int nb_proc, int* bounds, int interval, int N, double serial);

int main(int argc, char **argv){
    
    int size, rank;
    /* Initialize mpi */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    int N = atoi(argv[1]);
	
    /* Initialize P vector */
	int P[15][7] = {
	{	1, 0, 0, 0, 0, 0, 0		},
	{	-1, 0, 0, 0, 0, 0, 0	},
	{	-1, 0, 1, 0, 0, 0, 0	},
	{	0, 1, 0, 0, 0, 0, 0		},
	{	0,-1, 0, 0, 0, 0, 0		},
	{	0,-1, 0, 1, 0, 0, 0		},
	{	0, 0, -1, 0, 0, 0, 0	},
	{	0, 0, -1, 0, 1, 0, 0	},
	{	0, 0, 0, -1, 0, 0, 0	},
	{	0, 0, 0, -1, 0, 1, 0	},
	{	0, 0, 0, 0, -1, 0, 0	},
	{	0, 0, 0, 0, -1, 0, 1	},
	{	0, 0, 0, 0, 0, -1, 0	},
	{	1, 0, 0, 0, 0, 0, -1	},
	{  	0, 0, 0, 0, 0, 0, -1	}
    };

    int *results = (int*)malloc(7 * N * sizeof(int)); /* Array to store each local run of ssa at time T */
    int *total_results = (int*)malloc( N * sizeof(int));
    float *w=(float*)malloc(15*sizeof(float));
   
    /* Select number of experiments N and p processes such that N=n*p* for some n scalar so if i chose 100 = 25*5 N=100 n=25 p=processes */
    int exp_per_proc = N/size;
    int *humans = (int*)malloc(exp_per_proc*sizeof(int)); /* Array to store the x[1] element from each run of ssa at time T*/
    /* Initialize parallel runtime clock */
    double start, end, start_comm, end_comm;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    /*** MC ***/
    srand(time(NULL) + rank);
    for (int mc_exp =0; mc_exp < exp_per_proc; mc_exp++){
    /*** SSA ***/
    /* 1. Set a final simulation time T, current time t=0, initial state x=x0 */
    double T = 100, t = 0;
    int x0[7]={900, 900, 30, 330, 50, 270, 20};
    int x[7]= {0,0,0,0,0,0,0};
    memcpy(x, x0, sizeof(x));
    float a0,u1,u2,tau;
    int r;
    /* 2.  while ( t < T ) */
        while( t < T ){
            /* 3. Compute w = prop(x)*/
            prop(x,w);
            /* 4. Compute a0 */
            a0 = alpha_zero(15, w);
            /* 5. Generate two uniform numbers u1, u2 between 0 and 1 */
            u1 = uni_dist();
            u2 = uni_dist();
            /* 6. Set taf = -ln(u1)/a0 */
            tau = -log(u1)/a0;
            /* 7.  Find r such that: the sum of w(k), for k=0..r-1 < a0u2 <= then the sum of w(k) for k=1...r*/
            r = find_r(w,15,a0,u2);
            /* 8. Update the state vector */
            for (int i = 0; i < 7; i++)
                x[i] += (int)P[r][i];
            t = t + tau;
        } /*** SSA ENDS HERE***/
        
        for (int i=0; i < 7; i++)
            for (int j=mc_exp; j < exp_per_proc; j++)
                *(results + i*exp_per_proc + j) = x[i];
    } /************************ MC ENDS HERE ************************/
    
    /* Compute something */
    
    /* 1. Collect the susceptible humans from each procs MC results and store them in humans */
    // printf("RANK[%d]\n", rank);
    for (int j=0; j < exp_per_proc; j++){
        humans[j] = *(results + j);
        }
    /* 2. Find the min and max */ 
    /*
    qsort(humans, exp_per_proc, sizeof(int), cmpfunc);
    int max= humans[exp_per_proc-1];
    int min= humans[0]; 
    */
    int max = 0;
    int min = __INT_MAX__; 
    for (int i = 0;  i < exp_per_proc; i++){
        if (humans[i] > max){
              max = humans[i];
        }
        if (humans[i] < min){
            min = humans[i];
        }
    }
    start_comm = MPI_Wtime();
    /*Gather all the max and min and find the max and min from all the procs */
    int global_max; 
    int global_min;
    MPI_Reduce(&max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gather(humans, exp_per_proc, MPI_INT, total_results, exp_per_proc, MPI_INT, 0, MPI_COMM_WORLD); 
    end_comm = MPI_Wtime() - start_comm;


    end = MPI_Wtime() - start; //PARALLEL PART ENDS AFTER THIS LINE

    if ( rank == 0 ) { // SERIAL PART STARTS HERE
    double serial_start = get_wall_seconds();
    // printf("\n Global Max [%d] \t Global MIN [%d] \n", global_max, global_min);
    int bins = 20;
    int interval = (global_max-global_min)/bins;
    int hist_res[21]= { 0 };
    int interval_bounds[21] = { 0 };
    for (int i=0; i<21; i++){
        if ( i == 20){
            interval_bounds[i] = global_min + i*(int)interval + (int)interval;
        }else{
            interval_bounds[i] = global_min + i*(int)interval;
        }
    }
    /*
    for (int i=0; i<21; i++)
        printf("Bounds[%d] = [%d]\t", i, interval_bounds[i]);
    putchar('\n');
    */

    /*Count frequencies*/
    int idx=0;
    for (int i=1; i<21; i++){
        int counter =0;
        for (int j=0; j<N; j++){
           // printf("HUMANSZZ=[%d]", humans[j]);
            if ( i!=20 ){
                if ((int)total_results[j] >= (int)interval_bounds[i-1] && (int)total_results[j] < (int)interval_bounds[i]){
                    counter++;
                }
            }else if( i == 20){
                if ((int)total_results[j] >= (int)interval_bounds[i-1] || (int)total_results[j] > (int)interval_bounds[i]){
                    counter++;
                }
            }
        }
        hist_res[idx] += counter;
        idx++;
    }
    double serial_end = get_wall_seconds();
    double serial_time = serial_end-serial_start;
    write_output(hist_res, bins, end, end_comm, size, interval_bounds, interval, N, serial_time);
    
    //int *hist_res = (int*)malloc(bins*sizeof(int));
    //printf("INTERVAL:%d\n", interval );
    /*
    int total_hist=0;
    for (int i=0; i <20; i++){
        printf("bin[%d] has [%d] elements\n",i, hist_res[i]);
        total_hist += hist_res[i];
    }
    
    printf("Total sum of interval elements is %d", total_hist);
    putchar('\n');
    */
    // Plotting histogram
    /*
    int count=0;
    printf("\nHistogram  data, Interval:%d \n", interval);
    for (int i = 1; i < 21; i++)
    {
        count = hist_res[i-1];
        printf("%d\n|", interval_bounds[i-1]);
        
        for (int j = 0; j < count; j++)
        {
            printf("%c", '*');
        }
        printf("[%d]", hist_res[i-1]);
        printf("\n");
    }
    */
    }
free(total_results);
free(humans);
//free(hist_res);
free(results);
free(w);
MPI_Finalize();
return 0;
}


void write_output(  int* data, int len, double par_runtime, double comm_runtime, int nb_proc, int *bounds, int interval, int N, double serial)
{
    
    FILE* fp = fopen( "final_results.txt", "a" ); 
    if( fp == NULL ) 
        return;
    
    /*
    printf("Number of Procs [%d],\nParallel Runtime[%f],\nCommunication Runtime[%f],\nSerial Runtime[%f],\nTotal Runtime[%f],\n",nb_proc, par_runtime, comm_runtime, serial, par_runtime+serial );
    printf("Interval[%d]\n", interval);
    printf("BIN\tBounds\tElements\n");
    */
    
    fprintf(fp, "Number of Procs [%d],\nParallel Runtime[%f],\nCommunication Runtime[%f],\nSerial Runtime[%f],\nTotal Runtime[%f],\n",nb_proc, par_runtime, comm_runtime, serial, par_runtime+serial );
    fprintf(fp, "N=[%d]\tInterval[%d]\n",N, interval);
    fprintf(fp, "BIN\tBounds\tElements\n");
    

    for( int i = 0; i < len; ++i )
    {   
        // printf("%d\t\t%d\t\t%d\n",i+1, bounds[i], data[i]);
        fprintf( fp, "%d\t\t\t%d\t\t%d\n",i+1, bounds[i], data[i]);
        //fprintf( fp, "bin %d has %d \n",i+1, data[i]);
        //fprintf( fp, "%d ", data[i] );
    }
    // printf("\n******************************************************************************************************\n");
     fprintf(fp, "\n******************************************************************************************************\n");
     fclose( fp );
 // writeMatrix
}

int find_r(float *w, int R, float a0, float u2){
    int r=0,found=0;
    float sum=0;
    float mid = a0*u2;
    for (int i=0; i < R; i++){
        sum += (float)w[i];
        //printf("SUM[%f]\tw[%f]=\t MID=[%f]\n", sum, w[i], mid);
        r++;
        if ( (int)sum >= (int)mid ){
            found = r;
            break;
        }
    }
    return found-1;
}

double uni_dist() { 
    return (rand() + 1.0) / (RAND_MAX+2.0);
}

float alpha_zero(int R, float* w){
    float alpha =0;
    for (int i=0; i<15; i++){
        alpha += w[i];
    }
    return alpha;
}

void prop(int *x, float *w) {
	// Birth number, humans
	const float LAMBDA_H = 20;
	// Birth number, mosquitoes
	const float LAMBDA_M = 0.5;
	// Biting rate of mosquitoes
	const float B = 0.075;
	/* Probability that a bite by an infectious mosquito results in transmission
	   of disease to human*/
	const float BETA_H = 0.3;
	/* Probability that a bite results in transmission of parasite to a
	   susceptible mosquito*/
	const float BETA_M = 0.5;
	// Human mortality rate
	const float MU_H = 0.015;
	// Mosquito mortality rate
	const float MU_M = 0.02;
	// Disease induced death rate, humans
	const float DELTA_H = 0.05;
	// Disease induced death rate, mosquitoes
	const float DELTA_M = 0.15;
	// Rate of progression from exposed to infectious state, humans
	const float ALFA_H = 0.6;
	// Rate of progression from exposed to infectious state, mosquitoes
	const float ALFA_M = 0.6;
	// Recovery rate, humans
	const float R = 0.05;
	// Loss of immunity rate, humans
	const float OMEGA = 0.02;
	/* Proportion of an antibody produced by human in response to the incidence
	   of infection caused by mosquito. */
	const float NU_H = 0.5;
	/* Proportion of an antibody produced by mosquito in response to the
	   incidence of infection caused by human. */
	const float NU_M = 0.15;

	w[0] = LAMBDA_H;
	w[1] = MU_H * x[0];
	w[2] = (B * BETA_H * x[0] * x[5]) / (1 + NU_H * x[5]);
	w[3] = LAMBDA_M;
	w[4] = MU_M * x[1];
	w[5] = (B * BETA_M * x[1]*x[4]) / (1 + NU_M * x[4]);
	w[6] = MU_H * x[2];
	w[7] = ALFA_H * x[2];
	w[8] = MU_M * x[3];
	w[9] = ALFA_M * x[3];
	w[10] = (MU_H + DELTA_H) * x[4];
	w[11] = R * x[4];
	w[12] = (MU_M + DELTA_M) * x[5];
	w[13] = OMEGA * x[6];
	w[14] = MU_H * x[6];

	//return w;
}

static double get_wall_seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

/*
int cmpfunc (const void * a, const void * b)
{
    if (*(int*)a > *(int*)b)
        return 1;
    else if (*(int*)a < *(int*)b)
        return -1;
    else
        return 0;
}
*/