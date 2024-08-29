/*=================================================================
                                LICENSE
 *=================================================================
	This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *=================================================================/

/*=================================================================
 *
 * prox_ind_l1_norm.c 
 * 
 * This file provides two implementations for solving the proximal
 * operator of the induced l1 norm:
 * 
 *        X = argmin 1/2 * |X-V|_F^2 + lambda * |X|_1
 *              X
 *
 * The two methods are:
 *
 *  X = prox_l1_norm_active_set (V, lambda) 
 *  X = prox_l1_norm_column_sort(V, lambda) 
 *
 * Inputs:
 *
 *  V     : n-times-m input matrix
 *
 *  lambda: penalty parameter
 *
 * Outputs:
 *
 *  X     : n-times-m matrix such that
 *
 *        X = argmin 1/2 * |X-V|_F^2 + lambda * |X|_1
 *              X
 *=================================================================
 Author: Benjamin Bejar -- bbejar@cis.jhu.edu
 *=================================================================*/

#ifndef PROJCODE_INCLUDE_L1INFTY_BEJAR_HPP
#define PROJCODE_INCLUDE_L1INFTY_BEJAR_HPP

# include <stdio.h>
# include <math.h>
# include <stdlib.h>

namespace proj {
namespace l1infty {

namespace bejar {

/* definition of constants */
#define TRUE  1
#define FALSE 0

/* definition of data types */
typedef struct {
	double * data;
	unsigned int nrows;
	unsigned int ncols;
} Matrix;

typedef struct {
	double  val;
	int 	idx;
} indexed_double;

/* Function prototypes */
void 	prox_l1_norm_active_set ( Matrix *X, Matrix *V, double lambda );
void 	prox_l1_norm_column_sort( Matrix *X, Matrix *V, double lambda );
double 	l1_inf_norm		( Matrix *X );
double 	get_t			( double *psum, int *nj, int M, double lambda );
double 	get_lower_bound	( Matrix *X, double lambda, indexed_double *u, int *M );
void 	get_column_maxes( double *umax, Matrix *X );
char 	update_J_set	( char *J, double *psum, int *Ji, double *x, int n, double t_lb );
double 	sum_array		( double *input, int nelems );
void 	abs_array		( double *res, double *input, int nelems );
int 	compare_iarray 	( const void * a, const void * b );
int 	compare_array 	( const void * a, const void * b );
Matrix 	create_matrix	( int nrows, int ncols );
void 	destroy_matrix	( Matrix *A );
void 	print_matrix	( Matrix *A );


// /* Main program */
// int main() {
	
// 	/* create matrix */
// 	Matrix *A = create_matrix(4,3);
// 	A->data[0] = 1.0;  A->data[4] = 0.1;  A->data[8] = 2.1;
// 	A->data[1] = 1.2;  A->data[5] = 0.7;  A->data[9] = 2.2;
// 	A->data[2] = 2.1;  A->data[6] = -0.2; A->data[10] = 1.3;
// 	A->data[3] = -3.1; A->data[7] = 3.1;  A->data[11] = -1.0;

// 	/* create array */
// 	Matrix *U = create_matrix(A->nrows,A->ncols);

// 	/* compute proximal operator */
// 	print_matrix(A);
// 	prox_l1_norm_column_sort(U,A,1.0);
// 	print_matrix(U);

// 	/* active set method */
// 	prox_l1_norm_active_set(U,A,1.0);
// 	print_matrix(U);

// 	/* release memory */
// 	destroy_matrix(A);
// 	destroy_matrix(U);

// 	/* terminate */
// 	return 0;
// }

/**********************
	LOCAL FUNCTIONS 
***********************/

/** Proximal operator for (induced) l1 norm using active set method */
void prox_l1_norm_active_set( Matrix *X, Matrix *V, double lambda ) {

	// variables
	int ii,jj,M,offset;

	// compute absolute value
	int nelems = V->ncols * V->nrows;
	abs_array(X->data,V->data,nelems);
	
	// compute l1,oo norm
	double l1inf = l1_inf_norm( X );

	// return zero solution
	if(l1inf < lambda) {
		for(ii=0;ii<nelems;ii++)
			X->data[ii] = 0.0;
		return;
	}

	// compoute lower bound and set of columns affected by thresholding
	indexed_double u[X->ncols];
	double t_lb = get_lower_bound(X,lambda,u,&M);
	
	// check that the computed lower bound is a feasible point
	if(t_lb < 0) {
		// create an array of max values
		double umax[M];
		// get the max over columns
		get_column_maxes(umax,X);
		// start from a feasible point
		t_lb = 0;
		for(ii=0;ii<M;ii++)
			// add maximums over columns
			t_lb += umax[ii];
		// compute new lower bound
		t_lb = (t_lb - lambda)/M;
	}

	// create indicator variable (J sets)
	char J[X->nrows*M];
	double psum[M];
	int nzJ[M];
	for(ii=0;ii<M;ii++){
		nzJ[ii] = X->nrows;
		psum[ii]= u[ii].val;
	}

	// initialize sets
	for(ii=0;ii<X->nrows*M;ii++)
		J[ii] = TRUE;

	// variables
	double t_new, w;
	char update_Ji[M];
	char J_converged;

	// initialization
	for(ii=0;ii<M;ii++)
		update_Ji[ii] = TRUE;
	
	// loop until convergence
	while(1) {
	
		// initialize variables
		J_converged = TRUE;

		// update the sets of non-zero elements
		for(ii=0;ii<M;ii++){
			if(update_Ji[ii])
				update_Ji[ii] = update_J_set(&J[ii*X->nrows], &psum[ii], &nzJ[ii], &X->data[u[ii].idx*X->nrows], X->nrows, t_lb);
			if(update_Ji[ii])
				J_converged = FALSE;
		}

		// once converged update t and M
		if(J_converged){
			
			// compute t
			t_new = get_t(psum,nzJ,M,lambda);

			// update M
			if(t_new >= u[M-1].val){
				M = M - 1;
				while(t_new >= u[M-1].val)
					M = M - 1;
				// update t
				t_new = get_t(psum,nzJ,M,lambda);
			}

			// check for convergence
			if(t_lb >= t_new){
				break;
			}

			// update lower bound
			t_lb = t_new;

			// reset variables
			for(ii=0;ii<M;ii++){
				update_Ji[ii] = TRUE;
				nzJ[ii] = X->nrows;
				psum[ii]= u[ii].val;
				offset = ii*X->nrows;
				for(jj=0;jj<X->nrows;jj++)
					J[offset+jj] = TRUE;
			}
		}
	}
	
	// compute proximal operator
	double mu;
	for(ii=0;ii<M;ii++){
		// compute mu
		mu = (psum[ii]-t_new)/nzJ[ii];
		offset = u[ii].idx * X->nrows;
		for(jj=0;jj<X->nrows;jj++){
			X->data[offset+jj] -= mu;
			if(X->data[offset+jj]<0)
				X->data[offset+jj] = 0.0;
		}
	}
	// restore the sign
	for(ii=0;ii<nelems;ii++){
		if(V->data[ii] < 0)
			X->data[ii] = -X->data[ii];
	}
}

/** Proximal operator for induced l1 norm */
void prox_l1_norm_column_sort( Matrix *X, Matrix *V, double lambda ) {

	// variables
	int ii, jj, offset;

	// compute absolute value
	int nelems = V->ncols * V->nrows;
	abs_array(X->data,V->data,nelems);
	
	// compute l1,oo norm
	double l1inf = l1_inf_norm( X );

	// return zero solution
	if(l1inf < lambda) {
		for(ii=0;ii<nelems;ii++)
			X->data[ii] = 0.0;
		return;
	}

	// compoute lower bound and set of columns affected by thresholding
	int M,N;
	indexed_double u[X->ncols];
	double t_lb = get_lower_bound(X,lambda,u,&M);

	// sort those columns affected by thresholding
	double U[V->nrows*M];
	for(ii=0;ii<M;ii++) {
		// copy data
		for(int jj=0;jj<X->nrows;jj++)
			U[ii*X->nrows+jj] = X->data[u[ii].idx * X->nrows + jj];
		// sort data
		qsort( &U[ii*X->nrows], X->nrows, sizeof(double), compare_array );
	}

	// check that the computed lower bound is a feasible point
	if(t_lb < 0) {
		// start from a feasible point
		t_lb = 0;
		for(ii=0;ii<M;ii++)
			// add maximums over columns
			t_lb += U[ii*X->nrows];
		// compute new lower bound
		t_lb = (t_lb - lambda)/M;
	}

	// create variables
	double psum[M]; 	// store partial sums
	int    nj[M];		// store number of non-zero elements
	// initialize variables
	for(ii=0;ii<M;ii++){
		psum[ii] = U[ii*X->nrows];
		nj[ii]   = 1;
	}
	
	double tnew;

	// optimization loop
	while(1){

		// waterfilling over columns
		for(ii=0;ii<M;ii++) {

			// offset to column
			offset = ii*X->nrows;

			// waterfilling
			while( nj[ii]<X->nrows ){
				if( ((psum[ii]-t_lb)/nj[ii] ) < U[offset+nj[ii]] ){
					// update
					psum[ii] += U[offset+nj[ii]];
					nj[ii]   += 1;
				}else{
					break;
				}
			}
		}

		// update t
		tnew = get_t(psum,nj,M,lambda);
		
		// exit condition
		if(tnew==t_lb)
			break;
		else{
			// update t
			t_lb = tnew;
			// update M
			while(tnew > u[M-1].val)
				M -= 1;
		}
	}

	double mu;
	// compute proximal operator
	for(ii=0;ii<M;ii++){
		// compute mu
		mu = (psum[ii]-tnew)/nj[ii];
		offset = u[ii].idx * X->nrows;
		for(jj=0;jj<X->nrows;jj++){
			X->data[offset+jj] -= mu;
			if(X->data[offset+jj]<0)
				X->data[offset+jj] = 0.0;
		}
	}
	// restore the sign
	for(ii=0;ii<nelems;ii++){
		if(V->data[ii] < 0)
			X->data[ii] = -X->data[ii];
	}
}

/** Evaluate optimal value of t */
double get_t(double *psum, int *nj, int M, double lambda) {
	int ii;
	double tnew = 0, beta = 0;
	
	for(ii=0;ii<M;ii++){
		tnew += psum[ii]/nj[ii];
		beta += 1.0/nj[ii];
	}
	tnew = (tnew-lambda)/beta;
	
	return tnew;
}

/** Maximize lower bound on the norm */
double get_lower_bound(Matrix *X, double lambda, indexed_double *u, int *M) {

	// variables
	int ii;
	double w=0,t_lb=-1,t=0;
	
	// compute l1 norms
	for(ii=0;ii<X->ncols;ii++) {
		u[ii].val = sum_array( &X->data[ii*X->nrows], X->nrows );
		u[ii].idx = ii;
	}
	
	// sort the l1 norms
	qsort(u, X->ncols, sizeof(indexed_double), compare_iarray);
	
	// compute a lower bound
	w    = u[0].val; *M = 1;
	t_lb = (w - X->nrows*lambda);
	
	for(ii=1;ii<X->ncols;ii++) {

		w += u[ii].val;
		t = (w - X->nrows*lambda)/(ii+1);
		// store the maximum
		if(t_lb <= t){
			t_lb = t;
			*M   = ii+1;
		}
	}

	// return value
	return t_lb;
}

/* Update set of non-zero entries */
char update_J_set( char *J, double *psum, int *Ji, double *x, int n, double t_lb ){
	int ii;
	double mu;
	char updated = FALSE;

	// compute mu
	mu = (*psum - t_lb)/(*Ji);
	
	// update the sets
	for(ii=0;ii<n;ii++){
		if(J[ii] == TRUE){
			if(x[ii] <= mu){
				J[ii] = FALSE;
				*Ji -= 1;
				*psum -= x[ii];
				updated = TRUE;
			}
		}
		if(J[ii] == FALSE){
			if(x[ii]>mu)
				printf("error\n");
		}

	}
	//printf("mu %f nj %d psum %f\n",mu, *Ji, *psum);
	// return value
	return updated;
}

/** Get max over columns */
void get_column_maxes( double *umax, Matrix *X ) {
	int ii, jj, offset;
	double mval;

	// loop over Matrix elements to find the column-wise maximums
	for(ii=0;ii<X->ncols;ii++){
		offset   = ii*X->nrows;
		umax[ii] = X->data[offset];
		for(jj=1;jj<X->nrows;jj++){
			mval = X->data[offset+jj];
			if(umax[ii] < mval)
				umax[ii] = mval;
		}
	}
	return;
}

/** Sum array of double values */
double sum_array( double *input, int nelems) {
	int ii;
	double result = 0.0;

	for(ii=0;ii<nelems;ii++)
		result += input[ii];
	
	// return value
	return result;
}

/** Compare arrays of indexed doubles */
int compare_iarray (const void * a, const void * b)
{
    if ( (((indexed_double *)a)->val - ((indexed_double *)b)->val) > 0)
    	return -1;
    if ( (((indexed_double *)a)->val - ((indexed_double *)b)->val) < 0)
    	return 1;
    return 0;
}

/** Compare array of doubles */
int compare_array (const void * a, const void * b)
{
    if ( ( *((double *)a) - *((double *)b) ) > 0)
    	return -1;
    if ( ( *((double *)a) - *((double *)b) ) < 0)
    	return 1;
    return 0;
}

/** Compute l1,oo norm, the matrix
 is assumed to be non-negative */
double l1_inf_norm( Matrix *X ) {
	
	int ii, jj, offset;
	double t = 0, maxval;

	// get the max over columns
	for(jj=0;jj<X->ncols;jj++) {
		offset = jj*X->nrows;
		maxval = X->data[offset];
		for(ii=1;ii<X->nrows;ii++) {
			if( X->data[offset + ii] > maxval )
				maxval = X->data[offset+ii];
		}
		// add column max
		t += maxval;
	}
	// return l1,oo norm
	return t;
}

/** Compute absolute value of an array */
void abs_array(double *res, double *input, int nelems) {

	int ii;
	for(ii=0;ii<nelems;ii++)
		res[ii] = fabs(input[ii]);

}

/** Print values of a matrix */
void print_matrix(Matrix *A) {

	int ii, jj;
	printf("\n");
	for(ii=0;ii<A->nrows;ii++) {
		printf("[ ");
		for(jj=0;jj<A->ncols;jj++) {
			printf("%.2f ",A->data[ii+jj*A->nrows]);
		}
		printf("]\n");
	}
	printf("\n\n");
}

/** Allocate memory for array */
Matrix create_matrix(int nrows, int ncols) {

	// create variable
	Matrix A;
	// allocate memory for the array
	A.data = new double[nrows * ncols];
	// assigm dimensions
	A.ncols = ncols;
	A.nrows = nrows;
	// return value
	return A;
}

/** Release memory */
void destroy_matrix(Matrix *A) {

	delete [] A->data;
}

}  // namespace Bejar

inline void Bejar1(double *y, double *x, const int nrows, const int ncols,
                     const double C) {
  bejar::Matrix Y;
  Y.data = y;
  Y.ncols = ncols;
  Y.nrows = nrows;

  bejar::Matrix X;
  X.data = x;
  X.ncols = ncols;
  X.nrows = nrows;

  bejar::prox_l1_norm_column_sort(&Y, &X, C);
}


inline void Bejar2(double *y, double *x, const int nrows, const int ncols,
                     const double C) {
  bejar::Matrix Y;
  Y.data = y;
  Y.ncols = ncols;
  Y.nrows = nrows;

  bejar::Matrix X;
  X.data = x;
  X.ncols = ncols;
  X.nrows = nrows;

  bejar::prox_l1_norm_active_set(&Y, &X, C);
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_BEJAR_HPP */
