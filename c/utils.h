#ifndef _utils_h
#define _utils_h

// NOTE, to use MKL LAPACK define _MKL_USE
#ifdef _MKL_USE
#include "mkl_lapacke.h"
#endif


#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_cdf.h>


int set_matrix_block(gsl_matrix * dest,gsl_matrix *source,int left,int top){
	int i,j;
	int right = source->size1 + left;
	int bottom = source->size2 + top;
	for(i=left;i<right;i++){
		for(j=top;j<bottom;j++){
			double sourceij = gsl_matrix_get(source,i-left,j-top);
			gsl_matrix_set(dest,i,j,sourceij);
		}
	}
	return 0;
}
int outer(gsl_vector * a,gsl_vector* b, gsl_matrix* C){
	// computes a*b' = C
	int Na,Nb,r,c;
	Na = a->size;
	Nb = b->size;
	if (C->size1!=Na || C->size2!=Nb){
		printf("Matrix C is the wrong size, Na,Nb=(%d,%d), C=(%d,%d)\n",Na,Nb,(int)C->size1,(int)C->size2);
		return 1;
	}
	for(r=0;r<Na;r++){
		for(c=0;c<Nb;c++){
			double prod = gsl_vector_get(a,r)*gsl_vector_get(b,c);
			gsl_matrix_set(C,r,c,prod);
		}
	}
	return 0;
}

int find_eigens(gsl_matrix * subject, gsl_vector * eigenvals, gsl_matrix * eigenvecs){
	// note that subject is destroyed in here!
	int status=0;
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(subject->size1);
	status += gsl_eigen_symmv(subject, eigenvals, eigenvecs, w);
	gsl_eigen_symmv_free(w);
	//gsl_matrix_free(subject);
	return status;
}

double norm1mat(gsl_matrix *m){
	// sum of colmumn norms
	int nc = m->size2;
	int nr = m->size1;
	int c,r;
	double csum,mn = 0.0;
	for (c=0;c<nc;c++){
		csum = 0.0;
		for(r=0;r<nr;r++)
			csum+= fabs(gsl_matrix_get(m,r,c));
		mn = csum>mn ? csum : mn;
	}
	return mn;
}

int kron(int TransA,int TransB,gsl_matrix* A,gsl_matrix* B,int TransC,gsl_matrix* C){
//computes kronecker products of A \oprod B = C
	int rA,rB,cA,cB, ri1,ri2,ci1,ci2;
	rA = TransA==0 ? A->size1 : A->size2;
	cA = TransA==0 ? A->size2 : A->size1;
	rB = TransB==0 ? B->size1 : B->size2;
	cB = TransB==0 ? B->size2 : B->size1;

	if(rA*rB == C->size1 && cA*cB == C->size2){

	if (TransC == 0){
		for (ri1 = 0;ri1<rA;ri1++){
			for(ci1=0;ci1<cA;ci1++){

		for (ri2 = 0;ri2<rA;ri2++){
			for(ci2=0;ci2<cA;ci2++){
				gsl_matrix_set(C,rB*ri1 + ri2,cB*ci1 + ci2, gsl_matrix_get(A,ri1,ci1)*gsl_matrix_get(B,ri2,ci2) );
			}
		}

			}
		}
	}
	else{
		for (ri1 = 0;ri1<rA;ri1++){
			for(ci1=0;ci1<cA;ci1++){

		for (ri2 = 0;ri2<rA;ri2++){
			for(ci2=0;ci2<cA;ci2++){
				gsl_matrix_set(C,cB*ci1 + ci2, rB*ri1 + ri2, gsl_matrix_get(A,ri1,ci1)*gsl_matrix_get(B,ri2,ci2) );
			}
		}

			}
		}	
	}
	return 0;
	}
	else{
	return 1;
	}
}



double cov(gsl_vector* x1, gsl_vector * x2){
	double m1,m2,len,cov;
	int ri;
	if(x1->size != x2->size){
		printf("Cov vectors are the wrong size");
		return 0.0;
	}
	len = (double)x1->size;
	m1=0.0; m2=0.0;cov=0.0;
	for(ri =0;ri<x1->size;ri++)
		m1 += x1->data[ri] / len;
	for(ri =0;ri<x2->size;ri++)
		m2 += x2->data[ri] / len;
	for(ri =0;ri<x2->size;ri++)
		cov +=(x2->data[ri] -m2)*(x1->data[ri] -m1);
	cov /= len;
	return cov;
}

int WOLS(const gsl_vector* y, const gsl_matrix* x,const gsl_matrix* w, gsl_vector * coef, gsl_vector * e){
	int status =0;
	int r = x->size1;
	int c = x->size2;
	gsl_set_error_handler_off();
	if (r<2){
		gsl_vector_set(coef,0,0.0);
		gsl_vector_set(coef,1,0.0);
		return 0;
	}

	gsl_matrix * xx 	= gsl_matrix_alloc(c,c);
	gsl_vector * xpwy	= gsl_vector_alloc(c);
	gsl_vector * p 	= gsl_vector_alloc (c);
	gsl_matrix * xpw		= gsl_matrix_alloc(c,r);

	gsl_blas_dgemm( CblasTrans, CblasNoTrans,1.0,x,w,0.0,xpw);
	gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,1.0,xpw,x,0.0,xx);
	status = gsl_linalg_QR_decomp(xx, p);
	status+= gsl_blas_dgemv (CblasNoTrans, 1.0, xpw, y, 0.0, xpwy);
	status+= gsl_linalg_QR_solve(xx, p, xpwy,coef);
	gsl_blas_dgemv(CblasNoTrans,1.0,x,coef,0.0,e);

	gsl_vector_free(p);
	gsl_matrix_free(xx);
	gsl_vector_free(xpwy);
	gsl_matrix_free(xpw);

	return status;
}


int WOLS_LU(const gsl_vector* y, const gsl_matrix* x,const gsl_matrix* w, gsl_vector * coef, gsl_vector * e){
	int status =0,s;
	int r = x->size1;
	int c = x->size2;
	gsl_set_error_handler_off();
	if (r<2){
		gsl_vector_set(coef,0,0.0);
		gsl_vector_set(coef,1,0.0);
		return 0;
	}

	gsl_matrix * xx 	= gsl_matrix_alloc(c,c);
	gsl_vector * xpwy	= gsl_vector_alloc(c);
	gsl_permutation * p 	= gsl_permutation_alloc (c);
	gsl_matrix * xpw		= gsl_matrix_alloc(c,r);
	
	gsl_blas_dgemm( CblasTrans, CblasNoTrans,1.0,x,w,0.0,xpw);
	gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,1.0,xpw,x,0.0,xx);
	status = gsl_linalg_LU_decomp(xx, p, &s);
	status+= gsl_blas_dgemv (CblasNoTrans, 1.0, xpw, y, 0.0, xpwy);
	status+= gsl_linalg_LU_solve(xx, p, xpwy,coef);
	gsl_blas_dgemv(CblasNoTrans,1.0,x,coef,0.0,e);	
	gsl_permutation_free(p);

	gsl_matrix_free(xx);
	gsl_vector_free(xpwy);
	gsl_matrix_free(xpw);

	return status;
}

int OLS(const gsl_vector* y, const gsl_matrix* x, gsl_vector * coef, gsl_vector * e){
	int status =0,s;
//	gsl_matrix * W = gsl_matrix_alloc(x->size1,x->size1);
//	gsl_matrix_set_identity(W);
//	status = WOLS_LU(y,x,W,coef,e);

//	gsl_matrix_free(W);
	int r = x->size1;
	int c = x->size2;

	if (r<2){
		gsl_vector_set(coef,0,0.0);
		gsl_vector_set(coef,1,0.0);
		return 0;
	}

	gsl_vector * xpy	= gsl_vector_alloc(c);
	gsl_matrix * xx 	= gsl_matrix_alloc(c,c);
	gsl_permutation * p 	= gsl_permutation_alloc (c);

	gsl_blas_dgemm( CblasTrans, CblasNoTrans,1.0,x,x,0.0,xx);
	status = gsl_linalg_LU_decomp(xx, p, &s);
	status+= gsl_blas_dgemv (CblasTrans, 1.0, x, y, 0.0, xpy);
	status+= gsl_linalg_LU_solve(xx, p, xpy,coef);
	gsl_blas_dgemv(CblasNoTrans,1.0,x,coef,0.0,e);
	gsl_permutation_free(p);

	gsl_matrix_free(xx);
	gsl_vector_free(xpy);

	return status;
}

int inv_wrap(gsl_matrix * invx, gsl_matrix * x){
	int r = x->size1;
	int c = x->size2;
	int status =0,s;

#ifdef _MKL_USE
	int* ipiv = (int*)malloc(sizeof(int)*r);
	double * xdat;
	if(invx->tda!=c){// will have to recopy into a row-major array
		xdat = (double*)malloc(r*c*sizeof(double));
		int ri,ci;
		for(ri=0;ri<r;ri++){
			for(ci=0;ci<c;ci++)
				xdat[ri*c+ci]=x->data[ri*x->tda + ci];
		}
	}
	else{
		gsl_matrix_memcpy(invx,x);
		xdat = invx->data;
	}
	status += LAPACKE_dgetrf( LAPACK_ROW_MAJOR, r, c, xdat, c, ipiv );
	status += LAPACKE_dgetri( LAPACK_ROW_MAJOR, c, xdat, c, ipiv );
	if(invx->tda!=c){
		gsl_matrix_view invvw = gsl_matrix_view_array(xdat,r,c);
		gsl_matrix_memcpy(invx,&invvw.matrix);
		free(xdat);
	}
	free(ipiv);
#endif


#ifndef _MKL_USE
	gsl_permutation * p 	= gsl_permutation_alloc (c);
	status = gsl_linalg_LU_decomp(x, p, &s);
	status = gsl_linalg_LU_invert(x, p, invx);
	gsl_permutation_free(p);
#endif

	return status;
}

int pinv_sol(gsl_vector * x, const gsl_matrix * A, const gsl_vector * b){
	int status=0;
	int Nr,Nc,ri,ci;
	gsl_matrix * U, * V;
	gsl_vector * S, *work;
	
	Nr = A->size1;
	Nc = A->size2;
	
	U 	= gsl_matrix_alloc(Nr,Nc);
	V 	= gsl_matrix_alloc(Nc,Nc);
	S 	= gsl_vector_alloc(Nc);
	work 	= gsl_vector_alloc(Nc);

	gsl_matrix_memcpy(U,A);
	
	if(Nr>Nc){
		gsl_matrix * Xwork  = gsl_matrix_alloc(Nc,Nc);
		status += gsl_linalg_SV_decomp_mod(U, Xwork, V, S, work);
		gsl_matrix_free(Xwork);
	}
	else
		status += gsl_linalg_SV_decomp(U, V, S, work);	
	status += gsl_linalg_SV_solve(U,V,S,b,x);
	
	gsl_vector_free(S);
	gsl_matrix_free(V);
	gsl_matrix_free(U);
	gsl_vector_free(work);
	
	return status;
}

int sol_Axb_tri(gsl_matrix * A, gsl_vector * b, gsl_vector * x, gsl_permutation * p){
	// if tri==1 then assumes already LU decomp
	int s,status=0;
	int r = A->size1;
	int c = b->size;
	gsl_matrix* AA;

	if(r!=c)
		status = 1;
	int tri = 1;
	if(p==0){
		tri =0;
		p = gsl_permutation_alloc (c);
		AA = gsl_matrix_alloc(r,c);
		gsl_matrix_memcpy(AA,A);
		status = gsl_linalg_LU_decomp (AA, p, &s);
		status += gsl_linalg_LU_solve (AA, p, b, x);
	}
	else
		status = gsl_linalg_LU_solve (A, p, b, x);

	if(tri!=1){
		gsl_permutation_free (p);
		gsl_matrix_free(AA);
	}
	return status;
}
int sol_Axb(gsl_matrix * A, gsl_vector * b, gsl_vector * x){
	// this is just a wrapper that assumes that not already triangularized A
	int status = sol_Axb_tri(A, b, x, 0);
	return status;
}

int sol_AXM(gsl_matrix * A, gsl_matrix * M, gsl_matrix *X, void* tri_v){
	// if int tri==0 (null pointer) then assume not yet LU decomposed
	int status, r, c, rm,cm;
	r = A->size1;
	c = A->size2;
	rm= M->size1;
	cm= M->size2;



#ifndef _MKL_USE

	gsl_matrix * AA;
	gsl_permutation * p = (gsl_permutation*)tri_v;
	int s;
	AA = gsl_matrix_calloc(r,c);
	gsl_matrix_memcpy(AA,A);
	status = gsl_linalg_LU_decomp (AA, p, &s);

	int ci;
	for(ci=0;ci<cm;ci++){
		gsl_vector_view x = gsl_matrix_column(X,ci);
		gsl_vector_view b = gsl_matrix_column(M,ci);
		status = gsl_linalg_LU_solve (A, p, &b.vector, &x.vector);
	}
	gsl_matrix_free(AA);

#endif

#ifdef _MKL_USE
	int* tri = (int*)tri_v;
	int * ipiv;
	double *xdat,*ydat;
	if(A->tda!=c){// will have to recopy into a row-major array
		xdat = (double*)malloc(r*c*sizeof(double));
		int ri,ci;
		for(ri=0;ri<r;ri++){
			for(ci=0;ci<c;ci++)
				xdat[ri*c+ci]=gsl_matrix_get(A,ri,ci);
		}
	}
	else
		xdat = A->data;
	if(M->tda!=c){// will have to recopy into a row-major array
		ydat = (double*)malloc(r*c*sizeof(double));
		int ri,ci;
		for(ri=0;ri<rm;ri++){
			for(ci=0;ci<cm;ci++)
				ydat[ri*cm+ci]=gsl_matrix_get(M,ri,ci);
		}
	}
	else
		ydat = M->data;
	if(tri== 0){
		ipiv = (int*)malloc(sizeof(int)*c);
		status =LAPACKE_dgetrf(LAPACK_ROW_MAJOR, r,c, xdat, c,ipiv );
	}
	else
		ipiv = tri;

	status += LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', rm, c, xdat, c, ipiv, ydat, cm);
	if(A->tda!=c)
		free(xdat);
	if(M->tda!=c)
		free(ydat);

#endif


	return status;
}

int pca(const gsl_matrix * data, const int dimensions_we_want, const int demean, const int stdize,gsl_matrix * pc_space, gsl_matrix * projected){
	int ds = data->size2;
	int i,c,status=0;
	gsl_matrix * x;
	x = gsl_matrix_calloc(data->size1,data->size2);

	if (demean>0){
		gsl_matrix_memcpy(x,data);
		for(i=0;i<data->size1;i++){
			double rmean = 0.0;
			for (c=0;c<data->size2;c++)
				rmean += data->data[i*data->tda + c];
			rmean /= (double)data->size2;
			for (c=0;c<data->size2;c++)
				x->data[i*data->tda + c] -= rmean;
		}
	}
	else
		gsl_matrix_memcpy( x,data);
	if(stdize >0){
		for(i=0;i<data->size1;i++){
			double rvar =  0.;
			double rmean = 0.0;
			for (c=0;c<data->size2;c++)
				rmean += data->data[i*data->tda + c];
			rmean /= (double)data->size2;
			for (c=0;c<data->size2;c++)
				rvar += pow(data->data[i*data->tda + c] -rmean ,2);
			rvar /= (double)data->size2;
			for (c=0;c<data->size2;c++)
				x->data[i*data->tda + c] /= pow(rvar,0.5);
		}
	}

	gsl_matrix *cov_matrix = gsl_matrix_calloc(ds, ds);
	gsl_vector *eigenvals = gsl_vector_alloc(ds);
	gsl_matrix *eigenvecs = gsl_matrix_alloc(ds, ds);
	status += gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0, x, x, 0.0, cov_matrix);
	status += find_eigens(cov_matrix, eigenvals, eigenvecs);

	status += gsl_eigen_symmv_sort(eigenvals, eigenvecs,GSL_EIGEN_SORT_ABS_DESC);


	for (i=0;i<dimensions_we_want; i++){
		gsl_vector_view evec = gsl_matrix_column(eigenvecs, i);
		status += gsl_matrix_set_col(pc_space, i, &evec.vector);
	}
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0, x, pc_space, 0.0, projected);

	gsl_matrix_free(cov_matrix);
	gsl_vector_free(eigenvals);
	gsl_matrix_free(eigenvecs);
	gsl_matrix_free(x);
	return status;
}



int isfinmat(gsl_matrix * mat){
	// returns 0 if entire matrix is finite and o.w. the number of non-finite entries
	int r,c,Nr,Nc,status;
	status = 0;
	Nr = mat->size1;
	Nc = mat->size2;

	#pragma omp parallel for default(shared) private(r,c)
	for(r=0;r<Nr;r++){
		for(c=0;c<Nc;c++){
			if(gsl_finite(mat->data[r*mat->tda +c])==0){
				mat->data[r*mat->tda +c] = 0.0;
				status++;
			}
		}
	}
	return status;
}

int isfinvec(gsl_vector * vec){
	int r,Nr,status;
	status =0;
	Nr = vec->size;
	//#pragma omp parallel for default(shared) private(r)
	for(r=0;r<Nr;r++){
		if(gsl_finite(vec->data[r])==0){
			vec->data[r] = 0.0;
			status++;
		}
	}
	return status;
}

void printmat(char* name, gsl_matrix* mat){
	FILE* matfile;int yi,bi;
	int rows = mat->size1;
	int cols = mat->size2;
	matfile = fopen(name, "w");

	for(yi=0;yi<rows;yi++){
		for(bi=0;bi<cols-1;bi++){
			fprintf(matfile,"%f,",gsl_matrix_get(mat,yi,bi));
		}
		fprintf(matfile,"%f\n",gsl_matrix_get(mat, yi,cols-1));
	}
	printf("Printing matrix to, %s\n",name);
	fclose(matfile);
}
void printvec(char* name, gsl_vector* vec){
	FILE* matfile;int ri;
	int rows = vec->size;
	matfile = fopen(name, "w");

	for(ri=0;ri<rows;ri++){
		fprintf(matfile,"%f\n",gsl_vector_get(vec,ri));
	}
	printf("Printing matrix to, %s\n",name);
	fclose(matfile);
}

int readmat(char* name, gsl_matrix * mat){
	int status,Nr,Nc,r,c,rstatus,trashstatus;
	double trash;
	
	Nc = (int) mat->size2;
	Nr = (int) mat->size1;
	status =0;
	FILE * f;
	f = fopen(name,"r");
	if(f == NULL){
		status = 1;
		gsl_matrix_set_zero(mat);
		printf("Could not read from %s",name);
	}
	else{
		double dd;
		for(r=0;r<Nr;r++){
			for(c=0;c<Nc-1;c++){
				rstatus = fscanf(f,"%lf,",&dd);
				if(rstatus==0){
					rstatus = fscanf(f,"%lf\n",&dd);
				}
				if(rstatus==0)
					printf("Error, did not read from %s\n",name);

				gsl_matrix_set(mat,r,c,(double) dd);
				
			}			
			rstatus = fscanf(f,"%lf\n",&dd);
			/* if this doesn't work, then keep reading --- This does not work
			if(rstatus==0){
				rstatus = fscanf(f,"%lf,",&dd);
				if(rstatus>0){// it wasn't the end of the line, so now read until the end of the line
					for(c=0;c<100;c++){
						trashstatus = fscanf(f,"%lf,",&trash);
						if(trashstatus ==0){
							trashstatus = fscanf(f,"%lf\n",&trash);
							break; //came to the end
						}
					}
				}
			}*/
			if(rstatus==0)
				printf("Error, did not read from %s\n",name);

			gsl_matrix_set(mat,r,Nc-1,dd);
		}
		

		fclose(f);
	}
	return status;
}

int readvec(char* name, gsl_vector * vec){
	int status,N,i,rstatus;

	N = (int) vec->size;

	status =0;
	FILE * f;
	f = fopen(name,"r");
	status = f == NULL ? 1 : 0;
	float dd;
	for(i=0;i<N;i++){
		rstatus = fscanf(f,"%f\n",&dd);
		if(rstatus==0){
			printf("Error, did not read from %s",name);
		}
		gsl_vector_set(vec,i,(double)dd);
	}
	fclose(f);
	return status;
}


void vec(gsl_matrix * mat, gsl_vector * vec_ret){
// stacks by column
// requires 2x memory, as this is not a copy in place transformation
	int r,c;
	for(c=0;c < mat->size2;c++){
		for(r=0; r< mat->size1; r++)
			gsl_vector_set(vec_ret,r+c*(mat->size1),
				gsl_matrix_get(mat,r,c)); 
	}

}

void vech(gsl_matrix * mat, gsl_vector * vech_ret ){
// stacks by column, only using upper triangle of mat
// requires 2x memory, as this is not a copy in place transformation

	int r,c;
	for(c=0;c < mat->size2;c++){
		for(r=0; r< c+1; r++)
			gsl_vector_set(vech_ret,r+c*(mat->size1),
				gsl_matrix_get(mat,r,c)); 
	}
}

int randn(gsl_matrix * mat,unsigned long int seed){
	// fills mat with i.i.d random normal draws
	int r,c,Nr,Nc,status;
	gsl_rng * rng;
	double draw;
	Nr = mat->size1;
	Nc = mat->size2;
	//gsl_rng_env_setup(); // Reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED

	rng  = gsl_rng_alloc(gsl_rng_mt19937);
	status = 0;
	gsl_rng_set (rng, seed);
	for(r=0;r<Nr;r++){
		for(c=0;c<Nc;c++){
			draw = gsl_ran_gaussian_ziggurat(rng,1.0);
			gsl_matrix_set(mat,r,c,draw);
		}
	}

	gsl_rng_free(rng);
	return status;
}

#endif
