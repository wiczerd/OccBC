/* This program solves my dissertation by a hybrid k=order perturbation with an optimal COV and exact nonlinear policies
*  First, it solves the first order approximation by some straight forward linear algebra.  This can be used then for the 2nd, too.
*  For more on the optimal COV see J F-V J R-R (JEDC 2006)
*  The hybrid portion involves using the exactly computed decision rules given approximated state-dynamics: see Maliar, Maliar and Villemot (Dynare WP 6)
*
*  au : David Wiczer
*  mdy: 2-20-2011
*  rev: 6-21-2012
*
* compile line: icc /home/david/Documents/CurrResearch/OccBC/program/c/occBC_hybr.c -static -lgsl -openmp -mkl=parallel -O3 -lnlopt -I/home/david/Computation -I/home/david/Computation/gsl/gsl-1.15 -o occBC_hybr.out
*/

//#define _MKL_USE // this changes LAPACK options to use faster MKL versions of LAPACK
//#define _DFBOLS_USE
//#define _MPI_USE



#ifdef _MKL_USE
#include "mkl_lapacke.h"
#endif
#ifdef _MPI_USE
#include <mpi.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "utils.h"
#include <math.h>
#include <time.h>
#include <nlopt.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_errno.h>



int Nthread 	= 12;
int Nsolerr		= 0;
int Nobs,simT, nsim, Nskill;
int nshock;
int neq;
int verbose 	= 3;
int printlev 	= 3;
int use_anal	= 1;
int opt_alg		= 3; // 10x => Nelder-Meade, 5x => Subplex, 3x => DFBOLS, o.w => BOBYQA
int polish_alg	= 0;
double ss_tol	= 1e-8; // tighten this with iterations?
double dyn_tol	= 1e-8; // ditto?
int homosk_zeta	= 0;
int diag_zeta	= 1;
int gsl_fin_diffs=0;
int csize		= 8;
FILE * calhist, *solerr,*simerr ;
char soler_f[] = "solerrX.log";
char calhi_f[] = "calhistX.txt";
char simer_f[] = "simerrX.log";

int jac_ct=0;

double * dim_scale;

double shist[]  = {1.0,1.0,1.0,1.0,1.0,1.0};


//declare some parameters
int const Noccs	= 22;
int const Nfac	= 2;// how many factors in the z_i model?
int const Nllag	= 0;// how many lags in lambda
int const Nglag	= 1;// how many lags in gamma
int const Nagf	= 1;// Z with coefficients beyond just the 1
int Ns,Nc,Nx;
int Notz;

// checks on internals
int gpol_zeros = 0;
int t_zeros = 0;

double 	beta	= 0.9967;	// monthly discount
double	nu 		= 0.0; 		// switching cost --- I don't need this!
double 	fm_shr	= 0.67;	// firm's surplus share
double 	kappa	= 0.27;//.1306; corresponds to HM number
double *	b; // unemployment benefit
double 	brt		= 0.42;
double 	sbar	= 0.02;	// will endogenize this a la Cheremukhin (2011)


double ** chi;

//0.497773,0.006930,0.031176,0.049722,0.089230,1.170998,
//0.742965,0.005235,0.046462,0.022331,0.188347,0.500000,-3.323152,-3.503544,-2.079600,
//0.436605,0.023969,0.050260,0.032846,0.091122,0.953266,
//0.643555,0.035170,0.046250,0.023750,0.151250,0.500000,-1.242500,-3.747500,-3.747500


double 	phi		= 0.647020;	// : log(1/UE elasticity -1) = phi*log(theta)= log(1/.38-1)/log(5)
double 	sig_psi	= 0.014459; 	// higher reduces the switching prob
double 	tau 	= 0.041851;
double 	scale_s	= 0.026889;
double 	shape_s	= 0.194987;
double 	effic	= 0.495975;
double 	chi_co[]= {-0.006211,-0.038432,-2.717312};


double rhoZ		= 0.9732;
double rhozz	= 0.99;

double sig_eps	=  1.2610e-06;//0.00008557;//0.00002557;//7.2445e-7;  using mine, not FED Numbers
double sig_zet	= 1e-5;//1.0e-6; (adjustment for the FRED number)

gsl_vector * cov_ze;
gsl_matrix * GammaCoef, *var_eta; 	//dynamics of the factors
gsl_matrix * LambdaCoef,*var_zeta;	//factors' effect on z
gsl_matrix * LambdaZCoef;			//agg prod effect on z
gsl_matrix * f_skills;				//skills attached to occupations

// data moments that are not directly observable
double avg_fnd	= 0.3323252;
double avg_urt	= 0.06;
double avg_wg	= 0.006;	// this is the 5 year wage growth rate: make tau fit this
double avg_elpt	= 0.48;		// elasticity of p wrt theta, Barichon 2011 estimate
double avg_sdsep= 0.01256;	//the average standard deviation of sld across occupations
double med_dr	= 13;
double chng_pr	= 0.45;
double sk_wg[]	= {-0.330679,-0.2942328,-0.185577}; //{-0.610,-0.631,-0.399}; When in simplex (not sure why that regression worked)
double Zfcoef[] = {0.0233,-0.0117};
			// old : {-0.0028,-0.0089,0.0355,0.0083};

struct aux_coef{
	gsl_matrix * data_moments;
	gsl_matrix * moment_weights;
	gsl_matrix * draws;
	gsl_matrix * Xdat;
};


struct dur_moments{
	double	pr6mo;
	double	sd_dur;
	double	E_dur;
	double*	dur_l;
	double	chng_wg;
	gsl_matrix * dur_qtl;
	gsl_vector * Ft;
	gsl_vector * Ft_occ;
	gsl_vector * dur_hist;
	gsl_matrix * dur_l_hist;
};

struct sys_coef{
	gsl_matrix * A0, *A1, *A2, *A3;
	gsl_matrix * F0, *F1, *F2, *F3;
	gsl_matrix * N,*S;
	gsl_vector * Ul0;
	double * COV; // array of change-of-variables powers
};

struct sys_sol{
	gsl_matrix *P0, *P1, *P2;
	gsl_matrix *PP;
	gsl_matrix *gld,*tld,*sld,*ss_wld;
};
struct st_wr{
	struct sys_coef * sys;
	struct sys_sol * sol;
	struct aux_coef * sim;
	int cal_set,d;
	unsigned n;
	const double * x;
	gsl_vector * ss;
	gsl_matrix * xss;
	nlopt_opt opt0;
};

struct st_wr * g_st;

// Solve the steady state
int sol_ss(gsl_vector * ss, gsl_vector * Zz,gsl_matrix * xss, struct sys_sol *sol);

// Solving the dynamic model
int sol_shock(struct st_wr * st);
int sol_dyn(gsl_vector * ss, struct sys_sol * sol, struct sys_coef * sys);
int sys_st_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx);
int sys_co_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx);
int sys_ex_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco);
int sys_def(gsl_vector * ss, struct sys_coef *sys);

// Policies
int gpol(gsl_matrix * gld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int spol(gsl_matrix * sld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int theta(gsl_matrix * tld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int xprime(gsl_matrix * xp, gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_matrix * x, const gsl_vector * Zz);

// results:
double endog_std(double stdZz,gsl_matrix * x0, struct st_wr * st, int pos);
int ss_moments(struct aux_coef * ssdat, gsl_vector * ss, gsl_matrix * xss);
int sim_moments(struct st_wr * st, gsl_vector * ss,gsl_matrix * xss);
int dur_dist(struct dur_moments * s_mom, const gsl_matrix * gld_hist, const gsl_matrix * pld_hist, const gsl_matrix * x_u_hist);

int TGR(gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		double *urt, double *opt_urt,
		struct st_wr * st);

// Utilities
double gsl_endog_std(double x, void *p);
int Es_dyn(gsl_vector * Es, const gsl_vector * ss_W, const gsl_vector * Wlast ,const gsl_matrix * invP0P1, const gsl_matrix * invP0P2, const gsl_vector * Zz);
int Es_cal(gsl_vector * Es, const gsl_vector * ss_W, const gsl_matrix * PP, const gsl_vector * Zz);
int set_params(const double * x,int cal_set);
int alloc_econ(struct st_wr * st);
int free_econ(struct st_wr * st);
int clear_econ(struct st_wr *st);


double qmatch(const double theta);
double pmatch(const double theta);
double dqmatch(const double theta);
double dpmatch(const double theta);
double invq(const double q);
double invp(const double p);
double dinvq(const double q);

// old for solving
int QZwrap(gsl_matrix * A, gsl_matrix *B, //gsl_matrix * Omega, gsl_matrix * Lambda,
		gsl_matrix * Q, gsl_matrix * Z, gsl_matrix * S, gsl_matrix * T);
int QZdiv(double stake, gsl_matrix * S,gsl_matrix * T,gsl_matrix * Q,gsl_matrix * Z);
int QZswitch(int i, gsl_matrix * A,gsl_matrix * B,gsl_matrix * Q,gsl_matrix * Z);
void VARest(gsl_matrix * Xdat,gsl_matrix * coef, gsl_matrix * varcov);

// calibration functions
double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg0, int alg1);
double cal_dist_df(unsigned n, const double *x, double *grad, void* params);
double cal_dist(unsigned n, const double *x, double *grad, void* params);
int cal_dist_wrap(const gsl_vector * x, void* params, gsl_vector * f);
void dfovec_iface_(double * f, double * x, int * n);

double quad_solve(unsigned n, const double *x, double *grad, void*params){
	// to test the multi-start
	int i;
	double obj=0.0;
	for(i=0;i<n;i++)
		obj+= (double)(i+1) * pow(x[i],2);
	return obj;
}

int main(int argc,char *argv[]){
	// pass 0 to calibrate the whole thing, 1 for just calibration, 2 for wage parameters and 3+ for whole thing.
	// <0 does not calibrate
	int i,l,d,status,calflag;
 	gsl_set_error_handler_off();
	if(argc<=1)
		calflag = -1;
	if(argc >=2)
		calflag = atoi(argv[1]);
	if(argc >=3)
		opt_alg = atoi(argv[2]);
	if(argc >=4)
		polish_alg = atoi(argv[3]);
	status = 0;

	omp_set_num_threads(Nthread);
	//initialize parameters
	Nskill = 3;
#ifdef _MKL_USE
	printf("Begining, Calflag = %d, USE_MKL=%d,USE_DFBOLS=%d\n",calflag,_MKL_USE,_DFBOLS_USE);
#endif

	f_skills = gsl_matrix_alloc(Noccs,6);
	readmat("rf_skills.csv",f_skills);

	if(printlev>=2)	printmat("Readf_skills.csv" ,f_skills);
	FILE * readparams = fopen("params_in.csv","r+");
	fscanf(readparams,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&phi,&sig_psi,&tau,&scale_s,&shape_s,&effic,&chi_co[0],&chi_co[1],&chi_co[2]);
	fclose(readparams);
	printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",phi,sig_psi,tau,scale_s,shape_s,effic,chi_co[0],chi_co[1],chi_co[2]);
	b	= malloc(sizeof(double)*2);
	chi = malloc(sizeof(double*)*Noccs+1);
	for(l=0;l<Noccs+1;l++)
		chi[l] = malloc(sizeof(double*)*Noccs);
	// this is just to initialize
	double chi_coef0[] ={-3.323152,-3.503544,-2.079600};
		/*	{ // these come from the solution in matlab, where weights are equal for all possible switches and w=chi
			   -0.9014,
			   -0.8336,
			   -0.5038};
		*/

	set_params(chi_co, 2);


	cov_ze = gsl_vector_calloc(Noccs);
	//readvec("cov_ze.csv",cov_ze);
	GammaCoef	= gsl_matrix_calloc(Nfac,Nfac*Nglag);
	LambdaCoef	= gsl_matrix_calloc(Noccs,Nfac*(Nllag+1));
	LambdaZCoef = gsl_matrix_calloc(Noccs,Nagf);
	gsl_matrix * LamRead		= gsl_matrix_calloc(Noccs,Nagf+Nfac*(Nllag+1));
	var_eta		= gsl_matrix_calloc(Nfac,Nfac);
	var_zeta	= gsl_matrix_calloc(Noccs,Noccs);
	readmat("Gamma.csv",GammaCoef);
	readmat("Lambda.csv",LamRead); // this one file has both Lambda and coefs on Z
	
	gsl_matrix_view LC = gsl_matrix_submatrix(LamRead,0,0,Noccs,Nfac*(Nllag+1));
	gsl_matrix_memcpy(LambdaCoef,&LC.matrix);
	LC = gsl_matrix_submatrix(LamRead,0,Nfac*(Nllag+1),Noccs,1);
	gsl_matrix_memcpy(LambdaZCoef,&LC.matrix);
	readmat("var_eta.csv",var_eta);
	if(homosk_zeta==1){
		gsl_matrix_set_identity(var_zeta);
		gsl_matrix_scale(var_zeta,sig_zet);
	}
	else if(diag_zeta==1){
		readmat("var_zeta.csv",var_zeta);
		for(l=0;l<var_zeta->size1;l++){
			for(d=0;d<var_zeta->size2;d++){
				if(d!=l) gsl_matrix_set(var_zeta,l,d,0.0);
			}
		}
		gsl_matrix_scale(var_zeta,3.02); // adjust so that has the same unconditional variance as with the true persistence
	}
	else
		readmat("var_zeta.csv",var_zeta);

	if(printlev>=3){
		printmat("readGamma.csv",GammaCoef);
		printmat("readLambda.csv",LambdaCoef);
	}



	struct st_wr * st = malloc( sizeof(struct st_wr) );

	st->cal_set = calflag;

	/* Calibration Loop!
	*/
	double x0_0[]	= {phi	,scale_s	,shape_s	,effic,	chi_co[0]	,chi_co[1]	,chi_co[2]};
	double lb_0[]	= {0.25	,0.02		,0.005		,0.5	,-2.0			,-2.0			,-2.0	}; 
	double ub_0[]	= {0.8	,0.07		,0.20		,1.35	,.00			,.00			,.00	};

	for(i=0;i<4+Nskill;i++){
		if (lb_0[i]>=ub_0[i]){
			double lbi = lb_0[i];
			lb_0[i] = ub_0[i];
			ub_0[i] = lbi;
		}
	}
	gsl_vector_view x0vw = gsl_vector_view_array (x0_0, 7);
	gsl_vector_view lbvw = gsl_vector_view_array (lb_0, 7);
	gsl_vector_view ubvw = gsl_vector_view_array (ub_0, 7);

	if(calflag>=0){
		double objval = 100.0;
		int printlev_old = printlev;
		int verbose_old	= verbose;
		printlev =1;
		verbose = 1;
		int param_off = 4;
		double cal_xtol = 1e-9;
		double cal_ftol = 1e-8;
		if(calflag==1||calflag ==0){
			//first calibrate the parameters, fix those and then find chi

			st->cal_set = 1;
			st->n	= param_off;

			gsl_vector_view x0_1 = gsl_vector_subvector(&x0vw.vector, 0, param_off);//{x0_0[0],x0_0[1],x0_0[2],x0_0[3],x0_0[4]};
			gsl_vector_view lb_1 = gsl_vector_subvector(&lbvw.vector, 0, param_off);//{lb_0[0],lb_0[1],lb_0[2],lb_0[3],lb_0[4]};
			gsl_vector_view ub_1 = gsl_vector_subvector(&ubvw.vector, 0, param_off);//{ub_0[0],ub_0[1],ub_0[2],ub_0[3],ub_0[4]};

			strcpy(calhi_f,"calhist1.txt");
			calhist = fopen(calhi_f,"a+");
			fprintf(calhist,"phi,sig_psi,tau,sbar,effic,b1,b2,b3,b0\n");
			fprintf(calhist,"dist,wg,fnd,chng,urt,elpt,b1,b2,b3,b0\n");

			fprintf(calhist,"***Beginning Calibration of param set 1***\n");
			fclose(calhist);

			strcpy(soler_f,"solerr1.log");
			solerr = fopen(soler_f,"a+");
			fprintf(solerr,"Errors while solving model with param set 1\n");
			fclose(solerr);

			strcpy(simer_f,"simerr1.log");
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"Errors in policies while simulating param set 1\n");
			fclose(simerr);
			if(verbose >= 1)printf("Starting to calibrate on variable subset 1\n");
			//							f		lb ub n x0 param, ftol xtol, ctol , algo
			//							quad_solve

//			gsl_vector_set_all(&lb_1.vector,-1.0);
//			objval = bndCalMinwrap(&quad_solve,(lb_1.vector).data,(ub_1.vector).data,param_off ,(x0_1.vector).data,st,1e-7,1e-7,0.0,opt_alg,0);
			objval = bndCalMinwrap(&cal_dist,(lb_1.vector).data,(ub_1.vector).data,param_off ,(x0_1.vector).data,st,cal_xtol,cal_ftol,0.0,opt_alg,polish_alg);
			if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st->cal_set);

		}
		// FIND CHI
		if(calflag == 2 || calflag==0){
			st->n	= Nskill;
			gsl_vector_view x0_2 = gsl_vector_subvector(&x0vw.vector, param_off,st->n);
			gsl_vector_view lb_2 = gsl_vector_subvector(&lbvw.vector, param_off,st->n);
			gsl_vector_view ub_2 = gsl_vector_subvector(&ubvw.vector, param_off,st->n);
			st->cal_set = 2;

			strcpy(calhi_f,"calhist2.txt");
			calhist = fopen(calhi_f,"a+");
			fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3,b0\n");
			fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3,b0\n");
			fprintf(calhist,"***Beginning Calibration of param set 2***\n");
			fclose(calhist);


			solerr = fopen("solerr2.log","a+");
			fprintf(solerr,"Errors while solving model with param set 2\n");
			fclose(solerr);

			strcpy(simer_f,"simerr2.log");
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"Errors in policies while simulating param set 2\n");
			fclose(simerr);

			if(verbose >= 1)printf("Starting to calibrate on variable subset 2\n");
			//							f		lb ub n x0 param, ftol xtol, ctol , algo
			objval = bndCalMinwrap(&cal_dist,(lb_2.vector).data,(ub_2.vector).data,Nskill,(x0_2.vector).data,st,cal_xtol,cal_ftol,0.0,opt_alg,polish_alg);
			if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st->cal_set);

		}
		// THE WHOLE SHEBANG
		if(calflag==0||calflag>=3){
			st->cal_set = 0;
			st->n	= param_off+Nskill;

			strcpy(calhi_f,"calhist0.log");
			calhist = fopen(calhi_f,"a+");
			fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3\n");
			fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3\n");
			fprintf(calhist,"***Beginning Calibration of param set 0***\n");
			fclose(calhist);

			strcpy(soler_f,"solerr0.log");
			solerr = fopen(soler_f,"a+");
			fprintf(solerr,"Errors while solving model with param set 0\n");
			fclose(solerr);

			strcpy(simer_f,"simerr0.log");
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"Errors in policies while simulating param set 0\n");
			fclose(simerr);

			if(verbose >= 1)printf("Starting to calibrate on variable subset 0\n");
			//							f		lb ub n x0 param, ftol xtol, ctol , algo
			objval = bndCalMinwrap(&cal_dist,(lbvw.vector).data,(ubvw.vector).data,st->n,(x0vw.vector).data,st,cal_xtol,cal_ftol,0.0,opt_alg,polish_alg);
			if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st->cal_set);

		}
		printlev = printlev_old;
		verbose  = verbose_old;
	}

	alloc_econ(st);

	//comparative statics
	/*
	int printlev_old = printlev;
	printlev =0;
	gsl_vector * Zhl = gsl_vector_calloc(Nx);
	gsl_vector * dssdZ = gsl_vector_calloc(st->ss->size);
	Zhl->data[0] += sqrt(sig_eps);
	status += sol_ss(dssdZ,Zhl,st->xss,st->sol);
	Zhl->data[0] -= 2.0*sqrt(sig_eps);
	status += sol_ss(st->ss,Zhl,st->xss,st->sol);
	for(i=0;i<dssdZ->size;i++){
		dssdZ->data[i] -=st->ss->data[i];
		dssdZ->data[i] /= 2.0*sqrt(sig_eps);
	}
	printvec("dssdZ.csv",dssdZ);
	gsl_vector_free(dssdZ);
	gsl_vector_free(Zhl);
	printlev = printlev_old;
	*/

	status += sol_ss(st->ss,NULL,st->xss,st->sol);
	if(verbose>=1 && status ==0) printf("Successfully computed the steady state\n");
	if(verbose>=0 && status ==1) printf("Broke while computing steady state\n");
	status += sys_def(st->ss,st->sys);
	if(verbose>=1 && status >=1) printf("System not defined\n");
	if(verbose>=0 && status ==0) printf("System successfully defined\n");

	if(verbose>=2) printf("Now defining the 1st order solution to the dynamic model\n");

	int t;
	if(verbose>=1) t = clock();
	status += sol_dyn(st->ss, st->sol,st->sys);
	if(verbose>=0 && status >=1) printf("System not solved\n");
	if(verbose>=0 && status ==0) printf("System successfully solved\n");
	// compute numerical derivatives on theta
	if(status == 0){
		int printlev_old = printlev;
		printlev =0;
		gsl_vector * Zhl = gsl_vector_calloc(Nx);
		gsl_matrix * dtlddZ = gsl_matrix_calloc(Noccs+1,Noccs);
		Zhl->data[0] += sqrt(sig_eps);
		status += theta(dtlddZ, st->ss, st->sys, st->sol, Zhl);
		Zhl->data[0] -= 2.0*sqrt(sig_eps);
		st->sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += theta(st->sol->tld, st->ss, st->sys, st->sol, Zhl);
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				dtlddZ->data[l*Noccs+d] -=st->sol->tld->data[l*Noccs+d];
				dtlddZ->data[l*Noccs+d] /= 2.0*sqrt(sig_eps);
			}
		}
		gsl_matrix_free(st->sol->tld);
		st->sol->tld = NULL;
		printmat("dtlddZ.csv",dtlddZ);
		gsl_matrix_free(dtlddZ);
		gsl_vector_free(Zhl);
		printlev = printlev_old;
	}


//	if(status==0)
//		status += sol_shock(st);
//	if(verbose>=1 && status >=1) printf("Shocks not calibrated\n");
//	if(verbose>=0 && status ==0) printf("Shocks successfully calibrated\n");


	//status += ss_moments(&simdat, ss, xss);
	if(status ==0)
		status += sim_moments(st,st->ss,st->xss);
	if(verbose>=1 && status ==0){

		printf ("It took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	}

	//	TGR experiment
	gsl_vector * u_dur_dist = gsl_vector_alloc(5);
	gsl_vector * opt_dur_dist = gsl_vector_alloc(5);
	gsl_vector * fnd_dist	= gsl_vector_alloc(Noccs+1);
	gsl_vector * opt_fnd_dist = gsl_vector_alloc(Noccs+1);
	double urt,opt_urt;
	if(status ==0)
		status +=  TGR(u_dur_dist,opt_dur_dist,fnd_dist,opt_fnd_dist,&urt,&opt_urt,st);


	status += free_econ(st);

	gsl_matrix_free(GammaCoef);	gsl_matrix_free(LambdaCoef);
	gsl_matrix_free(var_eta); gsl_matrix_free(var_zeta);

	gsl_vector_free(u_dur_dist);gsl_vector_free(opt_dur_dist);	gsl_matrix_free(f_skills);


	return status;
}

/*
 * Solve the dynamic system, by my own method (no QZ decomp)
 * The technique follows page 3 & 6 of my Leuchtturm1917
 *
 */
int sol_dyn(gsl_vector * ss, struct sys_sol * sol, struct sys_coef *sys){
/* Using system matrices:
 * 	A0 s = A1 Es' + A2 c + A3 Z
 * 	F0 c = F1 Es' + F2 s + F3 Z
 *
 * 	Find solution:
 *	P0 s = P1 Es' + P2 Z
 *	c	= R(s,E[s'|s,Z],Z)
 *
 *	s	= PP z;
 */

	
	int Ns,Nc,Nx, i, maxrecur=5000; // these might be different from the globals of the same name.... but should not be
	int status = 0,nonfin=0;
	Ns = (sys->F1)->size2;
	Nc = (sys->F1)->size1;
	Nx = (sys->F3)->size2;

	if(verbose>=2) printf("Ns=%d,Nc=%d,Nx=%d\n",Ns,Nc,Nx);


	// Find P1
	gsl_matrix * A2invF0F1 = gsl_matrix_calloc(Ns,Ns);
	gsl_matrix * invF0F1 = gsl_matrix_calloc(Nc,Ns);

#ifdef _MKL_USE
	gsl_matrix * F0LU =gsl_matrix_calloc((sys->F0)->size1,(sys->F0)->size2);
	gsl_matrix_memcpy(F0LU,sys->F0);
	int * ipiv = malloc(sizeof(int)* ((sys->F0)->size2) );
	status =LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (sys->F0)->size1,(sys->F0)->size2, F0LU->data, (sys->F0)->size2,ipiv );
	gsl_matrix_memcpy(invF0F1,sys->F1);
	//							order	,	trans,	n row b or a	, nrhs,
	status += LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',  (sys->F0)->size1, (sys->F1)->size2,
	//		a	,		lda,			, work,	b,		ldb
			F0LU->data, (sys->F0)->size2, ipiv, invF0F1->data, invF0F1->size2);
	//status += sol_AXM(F0LU, sys->F1, invF0F1, ipiv);

#endif

//	status += sol_AXM(F0LU, sys->F1, invF0F1, ipiv);
#ifndef _MKL_USE

	gsl_matrix * invF0 	= gsl_matrix_calloc(Nc,Nc);
	status += inv_wrap(invF0,sys->F0);
	nonfin += isfinmat(invF0);

	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invF0,sys->F1,0.0,invF0F1);
#endif

	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sys->A2,invF0F1,0.0,A2invF0F1);
	gsl_matrix_memcpy(sol->P1,sys->A1);
	gsl_matrix_add(sol->P1,A2invF0F1);
	gsl_matrix_free(A2invF0F1);gsl_matrix_free(invF0F1);

	// Find P0
	gsl_matrix * A2invF0F2 = gsl_matrix_calloc(Ns,Ns);
	gsl_matrix * invF0F2 = gsl_matrix_calloc(Nc,Ns);
#ifdef _MKL_USE
	gsl_matrix_memcpy(invF0F2,sys->F2);

	status += LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',  (sys->F0)->size1, (sys->F2)->size2,
			F0LU->data, (sys->F0)->size2, ipiv, invF0F2->data, invF0F2->size2);
#endif

#ifndef _MKL_USE
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invF0,sys->F2,0.0,invF0F2);
#endif
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sys->A2,invF0F2,0.0,A2invF0F2);
	gsl_matrix_memcpy(sol->P0,sys->A0);
	gsl_matrix_sub(sol->P0,A2invF0F2);
	gsl_matrix_free(A2invF0F2);gsl_matrix_free(invF0F2);

	//Find P2
	gsl_matrix * A2invF0F3 = gsl_matrix_calloc(Ns,Nx);
	gsl_matrix * invF0F3 = gsl_matrix_calloc(Nc,Nx);
#ifdef _MKL_USE
	gsl_matrix_memcpy(invF0F3,sys->F3);
	status += LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',  (sys->F0)->size1, (sys->F3)->size2,
			F0LU->data, (sys->F0)->size2, ipiv, invF0F3->data, invF0F3->size2);
#endif

#ifndef _MKL_USE
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invF0,sys->F3,0.0,invF0F3);
#endif
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sys->A2,invF0F3,0.0,A2invF0F3);
	gsl_matrix_memcpy(sol->P2,sys->A3);
	gsl_matrix_add(sol->P2,A2invF0F3);
	gsl_matrix_free(A2invF0F3);gsl_matrix_free(invF0F3);

	if(printlev>=2){
		printmat("P0.csv",sol->P0);
		printmat("P1.csv",sol->P1);
		printmat("P2.csv",sol->P2);
	}
	nonfin += isfinmat(sol->P0);
	nonfin += isfinmat(sol->P1);
	nonfin += isfinmat(sol->P2);
	if(printlev>=2){
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"Non-finite %d times in P0-P2\n",nonfin);
		fclose(solerr);
	}

	// and forward looking for PP

	gsl_matrix * P1invP0= gsl_matrix_calloc(Ns,Ns);
#ifdef _MKL_USE
	// will have to solve P0^T X^T = P1 instead of  X P0 = P1

	gsl_matrix * P0tLU =gsl_matrix_calloc((sol->P0)->size1,(sol->P0)->size2);
	// this is now the transpose
	gsl_matrix_transpose_memcpy(P0tLU,sol->P0);
	int * ipiv_P0 = malloc(sizeof(int)* ((sol->P0)->size2) );
	status =LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (sol->P0)->size1,(sol->P0)->size2, P0tLU->data,
			(sol->P0)->size2,ipiv_P0 );
	gsl_matrix_transpose_memcpy(P1invP0,sol->P1);
	//							order	,	trans,	n row b or a	, nrhs,
	status += LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',  P0tLU->size1, P0tLU->size2,
	//		a	,		lda,			, work,	b,		ldb
			P0tLU->data, P0tLU->size2, ipiv_P0, P1invP0->data, P1invP0->size2);

	gsl_matrix_free(P0tLU);
	free(ipiv_P0);

	// this comes out!
	//status += inv_wrap(invP0,sol->P0);
	//gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,sol->P1,0.0,sol->invP0P1);
	//gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,sol->P2,0.0,sol->invP0P2);

	// !!!! WITH MKL: P1invP0 is actually (P1invP0)^T !!!!!
#endif
#ifndef _MKL_USE
	gsl_matrix * invP0	= gsl_matrix_calloc(Ns,Ns);
	status += inv_wrap(invP0,sol->P0);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,invP0,0.0,P1invP0);

	// this comes out!!
	//gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,sol->P1,0.0,sol->invP0P1);
	//gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,sol->P2,0.0,sol->invP0P2);
#endif



	gsl_matrix * P1invP0P2N 	= gsl_matrix_calloc(Ns,Nx);
	gsl_matrix * P1invP0P2NN 	= gsl_matrix_calloc(Ns,Nx);
	gsl_matrix * P0PP 			= gsl_matrix_calloc(Ns,Nx);
#ifndef _MKL_USE
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0,sol->P2,0.0,P1invP0P2NN);
#endif
#ifdef _MKL_USE
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,P1invP0,sol->P2,0.0,P1invP0P2NN);
#endif

	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0P2NN,sys->N,0.0,P1invP0P2N);
//	gsl_matrix_memcpy(P1invP0P2N,P1invP0P2NN);
	double mnorm = norm1mat(P1invP0P2N);
	double mnorm0= fabs(norm1mat(P1invP0P2N));
	double mnorml= 1e6;
	for(i = 0;i<maxrecur;i++){
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0P2N,sys->N,0.0,P1invP0P2NN);

#ifndef _MKL_USE
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0,P1invP0P2NN,0.0,P1invP0P2N);
#endif

#ifdef _MKL_USE
		gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,P1invP0,P1invP0P2NN,0.0,P1invP0P2N);
#endif
		// check convergence in the 1 norm
		mnorm = norm1mat(P1invP0P2N);
		if(mnorm>=1e10 || mnorm<=-1e10){
			status++;
			break;
		}
		// check that converging either in series or cauchy sense
		if((fabs(mnorm)/mnorm0<dyn_tol) || (fabs(mnorml-mnorm)<dyn_tol*10.0 && fabs(mnorm)/mnorm0<dyn_tol*10.0))
			break;
		gsl_matrix_add(P0PP,P1invP0P2N);
		mnorml = mnorm;
	}// end for(i=0;i<maxrecur;i++)
	//status = i<maxrecur-1? status : status+1;
	if(i>=maxrecur-1){
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"Forward expectations stopped at %f, did not converge\n",mnorm);
		fclose(solerr);
		if(verbose>=2)
			printf("Forward expectations stopped at %f, did not converge\n",mnorm);
		if(fabs(mnorm)>mnorm0)
			status++;
	}

	//gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,P0PP,0.0,sol->PP);
#ifdef _MKL_USE
	gsl_matrix * invP0	= gsl_matrix_calloc(Ns,Ns);
	status += inv_wrap(invP0,sol->P0);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,P0PP,0.0,sol->PP);
	gsl_matrix_free(invP0);
#endif


#ifndef _MKL_USE
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invP0,P0PP,0.0,sol->PP);
#endif


	gsl_matrix_add(sol->PP,sol->P2);


	gsl_matrix_free(P0PP);

	gsl_matrix_free(P1invP0);gsl_matrix_free(P1invP0P2NN);gsl_matrix_free(P1invP0P2N);
#ifndef _MKL_USE
	gsl_matrix_free(invP0);
	gsl_matrix_free(invF0);
#endif
#ifdef _MKL_USE
	gsl_matrix_free(F0LU);
	free(ipiv);

#endif

	if(printlev>=2)
		printmat("PP.csv",sol->PP);

	return status;
}

/*
 * Solve for the steady state
 */



int sol_ss(gsl_vector * ss, gsl_vector * Zz, gsl_matrix * xss, struct sys_sol * sol){
	// will solve the ss with
	//vars = [ x U_l0 U_ld J_ld gld wld thetald sld] in row-major order, i.e. {(0,1),(0,2),...,(J,J)}

	int status,l,d,dd,ll,iter,itermax,i,allocZz;
	
	gsl_vector * W_l0,*W_ld;
	gsl_vector * gld,*lgld,*thetald,*findrt, *sld;
	gsl_vector * ret_d;
	gsl_matrix * pld,*x;//,*lx;
	if(Zz == NULL){
		Zz = gsl_vector_calloc(Nx);
		allocZz = 1;
	}
	else
		allocZz = 0;
	x		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	//lx		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	W_l0 	= gsl_vector_calloc(Noccs+1);
	W_ld 	= gsl_vector_calloc(Noccs+Noccs*Noccs);

	gld 	= gsl_vector_calloc(Noccs+Noccs*Noccs);
	ret_d	= gsl_vector_calloc(Noccs);
	lgld 	= gsl_vector_calloc(Noccs+Noccs*Noccs);
	thetald	= gsl_vector_calloc(Noccs+Noccs*Noccs);
	findrt 	= gsl_vector_calloc(Noccs+1);
	sld		= gsl_vector_calloc(Noccs+Noccs*Noccs);

	pld	= gsl_matrix_alloc(Noccs+1,Noccs);
	
	// initialize the policies:
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1] : b[0];
		for(d=0;d<Noccs;d++){
		//	if(l==0)
				gsl_vector_set(gld,l*Noccs+d,1.0/(double)Noccs);
		//	else if(l==d+1)
		//		gsl_vector_set(gld,l*Noccs+d,1.0);
			gsl_vector_set(thetald,l*Noccs+d,invp(avg_fnd));
			gsl_vector_set(sld,l*Noccs+d,sbar);
			gsl_matrix_set(pld,l,d,pmatch(thetald->data[l*Noccs+d]));
			if(gsl_matrix_get(pld,l,d)>1.0)
				gsl_matrix_set(pld,l,d,1.0);
			W_l0->data[l] += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*(chi[l][d] - bl);
			W_ld->data[l*Noccs+d] = chi[l][d];
		}
		W_l0->data[l] += bl;
	}
	itermax = 1000;

	if(printlev>=2){
		printvec("gld0.csv",gld);
		printvec("thetald0.csv",thetald);
		printvec("W_l0_0.csv",W_l0);
		printvec("W_ld_0.csv",W_ld);
	}
	double maxdistg=1e-4;
	double lmaxdistg = 100.0;
	int break_flag =0;
	double Z = Zz->data[0];

	for(iter=0;iter<itermax;iter++){

		int vfiter;
		for(vfiter =0;vfiter<itermax/10;vfiter++){
			for(d=0;d<Noccs;d++){
				l = d+1;
				double zd = Zz->data[d+Notz];
				double sepld = sld->data[l*Noccs+d];
				gsl_vector_set(W_ld,l*Noccs+d,
						((1.0-sepld)*chi[l][d]*(exp(Z+zd)+beta*W_ld->data[l*Noccs+d])
								+ sepld*W_l0->data[l])
				//		/(1.0-(1.0-sepld)*beta)
						);
			}
			// W_ld
			for(l=0;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					double sepld = sld->data[l*Noccs+d];
					double zd = Zz->data[d+Notz];
					if(l-1!=d){
						gsl_vector_set(W_ld,l*Noccs+d,
							((1.0-tau)*((1.0-sepld )*(chi[l][d]*exp(Z+zd) + beta*W_ld->data[l*Noccs+d])
									+ sepld*W_l0->data[0] ) + tau*W_ld->data[(d+1)*Noccs+d]  )
				//			/(1.0-(1.0-sepld)*(1.0-tau)*beta)
									);

					}
				}
			}

			for(l=0;l<Noccs+1;l++){
				double findrt_ld =0.0;
				for(d=0;d<Noccs;d++)
					findrt_ld += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d);
				gsl_vector_set(findrt,l,findrt_ld);
			}
			// W_l0

			double maxdistW0 = 0.0;
			for(l=0;l<Noccs+1;l++){
				double bl = l>0 ? b[1]:b[0];
				double W_0 =0.0;
				for(d=0;d<Noccs;d++){
					double zd = Zz->data[d+Notz];
					double nud = l==d+1? 0.0:nu;
					double post = gsl_matrix_get(pld,l,d)>0? - kappa*thetald->data[l*Noccs+d]/gsl_matrix_get(pld,l,d) : 0.0;
					double ret 	= chi[l][d]*exp(Z+zd) - nud
			//				+ post
							+ beta*gsl_vector_get(W_ld,l*Noccs+d)
							- bl - beta*W_l0->data[l] ;
					//ret  = (l == d+1) ? ret  - bl - beta*W_l0->data[l] : ret  - bl - beta*W_l0->data[0];
					ret *= (1.0-fm_shr);
					ret += bl + beta*W_l0->data[l];
					W_0 += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*ret;
				}

				W_0 += (1.0-gsl_vector_get(findrt,l))*(bl + beta*W_l0->data[l]);
				//W_0 /= (1.0 - beta*(1.0-gsl_vector_get(findrt,l)));
				double distW0 = fabs(W_l0->data[l] - W_0)/W_l0->data[l];
				W_l0->data[l] = W_0;
				if(distW0 > maxdistW0)
					maxdistW0 = distW0;
			}
			double W0_tol = 2*maxdistg< 1e-4? 2*maxdistg : 1e-4;
			W0_tol = break_flag == 1 ? ss_tol : W0_tol;
			if(maxdistW0<W0_tol)// not particularly precise to start
				break;

		}//for vfiter


		// update policies

		maxdistg = 0.0;
		gsl_vector_memcpy(lgld,gld);

		//either make ret_d local, private or do not parallelize
		//#pragma omp parallel for default(shared) private(d,l,dd,ret_d)
		for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			for(d=0;d<Noccs;d++){
				double zd = Zz->data[d+Notz];
				double nud = l == d+1? 0:nu;
				//double W0 	= l == d+1?  W_l0->data[l] : W_l0->data[0];
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l] ;
				//Wdif = Wdif<0.0 ?  0.0 : Wdif;
				double pld_ld =gsl_matrix_get(pld,l,d);
				double post = pld_ld > 0 ? -kappa*gsl_vector_get(thetald,l*Noccs+d)/pld_ld : 0.0;

				double arr_d = (1.0-fm_shr)*(chi[l][d]*exp(Z+zd) -nud - bl + beta*Wdif)
						+ bl + W_l0->data[l];

				gsl_vector_set(ret_d,d,pld_ld*arr_d);

			}
			double sexpret_dd =0.0;
			for(dd=0;dd<Noccs;dd++)
				sexpret_dd = gsl_vector_get(ret_d,dd)<= 0.0 ? sexpret_dd :
					exp(sig_psi*gsl_vector_get(ret_d,dd)) + sexpret_dd;
			for(d=0;d<Noccs;d++){
				//gld
				double zd = Zz->data[d+Notz];
				// known return in occ d
				double gg = exp(sig_psi*gsl_vector_get(ret_d,d))/sexpret_dd;
				if(ret_d->data[d]<=0)
					gg =0;
				gsl_vector_set(gld,l*Noccs+d,gg);

				//theta
				double nud 	= l-1 !=d ? nu : 0.0;
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l];
				//Wdif = Wdif<0.0 ? 0.0 : Wdif;
				double surp = chi[l][d]*exp(Z+zd) - nud - bl +beta*Wdif;
				if(surp > 0.0){
					double qhere = kappa/(fm_shr*surp);
					gsl_vector_set(thetald,l*Noccs+d,invq(qhere) );
				}
				else
					gsl_vector_set(thetald,l*Noccs+d,0.0);

				// sld
				double cutoff = -(chi[l][d]*exp(Z+zd) + beta*W_ld->data[l*Noccs+d] - W_l0->data[l]);
				cutoff = cutoff >0.0 ? 0.0 : cutoff;
				double sep_ld = scale_s*exp(shape_s*cutoff);
				gsl_vector_set(sld,l*Noccs+d,sep_ld);
			}// for d=0:Noccs
			// adds to 1
			double gsum = 0.0;
			for(d =0 ;d<Noccs;d++)	gsum += gld->data[l*Noccs+d];
			if(gsum>0)
				for(d =0 ;d<Noccs;d++)	gld->data[l*Noccs+d] /= gsum;
			else
				for(d =0 ;d<Noccs;d++)	gld->data[l*Noccs+d] = 1.0/(double)Noccs;
		}// for l=0:Noccs
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double distg = fabs(lgld->data[l*Noccs+d] - gld->data[l*Noccs+d])/(1.0+lgld->data[l*Noccs+d]);
				maxdistg = distg>maxdistg ? distg : maxdistg;
			}
		}

		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(pld,l,d,pmatch(thetald->data[l*Noccs+d]) );
		}


		// this is actually testing the condition from W_l0 above
		if(break_flag ==1)
			break;
		else if(maxdistg < ss_tol && iter>=10)
			break_flag=1;
		else if((fabs(maxdistg - lmaxdistg) < ss_tol) & (iter>100)){
			break_flag=1;
			if(verbose>=2) printf("Not making progress in SS from %f\n",maxdistg);
			if(printlev>=1){
				solerr = fopen(soler_f,"a+");
				fprintf(solerr,"SS err stuck at %f", maxdistg);
				fclose(solerr);
			}
		}
		lmaxdistg = maxdistg;
		if(printlev>=4){
			printvec("W_l0_i.csv",W_l0);
			printvec("W_ld_i.csv",W_ld);
			printvec("sld_i.csv",sld);
			printvec("gld_i.csv",gld);
			printvec("thetald_i.csv",thetald);
			printvec("findrt_i.csv",findrt);
		}

	}// end iter=0:maxiter
	status = iter<itermax || (fabs(maxdistg - lmaxdistg)<ss_tol*5 && maxdistg<1.0) ? 0 : 1;

	// steady-state wages
	double * Jd = malloc(sizeof(double)*Noccs);

	for(d=0;d<Noccs;d++){
		double zd = Zz->data[d+Notz];
		double bl = b[1];
		int l = d+1;
		// need to use spol to get \bar \xi^{ld}, then evaluate the mean, \int_{-\bar \xi^{ld}}^0\xi sh e^{sh*\xi}d\xi
		//and then invert (1-scale_s)*0 + scale_s*log(sld/shape_s)/scale_s

		double barxi = -log(sld->data[l*Noccs+d]/scale_s )/shape_s;
		double Exi = scale_s*((1.0/shape_s+barxi)*exp(-shape_s*barxi)-1.0/shape_s);
		Exi = 0.0;
	//	double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(bl-Exi);
	//	double wld_ld = ((1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(W_ld->data[l*Noccs+d] - W_l0->data[l] + bl-Exi)
	//			+ beta*(1.0-sld->data[l*Noccs+d])*chi[l][d]*exp(Z+zd)/(1.0-beta*(1.0-sld->data[l*Noccs+d])) )
	//					/(1.0 + beta/(1.0-beta*(1.0-sld->data[l*Noccs+d])));
		double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd) + fm_shr*( Exi*(1.0-beta*(1.0-sld->data[l*Noccs+d]))
				+ bl + beta*W_l0->data[l] - beta*W_ld->data[l*Noccs+d])
				+ beta*kappa/pld->data[l*pld->tda+d]*thetald->data[l*Noccs+d]*(1.0 - sld->data[l*Noccs+d]);

		wld_ld = wld_ld > 0.0 && thetald->data[l*Noccs+d]>0.0 ? wld_ld : 0.0;
		wld_ld = wld_ld > chi[l][d]*exp(Z+zd) ? chi[l][d]*exp(Z+zd) : wld_ld;
	//	wld_ld = chi[l][d]*exp(Z+zd);
		gsl_matrix_set(sol->ss_wld,l,d,wld_ld);
		Jd[d] = (1.0-sld->data[l*Noccs+d])*(chi[l][d]*exp(Z+zd) - wld_ld)/(1.0-(1.0-sld->data[l*Noccs+d])*beta);
	}

	for(l=0;l<Noccs+1;l++){
		double bl = l==0 ? b[0] : b[1];
		for(d=0;d<Noccs;d++){
			double zd = Zz->data[d+Notz];
			if(d!=l-1){

				double Ephi = 0.0;
//				double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(bl - Ephi) ;

//				double wld_ld = ((1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(W_ld->data[l*Noccs+d] - W_l0->data[l] + bl - Ephi)
//						+ beta*((1.0-sld->data[l*Noccs+d])*(1.0-tau)*chi[l][d]*exp(Z+zd) +tau*Jd[d] )/(1.0-beta*(1.0-sld->data[l*Noccs+d])*(1.0-tau) ))
//								/(1.0+ beta/(1.0-beta*(1.0-tau)*(1.0-sld->data[l*Noccs+d])));
				double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd) + fm_shr*(
						bl + beta*W_l0->data[l] - beta*W_ld->data[l*Noccs+d])
						+ beta*kappa/pld->data[l*pld->tda+d]*thetald->data[l*Noccs+d]*(1.0-sld->data[l*Noccs+d])*(1.0-tau)
						+ tau*(1.0-sld->data[(d+1)*Noccs+d])*beta*kappa/pld->data[(d+1)*pld->tda+d]*thetald->data[(d+1)*Noccs+d];
				wld_ld = wld_ld > 0.0 && thetald->data[l*Noccs+d]>0.0 ? wld_ld : 0.0;
				wld_ld = wld_ld > chi[l][d]*exp(Z+zd) ? chi[l][d]*exp(Z+zd) : wld_ld;
		//		wld_ld = chi[l][d]*exp(Z+zd);
				gsl_matrix_set(sol->ss_wld,l,d,wld_ld);
			}
		}
	}
	free(Jd);

	if(printlev>=2){
		printvec("W_l0_i.csv",W_l0);
		printvec("W_ld_i.csv",W_ld);
		printvec("gld_i.csv",gld);
		printvec("sld_i.csv",sld);
		printmat("ss_wld.csv",sol->ss_wld);
		printvec("thetald_i.csv",thetald);
		printvec("findrt_i.csv",findrt);
	}

/*	I don't want to do this: there may be places with positive flow value to justify an opening
 * 	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			if(W_ld->data[l*Noccs+d]<= W_l0->data[l]){
	//			W_ld->data[l*Noccs+d] =W_l0->data[l];
				thetald->data[l*Noccs+d] = 0.0;
			}
	}
*/

	// calculate xss the right way!
	int Nxx = pow(Noccs+1,2);
	gsl_matrix * xtrans = gsl_matrix_calloc(Nxx,Nxx);
	gsl_vector_complex *eval = gsl_vector_complex_alloc(Nxx);
	gsl_matrix_complex *xxmat = gsl_matrix_complex_alloc (Nxx,Nxx);
	gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc(Nxx);
	//gsl_eigen_nonsymmv_params (1,w);
	gsl_matrix * xtransT = gsl_matrix_alloc(Nxx,Nxx);

	gsl_matrix * Pxx0 = gsl_matrix_calloc(Nxx,Nxx);
	gsl_matrix_set_identity(Pxx0);
	gsl_matrix * Pxx1 = gsl_matrix_calloc(Nxx,Nxx);

	gsl_matrix_set(Pxx1,0,0,1.0-findrt->data[0]);
	for(l=0;l<Noccs+1;l++){
		for(d=1;d<Noccs+1;d++){
			if(l!=d){
				gsl_matrix_set(Pxx0,0,l*(Noccs+1)+d,
					-(1.0-tau)*sld->data[l*Noccs+d]);
			}
		}
	}
	//x_l0 : l>0
	for(l=1;l<Noccs+1;l++){
		d=0;
		gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,
				(1.0-findrt->data[l]) );
		gsl_matrix_set(Pxx0,l*(Noccs+1)+d,l*(Noccs+1)+l, - sld->data[l*Noccs+d] );
	}
	//x_0d : d>0
	for(d=1;d<Noccs+1;d++){
		gsl_matrix_set(Pxx1,d,d,
				(1.0-tau)*(1.0- sld->data[0*Noccs+d]));
		gsl_matrix_set(Pxx1,d,0,
				gsl_vector_get(gld,d-1)*gsl_matrix_get(pld,0,d-1));
	}
	//x_ld : l,d>0
	for(l=1;l<Noccs+1;l++){
		for(d=1;d<Noccs+1;d++){
			if(l!=d){
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,
						(1.0-tau)*(1.0-sld->data[l*Noccs+d]));
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+0,
						gsl_vector_get(gld,l*Noccs + d-1)*gsl_matrix_get(pld,l,d-1));
			}else{
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,(1.0-sld->data[l*Noccs+d]));
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+0,
						gsl_vector_get(gld,l*Noccs + d-1)*gsl_matrix_get(pld,l,d-1));
				for(ll=0;ll<Noccs+1;ll++){
					if(ll!=l && ll!=d)
						gsl_matrix_set(Pxx0,l*(Noccs+1)+d,ll*(Noccs+1)+d,-tau);
				}
			}
		}
	}

	//xtrans = Pxx0\Pxx1;
	gsl_permutation * pp = gsl_permutation_alloc (Nxx);
	int s;
	status += gsl_linalg_LU_decomp (Pxx0, pp, &s);
	for(l=0;l<Nxx;l++){
		gsl_vector_view P1col = gsl_matrix_column(Pxx1,l);
		gsl_vector_view xtcol = gsl_matrix_column(xtrans,l);
		status += gsl_linalg_LU_solve (Pxx0, pp, &P1col.vector, &xtcol.vector);
	}
	gsl_permutation_free(pp);
	if(printlev>=2) printmat("xtrans.csv",xtrans);
	for(l=0;l<Nxx;l++){
		for(d=0;d<Nxx;d++)
			gsl_matrix_set(xtransT,d,l,gsl_matrix_get(xtrans,l,d));
	}
	gsl_eigen_nonsymmv(xtransT, eval, xxmat, w);

//	gsl_eigen_nonsymmv_sort (eval, xxmat, GSL_EIGEN_SORT_ABS_DESC);

	int ii =0;
	double dist1 = 1.0;
	gsl_complex eval_l = gsl_vector_complex_get(eval, ii);
	if(eval_l.dat[0]>1.0001 || eval_l.dat[0]<0.999){
		ii =0;
		dist1 = 1.0;
		for(l=0;l<Nxx;l++){
			gsl_complex eval_l = gsl_vector_complex_get(eval, l);
			if(fabs(eval_l.dat[0]-1.0 )<dist1 ){
				ii=l;
				dist1 = fabs(eval_l.dat[0]-1.0 );
			}
		}
	}

	if(dist1<5e-3){
		// it's a keeper!  use this x
		double xsum =0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs+1;d++){
				gsl_complex xxld = gsl_matrix_complex_get(xxmat,l*(Noccs+1)+d,ii);
				gsl_matrix_set(xss,l,d,fabs(xxld.dat[0]));
				xsum += fabs(xxld.dat[0]);
			}
		}
		gsl_matrix_scale(xss,1.0/xsum);
		if(printlev>=2) printmat("xss.csv",xss);
	}
	else{
		// just a uniform distribution
		for(l=0;l<Noccs+1;l++){
			gsl_matrix_set(xss,l,0,0.055/(double)(Noccs+1) );
			for(d=1;d<Noccs+1;d++){
				gsl_matrix_set(xss,l,d,0.945/(double)(Noccs*Noccs+Noccs) );
			}
		}
	}
	gsl_matrix_free(xtrans); gsl_matrix_free(xtransT);
	gsl_eigen_nonsymmv_free (w);
	gsl_matrix_complex_free(xxmat);
	gsl_matrix_free(Pxx0);gsl_matrix_free(Pxx1);
	gsl_vector_complex_free(eval);

	double urt = 0.0;
	for(l=0;l<Noccs+1;l++){
		urt+=gsl_matrix_get(xss,l,0);
	}


	// output into vars array
	//vars = [ x U_l0 U_ld J_ld gld wld thetald] in row-major order, i.e. {(0,1),(0,2),...,(J,J)}

	int ssi=0;

	for(i=0;i<Noccs+1;i++){
		gsl_vector_set(ss,ssi,W_l0->data[i]);
		ssi++;	
	}
	for(i=0;i<Noccs*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,W_ld->data[i]);
		ssi++;
	}

	for(i=0;i<Noccs*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,gld->data[i]);
		ssi++;
	}

	for(i=0;i<Noccs*(Noccs+1);i++){
		if(thetald->data[i]>0.0)
			gsl_vector_set(ss,ssi,thetald->data[i]);
		else
			gsl_vector_set(ss,ssi,0.0);
		ssi++;
	}
	for(i=0;i<Noccs*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,sld->data[i]);
		ssi++;
	}


	// free stuff!
	if(allocZz == 1) gsl_vector_free(Zz);
	gsl_vector_free(W_l0);
	gsl_vector_free(W_ld);
	gsl_matrix_free(x);
	gsl_vector_free(gld);
	gsl_vector_free(sld);
	gsl_vector_free(ret_d);
	gsl_vector_free(lgld);
	gsl_vector_free(thetald);
	gsl_matrix_free(pld);
	gsl_vector_free(findrt);

	return status;
}

int sol_shock(struct st_wr * st){
	int status, i,iter, biter,maxiter = 100,gsl_status;

	status =0;
	gsl_min_fminimizer * s = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
	double stdZz,stdZzL,stdZzH;
	gsl_function F;
	F.function = &gsl_endog_std;

	for(i=0;i<Noccs;i++){
		st->d = i+1;
		F.params = (void*)st;
		double stdZzL0 = 0.5;
		stdZzL = stdZzL0;
		stdZz  = 1.1;
		double stdZzH0 = 10.0;
		stdZzH = stdZzH0;

		iter =0,biter=0;
		int border = 0;
		do{
			gsl_min_fminimizer_set(s,&F,stdZz,stdZzL,stdZzH);
			if(s->f_lower<0.0){
				stdZzL0 *= 0.5;
				stdZzL = stdZzL0;
				border = 1;
			}
			else if(s->f_upper>0.0){
				stdZzH0 *= 2.0;
				stdZzH = stdZzH0;
				border = 1;
			}
			else{
				do{
					iter++;
					gsl_status = gsl_min_fminimizer_iterate(s);
					gsl_status = gsl_min_test_interval(stdZzL,stdZzH,1e-3,1e-3);
				}while(gsl_status == GSL_CONTINUE && iter<maxiter);
			}
		}while(border ==1 && biter<maxiter/10);

		if(i==-1)
			sig_eps *= stdZz*stdZz;
		else
			gsl_matrix_set(st->sys->S,i+Notz,i+Notz,gsl_matrix_get(st->sys->S,i+Notz,i+Notz)*stdZz);
	}

	gsl_min_fminimizer_free(s);
	return status;
}

/*
 *
 * The nonlinear system that will be solved
 */

int sys_def(gsl_vector * ss, struct sys_coef * sys){
	int l,d,status,f;
//	int wld_i, tld_i, x_i,gld_i;
	int Nvarex 	= (sys->A3)->size2;
//	int Nvaren 	= ss->size;
//	int Notz 	= 1+ Nagf +Nfac*(Nflag+1);


	gsl_matrix *N0,*N1,*invN0;
	if(Nvarex != Nx)
		printf("Size of N matrix is wrong!");

	N0 		= gsl_matrix_calloc(Nvarex,Nvarex);
	invN0 	= gsl_matrix_calloc(Nvarex,Nvarex);
	N1 		= gsl_matrix_calloc(Nvarex,Nvarex);
	gsl_matrix_set_identity(N0);

//	for(l=0;l<Noccs;l++)
//		gsl_matrix_set(N0,l+ Notz,0,0.0);
	// contemporaneous factors:
	for(f=0;f<Nfac;f++){
		for(d=0;d<Noccs;d++)
			gsl_matrix_set(N0,d+Notz,f+Nagf,
					-gsl_matrix_get(LambdaCoef,d,f));
	}
	// contemporaneous coefficient for z
	for(d=0;d<Noccs;d++)
		gsl_matrix_set(N0,d+Notz,0,
			-gsl_matrix_get(LambdaZCoef,d,0));
	/* First partition, N1_11 =
	 *
	 * (rhoZ  0 )
	 * ( I    0 )
	 *
	 * N1_12 =
	 * (Zfcoef 0 )
	 *
	 */
	gsl_matrix_set(N1,0,0,rhoZ);
	// dynamics for the lagged Z kept around
	for(l=0;l<Nagf-1;l++)
		gsl_matrix_set(N1,l+1,l+1,1.0);
	//for(f=0;f<Nfac*Nglag;f++)
	//	gsl_matrix_set(N1,0,1+Nagf,Zfcoef[f]);

	/* Second partition, N1_22 =
	 * (Gamma  0 )
	 * ( I     0 )
	 */
	set_matrix_block(N1,GammaCoef,Nagf,Nagf);
	for(f=0;f<Nfac*(Nllag);f++)
		gsl_matrix_set(N1,Nagf+Nfac+f,Nagf+f,1.0);
	/*
	 * Fourth partition, N1_32 =
	 * Lambda_{1:(Nflag-1)}
	 */
	if(Nllag>0){
		gsl_matrix_view N1_Lambda = gsl_matrix_submatrix(LambdaCoef, 0, Nfac, LambdaCoef->size1, LambdaCoef->size2-Nfac);
		set_matrix_block(N1,&N1_Lambda.matrix,Notz,1+Nagf);
	}
	/*
	 * Final partition, N1_33 =
	 *  rhozz*I
	 */
	for(l=0;l<Noccs;l++)
		gsl_matrix_set(N1,l+Notz,l+Notz,rhozz);

	inv_wrap(invN0,N0);
	status = gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invN0,N1,0.0,sys->N);

	gsl_matrix_set_identity(sys->S);
	gsl_matrix_set(sys->S,0,0,
			sig_eps);
	// set correlations
	for(l=0;l<Noccs;l++){
		gsl_matrix_set(sys->S,0,l+Notz,gsl_vector_get(cov_ze,l));
		gsl_matrix_set(sys->S,l+Notz,0,gsl_vector_get(cov_ze,l));
	}
	set_matrix_block(sys->S,var_eta,Nagf,Nagf);
	for(l=0;l<Noccs;l++){
		for(d=0;d<Noccs;d++){ //upper triangle only
			double Sld = gsl_matrix_get(var_zeta,l,d);
			//Sld = l!=d && (Sld>-5e-5 && Sld<5e-5) ? 0.0 : Sld;
			gsl_matrix_set(sys->S,Notz+l,Notz+d,Sld);
		}
	}

	status += gsl_linalg_cholesky_decomp(sys->S);
	// for some reason, this is not already triangular
	for(d=0;d<(sys->S)->size2;d++){
		for(l=0;l<(sys->S)->size1;l++){
			if(l<=d)
				gsl_matrix_set(sys->S,l,d,gsl_matrix_get(sys->S,l,d));
			else
				gsl_matrix_set(sys->S,l,d,0.0);
		}
	}

	// set as zero innovations to carried lag terms
	for(l=0;l<Nfac*Nllag;l++)
		gsl_matrix_set(sys->S,Nagf+Nfac+l,Nagf+Nfac+l,0.0);
	gsl_matrix * tmp = gsl_matrix_alloc(invN0->size1,invN0->size2);
	gsl_matrix_memcpy(tmp,sys->S);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invN0,tmp,0.0,sys->S);
	gsl_matrix_free(tmp);
	if(printlev>=2){
		printmat("N0.csv",N0);
		printmat("N1.csv",N1);
		printmat("invN0.csv",invN0);

	}

	gsl_matrix_free(N0);
	gsl_matrix_free(N1);
	gsl_matrix_free(invN0);

	// set system matrices for 1st order
	/* Using system matrices:
	 * 	A0 s = A1 Es' + A2 c + A3 Z
	 * 	F0 c = F1 Es' + F2 s + F3 Z
	 *
	 * 	Find solution:
	 *	P0 s = P1 Es' + P2 Z
	 *	c	= R(s,E[s'|s,Z],Z)
	 *
	 *	s	= PP z;
	 *
	 */

	gsl_vector * Zz = gsl_vector_calloc(Nvarex);
	status += sys_st_diff(ss,sys->A0,sys->A2,sys->A1,Zz);
	status += sys_co_diff(ss,sys->F2,sys->F0,sys->F1,Zz);
	status += sys_ex_diff(ss,sys->A3,sys->F3);

	// fill second derivs from finite differences on first derivs


	if(printlev>=2){
		printmat("A0.csv",sys->A0);
		printmat("A1.csv",sys->A1);
		printmat("A2.csv",sys->A2);
		printmat("A3.csv",sys->A3);
		printmat("F0.csv",sys->F0);
		printmat("F1.csv",sys->F1);
		printmat("F2.csv",sys->F2);
		printmat("F3.csv",sys->F3);
		printmat("N.csv",sys->N);
		printmat("S.csv",sys->S);
	}

	// check finite-ness of every element of every matrix
	int nonfin = 0;
	nonfin += isfinmat(sys->A0);
	nonfin += isfinmat(sys->A1);
	nonfin += isfinmat(sys->A2);
	nonfin += isfinmat(sys->A3);
	nonfin += isfinmat(sys->F0);
	nonfin += isfinmat(sys->F1);
	nonfin += isfinmat(sys->F2);
	nonfin += isfinmat(sys->F3);
	nonfin += isfinmat(sys->N);
	nonfin += isfinmat(sys->S);

	if(printlev>=1){
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"Defining the system had %d \n",nonfin);
		fclose(solerr);
	}

	gsl_vector_free(Zz);
	return status;
}

int sys_st_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	/* this function sets up coefficients for the matrix on the state-transition equation,
	 * i.e. \hat W = \sum f_c/W_ss * c_ss * c
	 */


	int l,d,status;
	int tld_i,sld_i,gld_i,Wld_i,Wl0_i;

	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);
	sld_i	= tld_i + Noccs*(Noccs+1);

	int ss_tld_i, ss_sld_i, ss_x_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;
	ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
	ss_sld_i	= ss_tld_i + Noccs*(Noccs+1);

	status =0;
	double Z = xx->data[0];


	for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			double findrtl = 0.0;
			for(d=0;d<Noccs;d++)
				findrtl += ss->data[ss_gld_i+l*Noccs+d]*pmatch(ss->data[ss_tld_i+l*Noccs+d]);
			double gdenom = 0.0;
			for(d=0;d<Noccs;d++){
				double nud = l ==d+1? 0.0 : nu;
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				double pld 	= pmatch(ss->data[ss_tld_i+l*Noccs+d]);
				double post = pld > 0.0? -kappa*ss->data[ss_tld_i+l*Noccs+d]/pld: 0.0;
				gdenom +=exp(sig_psi*pld*
						(chi[l][d]-bl+nud+ post + beta*contval));
			}
			// Wl0
			double Wl0_ss = ss->data[ss_Wl0_i + l];
			gsl_matrix_set(Dst_tp1,Wl0_i+l,Wl0_i+l, beta*(1.0-findrtl) );// /Wl0_ss*Wl0_ss
			gsl_matrix_set(Dst,Wl0_i+l,Wl0_i+l,1.0);
			for(d=0;d<Noccs;d++){
				double Wld_ss = ss->data[ss_Wld_i + l*Noccs+d];
				double zd = xx->data[d+Notz];
				// Wld
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,1.0);
				if(ss->data[ss_tld_i+l*Noccs+d]>0){
					if(d+1 ==l){
						gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d, Wld_i+l*Noccs+d, (1.0-ss->data[ss_sld_i + l*Noccs+d])*beta);
						gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l, - (ss->data[ss_sld_i + l*Noccs+d])
							*ss->data[ss_Wl0_i+l]/Wld_ss
							);
						double cont_dd =chi[l][d]*exp(Z+zd) -bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
						double dWds = -cont_dd + ss->data[ss_Wl0_i+l];
						gsl_matrix_set(Dco,Wld_i + l*Noccs+d, sld_i+l*Noccs+d,dWds
							/Wld_ss*ss->data[ss_sld_i+l*Noccs+d]
							);
					}
					else{
						gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,
								beta*(1.0-tau)*(1.0-ss->data[ss_sld_i + l*Noccs+d]) );

						gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+(d+1)*Noccs+d,-tau
								*ss->data[ss_Wld_i+(d+1)*Noccs+d]/Wld_ss
								);
						gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+0,-(1.0-tau)*ss->data[ss_sld_i + l*Noccs+d]
								*ss->data[ss_Wl0_i+l]/Wld_ss
								);
						double cont_ld =chi[l][d]*exp(Z+zd) - bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
						double dWds =-(1.0-tau)*(cont_ld - ss->data[ss_Wl0_i+0]);
						gsl_matrix_set(Dco,Wld_i + l*Noccs+d, sld_i+l*Noccs+d,dWds
								/Wld_ss*ss->data[ss_sld_i+l*Noccs+d]
								);
					}
				}// tss>0 market is open


				// Wl0

				// ss_ret = -nu - kappa/theta^ld/p^ld + mu^ld - mu^l0
				double pld		= pmatch(ss->data[ss_tld_i+l*Noccs+d]);
				double post 	=  - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld;
				double ss_ret 	= d+1==l ? 0.0 : -nu;
				ss_ret += chi[l][d]*exp(Z+zd) - bl;
				double contval = beta*(ss->data[ss_Wld_i+l*Noccs+d] -ss->data[ss_Wl0_i+l] );
				ss_ret += contval;
				ss_ret *= (1.0-fm_shr);
				ss_ret += bl + beta*ss->data[ss_Wl0_i+l];
				double u_ret = bl + beta*ss->data[ss_Wl0_i+l];
				//ss_ret = ss_ret>1e-5? ss_ret : 1e-5;
				double dtld = ss->data[ss_gld_i+l*Noccs+d]*dpmatch(ss->data[ss_tld_i+l*Noccs+d])*(
								ss_ret
								//- kappa*pow(ss->data[ss_tld_i+l*Noccs+d],phi)/(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi))
								- u_ret);

				if(gsl_finite(dtld) && pld>0.0)
					gsl_matrix_set(Dco,Wl0_i+l,tld_i+l*Noccs+d,dtld
							*ss->data[ss_tld_i+l*Noccs+d]/Wl0_ss
							);
				else
					gsl_matrix_set(Dco,Wl0_i+l,tld_i+l*Noccs+d,0.0);

				if(pld>0.0 && gsl_finite(pld*ss_ret))
					gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,pld*(ss_ret - u_ret)
						*ss->data[ss_gld_i+l*Noccs+d]/Wl0_ss
					);
				else
					gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,0.0);

				double disc_cont = (1.0-fm_shr)*beta*pld*ss->data[ss_gld_i+l*Noccs+d];
				if(pld>0.0 && gsl_finite(disc_cont))
					gsl_matrix_set(Dst_tp1,Wl0_i+l,Wld_i+l*Noccs+d,disc_cont
							/Wl0_ss*Wld_ss
						);
				else
					gsl_matrix_set(Dst_tp1,Wl0_i+l,Wld_i+l*Noccs+d,0.0);
			}
	}
	return status;
}


int sys_co_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	int l,d,dd,status;
	int tld_i,gld_i,sld_i,Wld_i,Wl0_i;//x_i,
	double Z,zd;
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);
	sld_i	= tld_i + Noccs*(Noccs+1);

	int ss_tld_i,ss_gld_i, ss_sld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i
	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
	ss_sld_i	= ss_tld_i + Noccs*(Noccs+1);

	gsl_vector * ret_d = gsl_vector_calloc(Noccs);
	status =0;
	Z = xx->data[0];

	// 1st order derivatives
	for(l=0;l<Noccs+1;l++){

		double Wl0_ss = ss->data[ss_Wl0_i+l];

		double bl = l>0 ? b[1]:b[0];
		double findrtl = 0.0;
		for(d=0;d<Noccs;d++)
			findrtl += effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+ pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi)*ss->data[ss_gld_i+l*Noccs+d];
		double gdenom = 0.0;
		for(d=0;d<Noccs;d++){
			double nud = l ==d+1? 0.0 : nu;
			zd = xx->data[d+Notz];
			double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			//contval = contval <0.0 ? 0.0 : contval;
			double pld = pmatch(ss->data[ss_tld_i+l*Noccs+d]);
			ret_d->data[d] = (1.0-fm_shr)*(chi[l][d]*exp(Z+zd)-bl+nud+beta*contval)
					+ bl + beta*ss->data[ss_Wl0_i+l];
			gdenom +=exp(sig_psi*effic*pld*ret_d->data[d]);
		}

		for(d=0;d<Noccs;d++){
			zd = xx->data[d+Notz];
			double nud = l==d+1 ? 0: nu;
			double Wld_ss = ss->data[ss_Wld_i +l*Noccs+d];
			// tld
			double tld_ss = ss->data[ss_tld_i+l*Noccs+d];
			gsl_matrix_set(Dco,tld_i + l*Noccs+d,tld_i + l*Noccs+d, 1.0);
			if(tld_ss>0.0){
				double surp = chi[l][d] - nud - bl + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				//dt/dWld
				double qhere = kappa/(fm_shr*surp);
				double dtld = -beta*kappa/fm_shr*pow(surp,-2)*dinvq(qhere);
				// DO I NEED THIS SAFETY?
				dtld = surp<=0 ? 0.0 : dtld;

				if(gsl_finite(dtld) && surp>0.0)
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, dtld
							*Wld_ss/tld_ss
							);
				else
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, 0.0);
				//dt/dWl0
				dtld *= -1.0;
				if(gsl_finite(dtld) && surp>0.0)
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wl0_i + l,dtld
							*Wl0_ss/tld_ss
							);
				else
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wl0_i + l,0.0);
			}//tld_ss >0
			else{
				gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wl0_i + l,0.0);
				gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, 0.0);
			}

			// gld
			double gld_ss = ss->data[ss_gld_i+l*Noccs+d];
			gsl_matrix_set(Dco,gld_i+l*Noccs+d,gld_i+l*Noccs+d,1.0);

			if(gld_ss>0 && tld_ss>0){
				// exp(sig_psi*effic*pow(ss->data[tld_i+l*Noccs+d],1.0-phi)*(ss->data[wld_i+l*Noccs+d]-b+nud+beta*(ss->data[Uld_i+l*Noccs+d]-ss->data[Ul0_i+l])))
				double pld		= effic*tld_ss/pow(1.0+pow(tld_ss,phi),1.0/phi);
				double post		= pld>0 ? - kappa*tld_ss/pld : 0.0;
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);

				double arr_d 	= (1.0-fm_shr)*(chi[l][d]*exp(Z+zd) -bl -nud + beta*contval)
									+ bl + ss->data[ss_Wl0_i+l];

				//double ret_d 	= pld*arr_d;
				double dgdWld = beta*sig_psi*pld*(1.0-fm_shr)*gld_ss*(1.0-gld_ss);
				if(ret_d->data[d]>0)
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,dgdWld
							*Wld_ss/gld_ss
							);
				else
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,0.0);
				/*
				 * dg/dWl0 = 0
				*/
				// dg/dt


				double dpdt = dpmatch(ss->data[ss_tld_i+l*Noccs+d]);
				double dgdt =sig_psi*(dpdt*ret_d->data[d]
				//		- kappa*pow(tld_ss,phi)/(1.0 + pow(tld_ss,phi))
						)
						*gld_ss*(1.0-gld_ss);
				if(tld_ss>0.0 && ret_d->data[d]>0.0)
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+d,-1.0*dgdt
						/gld_ss*tld_ss
						);
				else
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+d,0.0);
			}
			else{
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+d,0.0);
				gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,0.0);
			}//gld_ss>0

			for(dd=0;dd<Noccs;dd++){
			if(dd!=d){
				if(ss->data[ss_gld_i+l*Noccs+d]>0.0 && ss->data[ss_tld_i+l*Noccs+dd]>0.0){
					double nudd = l==dd+1 ? 0: nu;
					double zdd = xx->data[dd+Notz];
					double contval = (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
					double pldd		= pmatch(ss->data[ss_tld_i+l*Noccs+dd]);
					double postdd	=  pldd>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+dd]/pldd : 0.0;
					double arr_dd 	= (1.0-fm_shr)*(chi[l][dd]*exp(Z+zdd)-bl - nudd+beta*contval)
										+ bl + ss->data[ss_Wl0_i+l];
					//double ret_dd 	= pldd*arr_dd;

					double dpdt		= dpmatch(ss->data[ss_tld_i+l*Noccs+dd]);
					double dgdWld	= -pldd*beta*sig_psi*(1.0-fm_shr)
							*ss->data[ss_gld_i+l*Noccs+dd]*ss->data[ss_gld_i+l*Noccs+d];
					if(ret_d->data[dd]>0.0 && ss->data[ss_tld_i+l*Noccs+dd]>0.0)
						gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,dgdWld
							/gld_ss*ss->data[ss_Wld_i+l*Noccs+dd]
							);

					else
						gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,0.0);
					double dgdtdd = -sig_psi*(dpdt*ret_d->data[dd]
						//	-kappa*pow(ss->data[ss_tld_i+l*Noccs+dd],phi)/(1.0 + pow(ss->data[ss_tld_i+l*Noccs+dd],phi))
							)
							*gld_ss*ss->data[ss_gld_i+l*Noccs+dd];
					if(ret_d->data[dd]>0.0 && ss->data[ss_tld_i+l*Noccs+dd]>0.0)
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,-1.0*dgdtdd
							*ss->data[ss_tld_i +l*Noccs+dd]/gld_ss
							);
					else
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,0.0);
				} //gld_ss >0
				else{
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,0.0);
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,0.0);
				}
			}//if dd!=d
			}//for dd=1:J

			// sld
			gsl_matrix_set(Dco,sld_i +l*Noccs+d,sld_i +l*Noccs+d,1.0);
			double cut 	= ss->data[ss_Wl0_i+l]-exp(Z+zd)*chi[l][d]-beta*ss->data[ss_Wld_i +l*Noccs+d];
			cut = cut > 0.0 ? 0.0 : cut;
			double dF 	= scale_s*shape_s*exp(shape_s*cut);
			// ds/dWld
			double dsdWld = -beta*dF;
			if(ss->data[ss_tld_i+l*Noccs+d]>0.0)
				gsl_matrix_set(Dst_tp1,sld_i+l*Noccs+d,Wld_i+l*Noccs+d,dsdWld
					*ss->data[ss_Wld_i+l*Noccs+d]/ss->data[ss_sld_i+l*Noccs+d]
					);
			// ds/dWl0
			double dsdWl0 =  dF;
			if(ss->data[ss_tld_i+l*Noccs+d]>0.0)
				gsl_matrix_set(Dst,sld_i+l*Noccs+d,Wl0_i+l,dsdWl0
					*ss->data[ss_Wl0_i+l]/ss->data[ss_sld_i+l*Noccs+d]
					);

		}
	}

	gsl_vector_free(ret_d);
	return status;
}


int sys_ex_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco){
// derivatives for exog
	int tld_i,gld_i,sld_i,	Wld_i,Wl0_i;//x_i,
//	int Notz = 1 + Nagf + Nfac*(Nflag+1);
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);
	sld_i	= tld_i + Noccs*(Noccs+1);

	int ss_tld_i, ss_gld_i, ss_sld_i,ss_Wld_i,ss_Wl0_i;
	int l,d,dd,status;
	//ss_x_i	= 0;
	ss_Wl0_i	= 0; //x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
	ss_sld_i	= ss_tld_i + Noccs*(Noccs+1);

	status =0;

	for (l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double gdenom = 0.0;
		for(d=0;d<Noccs;d++){
			double pld 	= pmatch(ss->data[ss_tld_i+l*Noccs+d]);
			double post	= pld>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld : 0.0;
			double nud 	= l ==d+1? 0.0 : nu;
			double ret_d= (1.0-fm_shr)*(chi[l][d]-bl- nud + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]))
								+bl+ss->data[ss_Wl0_i+l];
			ret_d 		= ret_d <0.0 ? 0.0 : ret_d;
			gdenom +=exp(sig_psi*pld*ret_d);
		}
		double dWl0dZ =0.0;
		for(d=0;d<Noccs;d++)
			dWl0dZ += ss->data[ss_gld_i+l*Noccs+d]*chi[l][d]*(1.0-fm_shr)*pmatch(ss->data[ss_tld_i+l*Noccs+d]);
		gsl_matrix_set(Dst,Wl0_i+l,0,dWl0dZ
				/ss->data[ss_Wl0_i+l]
				);
		for(d=0;d<Noccs;d++){

			double nud = l ==d+1? 0.0 : nu;
			double pld 	= pmatch(ss->data[ss_tld_i+l*Noccs+d]);
			double post	= pld>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld : 0.0;
								// dWl0/dz
			double dWdz =ss->data[ss_gld_i+l*Noccs+d]*(1.0-fm_shr)*chi[l][d]*pld;

			gsl_matrix_set(Dst,Wl0_i+l,d+Notz,dWdz
					/ss->data[ss_Wl0_i+l]
					);

			if(ss->data[ss_tld_i+l*Noccs+d]>0.0){
				double surp = (chi[l][d]-nud- bl +beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]));
				//surp= surp<0 ? 0.0 : surp;
				double dqdz = -chi[l][d]*kappa/(fm_shr*surp*surp);
				double dtdz =dqdz*dinvq(kappa/(fm_shr*surp));
				if( surp>0 && gsl_finite(dtdz)){
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,dtdz   // dtheta/dZ
						/ss->data[ss_tld_i+l*Noccs+d]
						);
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,d+Notz,dtdz// dtheta/dz
						/ss->data[ss_tld_i+l*Noccs+d]
						);
				}
				else{
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,0.0);
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,d+Notz,0.0);
				}
			}//tld_ss >0
			else{
				gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,0.0);
				gsl_matrix_set(Dco,tld_i + l*Noccs+d,d+Notz,0.0);
			}

			//Wld
			if(ss->data[ss_tld_i + l*Noccs+d]>0.0){
				if(l!=d+1){
					// dWld/dZ
					double dWlddZ =(1.0- ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+
									(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d];
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,dWlddZ
							/ss->data[ss_Wld_i+l*Noccs+d]
							);
					double dWlddzd =((1.0-ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+
							(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d]) ;
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,dWlddzd
							/ss->data[ss_Wld_i+l*Noccs+d]
							);
				}
				else{
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,(1.0-ss->data[ss_sld_i + l*Noccs+d])*chi[l][d]
							/ss->data[ss_Wld_i+l*Noccs+d]
							);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,(1.0-ss->data[ss_sld_i + l*Noccs+d])*chi[l][d]
							// dWld/dzd
							/ss->data[ss_Wld_i+l*Noccs+d]
							);
				}
			}// Wld, tld_ss >0

			// gld
			if(ss->data[ss_gld_i+l*Noccs+d]>0.0 && ss->data[ss_tld_i+l*Noccs+d]>0.0){
				double contval  = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				//contval	= contval>0.0? contval : 0.0;
				//double ret_d 	= (1.0-fm_shr)*(chi[l][d] -bl - nud +beta*contval)
				//						+bl + ss->data[ss_Wl0_i+l];

				double dgdzd =sig_psi*pld*(1.0-fm_shr)*chi[l][d]*ss->data[ss_gld_i+l*Noccs+d]*(1.0-ss->data[ss_gld_i+l*Noccs+d]);
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,dgdzd
					/ss->data[ss_gld_i+l*Noccs+d]
					);
			}
			else{
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,0.0);
			}
			for(dd=0;dd<Noccs;dd++){
			if(dd!=d){
				if(ss->data[ss_gld_i+l*Noccs+d]>0.0 && ss->data[ss_tld_i+l*Noccs+dd]>0.0){
					double nudd 	= l==dd+1 ? 0: nu;
					double p_dd		= pmatch(ss->data[ss_tld_i+l*Noccs+dd]);
					double postdd	= p_dd > 0 ? - kappa*ss->data[ss_tld_i+l*Noccs+dd]/p_dd : 0.0;
					double contval	= (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
					//contval = contval<0.0 ? 0.0 : contval;
					//double ret_dd 	= (1.0-fm_shr)*(chi[l][dd]-bl - nudd +beta*contval)
					//					+bl + ss->data[ss_Wl0_i+l];

					double dgdzdd =-1.0*sig_psi*(1.0-fm_shr)*chi[l][dd]*
									ss->data[ss_gld_i+l*Noccs+d]*ss->data[ss_gld_i+l*Noccs+dd];
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,dd+Notz,dgdzdd
							/ss->data[ss_gld_i+l*Noccs+d]
							);
				}
				else
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,dd+Notz,0.0);
			}
			}

			// sld
			double cut 	= ss->data[ss_Wl0_i+l]-chi[l][d]-beta*ss->data[ss_Wld_i +l*Noccs+d];
			cut = cut > 0.0 ? 0.0 : cut;
			double dF 	= scale_s*shape_s*exp(shape_s*cut);
			// ds/dZ
			double dsdZ = -dF;
			if(ss->data[ss_tld_i+l*Noccs+d]>0.0){
				gsl_matrix_set(Dco,sld_i+l*Noccs+d,0,dsdZ
						/ss->data[ss_sld_i+l*Noccs+d]
					);
				// ds/dzd
				gsl_matrix_set(Dco,sld_i+l*Noccs+d,Notz+d,dsdZ
						/ss->data[ss_sld_i+l*Noccs+d]
					);
			}
		}
	}

	return status;
}
int ss_moments(struct aux_coef * ssdat, gsl_vector * ss, gsl_matrix * xss){
	/* will compute moments that match
	 * avg_sep
	 * avg_fnd
	 * avg_urt
	 * avg_wg
	 *
	 */
	int status,l,d;
	double s_fnd,s_wg, s_chng;
	int ss_tld_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i
	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
	double ** pld = malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		pld[l]= malloc(sizeof(double)*Noccs);


	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);
	status = 0;

	// calculate the unemployment rate:
	double urtss = 0.0;
	for(l=0;l<Noccs+1;l++)
		urtss += gsl_matrix_get(xss,l,0);
	gsl_vector * x_u = gsl_vector_calloc(Noccs+1);
	for(l=0;l<Noccs+1;l++)
		gsl_vector_set(x_u,l,gsl_matrix_get(xss,l,0)/urtss);
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			pld[l][d] = effic*ss->data[ss_tld_i+l*Noccs+d]/
				pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
	}

	s_fnd = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			fnd_l->data[l] += ss->data[ss_gld_i+l*Noccs+d]*pld[l][d];
		}
		s_fnd += fnd_l->data[l]*gsl_vector_get(x_u,l);
	}

	//generate the wage growth rate and subtract from tau

	s_wg = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			s_wg += (chi[d+1][d] - chi[l][d])*gsl_vector_get(x_u,l)*pld[l][d];

		}
	}
	s_wg *= tau; // dividing by expected time for that growth: 1.0/tau
	s_wg += 1.0;
	//generate the change probability:
	for(l=1;l<Noccs+1;l++){
		double s_chng_l = 1.0 - (ss->data[ss_gld_i+l*Noccs+(l-1)]*pld[l][d])/
				fnd_l->data[l];
		s_chng += s_chng_l*x_u->data[l];
	}
	s_chng /= (1.0 - x_u->data[0]);

	gsl_matrix_set(ssdat->data_moments,0,0,avg_wg  - s_wg);
	gsl_matrix_set(ssdat->data_moments,0,1,avg_urt - urtss);
	gsl_matrix_set(ssdat->data_moments,0,2,chng_pr - s_chng);

	for(l=0;l<Noccs+1;l++)
		free(pld[l]);
	free(pld);
	gsl_vector_free(fnd_l);
	return status;
}


int sim_moments(struct st_wr * st, gsl_vector * ss,gsl_matrix * xss){
	/* will compute moments that match
	 * avg_sep
	 * avg_fnd
	 * avg_elas
	 * avg_wg
	 *
	 */

	int status,l,d,di,si,Ndraw;
	double s_fnd,s_urt,s_wg,s_chng,m_Zz, s_cvu,s_sdu,s_sdZz,s_elpt,s_sdsep,s_sep;
	int ss_gld_i, ss_Wld_i,ss_Wl0_i;
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Noccs+1;
	gsl_matrix * xp 	= gsl_matrix_alloc(Noccs+1,Noccs+1);
	gsl_matrix * x 		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_matrix * wld 	= gsl_matrix_calloc(Noccs+1,Noccs);
	gsl_vector * Zz 	= gsl_vector_calloc(Nx);
	gsl_vector * fnd_l 	= gsl_vector_calloc(Noccs+1);
	double ** pld 		= malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		pld[l]=malloc(sizeof(double)*Noccs);
	struct sys_sol * sol 	= st->sol;
	struct sys_coef * sys 	= st->sys;
	struct aux_coef * simdat= st->sim;
	status = 0;
	int bad_coeffs=0;

	Ndraw = (simdat->draws)->size1;
	gsl_vector * shocks 	= gsl_vector_calloc(Nx);
	gsl_vector * Zzl		= gsl_vector_calloc(Nx);
	gsl_matrix * uf_hist	= gsl_matrix_calloc(Ndraw,4);
	gsl_matrix * urt_l, * urt_l_wt, * fnd_l_hist, * x_u_hist, * Zzl_hist, * s_wld,*tll_hist,*tld_hist,*gld_hist,*pld_hist,*fac_hist;
	if(printlev>=2){
		urt_l		= gsl_matrix_calloc(Ndraw,Noccs);
		fac_hist	= gsl_matrix_calloc(Ndraw,Nfac);
		fnd_l_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
		tll_hist	= gsl_matrix_calloc(Ndraw,Noccs);
		tld_hist	= gsl_matrix_calloc(Ndraw,Noccs*(Noccs+1));
		gld_hist	= gsl_matrix_calloc(Ndraw,Noccs*(Noccs+1));
		pld_hist	= gsl_matrix_calloc(Ndraw,Noccs*(Noccs+1));

		x_u_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
		Zzl_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
		s_wld		= gsl_matrix_calloc(Noccs+1,Noccs);
		urt_l_wt 	= gsl_matrix_calloc(Ndraw,Noccs);
	}
	struct dur_moments s_mom;
	if(printlev>=2){
		s_mom.Ft = gsl_vector_calloc(5);
		s_mom.Ft_occ = gsl_vector_calloc(5);
		s_mom.dur_hist = gsl_vector_calloc(Ndraw);
		s_mom.dur_l_hist = gsl_matrix_calloc(Ndraw,Noccs+1);
		s_mom.dur_l = malloc(sizeof(double)*(Noccs+1));
		for(l=0;l<Noccs+1;l++)
			s_mom.dur_l[l]=0.0;
	}
	double urtss = 0.0;
	for(l=0;l<Noccs+1;l++)
		urtss += gsl_matrix_get(xss,l,0);
	if(urtss>0.01 && urtss<0.1)
		gsl_matrix_memcpy(x,xss);
	else{
		for(l=0;l<Noccs+1;l++){
			gsl_matrix_set(x,l,0,0.055/((double)Noccs+1.0));
			for(d=1;d<Noccs+1;d++)
				gsl_matrix_set(x,l,d,0.945/((double)(Noccs*Noccs+Noccs)) );
		}
	}

	s_fnd 	= 0.0;
	s_elpt	= 0.0;
	s_mom.chng_wg	= 0.0;
	s_chng	= 0.0;
	s_urt	= 0.0;
	s_sep	= 0.0;
	s_sdsep	= 0.0;
	m_Zz	= 0.0;

	// allocate stuff for the regressions:
	gsl_matrix * XX = gsl_matrix_calloc(Noccs*(Noccs-1),Nskill+1);
	gsl_vector_view X0 = gsl_matrix_column(XX,Nskill);
	gsl_vector_set_all(&X0.vector,1.0);

	gsl_matrix * Wt	= gsl_matrix_calloc(Noccs*(Noccs-1),Noccs*(Noccs-1));
	gsl_vector * yloss = gsl_vector_alloc(Noccs*(Noccs-1));
	gsl_vector * coefs	= gsl_vector_calloc(Nskill + 1);
	gsl_vector * coefs_di	= gsl_vector_calloc(Nskill + 1);
	gsl_vector * er = gsl_vector_alloc(yloss->size);

	// Take a draw for Zz and initialize things
	int init_T = 500;

	gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,0);

	gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0, Zz);
	if(printlev>=3)
		printvec("Zdraws.csv",&Zdraw.vector);
	//run through a few draws without setting anything
	gsl_matrix_memcpy(x,xss);
	sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
	sol->gld = gsl_matrix_calloc(Noccs+1,Noccs);
	sol->sld = gsl_matrix_calloc(Noccs+1,Noccs);
	FILE * zzhist_init;
	if(printlev>=3) zzhist_init = fopen("zzhist.csv","a+");
	for(di=0;di<init_T;di++){

		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);
		status += xprime(xp,ss,sys,sol,x,Zz);

		gsl_matrix_memcpy(x,xp);

		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);
		if(printlev>=3){
			for(l=0;l< Zz->size;l++)
				fprintf(zzhist_init,"%f,",Zz->data[l]);
			fprintf(zzhist_init,"\n");
		}
	}
	if(printlev>=3) fclose(zzhist_init);
	for(di = 0;di<Ndraw;di++){
		//set up error logging
		gpol_zeros 	= 0;
		t_zeros 	= 0;

		double fac_ave = (double)Ndraw/((double)(di-1));

		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);

		if( printlev>=1 && (gpol_zeros>0 || t_zeros>0)){
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"%d times g^ld = 0 in sim # %d \n",gpol_zeros,di);
			fprintf(simerr,"%d times t^ld = 0 in sim # %d \n",t_zeros,di);
			fclose(simerr);
		}

		status += xprime(xp,ss,sys,sol,x,Zz);
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				pld[l][d] = pmatch(sol->tld->data[l*sol->tld->tda+d]);
		}
		if(printlev>=4){
			printmat("sol->tld.csv",sol->tld);
			printmat("sol->gld.csv",sol->gld);
			printmat("xp.csv",xp);
			printmat("xl.csv",x);
		}
		if(printlev>=2){
			for(d=0;d<Nfac;d++)
				gsl_matrix_set(fac_hist,di,d,Zz->data[1+d]);
			for(d=1;d<Noccs+1;d++){
				gsl_matrix_set(urt_l,di,d-1,gsl_matrix_get(xp,d,0)/(gsl_matrix_get(xp,d,0) + gsl_matrix_get(xp,d,d)));
				gsl_matrix_set(urt_l_wt,di,d-1,gsl_matrix_get(xp,d,0)+gsl_matrix_get(xp,d,d));
				gsl_matrix_set(tll_hist,di,d-1,0.0);
				tll_hist->data[di*tll_hist->tda+d-1] = gsl_matrix_get(sol->tld,d,d-1);
				for(l=0;l<Noccs+1;l++){
					tld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = gsl_matrix_get(sol->tld,l,d-1);
					gld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = gsl_matrix_get(sol->gld,l,d-1);
					pld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = pld[l][d-1];
				}

			}
		}

		// mean Z this period
		double m_Zz_t = 0.0;
		for(d=0;d<Noccs;d++)
			m_Zz_t += gsl_vector_get(Zz,Notz+d)/(double)Noccs;
		m_Zz_t 	/= 2.0;
		m_Zz_t 	+= gsl_vector_get(Zz,0)/2.0;
		m_Zz	+= m_Zz_t/(double)Ndraw;
		gsl_matrix_set(uf_hist,di,0,m_Zz_t);

		// calculate the unemployment rate:

		double urtp = 0.0;
		for(l=0;l<Noccs+1;l++)
			urtp += gsl_matrix_get(xp,l,0);
		s_urt += urtp/((double) Ndraw);
		gsl_matrix_set(uf_hist,di,1,urtp);

		gsl_vector * x_u = gsl_vector_calloc(Noccs+1);
		for(l=0;l<Noccs+1;l++){
			if(urtp>0.0)
				gsl_vector_set(x_u,l,gsl_matrix_get(xp,l,0)/urtp);
			else
				gsl_vector_set(x_u,l,0.0);
		}
		if(printlev>=2){
			gsl_vector_view x_u_d = gsl_matrix_row(x_u_hist,di);
			gsl_vector_memcpy(&x_u_d.vector,x_u);
		}

		// average tightness
		double m_tld =0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				m_tld += gsl_matrix_get(sol->gld,l,d)*gsl_vector_get(x_u,l)*sol->tld->data[l*Noccs+d];
		}
		gsl_matrix_set(uf_hist,di,3,m_tld);


		// get the finding rate:
		double d_fnd =0.0;
		for(l=0;l<Noccs+1;l++){
			fnd_l->data[l] = 0.0;
			for(d=0;d<Noccs;d++){
				fnd_l->data[l] += gsl_matrix_get(sol->gld,l,d)*pld[l][d];
			}
			d_fnd += fnd_l->data[l]*gsl_vector_get(x_u,l);
		}
		d_fnd = gsl_finite(d_fnd)==1? d_fnd : s_fnd*fac_ave;

		s_fnd += d_fnd/(double)Ndraw;

		gsl_matrix_set(uf_hist,di,2,d_fnd);
		if(printlev>=2){
			gsl_vector_view fnd_l_di = gsl_matrix_row(fnd_l_hist,di);
			gsl_vector_memcpy(&fnd_l_di.vector,fnd_l);
		}

		//elasticity of p wrt theta
		double x_elpt	= 0.0;
		double d_elpt	= 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				d_elpt += gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d)*
						1.0/(1.0+ pow(gsl_matrix_get(sol->tld,l,d),phi));
				x_elpt += gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d);
			}
		}
		if(x_elpt>0.0)
			d_elpt /= x_elpt;
		d_elpt = gsl_finite(d_elpt)==0 ? s_elpt*fac_ave : d_elpt;
		s_elpt += d_elpt/(double)Ndraw;

		//separation rate and sd of it
		double d_sep=0,d_sdsep=0;
		for(l=0;l<Noccs+1;l++){
			for(d=1;d<Noccs;d++){
				if(l!=d)
					d_sep += gsl_matrix_get(x,l,d)*((1.0-tau)*gsl_matrix_get(sol->sld,l,d-1)
							+ tau*gsl_matrix_get(sol->sld,d,d-1));
				else
					d_sep += gsl_matrix_get(x,l,d)*gsl_matrix_get(sol->sld,l,d-1);
			}
		}
		d_sep = gsl_finite(d_sep) == 1? d_sep : s_sep*fac_ave;
		s_sep += d_sep/(double)Ndraw;

		for(l=0;l<Noccs+1;l++){
			for(d=1;d<Noccs;d++){
				if(l!=d)
					d_sdsep += gsl_matrix_get(x,l,d)*pow( ((1.0-tau)*gsl_matrix_get(sol->sld,l,d-1)
						+ tau*gsl_matrix_get(sol->sld,d,d-1)) - d_sep, 2);
				else
					d_sdsep += gsl_matrix_get(x,l,d)*pow(gsl_matrix_get(sol->sld,l,d-1) -d_sep,2);
			}
		}
		d_sdsep =  pow(d_sdsep,0.5);
		d_sdsep = gsl_finite(d_sdsep) == 1 ? d_sdsep : s_sdsep*fac_ave;
		s_sdsep += d_sdsep/(double)Ndraw;


		//generate the change probability:
		double d_chng = 0.0;
		for(l=1;l<Noccs+1;l++){
			double d_chng_l = 1.0 - (gsl_matrix_get(sol->gld,l,l-1)*pld[l][l-1] )/
				fnd_l->data[l];
			d_chng += d_chng_l*x_u->data[l];
		}
		double d_chng_0=0.0;
		double inexper = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=1;d<Noccs+1;d++){
				if(l!=d)
					inexper += gsl_matrix_get(x,l,d);
			}
		}
		for(d=0;d<Noccs;d++){
			double d_chng_0x = 0.0;
			double inexp_d = 0.0;
			//find the number of inexperienced not in this occupation out of the total number of inexperienced
			for(l=1;l<Noccs+1;l++){
				if(l!=d+1)
					inexp_d += gsl_matrix_get(x,l,d)/inexper;
					//d_chng_0x += gsl_matrix_get(x,l,d)/gsl_vector_get(x_u,0);
			}
			d_chng_0x = (1.0-inexp_d);
			d_chng_0 += gsl_matrix_get(sol->gld,0,d)*pld[0][d]*d_chng_0x / fnd_l->data[0];
		}
		d_chng += d_chng_0*x_u->data[0];

		/*double max0 = 0.0;
		for(d=0;d<Noccs;d++)
			max0 = (gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi))>max0 ?
					(gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi)):max0;
		d_chng += (1.0-max0/fnd_l->data[0])*x_u->data[0];
		*/
		//d_chng /= (1.0 - x_u->data[0]);
		d_chng = gsl_finite(d_chng)==0 ? s_chng*fac_ave : d_chng;
		s_chng +=d_chng/(double)Ndraw;


		//	Wage growth

		double d_wg = 0.0;
		double x_wg = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l-1!=d){
					d_wg += gsl_vector_get(x_u,l)*(gsl_matrix_get(sol->gld,l,d))*pld[l][d]*
							(gsl_matrix_get(sol->ss_wld,d+1,d) - gsl_matrix_get(sol->ss_wld,l,d));
					x_wg += gsl_vector_get(x_u,l)*(gsl_matrix_get(sol->gld,l,d))*pld[l][d];
				}
			}
		}
		d_wg *= tau; // dividing by expected time for that growth: 1.0/tau
		if(x_wg>0.0)
			d_wg /= x_wg;
		else
			d_wg = 0;
		d_wg = gsl_finite(d_wg) == 0  ? s_wg*fac_ave : d_wg;
		s_wg += d_wg/(double)Ndraw;

		// wage loss due to change
		double d_wl = 0.0;
		double sx_wl= 0.0;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					d_wl +=gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d)*pld[l][d] *(gsl_matrix_get(sol->ss_wld,l,l-1) - gsl_matrix_get(sol->ss_wld,l,d));
					sx_wl +=gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d)*pld[l][d];
				}
			}
		}
		if(sx_wl>0.0)
			d_wl /=sx_wl;
		else
			d_wl = 0.0;
		s_mom.chng_wg += d_wl/(double)Ndraw;

	 	if((st->cal_set<=0 || st->cal_set==2)&& d_chng>1e-4){
		// do a regression on the changes in wages:
		// only look at experienced changers, and set chi[0][d] s.t. makes the avg wage loss match
			int Xrow =0;
			gsl_matrix_set_zero(Wt);

			for(l=1;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					if(l-1!=d){
						double ylossld = log(gsl_matrix_get(sol->ss_wld,l,d)/gsl_matrix_get(sol->ss_wld,d+1,d));

						double wgt = pld[l][d]*gsl_matrix_get(sol->gld,l,d)*x_u->data[l]/(1.0-x_u->data[0])/d_chng;
						if(gsl_finite(ylossld) && ylossld<0.0 && wgt>1e-5 && gsl_finite(wgt)){
							gsl_matrix_set(Wt,Xrow,Xrow,wgt);
							gsl_vector_set(yloss,Xrow,ylossld);
							for(si=0;si<Nskill;si++)
								gsl_matrix_set(XX,Xrow,si,
									(gsl_matrix_get(f_skills,d,si+1)-gsl_matrix_get(f_skills,l-1,si+1) ));
							Xrow++;
						}
					}
				}
			}
			gsl_vector_view y_v = gsl_vector_subvector (yloss, 0, Xrow);
			gsl_matrix_view X_v = gsl_matrix_submatrix(XX,0,0,Xrow,XX->size2);
			gsl_matrix_view W_v = gsl_matrix_submatrix(Wt,0,0,Xrow,Xrow);
			double Wsum=0.0;
			for(l=0;l<Xrow;l++) Wsum+=gsl_matrix_get(&W_v.matrix,l,l);
			gsl_matrix_scale(&W_v.matrix,1.0/Wsum);
			if(printlev>=4){
				printmat("XXloss.csv",&X_v.matrix);
				printmat("Wt.csv",&W_v.matrix);
				printvec("yloss.csv",&y_v.vector);
			}

			status += WOLS( &y_v.vector, &X_v.matrix ,&W_v.matrix, coefs_di, er);
			gsl_vector_scale(coefs_di,1.0/(double)Ndraw);
			int nfin_coefs = isfinvec(coefs_di);
			if(nfin_coefs == 0)
				gsl_vector_add(coefs,coefs_di);
			else{
				bad_coeffs++;
				gsl_vector_memcpy(coefs_di,coefs);
				gsl_vector_scale(coefs_di, fac_ave  );
				gsl_vector_add(coefs,coefs_di);
			}
		}


		if(printlev>=2){
		// compute wages in  this period

			gsl_vector * Es = gsl_vector_calloc(Ns);
			gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
			status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

			for(l=0;l<Noccs+1;l++){
				double bl = l==0 ? b[0] : b[1];
				for(d=0;d<Noccs;d++){
					// need to use spol to get \bar \xi^{ld}, then evaluate the mean, \int_{-\bar \xi^{ld}}^0\xi sh e^{sh*\xi}d\xi
					//and then invert (1-scale_s)*0 + scale_s*log(sld/shape_s)/scale_s
					double barxi = -log(gsl_matrix_get(sol->sld,l,d)/shape_s )/shape_s;
					double Exi = scale_s*((1.0/shape_s+barxi)*exp(-shape_s*barxi)-1.0/shape_s);
					double bEJ = beta*fm_shr*(Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i+l]);  // fill this in with shr*E(surp)::: not sure if this is correct

					double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Zz->data[0]+Zz->data[Notz+d]) +
							bEJ - fm_shr*beta*(Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i+l]) - fm_shr*(bl+Exi);
					wld_ld = wld_ld >0.0 ? wld_ld : 0.0;
					gsl_matrix_set(wld,l,d,wld_ld);
				}
			}
			s_wld->data[l*wld->tda+d] += wld->data[l*wld->tda+d]/(double) Noccs;
			gsl_vector_free(Es);
		}

		// advance a period:
		gsl_matrix_memcpy(x,xp);
		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);

		if(printlev>=2){
		gsl_matrix_set(Zzl_hist,di,0,Zz->data[0]);
			for(l=0;l<Noccs;l++)
				gsl_matrix_set(Zzl_hist,di,l+1,Zz->data[Notz+l]);
		}
		gsl_vector_free(x_u);
	}// for d<Ndraw


	if(printlev>=2){
		printmat("pld_hist.csv",pld_hist);
		printmat("gld_hist.csv",gld_hist);
		printmat("xp.csv",xp);
		printmat("wld.csv",wld);
		printmat("Zzl_hist.csv",Zzl_hist);
		printmat("tll_hist.csv",tll_hist);
		printmat("fac_hist.csv",fac_hist);
		printmat("tld_hist.csv",tld_hist);
		printmat("x_u_hist.csv",x_u_hist);
		printmat("fnd_l_hist.csv",fnd_l_hist);
		printmat("uf_hist.csv",uf_hist);
		printmat("s_wld.csv",s_wld);
		printmat("urt_l.csv",urt_l);
		printmat("urt_l_wt.csv",urt_l_wt);
	}


	int fincoefs = 1;
	for(di=0;di<coefs->size-1;di++)
		fincoefs *= gsl_finite(gsl_vector_get(coefs,di));
	if(fincoefs!=0){
		for(di=0;di<coefs->size-1;di++)
			gsl_matrix_set(simdat->data_moments,1,di,(coefs->data[di]- sk_wg[di])/fabs(sk_wg[di]));
	}
	else{
		gsl_vector_view momcoef = gsl_matrix_row(simdat->data_moments,1);
		gsl_vector_scale(&momcoef.vector,1.5);
	}


	if(printlev>=2){
		//for(di=0;di<pld_hist->size1;di++){
		//	for(l=0;l<pld_hist->size2;l++)
		//		gsl_matrix_set(pld_hist,di,l,gsl_matrix_get(pld_hist,di,l)/s_fnd*avg_fnd);
		//}
		status += dur_dist(&s_mom,gld_hist,pld_hist,x_u_hist);
		gsl_vector_view dur_lv = gsl_vector_view_array(s_mom.dur_l,Noccs+1);
		printvec("mod_dur_l.csv",&dur_lv.vector);
		printvec("Ft.csv",s_mom.Ft);
		printvec("Ft_occ.csv",s_mom.Ft_occ);
		printvec("dur_hist.csv",s_mom.dur_hist);
		printmat("dur_l_hist.csv",s_mom.dur_l_hist);
		FILE * durstats = fopen("durstats.csv","w+");
		printf("Printing durstats.csv\n");
		fprintf(durstats,"E[dur],Pr[>=6mo],sd(dur),D(wg)\n");
		fprintf(durstats,"%f,%f,%f,%f\n",s_mom.E_dur,s_mom.pr6mo,s_mom.sd_dur,s_mom.chng_wg);

		s_cvu	= 0.0;
		s_sdZz	= 0.0;
		for(di=0;di<uf_hist->size1;di++){
			s_cvu += pow(gsl_matrix_get(uf_hist,di,1) - s_urt,2);
			s_sdZz+= pow(gsl_matrix_get(uf_hist,di,0) - m_Zz,2);
		}
		s_cvu 	/=(double)uf_hist->size1;
		s_sdZz	/=(double)uf_hist->size1;
		s_sdu	= sqrt(s_cvu);
		s_sdZz	= sqrt(s_sdZz);
		s_cvu 	= s_sdu/s_urt;
	}
	if(verbose >=3) printf("The average wage-loss was %f\n",s_mom.chng_wg);
	int finmoments = 1;
	finmoments *= gsl_finite(s_wg);
	finmoments *= gsl_finite(s_fnd);
	finmoments *= gsl_finite(s_chng);
	finmoments *= gsl_finite(s_urt);
	finmoments *= gsl_finite(s_elpt);
	finmoments *= gsl_finite(s_sdsep);

	if(status == 0 && finmoments!=0){
//		gsl_matrix_set(simdat->data_moments,0,0,(avg_wg  - s_wg)/avg_wg);
		gsl_matrix_set(simdat->data_moments,0,0,(avg_fnd - s_fnd)/avg_fnd); // this one is much more important (arbitrary)
//		gsl_matrix_set(simdat->data_moments,0,2,(chng_pr - s_chng)/chng_pr);
		gsl_matrix_set(simdat->data_moments,0,1,(avg_urt - s_urt)/avg_urt);
		gsl_matrix_set(simdat->data_moments,0,2,(avg_elpt - s_elpt)/avg_elpt);
		gsl_matrix_set(simdat->data_moments,0,3,(avg_sdsep - s_sdsep)/avg_sdsep);
	}
	else{// penalize it for going in a region that can't be solved
		gsl_matrix_scale(simdat->data_moments,5.0);
		status ++;
	}

	if(verbose>=3){
		printf("(obj-sim)/obj : ");
//		printf("wg=%f, ",gsl_matrix_get(simdat->data_moments,0,0));
		printf("fnd=%f, ",gsl_matrix_get(simdat->data_moments,0,0));
//		printf("chng=%f, ",gsl_matrix_get(simdat->data_moments,0,2));
		printf("urt=%f, ",gsl_matrix_get(simdat->data_moments,0,1));
		printf("elpt=%f, ",gsl_matrix_get(simdat->data_moments,0,2));
		printf("sdsep=%f, ",gsl_matrix_get(simdat->data_moments,0,3));
		if(printlev>=2)printf("cvu=%f, ",s_cvu);
		printf("\n");
		double dispobj =0.0;
		for(l=0;l<(simdat->data_moments)->size2;l++)
			dispobj += (simdat->moment_weights)->data[l]*
				pow((simdat->data_moments)->data[l],2);

		printf("obj = %f\n",dispobj);
	}

	if(verbose>=3){
		printf("wage reg: ");
		for(di=0;di<Nskill;di++)printf("%f,",(sk_wg[di] - coefs->data[di])/sk_wg[di]);
		printf("\n");
	}
	printf(".");
	// Cleaning up!
	// !!!!!!!!!!!!
	gsl_matrix_free(sol->tld);
	sol->tld=NULL;
	gsl_matrix_free(sol->gld);
	sol->gld=NULL;
	gsl_matrix_free(sol->sld);
		sol->sld=NULL;

	for(l=0;l<Noccs+1;l++)
		free(pld[l]);
	free(pld);
	gsl_matrix_free(Wt);
	//gsl_matrix_free(XX);
	gsl_vector_free(coefs);
	gsl_vector_free(coefs_di);
	gsl_vector_free(yloss);
	gsl_vector_free(er);
	if(printlev>=2){
		gsl_vector_free(s_mom.Ft);
		gsl_vector_free(s_mom.dur_hist);
		gsl_matrix_free(s_mom.dur_l_hist);
		free(s_mom.dur_l);
	}
	gsl_vector_free(fnd_l);
	gsl_vector_free(Zzl);
	gsl_vector_free(shocks);
	gsl_matrix_free(uf_hist);
	if(printlev>=2){
		gsl_matrix_free(Zzl_hist);
		gsl_matrix_free(fac_hist);
		gsl_matrix_free(s_wld);
		gsl_matrix_free(urt_l);
		gsl_matrix_free(fnd_l_hist);
		gsl_matrix_free(tll_hist);
		gsl_matrix_free(tld_hist);
		gsl_matrix_free(x_u_hist);
	}
	gsl_matrix_free(x);
	gsl_matrix_free(xp);
	gsl_vector_free(Zz);
	gsl_matrix_free(wld);
	return status;
}


int dur_dist(struct dur_moments * s_mom, const gsl_matrix * gld_hist, const gsl_matrix * pld_hist, const gsl_matrix * x_u_hist){
	int status,l,d,duri,t,tp;
	s_mom->E_dur	= 0.0;
	s_mom->sd_dur	= 0.0;
	s_mom->pr6mo	= 0.0;

	status =0;

	int Ndraw = x_u_hist->size1;

	double durs[] = {1.0,3.0,6.0,12.0,24.0};
	double * d_dur_l = malloc(sizeof(double)*(Noccs+1));

	gsl_vector_set_zero(s_mom->Ft);

	for(duri =0;duri <5;duri ++){
		double Ft_ld = 0.0;
		double xt_ld = 0.0;
		double Ft_occ = 0.0;
		double xt_occ = 0.0;
		for(t=0;t<Ndraw;t++){
			for(l=0;l<Noccs+1;l++){
				double fnd_occ = 0.0;
				for(d=0;d<Noccs;d++){
					fnd_occ += gsl_matrix_get(gld_hist,t,l*Noccs+d)*gsl_matrix_get(pld_hist,t,l*Noccs+d);
					Ft_ld += gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d)
							*exp(- gsl_matrix_get(pld_hist,t,l*Noccs+d)*durs[duri])*gsl_matrix_get(pld_hist,t,l*Noccs+d);
					xt_ld += gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d)
							*exp(- gsl_matrix_get(pld_hist,t,l*Noccs+d)*durs[duri]);
				}
				Ft_occ += gsl_matrix_get(x_u_hist,t,l)*exp(- fnd_occ*durs[duri])*fnd_occ;
				xt_occ += gsl_matrix_get(x_u_hist,t,l)*exp(- fnd_occ*durs[duri]);
			}
		}
		Ft_ld	/= xt_ld;
		Ft_occ	/= xt_occ;
		gsl_vector_set(s_mom->Ft,duri,Ft_ld);
		gsl_vector_set(s_mom->Ft_occ,duri,Ft_occ);
	}
	for(t=0;t<Ndraw;t++){
		double d_dur = 0.0;
		for(l=0;l<Noccs+1;l++){
			d_dur_l[l] = 0.0;//gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d);

			for(tp=50;tp>0;tp--){
				int tidx = t-tp < 0 ? Ndraw + t - tp : t - tp;
				//int tidx = t+tp>Ndraw-1 ? t - Ndraw + tp : t + tp;
				double d_dur_ld = 0.0;
				double x_dur_l = 0.0;
				for(d=0;d<Noccs;d++){
					x_dur_l	 += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d));
					d_dur_ld += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d))
							*(gsl_matrix_get(pld_hist,tidx,l*Noccs+d)*((double)tp)
							+ (1.0 - gsl_matrix_get(pld_hist,tidx,l*Noccs+d))*d_dur_l[l]);
				}
				d_dur_l[l] = d_dur_ld/x_dur_l;
			}
			d_dur += d_dur_l[l]*gsl_matrix_get(x_u_hist,t,l);
			s_mom->dur_l[l] +=d_dur_l[l]/((double) Ndraw);

			gsl_matrix_set(s_mom->dur_l_hist,t,l,d_dur_l[l]);
		}
		gsl_vector_set(s_mom->dur_hist,t,d_dur);
		s_mom->E_dur += d_dur/((double) Ndraw);


		double d_sddur = 0.0;
		for(l=0;l<Noccs+1;l++){
			double d_sddur_l = 0.0;//gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d);
			double fnd_l = 0.0;
			for(tp=50;tp>0;tp--){
				int tidx = t-tp < 0 ? Ndraw + t - tp : t - tp;
				//int tidx = t+tp>Ndraw-1 ? t - Ndraw + tp : t + tp;
				double d_sddur_ld = 0.0;
				double x_sddur_ld = 0.0;
				for(d=0;d<Noccs;d++){
					x_sddur_ld += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d));
					d_sddur_ld += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d))
							*( gsl_matrix_get(pld_hist,tidx,l*Noccs+d)*pow((double)tp -d_dur_l[l],2)
							+ (1.0 - gsl_matrix_get(pld_hist,tidx,l*Noccs+d))*d_sddur_l);
				}
				d_sddur_l = d_sddur_ld/x_sddur_ld;
			}
			for(d=0; d<Noccs;d++)
				fnd_l += gsl_matrix_get(gld_hist,t,l*Noccs+d)*gsl_matrix_get(pld_hist,t,l*Noccs+d);

			//d_sddur_l = pow(fnd_l,-2);
			d_sddur +=(pow( d_dur_l[l] - d_dur,2) + d_sddur_l)*gsl_matrix_get(x_u_hist,t,l);
		}

		s_mom->sd_dur += d_sddur/(double)Ndraw;
		double d_6mo = 0.0;
		for(l=0;l<Noccs+1;l++){
			double d_6mo_l = 1.0;//gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d);
			for(tp=6;tp>0;tp--){
				int tidx = t+tp>Ndraw-1 ? t - Ndraw + tp : t + tp;
				for(d=0;d<Noccs;d++){
					d_6mo_l += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)
							*(1.0 - gsl_matrix_get(pld_hist,tidx,l*Noccs+d))*d_6mo_l;
				}
			}
			d_6mo += d_6mo_l*gsl_matrix_get(x_u_hist,t,l);
		}
		s_mom->pr6mo += d_6mo/(double)Ndraw;
	}
	s_mom->sd_dur =pow(s_mom->sd_dur,0.5);
	free(d_dur_l);
	return status;
}



int TGR(gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		double * urt, double * opt_urt,
		struct st_wr * st){

	int status,l,d,t,Tmo,di;
	status	= 0;
	Tmo 	= 24;
	int Ndraw = (st->sim->draws->size1)/Tmo;
	gsl_matrix * xp = gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_matrix * x = gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_matrix * r_occ_wr = gsl_matrix_alloc(9416,7);
	gsl_vector * Zz_2008 	= gsl_vector_alloc(Nx);
	gsl_vector * Zz_2009 	= gsl_vector_alloc(Nx);
	gsl_vector * Zz_2010 	= gsl_vector_alloc(Nx);
	gsl_vector * Zzl	= gsl_vector_alloc(Nx);
	gsl_matrix * Zhist	= gsl_matrix_alloc(24,Nx);
	gsl_matrix * Zave	= gsl_matrix_calloc(24,Nx);

	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);
	gsl_vector * av_fnd_l = gsl_vector_calloc(Noccs+1);
	gsl_vector * x_u 	= gsl_vector_calloc(Noccs+1);
	struct sys_sol * sol = st->sol;
	struct sys_coef * sys = st->sys;
	gsl_vector * ss = st->ss;

	double s_wl		= 0.0;
	double s_fnd	= 0.0;
	double s_sw 	= 0.0;

	gsl_vector * sw_hist = gsl_vector_calloc(Tmo*Ndraw);
	gsl_vector * wl_hist = gsl_vector_calloc(Tmo*Ndraw);
	gsl_matrix * fnd_l_hist	= gsl_matrix_calloc(Tmo*Ndraw,Noccs+1);
	gsl_matrix * pld_hist	= gsl_matrix_calloc(Tmo*Ndraw,(Noccs+1)*Noccs);
	gsl_matrix * gld_hist	= gsl_matrix_calloc(Tmo*Ndraw,(Noccs+1)*Noccs);
	gsl_matrix * x_l_hist	= gsl_matrix_calloc(Tmo*Ndraw,Noccs+1);
	gsl_vector * urt_hist	= gsl_vector_calloc(Tmo*Ndraw);
	gsl_vector * shocks 	= gsl_vector_calloc(Nx);
	readmat("occ_wr.csv",r_occ_wr);
	// skim until peak unemployment in June 2008
	for(l=0;l<r_occ_wr->size1;l++){
		int date = (int) rint(gsl_matrix_get(r_occ_wr,l,0));
		if( date == 200806 ){
			d = gsl_matrix_get(r_occ_wr,l,1);
			gsl_matrix_set(x,d,0,gsl_matrix_get(r_occ_wr,l,4));
			gsl_matrix_set(x,d,d,gsl_matrix_get(r_occ_wr,l,3));
		}
	}
	double fr_x00 = 0.4;
	double fr_xdd = 0.5;
	double usum = 0.0;
	double uread = 0.0;
	for(l=1;l<Noccs+1;l++){
		usum += gsl_matrix_get(x,l,0);
		uread+= gsl_matrix_get(x,l,0);
		usum += gsl_matrix_get(x,l,l);
	}
	gsl_matrix_set(x,0,0,uread/usum*fr_x00);
	for(l=1;l<Noccs+1;l++){
		gsl_matrix_set(x,l,0,(1.0-fr_x00)*gsl_matrix_get(x,l,0)/usum);
		double xll_t = gsl_matrix_get(x,l,l)/usum;
		gsl_matrix_set(x,l,l,(1.0-fr_xdd)*xll_t);
		d = l==1 ? 2 : 1;
		gsl_matrix_set(x,d,l,fr_xdd*xll_t);
	}
	// check the sum:
	if(printlev>=2)
		printmat("xTGR0.csv",x);
	double sx0 = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs+1;d++)
			sx0 += gsl_matrix_get(x,l,d);
	}
	*urt = 0.0;
	for(l=0;l<Noccs+1;l++)
		*urt+=gsl_matrix_get(x,l,0);
	//readvec("outzz.csv",Zz_TGR);
	readvec("outzz2008.csv",Zz_2008);
	readvec("outzz2009.csv",Zz_2009);
	readvec("outzz2010.csv",Zz_2010);

	if(sol->gld!=NULL)
		gsl_matrix_free(sol->gld);
	if(sol->sld!=NULL)
			gsl_matrix_free(sol->sld);
	if(sol->tld!=NULL)
			gsl_matrix_free(sol->tld);
	*urt = 0.0;
	sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
	sol->gld = gsl_matrix_calloc(Noccs+1,Noccs);
	sol->sld = gsl_matrix_calloc(Noccs+1,Noccs);
	for(di=0;di<Ndraw;di++){
		double d_wl		= 0.0;
		double d_fnd	= 0.0;
		double d_sw 	= 0.0;

		// Zz_TGR "stochastic" impulse response
		gsl_vector_view Zt = gsl_matrix_row(Zhist,0);
		// set this at 2008 levels
		gsl_vector_memcpy(&Zt.vector,Zz_2008);
		for(t=0;t<Tmo-1;t++){
			gsl_vector_view Zdraw = gsl_matrix_row(st->sim->draws,di);
			gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
			Zt = gsl_matrix_row(Zhist,t);
			gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, &Zt.vector, 0.0,Zzl);
			gsl_vector_add(Zzl,shocks);
			Zt = gsl_matrix_row(Zhist,t+1);
			gsl_vector_memcpy(&Zt.vector,Zzl);
		}
		if(printlev>=4)
				printmat("Zhist_TGR.csv",Zhist);

		// adjust to 2010 and trajectory
		for(l=0;l<Nx;l++){
			double zd2009 	= gsl_matrix_get(Zhist,11,l);
			double slope0	= (zd2009-gsl_matrix_get(Zhist,0,l))/12;
			gsl_matrix_set(Zhist,11,l,gsl_vector_get(Zz_2009,l));
			double slope1 	= (gsl_vector_get(Zz_2009,l)-gsl_vector_get(Zz_2008,l))/12;
			for(t=0;t<12;t++){
				double Zhist_d = gsl_matrix_get(Zhist,t,l);
				gsl_matrix_set(Zhist,t,l, Zhist_d-( slope0 - slope1)*((double)t)  );
				Zave->data[t*Nx + l] += Zhist->data[t*Nx +l]/(double)Ndraw;
			}
			double zd2010 	= gsl_matrix_get(Zhist,Tmo-1,l);
			slope0	= (zd2010-zd2009)/12;
			gsl_matrix_set(Zhist,Tmo-1,l,gsl_vector_get(Zz_2010,l));
			slope1 	= (gsl_vector_get(Zz_2010,l)- gsl_vector_get(Zz_2009,l))/12;
			/*for(t=12;t<Tmo;t++){
				double Zhist_d = gsl_matrix_get(Zhist,t,l);
				gsl_matrix_set(Zhist,t,l, Zhist_d -( slope0 - slope1)*((double)t) );
				Zave->data[t*Nx + l] += Zhist->data[t*Nx +l]/(double)Ndraw;
			}*/
			for(t=12;t<Tmo;t++){
				double Zhist_d = gsl_matrix_get(Zhist,t,l);
				gsl_matrix_set(Zhist,t,l, Zhist_d + zd2009 );
				Zave->data[t*Nx + l] += Zhist->data[t*Nx +l]/(double)Ndraw;
			}
		}
		if(printlev>=4)
			printmat("Zhist_TGR.csv",Zhist);

		for(t=0;t<Tmo;t++){// this is going to loop through 2 years of recession
			gsl_vector_view Zz_t = gsl_matrix_row(Zhist,t);
			status += theta(sol->tld,ss, sys, sol, &Zz_t.vector);
			status += gpol(sol->gld,ss, sys,  sol, &Zz_t.vector);
			status += spol(sol->sld,ss, sys,  sol, &Zz_t.vector);
			status += xprime(xp, ss, sys,  sol, x, &Zz_t.vector);

			// store the policies for the dur_dist later
			for(l=0;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					gsl_matrix_set(pld_hist,di*Tmo+t,l*Noccs+d,pmatch(gsl_matrix_get(sol->tld,l,d)) );
					gsl_matrix_set(gld_hist,di*Tmo+t,l*Noccs+d,gsl_matrix_get(sol->gld,l,d) );
				}
			}
		}
		s_wl /= x_wl;
		wl_hist->data[t] = s_wl;
		// advance to next step
		gsl_matrix_memcpy(x,xp);

		// Zz_TGR impulse response
		//Zz_TGR->data[0] *= -1.0;
//		for(d=0;d<Noccs;d++)
//			Zz_TGR->data[1+d+Nfac*Nflag] *= -1.0;
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz_TGR, 0.0,Zzl);
		gsl_vector_memcpy(Zz_TGR,Zzl);

			urt_hist->data[t] =0.0;
			for(l=0;l<Noccs+1;l++)
				urt_hist->data[t+di*Tmo] += gsl_matrix_get(xp,l,0);
			*urt += urt_hist->data[t+di*Tmo]/(double)Tmo;
			for(l=0;l<Noccs+1;l++)
				gsl_vector_set(x_u,l,gsl_matrix_get(xp,l,0)/urt_hist->data[t+di*Tmo]);
			gsl_vector_view x_l_t = gsl_matrix_row(x_l_hist,di*Tmo + t);
			gsl_vector_memcpy(&x_l_t.vector,x_u);
			for(l=0;l<Noccs+1;l++){
				double fnd_l_t_l = 0.0;
				for(d=0;d<Noccs;d++)
					fnd_l_t_l += gsl_matrix_get(sol->gld,l,d)*gsl_matrix_get(pld_hist,di*Tmo+t,l*Noccs+d);
				gsl_vector_set(fnd_l,l,fnd_l_t_l);
			}
			double d_fnd = 0.0;
			for(l=0;l<Noccs+1;l++)
				d_fnd += fnd_l->data[l]*x_u->data[l];
			s_fnd += d_fnd/(double)(Tmo*Ndraw);

			gsl_vector_view fnd_l_t = gsl_matrix_row(fnd_l_hist,di*Tmo + t);
			gsl_vector_memcpy(&fnd_l_t.vector,fnd_l);

			// calculate the number of switches:
			sw_hist->data[di*Tmo + t] = 0.0;

			for(l=0;l<Noccs+1;l++){
				double sw_l = 0.0;
				for(d=0;d<Noccs;d++){
					if(l!=d+1)
						sw_l+=gsl_matrix_get(sol->gld,l,d)*gsl_matrix_get(pld_hist,di*Tmo+t,l*Noccs+d);

				}
				sw_l /= fnd_l->data[l];
				d_sw += sw_l*x_u->data[l];
				sw_hist->data[di*Tmo + t]+= sw_l*gsl_vector_get(x_u,l);
			}
			s_sw += d_sw/(double)(Tmo*Ndraw);


			//wage loss USING STEADY STATE WAGES!!!!!!  Need to fix this by re-computing wages here
			double x_wl = 0.0;
			for(l=1;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					if(l!=d+1){
						d_wl += (gsl_matrix_get(st->sol->ss_wld,l,l-1) - gsl_matrix_get(st->sol->ss_wld,l,d))
								*gsl_vector_get(x_u,l)*gsl_matrix_get(pld_hist,di*Tmo+t,l*Noccs+d);
						x_wl += gsl_vector_get(x_u,l)*gsl_matrix_get(pld_hist,di*Tmo+t,l*Noccs+d);
					}
				}
			}
			d_wl /= x_wl;
			wl_hist->data[di*Tmo+t] = d_wl;
			s_wl += d_wl/(double)(Ndraw*Tmo);
			// advance to next step in x
			gsl_matrix_memcpy(x,xp);


		}// end for t=0:Tmo
	}// end for di =0:Ndraws
	gsl_matrix_free(sol->gld);gsl_matrix_free(sol->tld);gsl_matrix_free(sol->sld);
	sol->gld = NULL;sol->tld = NULL;sol->sld = NULL;

	struct dur_moments dur_TGR;
	dur_TGR.Ft = gsl_vector_alloc(5);
	dur_TGR.Ft_occ = gsl_vector_alloc(5);
	dur_TGR.dur_l_hist = gsl_matrix_calloc(Tmo*Ndraw,Noccs+1);
	dur_TGR.dur_hist = gsl_vector_calloc(Tmo*Ndraw);
	dur_TGR.dur_l = malloc(sizeof(double)*Noccs+1);
	status += dur_dist(&dur_TGR, gld_hist,pld_hist,x_l_hist);

	printvec("Ft_TGR.csv",dur_TGR.Ft);
	printvec("Ft_occ_TGR.csv",dur_TGR.Ft_occ);
	gsl_vector_view dur_v = gsl_vector_view_array(dur_TGR.dur_l,Noccs+1);
	printvec("dur_l_TGR.csv",&dur_v.vector);
	printvec("urt_TGR.csv",urt_hist);
	printvec("sw_TGR.csv",sw_hist);
	printvec("wl_TGR.csv",wl_hist);
	printmat("Zave_TGR.csv",Zave);
	printmat("fnd_l_TGR.csv",fnd_l_hist);
	printmat("x_l_TGR.csv",x_l_hist);
	printmat("dur_dist_TGR.csv",dur_dist_hist);

	// other duration stats
	FILE * durstatsTGR = fopen("durstats_TGR.csv","w+");
	fprintf(durstatsTGR,"E[dur],Pr[>=6mo],sd(dur),D(wg),ave fnd\n");
	fprintf(durstatsTGR,"%f,%f,%f,%f,%f\n",dur_TGR.E_dur,dur_TGR.pr6mo,dur_TGR.sd_dur,dur_TGR.chng_wg,s_fnd);

	fclose(durstatsTGR);

	gsl_vector_free(dur_TGR.Ft);gsl_vector_free(dur_TGR.Ft_occ);
	gsl_matrix_free(dur_TGR.dur_l_hist);free(dur_TGR.dur_l);
	gsl_vector_free(dur_TGR.dur_hist);

	gsl_vector_free(fnd_l);gsl_vector_free(av_fnd_l);gsl_matrix_free(fnd_l_hist);
	gsl_vector_free(x_u);
	gsl_matrix_free(r_occ_wr);gsl_matrix_free(xp);gsl_matrix_free(x);
	gsl_vector_free(Zzl);gsl_matrix_free(x_l_hist);
	gsl_matrix_free(Zhist);gsl_vector_free(Zz_2009);gsl_vector_free(Zz_2010);
	gsl_vector_free(sw_hist);gsl_vector_free(wl_hist);
	gsl_matrix_free(pld_hist);gsl_matrix_free(gld_hist);
	return status;
}



/*
 * Policy functions
 */
int gpol(gsl_matrix * gld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d;
	status=0;
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Noccs+1;
	int ss_gld_i,ss_tld_i, ss_Wld_i,ss_Wl0_i;

		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
		ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	double *gld_l 	= malloc(sizeof(double)*Noccs);

	double Z = Zz->data[0];
	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);

	status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

	gsl_matrix * pld = gsl_matrix_calloc(Noccs+1,Noccs);

	// make sure not negative value anywhere:
	/*for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			Es->data[Wld_i + l*Noccs+d] = Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l]<0 ?
				Es->data[Wl0_i + l] : Es->data[Wld_i + l*Noccs+d];
	}
	*/
	if(sol->tld==0)
		status += theta(pld,ss,sys,sol,Zz);
	else
		gsl_matrix_memcpy(pld,sol->tld);
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			double tld = gsl_matrix_get(pld,l,d);
			double pld_ld = pmatch(tld);
			//pld_ld = gsl_finite(pld_ld) ? pld_ld : 0.0;
			gsl_matrix_set(pld,l,d,pld_ld);
		}
	}

	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double gdenom =0;
		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0.0 : nu;
			double zd 	= Zz->data[d+Notz];
			double post = -kappa*gsl_matrix_get(sol->tld,l,d)/gsl_matrix_get(pld,l,d);
			double cont	= Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l];
			//double ret_d 	= (1.0 - fm_shr)*(chi[l][d]*exp(Z + zd) - bl - nud +post + beta*cont) + bl + Es->data[Wl0_i + l];
			double ret_d 	= (1.0-fm_shr)*(chi[l][d]*exp(Z + zd) - bl - nud + beta*cont)
									+ bl + Es->data[Wl0_i + l];
			double gld_ld 	= exp(sig_psi*gsl_matrix_get(pld,l,d)*ret_d);
			if(gsl_matrix_get(pld,l,d)>0  && ret_d>0 && gsl_finite(gld_ld))//  && ss->data[ss_tld_i+l*Noccs+d]>0.0)
				gld_l[d]= gld_ld;
			else
				gld_l[d] = 0.0;
			gdenom += gld_l[d];
		}
		if(gdenom>0){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(gld,l,d,gld_l[d]/gdenom);
		}
		else if(l>0){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(gld,l,d,0.0);
			gsl_matrix_set(gld,l,l-1,1.0);
			gpol_zeros ++;

		}
		else{
			// should never end up here.  If so, what's up?
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(gld,l,d,1.0/(double)Noccs);
			gpol_zeros ++;
		}
	}

	free(gld_l);
	gsl_vector_free(Es);
	gsl_matrix_free(pld);
	return status;
}
int spol(gsl_matrix * sld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d;
	status =0;
	gsl_matrix * gld;
	gsl_matrix * pld = gsl_matrix_calloc(Noccs+1,Noccs);

	int Wl0_i = 0;
		int Wld_i = Wl0_i + Noccs+1;
		int ss_gld_i, ss_tld_i, ss_sld_i,ss_Wld_i,ss_Wl0_i;
			//ss_x_i		= 0;
			ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
			ss_Wld_i	= ss_Wl0_i + Noccs+1;
			ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
			ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
			ss_sld_i 	= ss_tld_i + Noccs*(Noccs+1);
	double Z = Zz->data[0];
	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
	status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

	/*gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_blas_dgemv(CblasNoTrans, 1.0, sol->PP, Zz, 0.0, Es);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-Wl0_i);
	for(l=0;l<Ns;l++)
		Es->data[l] =exp(Es->data[l]);
	status += gsl_vector_mul(Es,&ss_W.vector);
	 */
	if(sol->tld==0)
		status += theta(pld,ss,sys,sol,Zz);
	else
		gsl_matrix_memcpy(pld,sol->tld);
	if(sol->gld==0){
		gld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += gpol(gld,ss,sys,sol,Zz);
	}
	else
		gld = sol->gld;

	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			gsl_matrix_set(pld,l,d,effic*gsl_matrix_get(pld,l,d)/
					pow(1.0+pow(gsl_matrix_get(pld,l,d),phi),1.0/phi));
	}
	//#pragma omp parallel for private(l,d)
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double W0 = 0.0;
		double fnd_rt = 0.0;
		for(d=0;d<Noccs;d++){
			fnd_rt += gsl_matrix_get(gld,l,d)*gsl_matrix_get(pld,l,d);
			double zd = Zz->data[d+Notz];
			double nud = l == d+1? 0:nu;
//			double post = -kappa*gsl_matrix_get(sol->tld,l,d)/gsl_matrix_get(pld,l,d);

			double ret	= chi[d][d]*exp(Z+zd) - nud
							+ beta*Es->data[Wld_i + l*Noccs+d]
							- bl - beta*Es->data[Wl0_i + l];
			ret *= (1.0- fm_shr);
			ret += bl + beta*Es->data[Wl0_i + l];
			W0+= gsl_matrix_get(gld,l,d)*gsl_matrix_get(pld,l,d)*ret;
		}
		W0 += (1.0-fnd_rt)*(bl + beta*Es->data[Wl0_i + l]);
		for(d=0;d<Noccs;d++){
			double zd = Zz->data[d+Notz];

			double Wdiff	= beta*Es->data[Wld_i+l*Noccs+d] - W0;

			double cutoff= - (chi[l][d]*exp(Z+zd)+ Wdiff);
			cutoff = cutoff >0.0 ? 0.0 : cutoff;
			double sep_ld = scale_s*exp(shape_s*cutoff);
			if(cutoff<0.0)
				gsl_matrix_set(sld,l,d,sep_ld);
			else
				gsl_matrix_set(sld,l,d,scale_s); // should this be lower?  This means negative value for being here...
		}
	}

	gsl_vector_free(Es);

	if(sol->gld==0)
		gsl_matrix_free(gld);
	gsl_matrix_free(pld);
	return status;
}

int theta(gsl_matrix * tld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d;
	status=0;
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Noccs+1;
	int ss_gld_i, ss_tld_i, ss_sld_i,ss_Wld_i,ss_Wl0_i;
		//ss_x_i		= 0;
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
		ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);
		ss_sld_i 	= ss_tld_i + Noccs*(Noccs+1);

	double Z = Zz->data[0];
	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
	status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double tld_s = 0.0;
		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0:nu;
			double zd = Zz->data[d+Notz];
			double cont	= (Es->data[Wld_i+l*Noccs+d] - Es->data[Wl0_i+l]);
			//cont = ss->data[ss_Wld_i+l*Noccs+d] -ss->data[ss_Wl0_i+l];
			double surp =chi[l][d]*exp(Z+zd) - bl - nud + beta*cont;

			double qhere = kappa/(fm_shr*surp);
			double tld_i = invq(qhere);

			//if(effic>1.0)
			//	tld_i = tld_i > pow(pow(effic,phi)-1.0,-1.0/phi) ? pow(pow(effic,phi)-1.0,-1.0/phi) : tld_i;
			if(gsl_finite(tld_i) && surp > 0.0 && ss->data[ss_tld_i+l*Noccs+d]>0.0){
				gsl_matrix_set(tld,l,d,tld_i);
				tld_s += tld_i;
			}
			else
				gsl_matrix_set(tld,l,d,0.0);
		}
		if(tld_s <= 0.0){
			t_zeros ++;
			if(l>0)
				gsl_matrix_set(tld,l,l-1,ss->data[ss_tld_i + l*Noccs + (l-1)]);
			else{
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(tld,l,d,ss->data[ss_tld_i+d]);
			}
		}
	}

	gsl_vector_free(Es);
	return status;
}

int xprime(gsl_matrix * xp, gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_matrix * x, const gsl_vector * Zz){
	int status,j,k,l,d;
	double ** ald;
	double * findrt;
	status=0;
	double ** pld	= malloc(sizeof(double*)*(Noccs+1));
	gsl_matrix * tld,*gld,*sld;
	for(l=0;l<Noccs+1;l++)
		pld[l] = malloc(sizeof(double)*Noccs);

	// define theta and g
	if(sol->tld==0){
		tld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += theta(tld, ss, sys, sol, Zz);
	}
	else
		tld = sol->tld;
	if(sol->gld==0){
		gld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += gpol(gld, ss, sys, sol, Zz);
	}
	else
		gld = sol->gld;
	if(sol->sld==0){
		sld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += spol(sld, ss, sys, sol, Zz);
	}
	else
		sld = sol->sld;
	double newdisp = 0.0;
	for(k=0;k<Noccs+1;k++){
		for(j=1;j<Noccs+1;j++){
			if(j!=k)
				newdisp += gsl_matrix_get(sld,k,j-1)*(1.0-tau)*gsl_matrix_get(x,k,j);
		}
	}

	ald  =  malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		ald[l] = malloc(sizeof(double)*Noccs);
	for(d=0;d<Noccs;d++){
		ald[0][d] = gsl_matrix_get(gld,0,d)*(gsl_matrix_get(x,0,0) + newdisp);
	}
	for(l=1;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			ald[l][d] = gsl_matrix_get(gld,l,d)*(gsl_matrix_get(x,l,0) + gsl_matrix_get(sld,l,l-1)*gsl_matrix_get(x,l,l));
			for(k=0;k<Noccs+1;k++){
				if(k!=l)
					ald[l][d]+= gsl_matrix_get(gld,l,d)*tau*gsl_matrix_get(sld,l,l-1)*gsl_matrix_get(x,k,l);
			}
		}
	}
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			pld[l][d]=effic*
					gsl_matrix_get(tld,l,d)/
					pow(1.0+ pow(gsl_matrix_get(tld,l,d),phi) ,1.0/phi);
	}

	findrt = malloc(sizeof(double)*(Noccs+1));
	for(l=0;l<Noccs+1;l++){
		findrt[l]=0.0;
		for(d=0;d<Noccs;d++)
			findrt[l] += pld[l][d]*gsl_matrix_get(gld,l,d);
	}

	//x00
	gsl_matrix_set(xp,0,0,
		(1.0-findrt[0])*(gsl_matrix_get(x,0,0) + newdisp)
		);
	//xl0
	for(l=1;l<Noccs+1;l++){
		double sxpjl = 0.0;
		for(j=0;j<Noccs+1;j++)
			sxpjl =j!=l? gsl_matrix_get(x,j,l)+sxpjl : sxpjl;
		gsl_matrix_set(xp,l,0,
			(1.0-findrt[l])*(gsl_matrix_get(x,l,0) + gsl_matrix_get(sld,l,l-1)*(gsl_matrix_get(x,l,l) + tau*sxpjl))
			);
	}


	//xld : d>0
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			if(l!=d+1)
				gsl_matrix_set(xp,l,d+1,
					(1.0-tau)*(1.0-gsl_matrix_get(sld,l,d))*gsl_matrix_get(x,l,d+1) + pld[l][d]*ald[l][d]
					);
			else{
				double newexp = 0.0;
				for(j=0;j<Noccs+1;j++)
					newexp = j!=d+1 ? tau*gsl_matrix_get(x,j,d+1)+newexp : newexp;
				gsl_matrix_set(xp,l,d+1,
					(1.0-gsl_matrix_get(sld,l,d))*(gsl_matrix_get(x,l,d+1)+newexp )+ pld[l][d]*ald[l][d]
					);
			}
		}
	}
	// normalize to sum to 1:
	double xsum = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs+1;d++){
			xsum += gsl_matrix_get(xp,l,d);
		}
	}
	gsl_matrix_scale(xp,1.0/xsum);
	if(xsum>1.00001 || xsum<0.99999){
		if(verbose >=1) printf("xsum = %f\n",xsum);
		simerr = fopen(simer_f,"a+");
		if(printlev >=1) fprintf(simerr,"xsum = %f\n",xsum);
		fclose(simerr);
	}
	for(l=0;l<Noccs+1;l++)
		free(ald[l]);
	free(ald);
	free(findrt);
	for(l=0;l<Noccs+1;l++)
		free(pld[l]);
	free(pld);
	if(sol->tld==0)
		gsl_matrix_free(tld);
	if(sol->gld==0)
		gsl_matrix_free(gld);
	if(sol->sld==0)
		gsl_matrix_free(sld);

	return status;
}
double gsl_endog_std(double x, void *p){
	struct st_wr * st = (struct st_wr *) p;
	double dev = endog_std(x,st->xss,st,st->d);
	return dev;
}


// Function takes in standard deviation of idiosync std z and std Z and computes the observed standard deviations
double endog_std(double stdZz, gsl_matrix * x0, struct st_wr * st, int pos){
	int status=0,l,i,d;
	// stdz scales the diagonal of the E[zeta zeta']

	(st->sol)->tld = gsl_matrix_calloc(Noccs+1,Noccs);
	(st->sol)->gld = gsl_matrix_calloc(Noccs+1,Noccs);
	(st->sol)->sld = gsl_matrix_calloc(Noccs+1,Noccs);

	gsl_matrix * xp = gsl_matrix_alloc(Noccs+1,Noccs+1);
	gsl_vector * Zzpstd = gsl_vector_calloc(Nx);
	double dev;

/*	steady state productivity:
 *  but this is actually not necessary
	double y_ss =0.0;
	double xe = 0.0;
	for(l=0;l<Noccs+1;l++){
		xe += gsl_matrix_get(x0,l,0);
		for(d=0;d<Noccs;d++){
			y_ss += gsl_matrix_get(x0,l,d+1)*chi[l][d];
		}
	}
	y_ss /=(1.0-xe);
*/
	if(pos==0){
		Zzpstd->data[0] = sqrt(sig_eps)*stdZz;
		status += theta((st->sol)->tld, st->ss, st->sys, st->sol, Zzpstd);
		status += gpol(st->sol->gld, st->ss, st->sys, st->sol, Zzpstd);
		status += spol(st->sol->sld, st->ss, st->sys, st->sol, Zzpstd);
		xprime(xp, st->ss, st->sys, st->sol, x0, Zzpstd);
		double xe = 0.0;
		double yp = 0.0;
		// compute productivity
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				xe += gsl_matrix_get(xp,l,d+1);
				yp += gsl_matrix_get(xp,l,d+1)*chi[l][d]*exp(Zzpstd->data[0]);
			}
		}
		yp /= xe;
		Zzpstd->data[0] = 0.0;
		//	and now negative productivity shocks
		Zzpstd->data[0] = -sqrt(sig_eps)*stdZz;
		status += theta((st->sol)->tld, st->ss, st->sys, st->sol, Zzpstd);
		status += gpol(st->sol->gld, st->ss, st->sys, st->sol, Zzpstd);
		status += spol(st->sol->sld, st->ss, st->sys, st->sol, Zzpstd);
		xprime(xp, st->ss, st->sys, st->sol, x0, Zzpstd);
		xe = 0.0;
		// compute productivity
		double ym0 = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				xe += gsl_matrix_get(xp,l,d+1);
				ym0 += gsl_matrix_get(xp,l,d+1)*chi[l][d]*exp(Zzpstd->data[0]);
			}
		}
		ym0 /= xe;
		yp -= ym0;
		yp /= 2.0;
		Zzpstd->data[0] = 0.0;
		double yp0 = (exp(sig_eps) -exp(-sig_eps))/2.0;
		dev = (yp0 - yp);
	}
	else{
		i = pos-1;
		double yp=0.0;
		Zzpstd->data[i+Notz] = gsl_matrix_get((st->sys)->S,i+Notz,i+Notz)*stdZz;
		status += theta((st->sol)->tld, st->ss, st->sys, st->sol, Zzpstd);
		status += gpol(st->sol->gld, st->ss, st->sys, st->sol, Zzpstd);
		status += spol(st->sol->sld, st->ss, st->sys, st->sol, Zzpstd);
		xprime(xp, st->ss, st->sys, st->sol, x0, Zzpstd);
		double xe = 0.0;
		for(l=0;l<Noccs+1;l++){
			xe += gsl_matrix_get(xp,l,i+1);
			yp += gsl_matrix_get(xp,l,i+1)*chi[l][i]*exp(Zzpstd->data[i+Notz]);
		}
		yp /= xe;
		Zzpstd->data[i+Notz] = 0.0;

// negative productivity shocks


		Zzpstd->data[i+Notz] = -gsl_matrix_get((st->sys)->S,i+Notz,i+Notz)*stdZz;
		status += theta((st->sol)->tld, st->ss, st->sys, st->sol, Zzpstd);
		status += gpol(st->sol->gld, st->ss, st->sys, st->sol, Zzpstd);
		status += spol(st->sol->sld, st->ss, st->sys, st->sol, Zzpstd);
		xprime(xp, st->ss, st->sys, st->sol, x0, Zzpstd);
		xe = 0.0;
		double ym0 = 0.0;
		for(l=0;l<Noccs+1;l++){
			xe += gsl_matrix_get(xp,l,i+1);
			ym0 += gsl_matrix_get(xp,l,i+1)*chi[l][i]*exp(Zzpstd->data[i+Notz]);
		}
		ym0 /= xe;
		yp -= ym0;
		yp /= 2.0;
		Zzpstd->data[i+Notz] = 0.0;
		double yp0 =(exp(gsl_matrix_get((st->sys)->S,i+Notz,i+Notz)) - exp(-gsl_matrix_get((st->sys)->S,i+Notz,i+Notz)))/2.0;
		dev = (yp0 - yp);
	}

	gsl_matrix_free((st->sol)->tld);(st->sol)->tld =NULL;
	gsl_matrix_free((st->sol)->sld);(st->sol)->sld =NULL;
	gsl_matrix_free((st->sol)->gld);(st->sol)->gld =NULL;
	return dev;
}


/*
 * Utilities
 *
 */


int Es_cal(gsl_vector * Es, const gsl_vector * ss_W, const gsl_matrix * PP, const gsl_vector * Zz){
	int status =0,l;
	gsl_blas_dgemv(CblasNoTrans, 1.0, PP, Zz, 0.0, Es);

	for(l=0;l<Ns;l++)
		Es->data[l] =exp(Es->data[l]);
	status += gsl_vector_mul(Es,ss_W);

//	status += gsl_vector_add(Es,ss_W);
	return status;
}
int Es_dyn(gsl_vector * Es, const gsl_vector * ss_W, const gsl_vector * Wlast ,const gsl_matrix * invP0P1, const gsl_matrix * invP0P2, const gsl_vector * Zz){
	int status =0,l;
	gsl_vector * Esp = gsl_vector_alloc(Es->size);
	gsl_blas_dgemv(CblasNoTrans, 1.0, invP0P2, Zz, 0.0, Es);
	gsl_blas_dgemv(CblasNoTrans, 1.0, invP0P1, Wlast, 0.0, Esp);
	gsl_vector_add(Es,Esp);
	for(l=0;l<Ns;l++)
		Es->data[l] =exp(Es->data[l]);
	status += gsl_vector_mul(Es,ss_W);
	//gsl_vector_set_zero(Es);
	gsl_vector_free(Esp);
	return status;
}

int alloc_econ(struct st_wr * st){
	int l,i,status;

	status =0;
	Ns = Noccs+1 + Noccs*(Noccs+1);//if including x: pow(Noccs+1,2) + Noccs+1 + 2*Noccs*(Noccs+1);
	Nc = 3*Noccs*(Noccs+1);
	Nx = Noccs + Nagf + Nfac*(Nllag+1);
	Notz 	= Nagf +Nfac*(Nllag+1);

	st->ss = gsl_vector_alloc(Ns + Nc);
	st->xss = gsl_matrix_calloc(Noccs+1,Noccs+1);

	// allocate system coefficients and transform:
	st->sys = malloc(sizeof(struct sys_coef));
	st->sol = malloc(sizeof(struct sys_sol));
	st->sim = malloc(sizeof(struct aux_coef));

	(st->sys)->COV = malloc(sizeof(double)*(Ns+Nc));
/*
 * System is:
 * 	A0 s =  A1 Es'+ A2 c + A3 Z
 * 	F0 c  = F1 Es'+ F2 s'+ F3 Z
 *
 */

	for(l=0;l<Ns + Nc;l++)
		(st->sys)->COV[l] = 1.0;
	(st->sys)->A0	= gsl_matrix_calloc(Ns,Ns);
	(st->sys)->A1	= gsl_matrix_calloc(Ns,Ns);
	(st->sys)->A2	= gsl_matrix_calloc(Ns,Nc);
	(st->sys)->A3	= gsl_matrix_calloc(Ns,Nx);

	(st->sys)->F0	= gsl_matrix_calloc(Nc,Nc);
	(st->sys)->F1	= gsl_matrix_calloc(Nc,Ns);
	(st->sys)->F2	= gsl_matrix_calloc(Nc,Ns);
	(st->sys)->F3	= gsl_matrix_calloc(Nc,Nx);

	(st->sys)->N  	= gsl_matrix_calloc(Nx,Nx);
	(st->sys)->S	= gsl_matrix_calloc(Nx,Nx);

/*
	P0*s' = P2*x + P1*s
*/
	(st->sol)->P0	= gsl_matrix_calloc(Ns,Ns);
	//(st->sol)->invP0P1	= gsl_matrix_calloc(Ns,Ns);
	(st->sol)->P1	= gsl_matrix_calloc(Ns,Ns);
	(st->sol)->P2	= gsl_matrix_calloc(Ns,Nx);
	//(st->sol)->invP0P2	= gsl_matrix_calloc(Ns,Nx);
	(st->sol)->PP	= gsl_matrix_calloc(Ns,Nx);
	(st->sol)->ss_wld = gsl_matrix_calloc(Noccs+1,Noccs);
	//(st->sol)->gld	= NULL;
	//(st->sol)->tld	= NULL;
	//(st->sol)->sld 	= NULL;
	simT = 2000;

	(st->sim)->data_moments 	= gsl_matrix_calloc(2,4); // pick some moments to match
	(st->sim)->moment_weights 	= gsl_matrix_calloc(2,4); // pick some moments to match
	(st->sim)->draws 			= gsl_matrix_calloc(simT,Nx); // do we want to make this nsim x simT

	randn((st->sim)->draws,6071984);
	if(printlev>2)
		printmat("draws.csv",(st->sim)->draws);

	gsl_matrix_set((st->sim)->data_moments,0,0,avg_fnd);
//	gsl_matrix_set((st->sim)->data_moments,0,2,chng_pr);
//	gsl_matrix_set((st->sim)->data_moments,0,0,avg_wg);
	gsl_matrix_set((st->sim)->data_moments,0,1,avg_urt);
	gsl_matrix_set((st->sim)->data_moments,0,2,avg_elpt);
	gsl_matrix_set((st->sim)->data_moments,0,3,avg_sdsep);
	// row 2 are the coefficients on 4 skills and a constant
	for(i=0;i<Nskill;i++)
		gsl_matrix_set((st->sim)->data_moments,1,i,sk_wg[i]);

//	gsl_matrix_set((st->sim)->moment_weights,0,0,1.0);
	gsl_matrix_set((st->sim)->moment_weights,0,0,10.0);
//	gsl_matrix_set((st->sim)->moment_weights,0,2,0.50);
	gsl_matrix_set((st->sim)->moment_weights,0,1,5.0);
	gsl_matrix_set((st->sim)->moment_weights,0,2,1.0);
	gsl_matrix_set((st->sim)->moment_weights,0,3,1.0);

	gsl_matrix_set((st->sim)->moment_weights,1,0,1.0);
	gsl_matrix_set((st->sim)->moment_weights,1,1,1.0);
	gsl_matrix_set((st->sim)->moment_weights,1,2,1.0);


	return status;
}


int free_econ(struct st_wr * st){
	int status;
	status = 0;
	if((st->xss)->block != NULL)
		gsl_matrix_free(st->xss);
	else
		status ++;
	if((st->ss)->block != NULL)
		gsl_vector_free(st->ss);
	else
		status ++;
	gsl_matrix_free( (st->sol)->P0 );
	gsl_matrix_free( (st->sol)->P1 );
	gsl_matrix_free( (st->sol)->P2 );
	//gsl_matrix_free( (st->sol)->invP0P2 );
	//gsl_matrix_free( (st->sol)->invP0P2 );
	gsl_matrix_free( (st->sol)->PP );
	gsl_matrix_free( (st->sol)->ss_wld );
	gsl_matrix_free( (st->sys)->A0 );
	gsl_matrix_free( (st->sys)->A1 );
	gsl_matrix_free( (st->sys)->A2 );
	gsl_matrix_free( (st->sys)->A3 );
	gsl_matrix_free( (st->sys)->F0 );
	gsl_matrix_free( (st->sys)->F1 );
	gsl_matrix_free( (st->sys)->F2 );
	gsl_matrix_free( (st->sys)->F3 );
	gsl_matrix_free( (st->sys)->N );
	gsl_matrix_free( (st->sys)->S );

	gsl_matrix_free( (st->sim)->data_moments );
	gsl_matrix_free( (st->sim)->moment_weights );
	gsl_matrix_free( (st->sim)->draws );
	free(st->sys);
	free(st->sol);
	free(st->sim);

	return status;
}
int clear_econ(struct st_wr *st){
	int status = 0;
	if((st->xss)->block != NULL)
		gsl_matrix_set_zero(st->xss);
	else
		status ++;
	if((st->ss)->block != NULL)
		gsl_vector_set_zero(st->ss);
	else
		status ++;
	gsl_matrix_set_zero( (st->sol)->P0 );
	gsl_matrix_set_zero( (st->sol)->P1 );
	gsl_matrix_set_zero( (st->sol)->P2 );
	gsl_matrix_set_zero( (st->sol)->PP );
	gsl_matrix_set_zero( (st->sol)->ss_wld );
	gsl_matrix_set_zero( (st->sys)->A0 );
	gsl_matrix_set_zero( (st->sys)->A1 );
	gsl_matrix_set_zero( (st->sys)->A2 );
	gsl_matrix_set_zero( (st->sys)->A3 );
	gsl_matrix_set_zero( (st->sys)->F0 );
	gsl_matrix_set_zero( (st->sys)->F1 );
	gsl_matrix_set_zero( (st->sys)->F2 );
	gsl_matrix_set_zero( (st->sys)->F3 );
	gsl_matrix_set_zero( (st->sys)->N );
	gsl_matrix_set_zero( (st->sys)->S );

	return status;
}


double qmatch(const double theta){
	double q = effic*pow(1.0+pow(theta,phi),-1.0/phi);
	q = q> 1.0? 1.0 : q;
	return q;
}
double pmatch(const double theta){
	double p = effic*theta*pow(1.0+pow(theta,phi),-1.0/phi);
	p = p>1.0 ? 1.0 : p;
	return p;
}
double dqmatch(const double theta){
	return -effic*pow(1.0+pow(theta,phi),-1.0/phi - 1.0)*pow(theta,phi-1.0);
}
double dpmatch(const double theta){
	return effic*pow(1.0+pow(theta,phi),-1.0/phi-1.0);
}
double invq(const double q){
	double t;
	t = pow(effic/q,phi)-1.0;
	t = t > 0.0 ? pow(t,1.0/phi) : 0.0;

	return t;
}
double invp(const double p){
	double pefphi = pow(p/effic,phi);
	return pow(pefphi/(1.0-pefphi),1.0/phi);
}
double dinvq(const double q){
	double dt;
	dt = pow(effic/q,phi)-1.0;
	dt = dt> 0.0 ? -1.0*pow(dt,1.0/phi-1.0)*pow(effic,phi)*pow(q,-phi-1.0) : 0.0;
	return dt;
}

/* This is not necessary when using utils.h
void OLS(gsl_vector * y, gsl_matrix * X, gsl_vector * coefs, gsl_vector * e){
	gsl_matrix * XpX = gsl_matrix_alloc(X->size2,X->size2);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans,1.0, X, X,0.0, XpX);
	gsl_matrix * invXpX = gsl_matrix_alloc(X->size2,X->size2);
	inv_wrap(XpX,invXpX);
	gsl_vector * Xpy = gsl_vector_alloc(X->size2);
	gsl_blas_dgemv(CblasTrans,1.0,X,y,0.0,Xpy);
	gsl_blas_dgemv(CblasNoTrans,1.0,invXpX,Xpy,0.0,coefs);
	gsl_blas_dgemv(CblasNoTrans,1.0,X,coefs,0.0,e);
	gsl_vector_scale(e,-1.0);
	gsl_vector_add(e,y);
}
 */
void VARest(gsl_matrix * X,gsl_matrix * coef, gsl_matrix * varcov){
	//gsl_matrix * Ydat = gsl_matrix_alloc(Xdat->size1,Xdat->size2);
	//gsl_matrix_memcpy(Ydat,Xdat);
	gsl_vector * Y = gsl_vector_alloc(X->size1-1);
	gsl_matrix * E = gsl_matrix_alloc(X->size1-1,X->size2);
	int i,t;
	gsl_matrix * Xreg = gsl_matrix_alloc(X->size1-1,X->size2+1);
	for (t=0;t<X->size1-1;t++){
		for(i=0;i<X->size2;i++)
			gsl_matrix_set(Xreg,t,i,gsl_matrix_get(X,t,i));
		gsl_matrix_set(Xreg,t,X->size2,1.0);
	}

	printmat("Xdata_VAR.txt",Xreg);

	for(i=0;i<coef->size2;i++){
		for(t=0;t<X->size1-1;t++)
			Y->data[t] = gsl_matrix_get(X,t+1,i);//Y leads 1
		gsl_vector_view bi = gsl_matrix_column(coef,i);
		gsl_vector_view ei = gsl_matrix_column(E,i);
		OLS(Y,Xreg,&bi.vector,&ei.vector);
	}
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,
			(1.0/(double)E->size1),E,E,0.0,varcov);
	gsl_matrix_free(E);
	gsl_vector_free(Y);
	gsl_matrix_free(Xreg);
}

int QZwrap(gsl_matrix * A, gsl_matrix *B,// gsl_matrix * Omega, gsl_matrix * Lambda,
		gsl_matrix * Q, gsl_matrix * Z, gsl_matrix * S, gsl_matrix * T){
	// wraps the QZ decomposition from GSL and returns Q,Z,S and T as Q^T A Z = S and Q^T B Z = T
	// throws out complex eigenvalues
	size_t n = A->size1;
	int status;
/*	gsl_matrix * A_s = gsl_matrix_alloc(n,n);
	gsl_matrix * B_s = gsl_matrix_alloc(n,n);
	gsl_matrix_memcpy(A_s,A);
	gsl_matrix_memcpy(B_s,B);
*/
	gsl_matrix_memcpy(S,A);
	gsl_matrix_memcpy(T,B);
	gsl_vector * lambda = gsl_vector_alloc(n);
	gsl_vector_complex * ilambda = gsl_vector_complex_alloc(n);
	gsl_matrix_complex * Omega_comp = gsl_matrix_complex_alloc(n,n);
	gsl_eigen_genv_workspace * w = gsl_eigen_genv_alloc(n);
	// be sure I don't need something like: gsl_eigen_genv_params (1, 1, 1, w);
	status = gsl_eigen_genv_QZ(S,T, ilambda, lambda, Omega_comp, Q, Z, w);
/*	for(i=0;i<n;i++){
		double alpha = GSL_REAL(gsl_vector_complex_get(ilambda,i));
		double beta = gsl_vector_get(lambda,i);
		double lam_i = beta/alpha;
		gsl_matrix_set(Lambda,i,i,lam_i);
	}
	for(j=0;j<n;j++){
		for(i=0;i<n;i++)
			gsl_matrix_set(Omega,i,j,GSL_REAL(gsl_matrix_complex_get(Omega_comp,i,j)));
	}
	*/
	gsl_matrix_transpose(Q); // is this needed?
	// check Q A Z = S?

	gsl_eigen_genv_free (w);
	gsl_vector_complex_free(ilambda);
	gsl_vector_free(lambda);
	gsl_matrix_complex_free(Omega_comp);
	return status;
}

int QZdiv(double stake, gsl_matrix * S,gsl_matrix * T,gsl_matrix * Q,gsl_matrix * Z){
	int i,j,m,k,status = 0;
	int N = S->size1;
	gsl_vector_view rootA =  gsl_matrix_diagonal(S); //needs to be considered in absolute value
	gsl_vector_view rootB =  gsl_matrix_diagonal(T);
	gsl_vector * root2 = gsl_vector_alloc((&rootA.vector)->size);
	for(i=0;i<root2->size;i++){
		root2->data[i] = fabs(gsl_vector_get(&rootA.vector,i))<1.0e-13 ?
				fabs(gsl_vector_get(&rootB.vector,i)):
				fabs(gsl_vector_get(&rootA.vector,i));
		root2->data[i] = fabs(gsl_vector_get(&rootB.vector,i))/root2->data[i];
	}
	for(i=N-1;i>=0;i--){
		m=0;
		for(j=i;j>=0;j--){
			if(root2->data[j] || root2->data[j]<-0.1){
				m=j;
				break;
			}
		}
		if(m==0)
			return status;
		for(k=m;k<=i-1;k++){
			status += QZswitch(k,S,T,Q,Z);
			double tmp = root2->data[k];
			root2->data[k] = root2->data[k+1];
			root2->data[k+1] = tmp;
		}
	}

	gsl_vector_free(root2);
	return status;
}

int QZswitch(int i, gsl_matrix *A,gsl_matrix *B,gsl_matrix *Q,gsl_matrix *Z){
	double a,c,d,b,e,f,
		sqrtproto;;
	int status;
	status =0;
	double small = 5*pow(2,-51);
	gsl_matrix * Ac = gsl_matrix_alloc(2,A->size2); // A(i:i+1,:)
	gsl_matrix * Ar = gsl_matrix_alloc(A->size1,2); // A(:,i:i+1)
	gsl_matrix * Bc = gsl_matrix_alloc(2,B->size2); // B(i:i+1,:)
	gsl_matrix * Br = gsl_matrix_alloc(B->size1,2); // B(:,i:i+1)
	gsl_matrix * Qc = gsl_matrix_alloc(2,Q->size2); // Q(i:i+1,:)
	gsl_matrix * Zr = gsl_matrix_alloc(Z->size1,2); // Z(:,i:i+1)

	a = gsl_matrix_get(A,i,i);
	d = gsl_matrix_get(B,i,i);
	b = gsl_matrix_get(A,i,i+1);
	e = gsl_matrix_get(B,i,i+1);
	c = gsl_matrix_get(A,i+1,i+1);
	f = gsl_matrix_get(B,i+1,i+1);

	gsl_matrix * xy = gsl_matrix_alloc(2,2);
	gsl_matrix * wz = gsl_matrix_alloc(2,2);

	gsl_vector * proto = gsl_vector_alloc(2);
	if(fabs(c)<small && fabs(f)<small){
		if (fabs(a)<small)
			return 0;
		else{
			proto->data[0] = b;
			proto->data[1] = -a;
			gsl_blas_ddot(proto, proto, &sqrtproto);
			sqrtproto = pow(sqrtproto,0.5);
			gsl_vector_scale(proto,1.0/sqrtproto);
			gsl_matrix_set(wz,0,0,proto->data[0]);gsl_matrix_set(wz,1,0,proto->data[1]);
			gsl_matrix_set(wz,1,0,-proto->data[1]);gsl_matrix_set(wz,1,1,proto->data[0]);
			gsl_matrix_set_identity(xy);
		}
	}
	else if(fabs(a)<small && fabs(d)<small){
		if(fabs(c)<small)
			return 0;
		else{
			proto->data[0] = c; proto->data[1] = -b;
			gsl_blas_ddot(proto, proto, &sqrtproto);
			sqrtproto = pow(sqrtproto,0.5);
			gsl_vector_scale(proto,1.0/sqrtproto);
			gsl_matrix_set(xy,0,0, proto->data[1]);gsl_matrix_set(xy,1,0,-proto->data[0]);
			gsl_matrix_set(xy,1,0, proto->data[0]);gsl_matrix_set(xy,1,1, proto->data[1]);
			gsl_matrix_set_identity(wz);
		}
	}
	else{
		//wz
		proto->data[0] = c*e-f*b;
		proto->data[1] = c*d-f*a;
		gsl_blas_ddot(proto, proto, &sqrtproto);
		sqrtproto = pow(sqrtproto,0.5);
		gsl_vector_scale(proto,1.0/sqrtproto);
		gsl_matrix_set(wz,0,0,proto->data[0]);gsl_matrix_set(wz,0,1,proto->data[1]);
		gsl_matrix_set(wz,1,0,-proto->data[1]);gsl_matrix_set(wz,1,1,proto->data[0]);
		//xy
		proto->data[0] = b*d-e*a;
		proto->data[1] = c*d-f*a;
		gsl_blas_ddot(proto, proto, &sqrtproto);
		sqrtproto = pow(sqrtproto,0.5);
		gsl_vector_scale(proto,1.0/sqrtproto);
		if(sqrtproto<small*10)
			return 0;
		gsl_matrix_set(xy,0,0,proto->data[0]);gsl_matrix_set(xy,0,1,proto->data[1]);
		gsl_matrix_set(xy,1,0,-proto->data[1]);gsl_matrix_set(xy,1,1,proto->data[0]);
	}
	gsl_matrix_view Asubc = gsl_matrix_submatrix(A,i,0,2,A->size2);
	gsl_matrix_memcpy(Ac,&Asubc.matrix);
	gsl_matrix_view Asubr = gsl_matrix_submatrix(A,0,i,A->size1,2);
	gsl_matrix_memcpy(Ar,&Asubr.matrix);
	gsl_matrix_view Bsubc = gsl_matrix_submatrix(B,i,0,2,B->size2);
	gsl_matrix_memcpy(Bc,&Bsubc.matrix);
	gsl_matrix_view Bsubr = gsl_matrix_submatrix(B,0,i,B->size1,2);
	gsl_matrix_memcpy(Br,&Bsubr.matrix);
	gsl_matrix_view Qsubc = gsl_matrix_submatrix(Q,i,0,2,A->size2);
	gsl_matrix_memcpy(Qc,&Qsubc.matrix);
	gsl_matrix_view Zsubr = gsl_matrix_submatrix(Z,0,i,A->size1,2);
	gsl_matrix_memcpy(Zr,&Zsubr.matrix);

	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,xy,Ac,0.0,&Asubc.matrix);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,xy,Bc,0.0,&Bsubc.matrix);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Ar,wz,0.0,&Asubr.matrix);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Br,wz,0.0,&Bsubr.matrix);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Zr,wz,0.0,&Zsubr.matrix);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,xy,Qc,0.0,&Qsubc.matrix);

	gsl_matrix_free(Ac);gsl_matrix_free(Ar);gsl_matrix_free(Bc);gsl_matrix_free(Br);
	gsl_matrix_free(Qc);gsl_matrix_free(Zr);
	return status;
}

void dfovec_iface_(double * f, double * x, int * n){
	// this is going to interface with dovec.f and call cal_dist using the global params
	unsigned nv = *n;
	int i;
	double *grad = NULL;
	double * x0 = malloc(sizeof(double)*nv);
	for(i=0;i<nv;i++)
		x0[i] = x[i] * dim_scale[i];
	cal_dist(nv, x0 ,grad, (void*) g_st);
	for(i=0;i<nv;i++)
		f[i] = (g_st->sim)->data_moments->data[i];
	free(x0);
}

int cal_dist_wrap(const gsl_vector * x, void* params, gsl_vector * f){
	// does not provide derivatives
	int i;
	unsigned n = ((struct st_wr * )params)->n;
	double * grad = NULL;
	cal_dist(n, x->data, grad, params);
	struct st_wr * st = (struct st_wr * )params;
	// note that this is going to take the matrix and vectorize in row-major order
	for(i=0;i<n;i++){
		gsl_vector_set(f,i,(st->sim)->data_moments->data[i] );
	}
	return GSL_SUCCESS;
}

int cal_df_wrap(const gsl_vector * x, void* params, gsl_matrix * Jac){
	// provides numerical derivatives
	int i,j;
	double * grad = NULL;
	double * xh,*xl,xd;
	double diff_eps = 1e-4;
	unsigned n = ((struct st_wr * )params)->n;
	gsl_matrix * moms =  (((struct st_wr * )params)->sim)->data_moments;
	xh	= malloc(sizeof(double)*n);
	xl	= malloc(sizeof(double)*n);
	for(i=0;i<n;i++){
		xh[i] = x->data[i];
		xl[i] = x->data[i];
	}
	// PUT IN PRAMGA TO PARALLELIZE HERE!
	for(i=0; i<n; i++){
		if(fabs(x->data[i])>diff_eps){
			xh[i] *= (1.0+diff_eps);
			xl[i] *= (1.0-diff_eps);
			xd 	= 2.0*x->data[i]*diff_eps;
		}
		else{
			xh[i] += diff_eps;
			xl[i] -= diff_eps;
			xd  = 2.0*diff_eps;
		}
		cal_dist(n, xh, grad, params);
		for(j=0;j<n;j++)
			gsl_matrix_set(Jac,i,j,moms->data[j]);
		cal_dist(n, xl, grad, params);
		for(j=0;j<n;j++)
			gsl_matrix_set(Jac,i,j, (gsl_matrix_get(Jac,i,j) -moms->data[j])/xd );

		if(fabs(x->data[i])>diff_eps){
			xh[i] /= (1.0+diff_eps);
			xl[i] /= (1.0-diff_eps);
		}
		else{
			xh[i] -= diff_eps;
			xl[i] += diff_eps;
		}
	}

	if(printlev>=2){
		char jacn[9];
		sprintf(jacn, "jac%d.csv", jac_ct);
		FILE* jac_f = fopen(jacn,"w+");
		gsl_matrix_fprintf (jac_f, Jac, "%f");
		fclose(jac_f);
	}

	return GSL_SUCCESS;
}

int cal_fdf_wrap(const gsl_vector * x, void *params, gsl_vector *f, gsl_matrix *Jac){
	cal_df_wrap(x, params, Jac);
	cal_dist_wrap(x, params, f);

	return GSL_SUCCESS;
}

double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg0, int alg1){

	int status,resi,starts,i,locstatus,dfbols_flag=0;

	status 	= 0;
	// csize 	= 8;  = how many nodes it's running on.
	starts 	= 8*csize; // this is how many will draw to begin, but will only start from a subset of these

	nlopt_opt opt0;
	gsl_multiroot_fdfsolver *sloc;//gsl_multiroot_solver *sloc;
	gsl_multiroot_function_fdf obj_wrap;//gsl_multiroot_function obj_wrap;
	double * w;
	int npt,mv,iprint,maxfun;
	double rhobeg, rhoend;
	double *dfbols_lb,*dfbols_ub;

	alloc_econ(ss);
	// this is just a checker: the idea was to leave ss and move around parameters
	/*status 	= sol_ss(ss->ss,ss->xss,ss->sol);
	if(printlev>=0 && status>=1) printf("Steady state not solved \n");
	status += sys_def(ss->ss,ss->sys);
	if(printlev>=0 && status>=1) printf("System not defined\n");
	status += sol_dyn(ss->ss,ss->sol,ss->sys);
	if(printlev>=0 && status>=1) printf("System not solved\n");
	 */

	if(alg0%10==0 && alg0<100)
		opt0 = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	else if(alg0%5 ==0 && alg0<100)
		opt0 = nlopt_create(NLOPT_LN_SBPLX,n);
	else if(alg0<100)
		opt0 = nlopt_create(NLOPT_LN_BOBYQA,n);
	else if(alg0>=100 && alg0%2==0){
		opt0 = nlopt_create(NLOPT_GN_ISRES,n);
	}
	else if(alg0>=100 && alg0%2!=0){
		opt0 = nlopt_create(NLOPT_GN_CRS2_LM,n);
	}
#ifdef _DFBOLS_USE
	if(alg0%3 == 0)
		dfbols_flag = 1;
#endif


	dim_scale = malloc(sizeof(double)*n);


	for(i=0;i<n;i++){
		dim_scale[i] = ub[i] - lb[i];
	}
	if(dfbols_flag == 1){
		// Derivative-free box-bounded non-linear least squares
		g_st = ss;
		npt = 2*n+1;
		mv  = n; // number of equations
		// rhobeg and rhoend these follow recommended guidelines in the dfbols software
		rhobeg = 0.25;
		rhoend = 1e-8;
		maxfun = 400*(n+1);
		int nspace = (npt+5)*(npt+n)+3*n*(n+5)/2;
		w = malloc(sizeof(double)*nspace);
		// rescale bounds in the x dimension, this is always going to be unit-box bounded
		dfbols_lb = malloc(sizeof(double)*n);
		dfbols_ub = malloc(sizeof(double)*n);
		for(i=0;i<n;i++){
			dfbols_ub[i] =ub[i]/dim_scale[i];
			dfbols_lb[i] =lb[i]/dim_scale[i];
		}
	}

	//  Non- deriv based, with NLOPT
	double xtol_abs[n];
	for(i=0;i<n;i++)
		xtol_abs[i] = x_tol*fabs(x0[i]);
	nlopt_set_xtol_rel(opt0, x_tol);
	nlopt_set_xtol_abs(opt0, xtol_abs);
	nlopt_set_ftol_rel(opt0, f_tol);
	nlopt_set_ftol_abs(opt0, f_tol); // because function value should be approaching zero, need to set this
	nlopt_set_maxeval(opt0, 50*pow(n,2));
	nlopt_set_stopval(opt0, 0.0);
	nlopt_set_lower_bounds(opt0,lb);
	nlopt_set_upper_bounds(opt0,ub);
	nlopt_set_min_objective(opt0, Vobj, (void*) ss);
	ss->opt0 = opt0;

	if(alg1>0){
		sloc = gsl_multiroot_fdfsolver_alloc(gsl_multiroot_fdfsolver_hybridsj,n);//sloc = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids,n);
		obj_wrap.f = &cal_dist_wrap;
		obj_wrap.df = &cal_df_wrap;
		obj_wrap.fdf= &cal_fdf_wrap;
		obj_wrap.n = n;
		obj_wrap.params = (void*)ss;
	}

	double opt_f=100;

	if(alg0>=10 && alg0<100){//random restarts
		// the queue of places to start
		FILE * f_startx, *f_startf;


		double ** x0queue = malloc(sizeof(double*)*starts);
		double *xmin;
		int nmax=0;
		xmin = malloc(sizeof(double)*n);
		gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol,n);
		double xqr[n];
		double ftol_ms = f_tol/10.0;
		double minf = 1000;
		// set the restart locations before hand
		for(resi=0;resi<starts;resi++){
			x0queue[resi] = malloc(sizeof(double)*n);
			gsl_qrng_get(q,xqr);
			for(i=0;i<n;i++)
				x0queue[resi][i] = xqr[i]*dim_scale[i] + lb[i];
		}
		// alloc space for the list of optimal points
		double ** x0_loc=malloc(sizeof(double*)*starts); // this is the set of local minima
		for(resi=0;resi<starts;resi++)
			x0_loc[resi] = malloc(sizeof(double)*n);
		double d_inner=0.0,d_outer=1000.0, stop_crit_N=starts; // take everyone to start
		int starti=0;
		double d_region[csize];
		for(resi=0;resi<starts;resi++){
			// update criteria for how many points we expect and for size of region of attraction each csize times
			int li;
			if(resi>0 && starti%csize==0){
				if(d_inner<1e-4)//if it gets too small, break out
					d_inner = d_outer;
				for(li=0;li<csize;li++)
					d_inner = d_region[li]<d_inner ? d_region[li] : d_inner;
				// Bayesian criteria for finding stopping point
				stop_crit_N = (double)(nmax*(resi-1)) / (double)(resi-nmax-2)-0.5;
				// the whole region is size 1, so this is narrowing in at rate sqrt(n/clustersize)
				d_outer = 1.0/ sqrt((double) (resi/csize));
			}

			// reject this start point?
			double d_xres=10.0;

			for(li=0;li<nmax;li++){
				double d_xres_li = 0.0;
				for(i=0;i<n;i++)
					d_xres_li += fabs(x0_loc[li][i]-x0queue[resi][i])/dim_scale[i];
				d_xres_li /= (double)n;
				d_xres = d_xres_li<d_xres? d_xres_li : d_xres;
			}
			double d_xmin =0.0;
			for(i=0;i<n;i++)//testing for close enough to the current min
				d_xmin += fabs(xmin[i]-x0queue[resi][i])/dim_scale[i];
			d_xmin /= (double)n;
			// this is the test, to make it explore and also narrow in: every 5 draws, don't worry about narrowing in, just in case
			if((d_xres>d_inner && d_xmin<d_outer) || resi%5==0){
				double xstart[n];
				for(i=0;i<n;i++)
					xstart[i] = x0queue[resi][i];

				// optimize!
				if(verbose >=1){
					printf("\n Starting an optimize\n");
				}

				if(dfbols_flag==1){
					for(i=0;i<n;i++)
						x0queue[resi][i] /= dim_scale[i];
#ifdef _DFBOLS_USE
					bobyqa_h_(&n,&npt,x0queue[resi],dfbols_lb,dfbols_ub,&rhobeg,&rhoend,&iprint,&maxfun,w,&mv);
#endif
					for(i=0;i<n;i++)
						x0queue[resi][i] *= dim_scale[i];

					opt_f = cal_dist(n,x0queue[resi],0,(void*)g_st);
				}
				else
					status = nlopt_optimize(opt0, x0queue[resi], &opt_f);
				if(opt_f<minf){
					minf = opt_f;
					for(i=0;i<n;i++)
						xmin[i] = x0queue[resi][i];
				}
				d_region[starti%csize] = 0.0;
				for(i=0;i<n;i++)
					d_region[starti%csize] += fabs(xstart[i] - x0queue[resi][i])/dim_scale[i] ;
				d_region[starti%csize] /= (double)n;

				// test if this is a new optim location
				int li=0;
				for(li=0;li<nmax;li++){
					double d_xres_li = 0.0;
					for(i=0;i<n;i++)
						d_xres_li += fabs(x0_loc[li][i]-x0queue[resi][i])/dim_scale[i];
					d_xres_li /= (double)n;
					if(d_xres_li<1e-5)
						break;
				}
				if(li>=nmax){//not close enough to call the same as one of these points
					f_startx = fopen("f_mstart_x.txt","a+");
					f_startf = fopen("f_mstart_f.txt","a+");

					for(i=0;i<n;i++)
						x0_loc[nmax][i]=x0queue[resi][i];
					nmax++;
					for(i=0;i<n;i++)
						fprintf(f_startx,"%f,", x0queue[resi][i]);

					fprintf(f_startx,"\n");
					fprintf(f_startf,"%f\n",opt_f);
					fclose(f_startx);
					fclose(f_startf);
				}
				// is it on the constraint? If so, push the boundary but don't cross zero
				for(i=0;i<n;i++){
					if((x0queue[resi][i] - lb[i])/dim_scale[i]<1e-4)
						lb[i] = lb[i] >0 ? lb[i]/2 : lb[i]*2;
					if((ub[i] - x0queue[resi][i])/dim_scale[i] < 1e-4)
						ub[i] = ub[i] >0 ? ub[i]*1.5 : ub[i]/1.5;
				}

				// time to stop the multi-start? This is the Bayesian criteria
				//if(nmax>=stop_crit_N || opt_f< ftol_ms)
				//	break;
				starti ++;
			}//if cluster
			else{
				if(verbose >=1) printf("Rejected start point inside a cluster\n");
			}
		}// for restarts
		printf("\n The overall best is:\n");
		for(i=0;i<n;i++)
			printf("%f,",xmin[i]);
		printf("\n");

		gsl_qrng_free(q);
		free(xmin);
		for(resi=0;resi<starts;resi++)
			free(x0queue[resi]);
		free(x0queue);
		for(resi=0;resi<starts;resi++)
			free(x0_loc[resi]);
		free(x0_loc);


	}// end multi start conditional
	else if(alg0>=100){// genetic algorithm!  Holy cow!
		nlopt_set_xtol_rel(opt0, 10*x_tol);
		for(i=0;i<n;i++)
				xtol_abs[i] = 10*x_tol*fabs(x0[i]);
		nlopt_set_xtol_abs(opt0, xtol_abs);
		nlopt_set_ftol_rel(opt0, 10*f_tol);
		nlopt_set_ftol_abs(opt0, 10*f_tol); // because function value should be approaching zero, need to set this


		status = nlopt_optimize(opt0,x0, &opt_f);
		printf("status of global solver %d is= %d\n",alg0 ,status);
		set_params( x0,ss->cal_set);
		FILE * f_startx, *f_startf;
		f_startx = fopen("f_mstart_x.txt","a+");
		f_startf = fopen("f_mstart_f.txt","a+");
		fprintf(f_startx,"\n");
		for(i=0;i<n;i++)
			fprintf(f_startx,"%f,", x0[i]);

		fprintf(f_startx,"\n");
		fprintf(f_startx,"\n");
		fprintf(f_startf,"Genetic Algorithm %d Result!\n", alg0);
		fprintf(f_startf,"%f\n",opt_f);
		fprintf(f_startf,"\n");
		fclose(f_startx);
		fclose(f_startf);
	}
	else if(alg0<10){// local search only
		if(dfbols_flag==1){
			for(i=0;i<n;i++)
				x0[i] /= dim_scale[i];
#ifdef _DFBOLS_USE
			bobyqa_h_(&n,&npt,x0,dfbols_lb,dfbols_ub,&rhobeg,&rhoend,&iprint,&maxfun,w,&mv);
#endif
			for(i=0;i<n;i++)
				x0[i] *= dim_scale[i];
			opt_f = cal_dist(n,x0,0,(void*)g_st);
		}
		else
			status = nlopt_optimize(opt0, x0, &opt_f);

		if((status == 4 || status == 3) && opt_f>0.1){
			// start over from the last stop point
			status = nlopt_optimize(opt0, x0, &opt_f);
		}
		set_params( x0,ss->cal_set);
		FILE * f_startx, *f_startf;
		f_startx = fopen("f_mstart_x.txt","a+");
		f_startf = fopen("f_mstart_f.txt","a+");
		fprintf(f_startx,"\n");
		for(i=0;i<n;i++)
			fprintf(f_startx,"%f,", x0[i]);

		fprintf(f_startx,"\n");
		fprintf(f_startx,"\n");
		fprintf(f_startf,"Local Algorithm %d Result!\n", alg0);
		fprintf(f_startf,"%f\n",opt_f);
		fprintf(f_startf,"\n");
		fclose(f_startx);
		fclose(f_startf);
	}

	// POLISH IT OFF!
	if(alg1>0 && opt_f>f_tol/10.0){
		calhist = fopen(calhi_f,"a+");
		fprintf(calhist,"**** Now solving by hybrid method equation solver ****\n");
		fclose(calhist);

		gsl_vector* x_vec = gsl_vector_calloc(n);
		gsl_vector_view x_view = gsl_vector_view_array(x0,n);
		gsl_vector_memcpy(x_vec,&x_view.vector) ;

		status += gsl_multiroot_fdfsolver_set(sloc,&obj_wrap,x_vec);
		resi = 0;
		do{
			resi++;
			locstatus = gsl_multiroot_fdfsolver_iterate(sloc);
			if(locstatus)
				break;
			locstatus =
					gsl_multiroot_test_residual(sloc->f,f_tol/10.0);
		}while(locstatus ==GSL_CONTINUE && resi<starts*100);

		if(gsl_multiroot_test_residual(sloc->f,n*opt_f)){
			for(i=0;i<n;i++)
				x0[i] = x_vec->data[i];
		}
		set_params(x0,ss->cal_set);
		FILE * cal_out = fopen(calhi_f,"a+");
		for(i=0;i<n;i++)
			fprintf(cal_out,"%f,",x0[i]);
		fprintf(cal_out,"\n");
		fclose(cal_out);

		gsl_multiroot_fdfsolver_free(sloc);
		gsl_vector_free(x_vec);
	}

	free_econ(ss);
	free(dim_scale);
	if(dfbols_flag==1){
		free(w);
		free(dfbols_ub);free(dfbols_lb);
	}

	else
		nlopt_destroy(opt0);
	return opt_f;
}

double cal_dist_df(unsigned n, const double *x, double *grad, void* params){
// this is for df for nlopt
	int i;
	//wraps for derivatives:
	double d = 100.0;
	double diff_eps, xd;
	double * xh,*xl;
	diff_eps = 1e-4;

	if(grad==NULL)
		d = cal_dist(n, x, grad, params);
	else{
		xh	= malloc(sizeof(double)*n);
		xl	= malloc(sizeof(double)*n);
		/*if(gsl_fin_diffs==1){
			((struct st_wr * )params)->n = n;
			((struct st_wr * )params)->x = x;

			gsl_function F;
			double result, abserr=100.0;

			F.function	= &cal_dist_wrap;
			F.params	= params;
			for(i=0;i<n;i++){
				gsl_deriv_central(F, x[i], diff_eps, result, abserr);
				grad[i] = result;
			}
		}
		else{*/
			for(i=0;i<n;i++){
				xh[i] = x[i];
				xl[i] = x[i];
			}
			// PUT IN PRAMGA TO PARALLELIZE HERE!
			for(i=0; i<n; i++){
				if(fabs(x[i])>diff_eps){
					xh[i] *= (1.0+diff_eps);
					xl[i] *= (1.0-diff_eps);
					xd 	= 2.0*x[i]*diff_eps;
				}
				else{
					xh[i] += diff_eps;
					xl[i] -= diff_eps;
					xd  = 2.0*diff_eps;
				}
				grad[i]   = cal_dist(n, xh, grad, params);
				grad[i]  -= cal_dist(n, xl, grad, params);
				grad[i]  /= xd;
				if(fabs(x[i])>diff_eps){
					xh[i] *= (1.0-diff_eps);
					xl[i] *= (1.0+diff_eps);
				}
				else{
					xh[i] -= diff_eps;
					xl[i] += diff_eps;
				}
			}

		//}
		free(xh);free(xl);
	}

	return d;
}

int set_params(const double * x, int cal_set){
	int status,i,d,l;
	status =0;
	int param_offset = 0;
	if(cal_set==0 || cal_set == 1){
		phi 	= x[0];
		//sig_psi = x[1];
		//tau 	= x[2];
		scale_s	= x[1];
		shape_s = x[2];
		effic	= x[3];
		param_offset = 4;
	}
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			chi[l][d] = 1.0;
	}
	
	/*
	if(cal_set==0 || cal_set==2){
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				chi[l][d] = 0.0;
				for(i=0;i<Nskill;i++){
					chi[l][d] += x[param_offset+i]*(gsl_matrix_get(f_skills,d,i+1) - gsl_matrix_get(f_skills,l-1,i+1));
				}
				//chi[l][d] += 1.0;
				chi[l][d] 	= exp(chi[l][d]);
				//chi[l][d] += x[param_offset+Nskill];
			}
		}
		double chi_lb = 0.50;
		double chi_ub = 0.90;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					if(chi[l][d]<=chi_lb) chi[l][d] = chi_lb;
					if(chi[l][d]>=chi_ub) chi[l][d] = chi_ub;
				}
			}
		}

		for(d=0;d<Noccs;d++){
			double mean_chi = 0.0;
			for(l=1;l<Noccs+1;l++)
				mean_chi = l!=d+1 ? chi[l][d] + mean_chi : mean_chi;
			chi[0][d] = mean_chi/((double)Noccs-1.0);

		}
		//for(d=0;d<Noccs;d++)
		//	chi[d+1][d] = 1.0;
	}*/

	// recompute b[0]
	//b[1]=0.71;
	b[1] = brt;
	b[0]=0.0;
	for(d=0;d<Noccs;d++){
		for(l=0;l<Noccs+1;l++)
			b[0] += chi[l][d];
		b[0] -= chi[d+1][d];
	}
	b[0] /= (double)(Noccs*Noccs);
	b[0] *= brt;
	//b[0] += 0.21; // 50% replacement ratio plus 0.21 flat for home production: Hall & Milgrom (2008)

	//do something about chi below b --- this might screw everything up?
	for(d=0;d<Noccs;d++){
		for(l=1;l<Noccs+1;l++)
	 		chi[l][d] = chi[l][d]<b[1]? b[1] :chi[l][d];
		chi[0][d] =chi[0][d]<b[0]? b[0] :  chi[0][d];
	}

	if(printlev>=2){
		gsl_matrix * cm = gsl_matrix_calloc(Noccs+1,Noccs);
	//	gsl_matrix_view chimat = gsl_matrix_view_array(chi,Noccs+1,Noccs); why doesn't this work?
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(cm,l,d,chi[l][d]);
		}
		printmat("chimat.csv",cm);
		gsl_matrix_free(cm);
	}

	return status;
}

double cal_dist(unsigned n, const double *x, double *grad, void* params){

	int status=0,i,d;
	struct st_wr * st = (struct st_wr * )params;
	clear_econ(st);
	struct aux_coef  * simdat	=  st->sim;
	struct sys_coef  * sys		= st->sys;
	struct sys_sol   * sol		= st->sol;
	gsl_matrix * xss = st->xss;
	gsl_vector * ss = st->ss;
	double dist;

	set_params(x, st->cal_set);

	int param_offset = st->cal_set == 1 || st->cal_set == 0 ? ((st->sim)->data_moments)->size2 : 0;


	/* zero everything out
	gsl_matrix_set_zero(sys->A0);gsl_matrix_set_zero(sys->A1);gsl_matrix_set_zero(sys->A2);gsl_matrix_set_zero(sys->A3);
	gsl_matrix_set_zero(sys->F0);gsl_matrix_set_zero(sys->F1);gsl_matrix_set_zero(sys->F2);gsl_matrix_set_zero(sys->F3);
	gsl_matrix_set_zero(sol->P0);gsl_matrix_set_zero(sol->P1);gsl_matrix_set_zero(sol->P2);gsl_matrix_set_zero(sol->PP);
	 */

	status 	= sol_ss(ss,NULL,xss,sol);
	if(printlev>=0 && status>=1) printf("Steady state not solved \n");
	status += sys_def(ss,sys);
	if(printlev>=0 && status>=1) printf("System not defined\n");
	status += sol_dyn(ss,sol,sys);
	if(printlev>=0 && status>=1) printf("System not solved\n");

	if(status == 0){
		status += sim_moments(st,ss,xss);
		Nsolerr =0;
	}
	else{
		++ Nsolerr;
		gsl_matrix_scale(simdat->data_moments,5.0);
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"System did not solve at: (%f,%f,%f,%f)\n",x[0],x[1],x[2],x[3]);
		fclose(solerr);
	}
	int Nmo = (simdat->data_moments)->size2;

	dist = 0.0;
	if(st->cal_set != 2){
		for(i=0;i<Nmo;i++)
			dist += gsl_matrix_get(simdat->moment_weights,0,i)*pow(gsl_matrix_get(simdat->data_moments,0,i),2);
		if(verbose>=2){
			printf("At phi=%f,scale_s=%f,shape_s=%f,effic=%f\n",x[0],x[1],x[2],x[3]);
			if(status>0) printf("System did not solve \n");
			printf("Calibration distance is %f\n",dist);
		}
	}
	if(st->cal_set!=1){
		int n2 = st->cal_set == 0 ? st->n - param_offset : st->n;
		// regression distance
		for(i=0;i<n2;i++)
			dist += gsl_matrix_get(simdat->moment_weights,1,i)*pow(gsl_matrix_get(simdat->data_moments,1,i),2);
		if(verbose>=2){
			for(i=0;i<n2;i++){
				printf("At b%d=%f\n",i,x[i+param_offset]);
			}
			printf("Calibration distance is %f\n",dist);
		}
	}
	if(printlev>=1){
		calhist = fopen(calhi_f,"a+");
		for(i=0;i<n;i++)
			fprintf(calhist,"%f,",x[i]);
		fprintf(calhist,"\n");
		fprintf(calhist,"%f,",dist);
		if(st->cal_set!=2){
			for(i=0;i<(simdat->data_moments)->size2;i++)
				fprintf(calhist,"%f,",(simdat->data_moments)->data[i]);
		}
		if(st->cal_set!=1){
			for(i=0;i<(simdat->data_moments)->size2;i++)
					fprintf(calhist,"%f,",gsl_matrix_get((simdat->data_moments),1,i));
		}
		fprintf(calhist,"\n");
		fclose(calhist);
	}
	for(i=5;i>0;i--)
		shist[i] = shist[i-1];
	shist[0]= dist;
	int exploding =0;
	for(i=1;i<6;i++){
		exploding = shist[i-1]/shist[i]>=1.5 ? exploding +1 : exploding;
	}
	if(exploding>=5)
		nlopt_force_stop(st->opt0);
	if(Nsolerr>2 || (dist>1000 && Nsolerr>1 ))
		nlopt_force_stop(st->opt0);


	return dist;
}
