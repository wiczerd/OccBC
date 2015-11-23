/* This program solves my dissertation by a hybrid k=order perturbation with an optimal COV and exact nonlinear policies
*  First, it solves the first order approximation by some straight forward linear algebra.  This can be used then for the 2nd, too.
*  For more on the optimal COV see J F-V J R-R (JEDC 2006)
*  The hybrid portion involves using the exactly computed decision rules given approximated state-dynamics: see Maliar, Maliar and Villemot (Dynare WP 6)
*
*  au : David Wiczer
*  mdy: 2-20-2011
*  rev: 6-21-2012
*
* old compile line: icc /home/david/Documents/CurrResearch/OccBC/program/c/occBC_hybr.c -static -lgsl -openmp -mkl=parallel -O3 -lnlopt -I/home/david/Computation -I/home/david/Computation/gsl/gsl-1.15 -o occBC_hybr.out
* new compile line:  icc  occBC_hybr.c -mkl -openmp -lgsl -D_MKL_USE=1 -D_DFBOLS_USE=0 -g -o occBC_hybr.out
* valgrind line: valgrind --leak-check=yes --log-file=occBC_valgrind.log --error-limit=no ./occBC_hybr.out
*/
//#define _MKL_USE // this changes LAPACK options to use faster MKL versions of LAPACK
//#define _DFBOLS_USE
//#define _MPI_USE

// To run, pass arguments first what to calibrate (if 0 or 4, does everything, otherwise 1,2,3 do only non HC params, only HC params or a model w/o any HC)
// then pass coarse optimization and fine optimization algorithm selections: 0 for fine (polishing) does not polish, otherwise uses Quasi-Newton

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
#include <gsl/gsl_roots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_statistics.h>




int Nsolerr		= 0;
int Nobs;
int simT = 218;
int nsim, Nskill;
int nshock;
int neq;
int verbose 	= 3;
int printlev 	= 3;
int rescale_var	= 0;
int check_fin	= 1;
int fd_dat_read = 0; // read in the actual data
int homosk_psi	= 1; // homskedastic psi assumption during the simulations
int sym_occs	= 0; // debugging: make all occupations symmetric
int szt_gsl		= 0; // use gls brent solver, or own?
int dbg_iters	= 0; // limits iterations for debugging purposes
int use_anal	= 1; // this governs whether the perturbation uses analytic derivatives
int fix_fac		= 0; // this allows f_t to be free when estimating.  o.w. the path of f_t, gamma and var_eta are fixed.
int opt_alg		= 3; // 10x => Nelder-Meade, 5x => Subplex, 3x => DFBOLS, o.w => BOBYQA
int polish_alg	= 0;
double ss_tol	= 1e-7; // tighten this with iterations?
double dyn_tol	= 1e-7; // ditto?
double zmt_upd	= .05;
double zub_fac	= 1.5; // if this is 1, then bounds on z are symmmetric, >1 implies more upside
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
int const Noccs	= 18;
int const Nfac	= 2;// how many factors in the z_i model?
int const Nllag	= 0;// how many lags in lambda
int const Nglag	= 1;// how many lags in gamma
int const Nagf	= 1;// Z with coefficients beyond just the 1
int Ns,Nc,Nx;
int Notz;
int Nt_dat = 63;

// checks on internals
int gpol_zeros = 0;
int t_zeros = 0;

double 	beta	= 0.9967;	// monthly discount
double	nu 		= 0.0; 		// switching cost --- I don't need this!
double 	fm_shr	= 0.5;	// firm's surplus share
double 	kappa	= 0.13;//.1306; corresponds to HM number, was 0.27
double *b; // unemployment benefit
double 	brt		= 0.42;
double 	bdur	= 6.0;
double 	privn	= 0.18;
double 	sbar	= 0.02333;	// will endogenize this a la Cheremukhin (2011)


double ** chi;

double 	phi		= 0.647020;	// : log(1/UE elasticity -1) = phi*log(theta)= log(1/.38-1)/log(5)
double 	sig_psi	= 0.014459; 	// higher reduces the switching prob
double 	tau 	= 0.041851;
double 	scale_s	= 0.026889;
double 	shape_s	= 0.194987;
double 	effic	= 0.495975;
double 	chi_co[]= {1.45,1.25,0.717312,-0.2 };


double rhoZ		= 0.9621;
double rhozz	= 0.1;

double sig_eps	= 2.526e-05;//1.2610e-06;//0.00008557;//0.00002557;//7.2445e-7;  using mine, not FED Numbers
double sig_zet	= 1e-5;//1.0e-6; (adjustment for the FRED number)

// as read, the process coefficients
gsl_vector * cov_ze;
gsl_matrix * GammaCoef, *var_eta; 	//dynamics of the factors
gsl_matrix * LambdaCoef,*var_zeta;	//factors' effect on z
gsl_matrix * LambdaZCoef;			//agg prod effect on z

gsl_matrix * f_skills;				//skills attached to occupations

// data moments that are not directly observable
double avg_fnd	= 0.3229178;
double avg_urt	= 0.06;
double avg_wg	= 0.006;	// this is the 5 year wage growth rate: make tau fit this
double avg_elpt	= 0.48;		// elasticity of p wrt theta, Barichon 2011 estimate
double avg_sdsep= 0.01256;	//the average standard deviation of sld across occupations
double med_dr	= 2.3;
double chng_pr	= 0.45;

double* sk_wg;
//double Zfcoef[] = {0.0233,-0.0117};
			// old : {-0.0028,-0.0089,0.0355,0.0083};

struct aux_coef{
	gsl_matrix * data_moments;
	gsl_matrix * moment_weights;
	gsl_matrix * draws;
	gsl_matrix * Xdat;
	gsl_matrix * fd_hist_dat;
};


struct dur_moments{
	double	pr6mo;
	double	sd_dur;
	double	E_dur;
	double*	dur_l;
	double	chng_wg;
	gsl_matrix * dur_qtl;
	gsl_vector * Ft;
	gsl_vector * xFt;
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
	gsl_matrix *gld,*tld,*sld,*ald,*ss_wld;
};
struct st_wr{
	struct sys_coef * sys;
	struct sys_sol * sol;
	struct aux_coef * sim;
	struct shock_mats* mats;
	int cal_set,d;
	unsigned n;
	double cal_worst,cal_best;
	const double * x;
	gsl_vector * ss;
	gsl_matrix * xss;
	nlopt_opt opt0;
};
struct shock_mats{
	gsl_matrix * Gamma, * Lambda, * var_zeta, *var_eta;
	double rhozz, rhoZ, sig_eps2;
	gsl_matrix * facs;
	gsl_vector * ag;
	gsl_matrix * Zz_hist;
};
struct zsol_params{
	double *fd_xg_surp;
	int d;
	gsl_matrix * PP;
};

struct st_wr * g_st;
struct shock_mats * fd_mats;

// Solve the steady state
int sol_ss(gsl_vector * ss, gsl_vector * Zz,gsl_matrix * xss, struct sys_sol *sol);

// Solving the dynamic model
int sol_dyn(gsl_vector * ss, struct sys_sol * sol, struct sys_coef * sys, int sol_shocks);
int sys_ex_set(gsl_matrix * N, gsl_matrix * S,struct shock_mats * mats);
int sys_st_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx);
int sys_co_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx);
int sys_ex_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco);
int sys_def(gsl_vector * ss, struct sys_coef *sys, struct shock_mats * mat0);

// Policies
int gpol(gsl_matrix * gld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int spol(gsl_matrix * sld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int theta(gsl_matrix * tld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int xprime(gsl_matrix * xp, gsl_matrix * ald, gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_matrix * x, const gsl_vector * Zz);
int invtheta_z(double * zz_fd, const double * fd_dat, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const double * Zz_in);

// results:
int sol_zproc(struct st_wr *st, gsl_vector * ss, gsl_matrix * xss, struct shock_mats * mats0);
int ss_moments(struct aux_coef * ssdat, gsl_vector * ss, gsl_matrix * xss);
int sim_moments(struct st_wr * st, gsl_vector * ss,gsl_matrix * xss);
int dur_dist(struct dur_moments * s_mom, const gsl_matrix * gld_hist, const gsl_matrix * pld_hist, const gsl_matrix * x_u_hist);

int fd_dat_sim(gsl_matrix * fd_hist_dat, struct shock_mats * fd_mats);

int TGR(//gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		//double *urt, double *opt_urt,
		struct st_wr * st);
double fr00, frdd; // THIS IS REALLY BAD PROGRAMMING

// Utilities
int est_fac_pro(gsl_matrix* occ_prod,gsl_vector* mon_Z, struct shock_mats * mats);
int Es_dyn(gsl_vector * Es, const gsl_vector * ss_W, const gsl_vector * Wlast ,const gsl_matrix * invP0P1, const gsl_matrix * invP0P2, const gsl_vector * Zz);
int Es_cal(gsl_vector * Es, const gsl_vector * ss_W, const gsl_matrix * PP, const gsl_vector * Zz);
int interp_shock_seq(const gsl_matrix * obs_prod, gsl_matrix * mon_prod);
int sol_calloc(struct sys_sol * new_sol);
int sol_free(struct sys_sol * new_sol);
int sol_memcpy(struct sys_sol * cpy, const struct sys_sol * base);
int VARest(gsl_matrix * Xdat,gsl_matrix * coef, gsl_matrix * varcov);

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
double hetero_ev(double eps, void * Vp);
double dhetero_ev(double eps, void * Vp);

// calibration functions
double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg0, int alg1);
double cal_dist_df(unsigned n, const double *x, double *grad, void* params);
double cal_dist(unsigned n, const double *x, double *grad, void* params);
int cal_dist_wrap(const gsl_vector * x, void* params, gsl_vector * f);
void dfovec_iface_(double * f, double * x, int * n);



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


	//initialize parameters
	Nskill = 4;
#ifdef _MKL_USE
	printf("Begining, Calflag = %d, USE_MKL=%d,USE_DFBOLS=%d\n",calflag,_MKL_USE,_DFBOLS_USE);
#endif

	f_skills = gsl_matrix_alloc(Noccs,5);
	readmat("rf_skills.csv",f_skills);

	if(printlev>=2)	printmat("Readf_skills.csv" ,f_skills);
	FILE * readparams = fopen("params_in.csv","r+");
	fscanf(readparams,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
		   &phi,&sig_psi,&tau,&scale_s,&shape_s,&effic,&chi_co[0],&chi_co[1],&chi_co[2],&chi_co[3]);
	fclose(readparams);
	printf("%f,%f,%f,%f,%f,%f",phi,sig_psi,tau,scale_s,shape_s,effic);

	sk_wg = malloc(sizeof(double)*Nskill);
	b	  = malloc(sizeof(double)*2);
	chi   = malloc((Noccs+1)*sizeof(double*));
	for(l=0;l<Noccs+1;l++)
		chi[l] = malloc(sizeof(double)*Noccs);

	// read in chi_co to initialize
	double chi_co_read[Nskill];
	FILE * readchi = fopen("chi_co.csv","r+");
	fscanf(readchi,"%lf,%lf,%lf,%lf",&chi_co_read[0],&chi_co_read[1],&chi_co_read[2],&chi_co_read[3]);
	fclose(readchi);

	// sk_wg are the targets
	for(l=0;l<Nskill;l++) sk_wg[l] = chi_co_read[l];

	set_params(chi_co, 2);
	if(fabs(calflag)==3){
		double x3[] = {phi,scale_s,shape_s,effic};
		set_params(x3,3);
	}

	// read in the matrices that are from finding rates in the data
	fd_mats = malloc(sizeof(struct shock_mats));
	fd_mats->Gamma = gsl_matrix_calloc(Nfac,Nfac);
	fd_mats->Lambda = gsl_matrix_calloc(Noccs,Nfac+1);
	fd_mats->var_eta = gsl_matrix_calloc(Nfac,Nfac);
	fd_mats->var_zeta= gsl_matrix_calloc(Noccs,Noccs);


	status += readmat("Gamma_fd.csv",fd_mats->Gamma);
	status += readmat("Lambda_fd.csv",fd_mats->Lambda);
	status += readmat("var_eta_fd.csv",fd_mats->var_eta);
	status += readmat("var_zeta_fd.csv",fd_mats->var_zeta);
	FILE * readfdparams = fopen("coefs_in_fd.csv","r+");
	fscanf(readfdparams, "%lf,%lf,%lf",&fd_mats->rhoZ,&fd_mats->rhozz,&fd_mats->sig_eps2);
	fclose(readfdparams);



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
	gsl_matrix_memcpy(LambdaZCoef,&(LC.matrix));
	readmat("var_eta.csv",var_eta);
	if(homosk_zeta==1){
		// zeta covariance matrix is diagonal and maybe homoskedastic
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
		for(l=0;l<var_zeta->size1;l++){
			for(d=0;d<var_zeta->size2;d++)
				if(d!=l) gsl_matrix_set(fd_mats->var_zeta,l,d,0.);
		}

	}
	else // arbitrary, potentially non-diagonal
		readmat("var_zeta.csv",var_zeta);

	if(rescale_var ==1){
		// fix variance so that it can solve
		double sigzeta_2 = (1.0-0.99*0.99)/(1.0 - rhozz*rhozz);
		double sigepsi_2 = (1.0-0.98*0.98)/(1.0 - rhoZ*rhoZ);
		rhozz = 0.99; rhoZ = 0.98;

		gsl_matrix_scale(var_zeta,sigzeta_2); // adjust so that has the same unconditional variance as with the true persistence
		sig_eps *= sigepsi_2;
	}

	if(printlev>=3){
		printmat("readGamma.csv",GammaCoef);
		printmat("readLambda.csv",LambdaCoef);
	}

	struct st_wr * st = malloc( sizeof(struct st_wr) );

	st->cal_set = calflag;
	st->cal_best= 10.0;
	st->cal_worst= 10.0;


	/* Calibration Loop!
	*/
	double x0_0[]	= {phi  ,sig_psi    ,tau    ,scale_s    ,shape_s ,effic	,chi_co[0] ,chi_co[1] ,chi_co[2] ,chi_co[3]	};
	double lb_0[]	= {0.25 ,0.25       ,0.001  ,0.035      ,0.015	 ,0.75   ,0.0       ,0.0      ,0.0      ,-1.0};
	double ub_0[]	= {0.6  ,5.0       ,0.1     ,0.15	    ,0.20    ,1.75   ,1.0       ,1.0      ,1.       , 0.0};

	int Ntotx = sizeof(x0_0)/sizeof(double);
	for(i=0;i<Ntotx;i++){
		if (lb_0[i]>=ub_0[i]){
			double lbi = lb_0[i];
			lb_0[i] = ub_0[i];
			ub_0[i] = lbi;
		}
	}
	gsl_vector_view x0vw = gsl_vector_view_array (x0_0, 6+Nskill);
	gsl_vector_view lbvw = gsl_vector_view_array (lb_0, 6+Nskill);
	gsl_vector_view ubvw = gsl_vector_view_array (ub_0, 6+Nskill);

	if(calflag>=0){
		double objval = 100.0;
		int printlev_old = printlev;
		int verbose_old	= verbose;
		printlev =1;
		verbose = 1;
		int param_off = 6; // the number of non-wage regression parameters
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
		if(calflag==0||calflag>=4){
			st->cal_set = 0;
			st->n	= param_off+Nskill;

			strcpy(calhi_f,"calhist0.log");
			calhist = fopen(calhi_f,"a+");
			fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3,b0\n");
			fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3,b0\n");
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
		if(calflag==3){
			// this is the no-hc version of the model
			st->cal_set = 3;
			st->n	= param_off - 2;
			gsl_vector * ub3 	= gsl_vector_calloc(st->n);
			ub3->data[0]		=	ub_0[0];
			ub3->data[1]		=	ub_0[3];	ub3->data[2]	=	ub_0[4];	ub3->data[3]	=	ub_0[5];
			gsl_vector * lb3	= gsl_vector_calloc(st->n);
			lb3->data[0]		=	lb_0[0];
			lb3->data[1]		=	lb_0[3];	lb3->data[2]	=	lb_0[4];	lb3->data[3]	=	lb_0[5];
			gsl_vector * x03 = gsl_vector_calloc(st->n);
			x03->data[0]	=	phi	; x03->data[1]	=	scale_s	;x03->data[2]	=	shape_s	;	x03->data[3]	=	effic;
		/*	strcpy(calhi_f,"calhist3.log");
			calhist = fopen(calhi_f,"a+");
			fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3\n");
			fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3\n");
			fprintf(calhist,"***Beginning Calibration of param set 3***\n");
			fclose(calhist);

			strcpy(soler_f,"solerr3.log");
			solerr = fopen(soler_f,"a+");
			fprintf(solerr,"Errors while solving model with param set 3\n");
			fclose(solerr);

			strcpy(simer_f,"simerr3.log");
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"Errors in policies while simulating param set 3\n");
			fclose(simerr);
		 	 */
			if(verbose >= 1)printf("Starting to calibrate on variable subset 3\n");
			//							f		lb ub n x0 param, ftol xtol, ctol , algo
			objval = bndCalMinwrap(&cal_dist,lb3->data,ub3->data,st->n,x03->data,st,cal_xtol,cal_ftol,0.0,opt_alg,polish_alg);
			if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st->cal_set);
			gsl_vector_free(ub3);
			gsl_vector_free(lb3);
			gsl_vector_free(x03);

		}

		printlev = printlev_old;
		verbose  = verbose_old;
	}

	alloc_econ(st);

	status += sol_ss(st->ss,0,st->xss,st->sol);
	if(verbose>=1 && status ==0) printf("Successfully computed the steady state\n");
	if(verbose>=0 && status ==1) printf("Broke while computing steady state\n");
	status += sys_def(st->ss,st->sys,st->mats);
	if(verbose>=1 && status >=1) printf("System not defined\n");
	if(verbose>=0 && status ==0) printf("System successfully defined\n");

	if(verbose>=2) printf("Now defining the 1st order solution to the dynamic model\n");

	int t;
	if(verbose>=1) t = clock();
	status += sol_dyn(st->ss, st->sol,st->sys,0);
	if(verbose>=0 && status >=1) printf("System not solved\n");
	if(verbose>=0 && status ==0) printf("System successfully solved\n");
 	// update and solve the stochastic process
	status += sol_zproc(st, st->ss, st->xss,st->mats);

	//status += ss_moments(&simdat, ss, xss);
	if(status ==0)
		status += sim_moments(st,st->ss,st->xss);
	if(verbose>=1 && status ==0){
		printf ("It took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	}

	// try out some parameter values for the wage regression

	/*/	TGR experiment

	if(status ==0)
		status +=  TGR(st);//TGR(u_dur_dist,opt_dur_dist,fnd_dist,opt_fnd_dist,&urt,&opt_urt,st);
	*/

	status += free_econ(st);

	if(calflag==-2){
		FILE * f_startx, *f_startf;

		// try out some parameter values for the wage regression
		int Ntestpts = 4;
		int ii,iii,iiii;
		ii = pow(Ntestpts,Nskill);
		double testpts[ii][Nskill];

		for(i=0;i<Ntestpts;i++){
			for(ii=0;ii<Ntestpts;ii++){
				for(iii=0;iii<Ntestpts;iii++){
					for(iiii=0;iiii<Ntestpts;iiii++){
						d = i*pow(Ntestpts,3)+ ii*pow(Ntestpts,2) + iii*Ntestpts + iiii;
						testpts[d][0]
								= (double)(i/Ntestpts)*   (ub_0[6]  -lb_0[6])   + lb_0[6];
						testpts[d][1]
								= (double)(ii/Ntestpts)*  (ub_0[6+1]-lb_0[6+1]) + lb_0[6+1];
						testpts[d][2]
								= (double)(iii/Ntestpts)* (ub_0[6+2]-lb_0[6+2]) + lb_0[6+2];
						testpts[d][3]
								= (double)(iiii/Ntestpts)*(ub_0[6+3]-lb_0[6+3]) + lb_0[6+3];

						set_params(testpts[d],2);

						alloc_econ(st);

						status += sol_ss(st->ss,0,st->xss,st->sol);
						if(verbose>=1 && status ==0) printf("Successfully computed the steady state\n");
						if(verbose>=0 && status ==1) printf("Broke while computing steady state\n");
						status += sys_def(st->ss,st->sys,st->mats);
						if(verbose>=1 && status >=1) printf("System not defined\n");
						if(verbose>=0 && status ==0) printf("System successfully defined\n");

						if(verbose>=2) printf("Now defining the 1st order solution to the dynamic model\n");

						int t;
						if(verbose>=1) t = clock();
						status += sol_dyn(st->ss, st->sol,st->sys,0);
						if(verbose>=0 && status >=1) printf("System not solved\n");
						if(verbose>=0 && status ==0) printf("System successfully solved\n");
						// update and solve the stochastic process
						status += sol_zproc(st, st->ss, st->xss,st->mats);

						if(status ==0)
							status += sim_moments(st,st->ss,st->xss);



						f_startx = fopen("testpts_chi_x.txt","a+");
						f_startf = fopen("testpts_chi_f.txt","a+");

						for(l=0;l<Nskill;l++)
							fprintf(f_startx,"%f,",testpts[d][l]);
						fprintf(f_startx,"\n");

						for(l=0;l<6;l++)
							fprintf(f_startf,"%f,", gsl_matrix_get(st->sim->data_moments,0,l));
						for(l=0;l<Nskill;l++)
							fprintf(f_startf,"%f,", gsl_matrix_get(st->sim->data_moments,1,l));

						fprintf(f_startf,"\n");
						fclose(f_startx);
						fclose(f_startf);


						status += free_econ(st);

					}
				}
			}
		}


	}




	gsl_matrix_free(GammaCoef);	gsl_matrix_free(LambdaCoef);
	gsl_matrix_free(var_eta); gsl_matrix_free(var_zeta);
	gsl_matrix_free(f_skills);
	free(b);
	for(l=0;l<Noccs+1;l++)
		free(chi[l]);
	free(chi);
	free(st);
	gsl_matrix_free(fd_mats->Gamma);
	gsl_matrix_free(fd_mats->Lambda);
	gsl_matrix_free(fd_mats->var_eta);
	gsl_matrix_free(fd_mats->var_zeta);
	free(fd_mats);
	free(sk_wg);

	return status;
}

/*
 * Solve the dynamic system, by my own method (no QZ decomp)
 * The technique follows page 3 & 6 of my Leuchtturm1917
 *
 */
int sol_dyn(gsl_vector * ss, struct sys_sol * sol, struct sys_coef *sys, int sol_shocks){

	// sol_shocks = 1 assumes that P1 has already been solved for and skips a big inversion
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
	int status = 0,s,nonfin=0;
	Ns = (sys->F1)->size2;
	Nc = (sys->F1)->size1;
	Nx = (sys->F3)->size2;



	if(verbose>=2) printf("Ns=%d,Nc=%d,Nx=%d\n",Ns,Nc,Nx);

	if(sol_shocks != 1){
		// Find P1
		gsl_matrix * A2invF0F1 = gsl_matrix_calloc(Ns,Ns);
		gsl_matrix * invF0F1 = gsl_matrix_calloc(Nc,Ns);

	#ifdef _MKL_USE
		gsl_matrix * F0LU =gsl_matrix_calloc((sys->F0)->size1,(sys->F0)->size2);
		gsl_matrix_memcpy(F0LU,sys->F0);
		int * ipiv = malloc(sizeof(int)* ((sys->F0)->size2) );
		for(s=0;s< sys->F0->size2; s++ ) ipiv[s] =0.;
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
		if(check_fin>0){
			nonfin += isfinmat(sol->P0);
			nonfin += isfinmat(sol->P1);
			nonfin += isfinmat(sol->P2);

			if(printlev>=1){
				solerr = fopen(soler_f,"a+");
				fprintf(solerr,"Non-finite %d times in P0-P2\n",nonfin);
				fclose(solerr);
			}
		}

	#ifndef _MKL_USE
		gsl_matrix_free(invF0);
		free(ipiv);
	#endif
	#ifdef _MKL_USE
		gsl_matrix_free(F0LU);
		free(ipiv);
	#endif

	}

	// and forward looking for PP --- everyone has to solve this because it depends on N

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


#endif
#ifndef _MKL_USE
	gsl_matrix * invP0	= gsl_matrix_calloc(Ns,Ns);
	status += inv_wrap(invP0,sol->P0);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,invP0,0.0,P1invP0);

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
	double mnorm = norm1mat(P1invP0P2N),dnorm =0 ;
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
		dnorm  = mnorm-mnorml;
		mnorml = mnorm;
	}// end for(i=0;i<maxrecur;i++)
	//status = i<maxrecur-1? status : status+1;
	if(i>=maxrecur-1){
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"Forward expectations stopped at %f, did not converge\n",mnorm);
		fclose(solerr);
		if(verbose>=2)
			printf("Forward expectations stopped at %f (diff = %f), did not converge\n",mnorm,dnorm);
		if(fabs(mnorm)>mnorm0 && dnorm>0)
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

	int status,l,d,dd,ll,iter,itermax,itermin,i,allocZz, JJ1;
	itermax = 2500;
	if(dbg_iters ==1) itermax =100;
	itermin = 100;
	gsl_vector 	* W_l0,*W_ld, *lW_l0;
	gsl_vector 	* Uw_ld;
	gsl_vector 	* gld,*lgld,*thetald,*findrt, *sld;
	gsl_matrix 	* m_distg; //max dist g, ave dist g, max dist W0
	gsl_matrix 	* pld,*x;//,*lx;

	if(Zz == 0 || Zz == NULL){
		Zz = gsl_vector_calloc(Nx);
		allocZz = 1;
	}
	else
		allocZz = 0;
	JJ1 = 	Noccs*(Noccs+1);
	x		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	//lx		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	W_l0 	= gsl_vector_calloc(2*Noccs+2);
	lW_l0 	= gsl_vector_calloc(2*Noccs+2);
	W_ld 	= gsl_vector_calloc(JJ1);
	Uw_ld 	= gsl_vector_calloc(JJ1*2);
	gld 	= gsl_vector_calloc(2*JJ1);
	lgld 	= gsl_vector_calloc(gld->size);
	thetald	= gsl_vector_calloc(2*JJ1);
	findrt 	= gsl_vector_calloc(2*(Noccs+1));
	sld		= gsl_vector_calloc(JJ1);

	pld	= gsl_matrix_alloc(2*(Noccs+1),Noccs);
	
	if(printlev>=2){
		m_distg = gsl_matrix_calloc(itermax,3);
	}
	
	// initialize the policies:
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1] : b[0];
		for(d=0;d<Noccs;d++){

			gsl_vector_set(sld,l*Noccs+d,sbar);
			for(ll=0;ll<2;ll++){
				//	if(l==0)
					gsl_vector_set(gld,ll*JJ1+l*Noccs+d,1.0/(double)Noccs);
				//	else if(l==d+1)
				//		gsl_vector_set(gld,l*Noccs+d,1.0);
				gsl_vector_set(thetald,ll*JJ1+l*Noccs+d,invp(avg_fnd));

				gsl_matrix_set(pld,ll*(Noccs+1)+l,d,pmatch(thetald->data[l*Noccs+d+ll*JJ1]));
				if(gsl_matrix_get(pld,ll*(Noccs+1)+l,d)>1.0)
					gsl_matrix_set(pld,ll*(Noccs+1)+l,d,1.0);
				W_l0->data[l+ll*(Noccs+1)] += gld->data[l*Noccs+d+ll*JJ1]*gsl_matrix_get(pld,l+ll*(Noccs+1),d)*(chi[l][d] - (1.0-(double)ll)*bl - (double)ll*privn);
				gsl_matrix_set(sol->ss_wld,ll*JJ1+l,d,chi[l][d]-.05);
			}
			W_ld->data[l*Noccs+d] = chi[l][d];///(1.0-beta);
		}
		W_l0->data[l] += bl;			W_l0->data[l] /= (1.0-beta);
		W_l0->data[l+Noccs+1] += privn;	W_l0->data[l+Noccs+1] /= (1.0-beta);
	}

	gsl_vector_view W_l0I = gsl_vector_subvector(W_l0,0,Noccs+1);
	gsl_vector_set_all(&W_l0I.vector,b[0]/(1.0-beta));
	gsl_vector_view W_l0E = gsl_vector_subvector(W_l0,Noccs+1,Noccs+1);
	gsl_vector_set_all(&W_l0E.vector,privn/(1.0-beta));

	if(printlev>=2){
		printvec("gld0.csv",gld);
		printvec("thetald0.csv",thetald);
		printvec("W_l0_0.csv",W_l0);
		printvec("W_ld_0.csv",W_ld);
	}
	double maxdistg=1e-4,maxdistW=1e-4,adistg=1e-4,lmaxdistW0=1.0;
	double lmaxdistg = 100.0;


	int break_flag =0;
	double Z = Zz->data[0];

	for(iter=0;iter<itermax;iter++){

		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++){
				double findrt_ld =0.0;
				for(d=0;d<Noccs;d++)
					findrt_ld += gld->data[JJ1*ll + l*Noccs+d]*gsl_matrix_get(pld,l+ll*JJ1,d);
				gsl_vector_set(findrt,l + ll*(Noccs+1),findrt_ld);
			}
		}
		double maxdistW0=0;
		gsl_vector_memcpy(lW_l0,W_l0);
		int vfiter;
		int vfitermax = break_flag == 1 ? itermax : itermax/2;
		if( vfitermax <itermin) vfitermax = itermin;
		if(iter==0){
		for(vfiter =0;vfiter<itermax/2;vfiter++){
			maxdistW0 = 0.0;
			for(d=0;d<Noccs;d++){
				l = d+1;
				double zd = Zz->data[d+Notz];
				double sepld = sld->data[l*Noccs+d];
				gsl_vector_set(W_ld,l*Noccs+d,
						((1.0-sepld)*(chi[l][d]*(1.0+zd)+beta*W_ld->data[l*Noccs+d])
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
							((1.0-tau)*((1.0-sepld )*(chi[l][d]*(1.0+zd)+ beta*W_ld->data[l*Noccs+d])
									+ sepld*W_l0->data[0] ) + tau*W_ld->data[(d+1)*Noccs+d]  )
				//			/(1.0-(1.0-sepld)*(1.0-tau)*beta)
									);

					}
				}
			}

			// W_l0
			for(l=0;l<Noccs+1;l++){
				double bl = l>0 ? b[1]:b[0];
				for(ll=0;ll<2;ll++){
					double findrt_ld =0.0;
					double W_0 =0.0;
					for(d=0;d<Noccs;d++){
						findrt_ld += gld->data[JJ1*ll + l*Noccs+d]*gsl_matrix_get(pld,l+ll*(Noccs+1),d);
						double zd = Zz->data[d+Notz];
						double nud = l==d+1? 0.0:nu;
						double post = gsl_matrix_get(pld,l,d)>0? - kappa*thetald->data[l*Noccs+d]/gsl_matrix_get(pld,l+ll*(Noccs+1),d) : 0.0;
						double ret 	= chi[l][d]*(1.0+zd) - nud
				//				+ post
								+ beta*gsl_vector_get(W_ld,l*Noccs+d)
								- bl*(1.0-(double)ll) - (double)ll*privn - beta*((1.0-1.0/bdur)*W_l0->data[l + ll*(Noccs+1)] + 1.0/bdur*W_l0->data[l+Noccs+1])
								;
						//ret  = (l == d+1) ? ret  - bl - beta*W_l0->data[l] : ret  - bl - beta*W_l0->data[0];
						ret 	*= (1.0-fm_shr);
						ret 	+= (bl*(1.0-(double)ll) + ((double)ll)*privn  + beta*(1.0-1.0/bdur)*W_l0->data[l+ ll*(Noccs+1)] + beta*1.0/bdur*W_l0->data[l+Noccs+1]);
						//ret		 = ret> W_ld->data[l*Noccs+d] ? W_ld->data[l*Noccs+d] : ret;
						W_0 	+= gld->data[ll*JJ1+l*Noccs+d]*gsl_matrix_get(pld,l+ll*(Noccs+1),d)*ret;
					}// d=0:Noccs

					W_0 	+= (1.0-findrt_ld)*(bl*(1.0-(double)ll) +privn*((double)ll)
							+ beta*((1.0-1.0/bdur)*W_l0->data[l+ll*(Noccs+1)] + 1.0/bdur*W_l0->data[l+Noccs+1]));
			//		W_0 	-= W_l0->data[l+ll*(Noccs+1)]*
			//				beta*(1.0-(1.0-(double)ll)*1.0/bdur)*(fm_shr*findrt_ld + (1.0-findrt_ld));
			//		W_0		/= (1.0 - beta*(1.0-(1.0-(double)ll)*1.0/bdur)*(fm_shr*findrt_ld + (1.0-findrt_ld)) );
					double distW0 = fabs(W_l0->data[l+ll*(Noccs+1)] - W_0)/W_l0->data[l];
					W_l0->data[l+ll*(Noccs+1)] = W_0;

					if(distW0 > maxdistW0)
						maxdistW0 = distW0;
				}
			}
			double W0_tol = 2*maxdistg< 1e-4? 2*maxdistg : 1e-4;
			W0_tol = break_flag == 1 ? ss_tol : W0_tol;
			if(maxdistW0<W0_tol)// not particularly precise to start
				break;
		}//for vfiter
		}

		// update policies

		maxdistg = 0.0;
		gsl_vector_memcpy(lgld,gld);

		#pragma omp parallel for private(d,ll,l,dd)
		for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			for(ll=0;ll<2;ll++){
				gsl_integration_workspace * integw = gsl_integration_workspace_alloc(1000);
				gsl_vector 	* ret_d = gsl_vector_calloc(Noccs);
				double 		* R_dP_ld = malloc((Noccs*2+1)*sizeof(double));
				for(d=0;d<Noccs;d++){
					double zd = Zz->data[d+Notz];
					double nud = l == d+1? 0:nu;

						double Wdif	= W_ld->data[l*Noccs+d] - (1.0-1.0/bdur)*W_l0->data[l + ll*(Noccs+1)] - 1.0/bdur*W_l0->data[l + Noccs+1];
						//Wdif = Wdif<0.0 ?  0.0 : Wdif;
						double pld_ld = gsl_matrix_get(pld,l+ll*(Noccs+1),d);
						double post = pld_ld > 0 ? -kappa*gsl_vector_get(thetald,ll*JJ1 + l*Noccs+d)/pld_ld : 0.0;

						double arr_d = (1.0-fm_shr)*(chi[l][d]*exp(zd) -nud - (1.0-(double)ll)*bl - ((double)ll)*privn + beta*Wdif);
						//		+ (1.0-(double)ll)*bl + ((double)ll)*privn + beta*((1.0-1.0/bdur)*W_l0->data[l+ ll*(Noccs+1)] + 1.0/bdur*W_l0->data[l + Noccs+1]);
						arr_d = arr_d > W_ld->data[l*Noccs+d] ? W_ld->data[l*Noccs+d]: arr_d;
						gsl_vector_set(ret_d,d,pld_ld*arr_d);
				}
				//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				// gld

				// calculate choice probs and integrate to be sure all options integrate to 1:
				for(d=0;d<Noccs+1;d++) R_dP_ld[d] = 0.;
				for(dd=0;dd<Noccs;dd++){ // values in any possible direction
					double ret_d_dd = gsl_vector_get(ret_d,dd);
					R_dP_ld[dd+1] =  ret_d_dd > 0 &&  ret_d_dd < 1.e10 ? ret_d_dd : 0.0;
					R_dP_ld[dd+1+Noccs] = sig_psi*gsl_matrix_get(pld,l+ll*(Noccs+1),dd);
				}

				double sexpret_dd = 0.0;
				if(homosk_psi != 1){
					gsl_function gprob;
					gprob.function = &hetero_ev;
					gprob.params = R_dP_ld;
					for (d = 0; d < Noccs; d++) { // choice probabilities
						if (R_dP_ld[d + 1] > 0. && gsl_matrix_get(pld, l + ll * (Noccs + 1), d) > 0.) {
							R_dP_ld[0] = (double) d;
							double gg, ggerr;
							#pragma omp critical
							{
								//	gsl_integration_qags(&gprob, 1.e-5, 20, 1.e-5, 1.e-5, 1000, integw, &gg, &ggerr);
								size_t neval;
								gsl_integration_qng(&gprob, 1.e-5, 20, 1.e-5, 1.e-5, &gg, &ggerr,&neval);
							}
							sexpret_dd += gg;
							gsl_vector_set(gld, l * Noccs + d + ll * JJ1, gg);
						}
						else
							gsl_vector_set(gld, l * Noccs + d + ll * JJ1, 0.);
					}
				}else{
					// total choice probability
					double gdenom =0;
					for(dd=0;dd<Noccs;dd++) {
						if (R_dP_ld[dd + 1] > 0. && gsl_matrix_get(pld, l + ll * (Noccs + 1), dd) > 0.)
							gdenom += exp(R_dP_ld[dd + 1]);
					}

					double gg;
					for(d=0;d<Noccs;d++){
						if (R_dP_ld[d + 1] > 0. && gsl_matrix_get(pld, l + ll * (Noccs + 1), d) > 0.) {
							gg = exp(R_dP_ld[d + 1]) / gdenom;
							sexpret_dd += gg;
							gsl_vector_set(gld, l * Noccs + d + ll * JJ1, gg);
						}else
							gsl_vector_set(gld, l * Noccs + d + ll * JJ1, 0.);
					}
				}
				for(d=0;d<Noccs;d++){
					double g_updaterate = 1.0;
					// convex combination between gld and lgld
					gsl_vector_set(gld, l*Noccs+d+ll*JJ1,
						(1.- g_updaterate)* gsl_vector_get(lgld,l*Noccs+d+ll*JJ1)
						// rescale by sexpret_dd because not exactly add to 1
						+ g_updaterate* gsl_vector_get(gld,l*Noccs+d+ll*JJ1)/sexpret_dd);
				}
				// check it was pretty close to 1
				if( fabs(sexpret_dd-1.) > 1.e-2){
					solerr = fopen(soler_f,"a+");
					fprintf(solerr,"SS choice probabilities added to %f != 1 at (l,ll)=(%d,%d)\n",sexpret_dd,l,ll);
					fclose(solerr);
					if(verbose>=2)
						printf("SS choice probabilities add to %f != 1 at (l,ll)=(%d,%d)\n",sexpret_dd,l,ll);
				}

				//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				// theta & sld
				for(d=0;d<Noccs;d++){
					double zd = Zz->data[d+Notz];

					double nud 	= l-1 !=d ? nu : 0.0;
					double Wdif	= W_ld->data[l*Noccs+d] - (1.0-1.0/bdur)*W_l0->data[ll*(Noccs+1)+l]- 1.0/bdur*W_l0->data[Noccs+1+l];
					double surp = chi[l][d]*exp(zd) - nud - (1.0-(double)ll)*bl - ((double)ll)*privn +beta*Wdif;
					if(surp > 0.0){
						double qhere = kappa/(fm_shr*surp);
						double tldi = invq(qhere);
						if(gsl_finite(tldi)==1)
							gsl_vector_set(thetald,JJ1*ll+l*Noccs+d,tldi );
						else
							gsl_vector_set(thetald,JJ1*ll+l*Noccs+d,0.0);
					}
					else
						gsl_vector_set(thetald,JJ1*ll+l*Noccs+d,0.0);

					// sld
					double cutoff = -(chi[l][d]*exp(zd) + beta*W_ld->data[l*Noccs+d] - W_l0->data[l]);
					cutoff = cutoff >0.0 ? 0.0 : cutoff;
					double sep_ld = scale_s*exp(shape_s*cutoff);
					gsl_vector_set(sld,l*Noccs+d,sep_ld);
				}// for d=0:Noccs
				// if there're vacancies open, be sure at least someone's going there:
				for(d=0;d<Noccs;d++){
					if( gsl_vector_get(thetald,JJ1*ll+l*Noccs+d)>0. && gsl_vector_get(gld,JJ1*ll+l*Noccs+d)<1.e-5 )
						gsl_vector_set(gld,JJ1*ll+l*Noccs+d,1.e-5) ;
				}

				// g adds to 1 (may not because of the prior step)
				double gsum = 0.0;
				for(d =0 ;d<Noccs;d++)	gsum += gld->data[ll*JJ1 + l*Noccs+d];
				for(d =0 ;d<Noccs;d++)	gld->data[ll*JJ1 + l*Noccs+d] /= gsum;
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(pld,l+ll*(Noccs+1),d,pmatch(thetald->data[ll*JJ1+l*Noccs+d]));


				// Calc VFs
 				if(l>0){
					d = l-1;
					double zd = Zz->data[d+Notz];
					double sepld = sld->data[l*Noccs+d];

					gsl_vector_set(W_ld,l*Noccs+d,
							((1.0-sepld)*(chi[l][d]*(1.0+zd)+beta*W_ld->data[l*Noccs+d])
									+ sepld*W_l0->data[l]));
				}
				for(d=0;d<Noccs;d++){
					double sepld = sld->data[l*Noccs+d];
					double zd = Zz->data[d+Notz];
					if(l-1!=d){
						gsl_vector_set(W_ld,l*Noccs+d,
							((1.0-tau)*((1.0-sepld )*(chi[l][d]*exp(zd)+ beta*W_ld->data[l*Noccs+d])
									+ sepld*W_l0->data[0] ) + tau*W_ld->data[(d+1)*Noccs+d]  ) );
						double bhere = l>0 ? b[1]:b[0];
						bhere = ll>0? privn:bhere;
						gsl_vector_set(Uw_ld,ll*(JJ1)+l*Noccs+d,
								((1.0-tau)*((1.0-sepld )*(gsl_matrix_get(sol->ss_wld,ll*JJ1+l,d)
									+ beta*Uw_ld->data[ll*(JJ1)+l*Noccs+d])
									+ sepld*W_l0->data[0] ) + tau*Uw_ld->data[(d+1)*Noccs+d]  ));
						gsl_matrix_set(sol->ss_wld,ll*(Noccs+1)+l,d,
								(1.0-fm_shr)*(chi[l][d]+beta*W_ld->data[l*Noccs+d])
								-fm_shr*(-bhere-beta*W_l0->data[l]) -beta*Uw_ld->data[l*Noccs+d]);// put in E[phi]
						// cannot get more than the full surplus
						if(gsl_matrix_get(sol->ss_wld,ll*(Noccs+1)+l,d)>chi[l][d])
								gsl_matrix_set(sol->ss_wld,ll*(Noccs+1)+l,d,chi[l][d]);

					}
					else
							gsl_vector_set(Uw_ld,ll*(JJ1)+l*Noccs+d,
								(1-sepld)*(gsl_matrix_get(sol->ss_wld,l,d)
								+ beta*Uw_ld->data[l*Noccs+d])
								+sepld*W_l0->data[l]);

							gsl_matrix_set(sol->ss_wld,ll*(Noccs+1)+l,d,
								(1.0-fm_shr)*(chi[l][d]+beta*W_ld->data[l*Noccs+d])
								-fm_shr*(-b[1]-beta*W_l0->data[l])-beta*Uw_ld->data[l*Noccs+d]);
							if(gsl_matrix_get(sol->ss_wld,ll*(Noccs+1)+l,d)>chi[l][d])
								gsl_matrix_set(sol->ss_wld,ll*(Noccs+1)+l,d,chi[l][d]);
				}

				double findrt_ld =0.0;
				double W_0 =0.0;
				for(d=0;d<Noccs;d++){
					findrt_ld += gld->data[JJ1*ll + l*Noccs+d]*gsl_matrix_get(pld,l+ll*(Noccs+1),d);
					double zd = Zz->data[d+Notz];
					double nud = l==d+1? 0.0:nu;
					double post = gsl_matrix_get(pld,l,d)>0? - kappa*thetald->data[l*Noccs+d]/gsl_matrix_get(pld,l+ll*(Noccs+1),d) : 0.0;
					double ret 	= chi[l][d]*exp(zd) - nud
							+ beta*gsl_vector_get(W_ld,l*Noccs+d)
							- bl*(1.0-(double)ll) - (double)ll*privn - beta*((1.0-1.0/bdur)*gsl_vector_get(W_l0,l+ll*(Noccs+1)) + 1.0/bdur*gsl_vector_get(W_l0,l+Noccs+1));
					ret 	*= (1.0-fm_shr);
					ret 	+= (bl*(1.0-(double)ll) + ((double)ll)*privn  + beta*(1.0-1.0/bdur)*gsl_vector_get(W_l0,l+ ll*(Noccs+1)) + beta*1.0/bdur*gsl_vector_get(W_l0,l+Noccs+1));
					W_0 	+= gld->data[ll*JJ1+l*Noccs+d]*gsl_matrix_get(pld,l+ll*(Noccs+1),d)*ret;
				}// d=0:Noccs

				W_0 	+= (1.0-findrt_ld)*(bl*(1.0-(double)ll) +privn*((double)ll)
						+ beta*((1.0-1.0/bdur)*gsl_vector_get(W_l0,l+ll*(Noccs+1)) + 1.0/bdur*gsl_vector_get(W_l0,l+Noccs+1)));
				double distW0 = fabs(W_l0->data[l+ll*(Noccs+1)] - W_0)/W_l0->data[l];
				gsl_vector_set(W_l0,l+ll*(Noccs+1),W_0);

				if(distW0 > maxdistW0)
					maxdistW0 = distW0;
					
				// free d-specific things
				gsl_vector_free(ret_d);
				free(R_dP_ld);
				gsl_integration_workspace_free(integw);
			}// for ll=0:1

		}// for l=0:Noccs
		
		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++){
				double distW = fabs(gsl_vector_get(lW_l0,l*Noccs + ll*(Noccs+1)) - gsl_vector_get(W_l0,l*Noccs + ll*(Noccs+1)) )
							   /(1.0+gsl_vector_get(lW_l0,l*Noccs + ll*(Noccs+1)));
				maxdistW = distW>maxdistg ? distW : maxdistW;
				for(d=0;d<Noccs;d++){
					double distg = fabs(gsl_vector_get(lgld,l*Noccs+d + ll*JJ1) - gsl_vector_get(gld,l*Noccs+d + ll*JJ1))
								   /(1.0+gsl_vector_get(lgld,l*Noccs+d + ll*JJ1));
					maxdistg = distg>maxdistg ? distg : maxdistg;
					adistg += distg*distg;
				}
			}
		}
		if(printlev>=2){
			gsl_matrix_set(m_distg,iter,0,maxdistg);
			gsl_matrix_set(m_distg,iter,1,sqrt(adistg/JJ1/2));
			gsl_matrix_set(m_distg,iter,2,maxdistW0);
		}

		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++){
				double fnd_lll = 0.0;
				for(d=0;d<Noccs;d++){
					double fllld = pmatch(thetald->data[JJ1*ll + l*Noccs+d]);
					fnd_lll += fllld;
					gsl_matrix_set(pld,l+ll*(Noccs+1),d, fllld);
				}
				if(fnd_lll <= 0.0){
					gsl_vector_view pld_d = gsl_matrix_row(pld,l+ll*(Noccs+1));
					gsl_vector_set_all(&pld_d.vector,0.001);
				}
			}
		}

		// this is actually testing the condition for iteration convergence above
		if(break_flag ==1)
			break;
		else if(maxdistW0 < ss_tol && iter>=10)
			break_flag=1;
		else if((fabs(maxdistW0 - lmaxdistW0) < ss_tol) & (maxdistW0< ss_tol*20) & (iter>itermin) ){
			break_flag=1;
			//if(verbose>=2) printf("Not making progress in SS from %f\n",maxdistW0);
			//if(printlev>=1){
			//	solerr = fopen(soler_f,"a+");
			//	fprintf(solerr,"SS err stuck at %f\n", maxdistW0);
			//	fclose(solerr);
			//}
		}


		lmaxdistg = maxdistg;
		lmaxdistW0= maxdistW0;
		if(printlev>=4){
			printvec("W_l0_i.csv",W_l0);
			printvec("W_ld_i.csv",W_ld);
			printvec("Uw_ld_i.csv",Uw_ld);
			printvec("sld_i.csv",sld);
			printvec("gld_i.csv",gld);
			printvec("thetald_i.csv",thetald);
			printvec("findrt_i.csv",findrt);
		}


	}// end iter=0:maxiter
	status = (iter<itermax || (fabs(maxdistg - lmaxdistg)<ss_tol*5 && maxdistg<1.0)) && W_l0->data[0]>0.0 ? 0 : 1;

	/* steady-state wages: trying to solve in closed form:
	 *
	double * Jd = malloc(sizeof(double)*Noccs);

	for(d=0;d<Noccs;d++){
		double zd = Zz->data[d+Notz];
		double bl = b[1];
		int l = d+1;
		// need to use spol to get \bar \xi^{ld}, then evaluate the mean, \int_{-\bar \xi^{ld}}^0\xi sh e^{sh*\xi}d\xi
		//and then invert (1-scale_s)*0 + scale_s*log(sld/shape_s)/scale_s

		double barxi = -log(sld->data[l*Noccs+d]/scale_s )/shape_s;
		double Exi = scale_s*((1.0/shape_s+barxi)*exp(-shape_s*barxi)-1.0/shape_s);
	//	Exi = 0.0;
	//	double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(bl-Exi);
	//	double wld_ld = ((1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(W_ld->data[l*Noccs+d] - W_l0->data[l] + bl-Exi)
	//			+ beta*(1.0-sld->data[l*Noccs+d])*chi[l][d]*exp(Z+zd)/(1.0-beta*(1.0-sld->data[l*Noccs+d])) )
	//					/(1.0 + beta/(1.0-beta*(1.0-sld->data[l*Noccs+d])));
		double wld_ld = (1.0-fm_shr)*chi[l][d]*(1.0+zd) + fm_shr*( Exi*(1.0-beta*(1.0-sld->data[l*Noccs+d]))
				+ bl + beta*W_l0->data[l] - beta*W_ld->data[l*Noccs+d])
				+ beta*kappa/pld->data[l*pld->tda+d]*thetald->data[l*Noccs+d]*(1.0 - sld->data[l*Noccs+d]);

		wld_ld = wld_ld > 0.0 && thetald->data[l*Noccs+d]>0.0 ? wld_ld : 0.0;
		wld_ld = wld_ld > chi[l][d]*(1.0+zd) ? chi[l][d]*(1.0+zd) : wld_ld;
	//	wld_ld = chi[l][d]*exp(Z+zd);
		gsl_matrix_set(sol->ss_wld,l,d,wld_ld);
		Jd[d] = (1.0-sld->data[l*Noccs+d])*(chi[l][d]*(1.0+zd) - wld_ld)/(1.0-(1.0-sld->data[l*Noccs+d])*beta);
	}

	for(l=0;l<Noccs+1;l++){
		double bl = l==0 ? b[0] : b[1];
		for(d=0;d<Noccs;d++){
			double zd = Zz->data[d+Notz];
			if(d!=l-1){
			for(ll=0;ll<2;ll++){
				double Ephi = 0.0;
//				double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(bl - Ephi) ;

//				double wld_ld = ((1.0-fm_shr)*chi[l][d]*exp(Z+zd)- fm_shr*beta*(W_ld->data[l*Noccs+d] - W_l0->data[l] + bl - Ephi)
//						+ beta*((1.0-sld->data[l*Noccs+d])*(1.0-tau)*chi[l][d]*exp(Z+zd) +tau*Jd[d] )/(1.0-beta*(1.0-sld->data[l*Noccs+d])*(1.0-tau) ))
//								/(1.0+ beta/(1.0-beta*(1.0-tau)*(1.0-sld->data[l*Noccs+d])));
				double wld_ld = (1.0-fm_shr)*chi[l][d]*(1.0+zd) + fm_shr*(
						bl*(1.0-(double)ll) + privn*((double)ll) + beta*W_l0->data[l] - beta*W_ld->data[l*Noccs+d])
						+ beta*kappa/gsl_matrix_get(pld,ll*(Noccs+1)+l,d)*thetald->data[ll*JJ1+l*Noccs+d]*(1.0-sld->data[l*Noccs+d])*(1.0-tau)
						+ tau*(1.0-sld->data[(d+1)*Noccs+d])*beta*kappa/gsl_matrix_get(pld,ll*(Noccs+1)+d+1,d)*thetald->data[ll*JJ1+(d+1)*Noccs+d];
				wld_ld = wld_ld > 0.0 && thetald->data[ll*JJ1+l*Noccs+d]>0.0 ? wld_ld : 0.0;
				wld_ld = wld_ld > chi[l][d]*(1.0+zd) ? chi[l][d]*(1.0+zd) : wld_ld;
		//		wld_ld = chi[l][d]*exp(Z+zd);
				gsl_matrix_set(sol->ss_wld,ll*(Noccs+1)+l,d,wld_ld);
			}
			}
		}
	}
	free(Jd);
	*/
	if(printlev>=2){
		printvec("W_l0_i.csv",W_l0);
		printvec("W_ld_i.csv",W_ld);
		printvec("gld_i.csv",gld);
		printvec("sld_i.csv",sld);
		printmat("ss_wld.csv",sol->ss_wld);
		printvec("thetald_i.csv",thetald);
		printmat("m_distg.csv",m_distg);
		printvec("findrt_i.csv",findrt);
		if(verbose>=3) printf("Converged in %d iters\n", iter-1);
	}

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
					-(1.0-tau)*sld->data[l*Noccs+d-1]);
			}
		}
	}
	//x_l0 : l>0
	for(l=1;l<Noccs+1;l++){
		d=0;
		gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,
			(1.0-findrt->data[l]) );
		gsl_matrix_set(Pxx0,l*(Noccs+1)+d,l*(Noccs+1)+l, - sld->data[l*Noccs+d-1] );
	}
	//x_0d : d>0
	for(d=1;d<Noccs+1;d++){
		gsl_matrix_set(Pxx1,d,d,
			(1.0-tau)*(1.0- sld->data[0*Noccs+d-1]));
		gsl_matrix_set(Pxx1,d,0,
			gsl_vector_get(gld,d-1)*gsl_matrix_get(pld,0,d-1));
	}
	//x_ld : l,d>0
	for(l=1;l<Noccs+1;l++){
		for(d=1;d<Noccs+1;d++){
			if(l!=d){
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,
						(1.0-tau)*(1.0-sld->data[l*Noccs+d-1]));
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+0,
						gsl_vector_get(gld,l*Noccs + d-1)*gsl_matrix_get(pld,l,d-1));
			}else{
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,(1.0-sld->data[l*Noccs+d-1]));
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
	// the expired group:
	for(l=0;l<Noccs+1;l++){
		gsl_matrix_set(xss,l,Noccs+1,gsl_matrix_get(xss,l,0)*1.0/bdur);
		gsl_matrix_set(xss,l,0,gsl_matrix_get(xss,l,0)*(1.0-1.0/bdur));
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

	for(i=0;i<2*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,W_l0->data[i]);
		ssi++;	
	}
	for(i=0;i<Noccs*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,W_ld->data[i]);
		ssi++;
	}

	for(i=0;i<2*Noccs*(Noccs+1);i++){
		gsl_vector_set(ss,ssi,gld->data[i]);
		ssi++;
	}

	for(i=0;i<2*Noccs*(Noccs+1);i++){
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
	gsl_vector_free(lW_l0);
	gsl_vector_free(W_ld);
	gsl_vector_free(Uw_ld);
	gsl_matrix_free(x);
	gsl_vector_free(gld);
	gsl_vector_free(sld);

	gsl_vector_free(lgld);
	gsl_vector_free(thetald);
	gsl_matrix_free(pld);
	gsl_vector_free(findrt);
	if(printlev>=2){
		gsl_matrix_free(m_distg);
	}
	return status;
}

/*
 *
 * The nonlinear system that will be solved
 */

int sys_def(gsl_vector * ss, struct sys_coef * sys, struct shock_mats *mat0){
	int l,d,status=0,f;
//	int wld_i, tld_i, x_i,gld_i;
	int Nvarex 	= (sys->A3)->size2;
//	int Nvaren 	= ss->size;
//	int Notz 	= 1+ Nagf +Nfac*(Nflag+1);


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


	/* used to read these globals around which to solve things
	 * struct shock_mats mat0;
	mat0.Gamma = GammaCoef;
	mat0.rhoZ	= rhoZ;
	mat0.rhozz	= rhozz;
	mat0.sig_eps2 = sig_eps;
	mat0.var_eta = var_eta;
	mat0.var_zeta= var_zeta;
	mat0.Lambda	= gsl_matrix_alloc(Noccs,Nfac+1);
	gsl_matrix_view LC = gsl_matrix_submatrix(mat0.Lambda,0,0,Noccs,Nfac*(Nllag+1));
	gsl_matrix_memcpy(&LC.matrix,LambdaCoef);
	LC = gsl_matrix_submatrix(mat0.Lambda,0,Nfac*(Nllag+1),Noccs,1);
	gsl_matrix_memcpy(&LC.matrix,LambdaZCoef);
	*/

	status += sys_ex_set(sys->N,sys->S,mat0);


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
	if(check_fin>0){
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
	}
	gsl_vector_free(Zz);
	return status;
}

int sys_ex_set(gsl_matrix * N, gsl_matrix * S,struct shock_mats * mats){

	int status=0,l,d,f;
	gsl_matrix *N0,*N1,*invN0;
	int Nvarex = N->size1;
	if(Nvarex != Nx)
		printf("Size of N matrix is wrong!");

	N0 		= gsl_matrix_calloc(Nvarex,Nvarex);
	invN0 	= gsl_matrix_calloc(Nvarex,Nvarex);
	N1 		= gsl_matrix_calloc(Nvarex,Nvarex);
	gsl_matrix_set_identity(N0);
	//gsl_matrix_set_zero(N1);
	//gsl_matrix_set_zero(invN0);


	//partitions of Lambda:
	gsl_matrix_view Lambdarf = gsl_matrix_submatrix(mats->Lambda,0,0,Noccs,Nfac*(Nllag+1)); // the unobserved factors come first
	gsl_vector_view LambdarZ = gsl_matrix_column(mats->Lambda,Nfac*(Nllag+1)); // the aggregate comes second


//	for(l=0;l<Noccs;l++)
//		gsl_matrix_set(N0,l+ Notz,0,0.0);
	// contemporaneous factors:

	for(f=0;f<Nfac;f++){
		for(d=0;d<Noccs;d++)
			gsl_matrix_set(N0,d+Notz,f+Nagf,
					-gsl_matrix_get(&Lambdarf.matrix,d,f));
	}
	// contemporaneous coefficient for z
	for(d=0;d<Noccs;d++)
		gsl_matrix_set(N0,d+Notz,0,
			-gsl_vector_get(&LambdarZ.vector,d));

	/* First partition, N1_11 =
	 *
	 * (rhoZ  0 )
	 * ( I    0 )
	 *
	 * N1_12 =
	 * (Zfcoef 0 )
	 *
	 */
	gsl_matrix_set(N1,0,0,mats->rhoZ);
	// dynamics for the lagged Z kept around
	for(l=0;l<Nagf-1;l++)
		gsl_matrix_set(N1,l+1,l+1,1.0);
	//for(f=0;f<Nfac*Nglag;f++)
	//	gsl_matrix_set(N1,0,1+Nagf,Zfcoef[f]);

	/* Second partition, N1_22 =
	 * (Gamma  0 )
	 * ( I     0 )
	 */
	set_matrix_block(N1,mats->Gamma,Nagf,Nagf);
	for(f=0;f<Nfac*(Nllag);f++)
		gsl_matrix_set(N1,Nagf+Nfac+f,Nagf+f,1.0);
	/*
	 * Fourth partition, N1_32 =
	 * Lambda_{1:(Nllag-1)}
	 */
	if(Nllag>0){
		gsl_matrix_view N1_Lambda = gsl_matrix_submatrix(&Lambdarf.matrix, 0, Nfac, Noccs, Nfac*(Nllag+1)-Nfac);
		set_matrix_block(N1,&N1_Lambda.matrix,Notz,1+Nagf);
	}
	/*
	 * Final partition, N1_33 =
	 *  rhozz*I
	 */
	for(l=0;l<Noccs;l++)
		gsl_matrix_set(N1,l+Notz,l+Notz,mats->rhozz);

	inv_wrap(invN0,N0);
	status = gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invN0,N1,0.0,N);

	gsl_matrix_set_identity(S);
	gsl_matrix_set(S,0,0,
			mats->sig_eps2);
	// set correlations
	for(l=0;l<Noccs;l++){
		gsl_matrix_set(S,0,l+Notz,gsl_vector_get(cov_ze,l));
		gsl_matrix_set(S,l+Notz,0,gsl_vector_get(cov_ze,l));
	}
	set_matrix_block(S,mats->var_eta,Nagf,Nagf);
	for(l=0;l<Noccs;l++){
		for(d=0;d<Noccs;d++){ //upper triangle only
			double Sld = gsl_matrix_get(mats->var_zeta,l,d);
			//Sld = l!=d && (Sld>-5e-5 && Sld<5e-5) ? 0.0 : Sld;
			gsl_matrix_set(S,Notz+l,Notz+d,Sld);
		}
	}

	status += gsl_linalg_cholesky_decomp(S);
	// for some reason, this is not already triangular
	for(d=0;d<S->size2;d++){
		for(l=0;l<S->size1;l++){
			if(l<=d)
				gsl_matrix_set(S,l,d,gsl_matrix_get(S,l,d));
			else
				gsl_matrix_set(S,l,d,0.0);
		}
	}

	// set as zero innovations to carried lag terms
	for(l=0;l<Nfac*Nllag;l++)
		gsl_matrix_set(S,Nagf+Nfac+l,Nagf+Nfac+l,0.0);
	gsl_matrix * tmp = gsl_matrix_alloc(invN0->size1,invN0->size2);
	gsl_matrix_memcpy(tmp,S);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invN0,tmp,0.0,S);
	gsl_matrix_free(tmp);
	if(printlev>=2){
		printmat("N0.csv",N0);
		printmat("N1.csv",N1);
		printmat("invN0.csv",invN0);

	}

	gsl_matrix_free(N0);
	gsl_matrix_free(N1);
	gsl_matrix_free(invN0);
	return status;
}

int sys_st_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	/* this function sets up coefficients for the matrix on the state-transition equation,
	 * i.e. \hat W = \sum f_c/W_ss * c_ss * c
	 */


	int l,d,status,ll;
	int tld_i,sld_i,gld_i,Wld_i,Wl0_i;
	int Nl	= 2*(Noccs+1);
	int JJ1	= Noccs*(Noccs+1);
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Nl;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*Nl;
	sld_i	= tld_i + Noccs*Nl;

	int ss_tld_i, ss_sld_i, ss_x_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;
	ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Nl;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
	ss_tld_i	= ss_gld_i + Noccs*Nl;//note, Noccs*Nl = 2*JJ1
	ss_sld_i	= ss_tld_i + Noccs*Nl;//note, Noccs*Nl = 2*JJ1

	status =0;
	double Z = xx->data[0];


	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double findrtl = 0.0;
		for(d=0;d<Noccs;d++)	findrtl += ss->data[ss_gld_i+l*Noccs+d]*pmatch(ss->data[ss_tld_i+l*Noccs+d]);
		double findrtle	= 0.0;
		for(d=0;d<Noccs;d++)	findrtle += ss->data[ss_gld_i+JJ1+l*Noccs+d]*pmatch(ss->data[ss_tld_i+JJ1+l*Noccs+d]);
		/*double gdenom = 0.0;
		for(d=0;d<Noccs;d++){
			double nud = l ==d+1? 0.0 : nu;
			double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			double pld 	= pmatch(ss->data[ss_tld_i+l*Noccs+d]);
			double post = pld > 0.0? -kappa*ss->data[ss_tld_i+l*Noccs+d]/pld: 0.0;
			gdenom +=exp(sig_psi*pld*
					(chi[l][d]-bl+nud+ post + beta*contval));
		}*/
		// Wl0
		double Wl0_ss 	= ss->data[ss_Wl0_i + l];
		double Wl0e_ss 	= ss->data[ss_Wl0_i + l+Noccs+1];
		gsl_matrix_set(Dst_tp1,Wl0_i+l,Wl0_i+l, beta*(1.0-findrtl)*(1.0-1.0/bdur) );
		double dWl0dWl0e = beta*(1.0-findrtl)*1.0/bdur;
		gsl_matrix_set(Dst_tp1,Wl0_i+l,Wl0_i+l +Noccs+1, dWl0dWl0e*Wl0e_ss/Wl0_ss );
		gsl_matrix_set(Dst_tp1,Wl0_i+l +Noccs+1,Wl0_i+l +Noccs+1, beta*(1.0-findrtle) );

		gsl_matrix_set(Dst,Wl0_i+l,Wl0_i+l,1.0);
		gsl_matrix_set(Dst,Wl0_i+l+Noccs+1,Wl0_i+l+Noccs+1,1.0);
		for(d=0;d<Noccs;d++){
			double Wld_ss = ss->data[ss_Wld_i + l*Noccs+d];
			double zd = xx->data[d+Notz];
			// Wld
			gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,1.0);
			if(ss->data[ss_tld_i + l*Noccs+d]>0.0 || ss->data[ss_tld_i + l*Noccs+d+JJ1]>0.0){
				if(d+1 ==l){
					gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d, Wld_i+l*Noccs+d, (1.0-ss->data[ss_sld_i + l*Noccs+d])*beta);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l, - (ss->data[ss_sld_i + l*Noccs+d])
						*ss->data[ss_Wl0_i+l]/Wld_ss
						);
					double cont_dd =chi[l][d]*(1.0+zd) -bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
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
					double cont_ld =chi[l][d]*(1.0+zd) - bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
					double dWds =-(1.0-tau)*(cont_ld - ss->data[ss_Wl0_i+0]);
					gsl_matrix_set(Dco,Wld_i + l*Noccs+d, sld_i+l*Noccs+d,dWds
							/Wld_ss*ss->data[ss_sld_i+l*Noccs+d]
							);
				}
			}// tss>0 market is open


			// Wl0
			for(ll=0;ll<2;ll++){
				// ss_ret = -nu - kappa/theta^ld/p^ld + mu^ld - mu^l0
				double Wl0_ss	= ss->data[ss_Wl0_i+l + ll*(Noccs+1)];
				double pld		= pmatch(ss->data[ss_tld_i+l*Noccs+d + JJ1*ll]);
				double post 	=  - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld;
				double ss_ret 	= d+1==l ? 0.0 : -nu;
				ss_ret += chi[l][d]*(1.0+zd) - bl*(1.0-(double)ll) - privn*((double)ll);
				double contval = beta*(ss->data[ss_Wld_i+l*Noccs+d] -(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] -
						1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1]);
				ss_ret += contval;
				ss_ret *= (1.0-fm_shr);
				ss_ret += bl*(1.0-(double)ll) + privn*((double)ll) + beta*(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] +
						beta*1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1];
				double u_ret = bl*(1.0-(double)ll) + privn*((double)ll) + beta*(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)]+
						beta*1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1];

				double dtld = ss->data[ss_gld_i+l*Noccs+d+ll*JJ1]*dpmatch(ss->data[ss_tld_i+l*Noccs+d+ll*JJ1])*(
								ss_ret
								//- kappa*pow(ss->data[ss_tld_i+l*Noccs+d],phi)/(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi))
								- u_ret);

				if(gsl_finite(dtld)==1 && pld>0.0)
					gsl_matrix_set(Dco,Wl0_i+l+ll*(Noccs+1),tld_i+l*Noccs+d+ll*JJ1,dtld
							*ss->data[ss_tld_i+l*Noccs+d+ll*JJ1]/Wl0_ss
							);
				else
					gsl_matrix_set(Dco,Wl0_i+l+ll*(Noccs+1),tld_i+l*Noccs+d+ll*JJ1,0.0);

				if(pld>0.0 && gsl_finite(pld*ss_ret)==1)
					gsl_matrix_set(Dco,Wl0_i+l+ll*(Noccs+1),gld_i+l*Noccs+d+ll*JJ1,pld*(ss_ret - u_ret)
						*ss->data[ss_gld_i+l*Noccs+d+ll*JJ1]/Wl0_ss
					);
				else
					gsl_matrix_set(Dco,Wl0_i+l+ll*(Noccs+1),gld_i+l*Noccs+d+ll*JJ1,0.0);

				double disc_cont = (1.0-fm_shr)*beta*pld*ss->data[ss_gld_i+l*Noccs+d + JJ1*ll];
				if(pld>0.0 && gsl_finite(disc_cont)==1)
					gsl_matrix_set(Dst_tp1,Wl0_i+l+ll*(Noccs+1),Wld_i+l*Noccs+d,disc_cont
							/Wl0_ss*Wld_ss
						);
				else
					gsl_matrix_set(Dst_tp1,Wl0_i+l+ll*(Noccs+1),Wld_i+l*Noccs+d,0.0);
			}
		}
	}
	return status;
}


int sys_co_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	int l,d,dd,status,ll;
	int tld_i,gld_i,sld_i,Wld_i,Wl0_i;//x_i,
	double Z,zd;
	int JJ1 = Noccs*(Noccs+1);
	int Nl	= 2*(Noccs+1);
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Nl;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Nl*Noccs;
	sld_i	= tld_i + Nl*Noccs;

	int ss_tld_i,ss_gld_i, ss_sld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i

	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Nl;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*Nl;
	ss_sld_i	= ss_tld_i + Noccs*Nl;

	gsl_vector * ret_d = gsl_vector_calloc(Noccs);
	gsl_vector * pXret_d = gsl_vector_calloc(Noccs);
	gsl_vector * pld_d = gsl_vector_calloc(Noccs);
	double * Vp = malloc((Noccs*2+2)*sizeof(double));
	status =0;
	Z = xx->data[0];
	gsl_integration_workspace * dgwksp = gsl_integration_workspace_alloc (1000);

	// 1st order derivatives
	for(l=0;l<Noccs+1;l++){

		double bl = l>0 ? b[1]:b[0];
		for(ll=0;ll<2;ll++){

			double Wl0_ss = ss->data[ss_Wl0_i+l +(Noccs+1)*ll];
			double findrtl = 0.0;
			for(d=0;d<Noccs;d++)
				findrtl += pmatch(ss->data[ss_tld_i+l*Noccs+d + ll*JJ1])*ss->data[ss_gld_i+l*Noccs+d + ll*JJ1];
			double gdenom = 0.0;
			for(d=0;d<Noccs;d++){
				double nud = l ==d+1? 0.0 : nu;
				zd = xx->data[d+Notz];
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1) ]-
						(1.0/bdur)*ss->data[ss_Wl0_i+l + Noccs+1 ]);
				//contval = contval <0.0 ? 0.0 : contval;
				double pld = pmatch(ss->data[ss_tld_i+l*Noccs+d + JJ1*ll]);
				ret_d->data[d] = (1.0-fm_shr)*(chi[l][d]*(1.0+zd)-bl*(1.0-(double)ll)- privn*((double)ll) +nud+beta*contval);
				//		+ bl*(1.0-(double)ll) + privn*((double)ll)+ beta*( (1.0-1.0/bdur)*ss->data[ss_Wl0_i+l +ll*(Noccs+1)]+(1.0/bdur)*ss->data[ss_Wl0_i+l +Noccs+1]);
				gsl_vector_set(pXret_d,d,gsl_vector_get(ret_d,d)*pld);
				gsl_vector_set(pld_d,d,pld);
				gdenom += exp(sig_psi*pXret_d->data[d]);
			}

			// set up Vp for integral to find derivatives of g
			for(d=0;d<Noccs;d++){
				Vp[d+2] = gsl_vector_get(pXret_d,d);
				Vp[d+2+Noccs] = sig_psi*gsl_vector_get(pld_d,d);
			}

			for(d=0;d<Noccs;d++){
				zd = xx->data[d+Notz];
				double nud = l==d+1 ? 0: nu;
				double Wld_ss = ss->data[ss_Wld_i +l*Noccs+d];
			// tld

				int tld_ldll = tld_i + l*Noccs+d + JJ1*ll;
				int ss_tld_ldll = ss_tld_i + l*Noccs+d + JJ1*ll;

				double tld_ss = ss->data[ss_tld_ldll];
				gsl_matrix_set(Dco,tld_ldll,tld_ldll, 1.0);
				if(tld_ss>0.0){
					double W_l0 = (1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] + 1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1];
					double surp = chi[l][d] - nud - bl*(1.0-(double)ll) -privn*((double)ll) + beta*(ss->data[ss_Wld_i+l*Noccs+d]-W_l0);
					//dt/dWld
					double qhere = kappa/(fm_shr*surp);
					double dtld = -beta*kappa/fm_shr*pow(surp,-2)*dinvq(qhere);
					// DO I NEED THIS SAFETY?
					dtld = surp<=0 ? 0.0 : dtld;

					if(gsl_finite(dtld) && surp>0.0)
						gsl_matrix_set(Dst_tp1,tld_ldll,Wld_i + l*Noccs+d, dtld
								*Wld_ss/tld_ss
								);
					else
						gsl_matrix_set(Dst_tp1,tld_ldll,Wld_i + l*Noccs+d, 0.0);
					//dt/dWl0
					double dtld_W_l0 = -dtld*(1.0 - 1.0/bdur*(1.0-(double)ll));
					if(gsl_finite(dtld) && surp>0.0)
						gsl_matrix_set(Dst_tp1,tld_ldll,Wl0_i+l + ll*(Noccs+1),dtld_W_l0
								*Wl0_ss/tld_ss
								);
					else
						gsl_matrix_set(Dst_tp1,tld_ldll,Wl0_i+l + ll*(Noccs+1),0.0);
					//dt/dWl0e
					if(ll==0){
						double dtld_W_l0e = -dtld*1.0/bdur;
						if(gsl_finite(dtld)&& surp>0.0)
						gsl_matrix_set(Dst_tp1,tld_ldll,Wl0_i+l + Noccs+1,dtld_W_l0e
							*ss->data[ss_Wld_i+l* Noccs+1]/tld_ss
							);
						else
							gsl_matrix_set(Dst_tp1,tld_ldll,Wl0_i+l + Noccs+1,0.0);
					}
				}//tld_ss >0
				else{
					gsl_matrix_set(Dst_tp1,tld_ldll,Wl0_i + l,0.0);
					gsl_matrix_set(Dst_tp1,tld_ldll,Wld_i + l*Noccs+d, 0.0);
				}

				// gld
				double gld_ss = ss->data[ss_gld_i+l*Noccs+d + ll*JJ1];
				gsl_matrix_set(Dco,gld_i+l*Noccs+d + ll*JJ1,gld_i+l*Noccs+d + ll*JJ1,1.0);

				if(gld_ss>0 && tld_ss>0){

					double pld		= pmatch(tld_ss);//effic*tld_ss/pow(1.0+pow(tld_ss,phi),1.0/phi);
					double post		= pld>0 ? - kappa*tld_ss/pld : 0.0;
					double contval = ss->data[ss_Wld_i+l*Noccs+d]-(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l+ll*(Noccs+1)]-
							1.0/bdur*ss->data[ss_Wl0_i+l+Noccs+1];

					double arr_d 	= (1.0-fm_shr)*(chi[l][d]*(1.0+zd) -bl*(1.0-(double)ll) -privn*((double)ll) -nud + beta*contval)
										+ bl*(1.0-(double)ll) -privn*((double)ll) + (1.0-1.0/bdur)*ss->data[ss_Wl0_i+l+ll*(Noccs+1)]+
										1.0/bdur*ss->data[ss_Wl0_i+l+Noccs+1];

					//double ret_d 	= pld*arr_d;
					// homosked errors: double dgdWld = beta*sig_psi*pld*(1.0-fm_shr)*gld_ss*(1.0-gld_ss);
					
					double dgdret[Noccs],dgdreterr;
					gsl_function dgprob;
					dgprob.function = dhetero_ev;
					dgprob.params = Vp;
					Vp[0] = (double) d;
					for(dd=0;dd<Noccs;dd++){
						dgdret[dd]  =0.;
						double pldd	= pmatch(ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]);
						double dpldd	= dpmatch(ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]);
						if(dd!=d){
						if(ss->data[ss_gld_i+l*Noccs+d]>1.0e-4 && ss->data[ss_gld_i+l*Noccs+dd+ll*JJ1]>1.0e-3
							&& ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]>1.0e-4){

							Vp[1] = (double) dd;
							double dgdret_dd;
							gsl_integration_qags(&dgprob,1.e-5,20,1e-6,1e-6,1000,dgwksp,&dgdret_dd,&dgdreterr);
							dgdret[dd] = dgdret_dd;
							
							//**********************
							// dgld / dWldd
							
							//		dret/dWldd		dg/dret
							double dgdWld = beta*pldd*(1.0-fm_shr)	*dgdret[dd];
							
							if(ret_d->data[dd]>0. && ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]>0.0)
								gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d + ll*JJ1,Wld_i+l*Noccs+dd,dgdWld
										*ss->data[ss_Wld_i+l*Noccs+dd]/gld_ss
										);
							else
								gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d + ll*JJ1,Wld_i+l*Noccs+dd,0.0);
							
							//**********************
							// dgld / dtldd
							//			dret/dtldd		dg/dret
							double dgdtldd =	dpldd*ret_d->data[dd]	*dgdret[dd];
							
							
							if(ret_d->data[dd]>0.0 && ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]>0.0)
								gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+dd+ll*JJ1,-1.0*dgdtldd
									*ss->data[ss_tld_i +l*Noccs+dd+ll*JJ1]/gld_ss
									);
							else
								gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+dd,0.0);
								

						}
						else{
							gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d + ll*JJ1,Wld_i+l*Noccs+dd,0.0);
							gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+dd,0.0);
						}//if gld>0 & pld>0
						}//if dd!=d
					}//for dd=1:J
					
					double dgdret_d = 0.;
					// computes the derivative w.r.t change in own value (see Bhat 1995 for derivation)
					for(dd = 0;dd<Noccs;dd++)  dgdret_d += -dgdret[dd];

					
					//**********************
					// dgld / dWld
					
					//		dret/dWld		dg/dret
					double dgdWld = beta*pld*(1.0-fm_shr)	*dgdret_d;
					
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d + ll*JJ1,Wld_i+l*Noccs+d,dgdWld
						*Wld_ss/gld_ss
						);
					//**********************
					// dgld / dWl0 = 0.

					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d + ll*JJ1,Wl0_i+l,0.0);

					//**********************
					// dgld / dtld
					//******************** NOTE: I have ignored the change to the variance of the shock that comes from dg/dt
					double dpdt = dpmatch(ss->data[ss_tld_i+l*Noccs+d + ll*JJ1]);
					double dgdt = (dpdt*ret_d->data[d])*dgdret_d;
					if(tld_ss>0.0 && ret_d->data[d]>0.0)
						gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+d+ll*JJ1,-1.0*dgdt
							/gld_ss*tld_ss
							);
					else
						gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+d+ll*JJ1,0.0);
				}
				else{
					gsl_matrix_set(Dco,gld_i+l*Noccs+d+ll*JJ1,tld_i+l*Noccs+d+ll*JJ1,0.0);
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d+ll*JJ1,Wld_i+l*Noccs+d+ll*JJ1,0.0);
				}//gld_ss>0

				// sld
				if(ll==0){//only need to do this once!
					gsl_matrix_set(Dco,sld_i +l*Noccs+d,sld_i +l*Noccs+d,1.0);
					double cut 	= ss->data[ss_Wl0_i+l]-(1.0+zd)*chi[l][d]-beta*ss->data[ss_Wld_i +l*Noccs+d];
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

			}// for d=1:Noccs+1
		}//for ll=1:2
	}// for l=1:Noccs+1
	free(Vp);
	gsl_vector_free(ret_d);
	gsl_vector_free(pXret_d);
	gsl_vector_free(pld_d);
	gsl_integration_workspace_free(dgwksp);
	return status;
}


int sys_ex_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco){
// derivatives for exog
	int tld_i,gld_i,sld_i,	Wld_i,Wl0_i;//x_i,
	int l,d,dd,status,ll;

	int JJ1 = Noccs*(Noccs+1);
	int Nl	= 2*(Noccs+1);
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Nl;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Nl*Noccs;
	sld_i	= tld_i + Nl*Noccs;

	int ss_tld_i,ss_gld_i, ss_sld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i

	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Nl;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*Nl;
	ss_sld_i	= ss_tld_i + Noccs*Nl;

	status =0;
	double * Vp = malloc((Noccs*2+2)*sizeof(double));
	double * ret_d = malloc(Noccs*sizeof(double));
	double * pld = malloc(Noccs*sizeof(double));
	gsl_integration_workspace * dgwksp = gsl_integration_workspace_alloc (1000);

	
	for (l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		for(ll=0;ll<2;ll++){
			
			double gdenom = 0.0;
			for(d=0;d<Noccs;d++){
				int ss_tldldll = ss_tld_i+l*Noccs+d + JJ1*ll;
				pld[d]	= pmatch(ss->data[ss_tldldll]);
				double post	= pld[d]>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d + JJ1*ll]/pld[d] : 0.0;
				double nud 	= l ==d+1? 0.0 : nu;
				ret_d[d]= (1.0-fm_shr)*(chi[l][d]-bl*(1.0-(double)ll) -privn*(double)ll- nud +
						beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]));
				//					+bl*(1.0-(double)ll) + privn*((double)ll)+ss->data[ss_Wl0_i+l];
				ret_d[d]	= ret_d[d] <0.0 ? 0.0 : ret_d[d];
				gdenom +=exp(sig_psi*pld[d]*ret_d[d]);
			}
			double dWl0dZ =0.0;
			for(d=0;d<Noccs;d++){
				Vp[d+2] = pld[d]*ret_d[d];
				Vp[d+2+Noccs] = sig_psi*pld[d];
			}
				
			
			
			for(d=0;d<Noccs;d++)
				dWl0dZ += ss->data[ss_gld_i+l*Noccs+d]*chi[l][d]*(1.0-fm_shr)*pmatch(ss->data[ss_tld_i+l*Noccs+d+ JJ1*ll]);
		//	gsl_matrix_set(Dst,Wl0_i+l,0,dWl0dZ
		//			/ss->data[ss_Wl0_i+l]
		//			);


			for(d=0;d<Noccs;d++){

				double nud = l ==d+1? 0.0 : nu;
				double post	= pld[d]>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d + ll*JJ1]/pld[d] : 0.0;
									// dWl0/dz
				double dWdz =ss->data[ss_gld_i+l*Noccs+d + ll*JJ1]*(1.0-fm_shr)*chi[l][d]*pld[d];

				gsl_matrix_set(Dst,Wl0_i+l+ll*(Noccs+1),d+Notz,dWdz
						/ss->data[ss_Wl0_i+l + ll*(Noccs+1)]
						);

				if(ss->data[ss_tld_i+l*Noccs+d+ ll*JJ1]>0.0){
					double surp = (chi[l][d]-nud- bl*(1.0-(double)ll) - bl*(1.0-(double)ll)
							+beta*(ss->data[ss_Wld_i+l*Noccs+d]-(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] -
							1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1]));
					//surp= surp<0 ? 0.0 : surp;
					double dqdz = -chi[l][d]*kappa/(fm_shr*surp*surp);
					double dtdz =dqdz*dinvq(kappa/(fm_shr*surp));
					if( surp>0 && gsl_finite(dtdz)){
		//				gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,dtdz   // dtheta/dZ
		//					/ss->data[ss_tld_i+l*Noccs+d]
		//					);
						gsl_matrix_set(Dco,tld_i + l*Noccs+d+ ll*JJ1,d+Notz,dtdz// dtheta/dz
							/ss->data[ss_tld_i+l*Noccs+d+ ll*JJ1]
							);
					}
					else{
						gsl_matrix_set(Dco,tld_i + l*Noccs+d+ ll*JJ1,d+Notz,0.0);
					}
				}//tld_ss >0
				else{
					gsl_matrix_set(Dco,tld_i + l*Noccs+d + ll*JJ1,d+Notz,0.0);
				}

				//Wld
				if((ss->data[ss_tld_i + l*Noccs+d]>0.0 || ss->data[ss_tld_i + l*Noccs+d+JJ1]>0.0) && ll==0){
					if(l!=d+1){
						// dWld/dZ
						double dWlddZ =(1.0- ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+
							(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d];
						double dWlddzd =((1.0-ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+
							(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d]) ;
						gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,dWlddzd
							/ss->data[ss_Wld_i+l*Noccs+d]
							);
					}
					else{

						gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,(1.0-ss->data[ss_sld_i + l*Noccs+d])*chi[l][d]
								// dWld/dzd
								/ss->data[ss_Wld_i+l*Noccs+d]
								);
					}
				}// Wld, tld_ss >0

				// gld
				double dgdret[Noccs];
				if(homosk_psi!=1){
					gsl_function dgprob;
					dgprob.function = dhetero_ev;
					dgprob.params = Vp;
					if(ss->data[ss_gld_i+l*Noccs+d + ll*JJ1]>0.0 && ss->data[ss_tld_i+l*Noccs+d+ ll*JJ1]>0.0){
						double contval  = (ss->data[ss_Wld_i+l*Noccs+d]-(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] -
										   1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1]);
						for(dd=0;dd<Noccs;dd++){
							dgdret[dd]  =0.;
							double pldd	= pmatch(ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]);
							double dpldd	= dpmatch(ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]);
							if(dd!=d){
								if(ss->data[ss_gld_i+l*Noccs+d]>1.e-4 && ss->data[ss_gld_i+l*Noccs+dd+ll*JJ1]>1.e-3
								   && ss->data[ss_tld_i+l*Noccs+dd+ll*JJ1]>1.e-4){
									Vp[1] = (double) dd;

									double dgdret_dd, dgdreterr;
									gsl_integration_qags(&dgprob,1.e-5,20,1e-6,1e-6,1000,dgwksp,&dgdret_dd,&dgdreterr);
									dgdret[dd] = dgdret_dd;

									double dgdzdd = pldd*(1.-fm_shr)*chi[l][dd]*dgdret_dd;
									gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,dd+Notz,dgdzdd
																					   /ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]
									);

								}else
									gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,dd+Notz,0.);
							}
						}
						double dgdzd = 0.;
						for(dd=0;dd<Noccs;dd++) dgdzd += -dgdret[dd];
						dgdzd *= pld[d]*(1.-fm_shr)*chi[l][d];
						gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,d+Notz,dgdzd
																		  /ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]
						);
					}
					else{
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,0.0);
						for(dd=0;dd<Noccs;dd++) gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,dd+Notz,0.);
					}

				}

				if(homosk_psi==1){

					if(ss->data[ss_gld_i+l*Noccs+d + ll*JJ1]>0.0 && ss->data[ss_tld_i+l*Noccs+d+ ll*JJ1]>0.0){
						// homog dgdzd:
						double dgdzd =pld[d]*(1.0-fm_shr)*chi[l][d]*ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]*(1.0-ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]);
						gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,d+Notz,dgdzd
																		  /ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]
						);
						// homog dgdzdd
						for(dd=0;dd<Noccs;dd++){
							if(dd!=d){
								if(ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]>0.0 && ss->data[ss_tld_i+l*Noccs+dd+ ll*JJ1]>0.0){
									double nudd 	= l==dd+1 ? 0: nu;
									double p_dd		= pmatch(ss->data[ss_tld_i+l*Noccs+dd+ ll*JJ1]);
									double postdd	= p_dd > 0 ? - kappa*ss->data[ss_tld_i+l*Noccs+dd+ ll*JJ1]/p_dd : 0.0;
									double contval	= (ss->data[ss_Wld_i+l*Noccs+dd]-(1.0-1.0/bdur)*ss->data[ss_Wl0_i+l + ll*(Noccs+1)] -
														 1.0/bdur*ss->data[ss_Wl0_i+l + Noccs+1]);
									//contval = contval<0.0 ? 0.0 : contval;
									//double ret_dd 	= (1.0-fm_shr)*(chi[l][dd]-bl - nudd +beta*contval)
									//					+bl + ss->data[ss_Wl0_i+l];

									double dgdzdd =-1.0*sig_psi*(1.0-fm_shr)*chi[l][dd]*
												   ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]*ss->data[ss_gld_i+l*Noccs+dd+ ll*JJ1];
									gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,dd+Notz,dgdzdd
																					   /ss->data[ss_gld_i+l*Noccs+d+ ll*JJ1]
									);
								}
								else
									gsl_matrix_set(Dco,gld_i+l*Noccs+d + ll*JJ1,dd+Notz,0.0);
							}
						}
					}
					else{
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,0.0);
						for(dd=0;dd<Noccs;dd++) gsl_matrix_set(Dco,gld_i+l*Noccs+d+ ll*JJ1,dd+Notz,0.);
					}

				}


				// sld
				double cut 	= ss->data[ss_Wl0_i+l]-chi[l][d]-beta*ss->data[ss_Wld_i +l*Noccs+d];
				cut = cut > 0.0 ? 0.0 : cut;
				double dF 	= scale_s*shape_s*exp(shape_s*cut);
				// ds/dZ
				double dsdZ = -dF;
				if(ss->data[ss_tld_i+l*Noccs+d]>0.0){
			//		gsl_matrix_set(Dco,sld_i+l*Noccs+d,0,dsdZ
			//				/ss->data[ss_sld_i+l*Noccs+d]
			//			);
					// ds/dzd
					gsl_matrix_set(Dco,sld_i+l*Noccs+d,Notz+d,dsdZ
							/ss->data[ss_sld_i+l*Noccs+d]
						);
				}
			}//for d=1:Noccs
		}//for ll=0:1
	}//for l=1:Noccs+1


	free(Vp);
	free(ret_d);
	free(pld);
	gsl_integration_workspace_free(dgwksp);
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

int sol_zproc(struct st_wr *st, gsl_vector * ss, gsl_matrix * xss, struct shock_mats * mats0){
	// this function simulates a series and then computes productivity histories on it
	// from these productivity histories, it estimates parameters of the process and then updates the decision rules
	// this iteratively is estimating the productivity proccess, because each time it's called it updates the sequence slightly
	// it will over-write mats0 with the current best estimate for the process coefficients

	// Does not use mats space of st, only mats0

	//run through a few draws without setting anything
	struct sys_sol * sol = st->sol;
	struct sys_coef * sys = st->sys;
	struct aux_coef * simdat = st->sim;
	int init_T = 200, l,d,ll,zbad=0;
	int Nl = 2*(Noccs+1);
	int di,fi,estiter,status =0,maxestiter=200;
	if(dbg_iters == 1) maxestiter = 20;
	double ** pld	= malloc(sizeof(double*)*Nl);
	for(l = 0;l<Nl;l++)
		pld[l] = malloc(sizeof(double)*Noccs);
	double ** pd_hist = malloc(sizeof(double)*simT);
	for(di=0;di<simT;di++)
		pd_hist[di] = malloc(sizeof(double)*Noccs);

	gsl_matrix * x		= gsl_matrix_alloc(Noccs+1,Noccs+2);
	gsl_matrix * xp		= gsl_matrix_alloc(Noccs+1,Noccs+2);
	gsl_matrix * xhist  = gsl_matrix_alloc(simT, (Noccs+1)*(Noccs+2));
	gsl_matrix * xd_hist= gsl_matrix_alloc(simT, (Noccs+2));


	gsl_matrix * zhist	= gsl_matrix_alloc(simT,Noccs);
	gsl_matrix * zhist0 = gsl_matrix_alloc(simT,Noccs);
	gsl_vector * Zhist	= gsl_vector_alloc(simT);

	gsl_vector * Zz		= gsl_vector_calloc(Nx);
	gsl_vector * Zzl	= gsl_vector_calloc(Nx);
	gsl_matrix * Zz_hist= gsl_matrix_calloc(simT,Nx);


	gsl_vector * shocks	= gsl_vector_alloc(Nx);

	// finding rate data
	gsl_vector * fnd_d	= gsl_vector_calloc(Noccs);
	gsl_matrix * fnd_d_hist  = gsl_matrix_calloc(simT,Noccs);
	gsl_vector * x_u	= gsl_vector_calloc(Nl);
	gsl_matrix * uf_hist= gsl_matrix_calloc(simT,4);

	gsl_matrix_memcpy(x,xss);
	sol->tld = gsl_matrix_calloc(Nl,Noccs);
	sol->gld = gsl_matrix_calloc(Nl,Noccs);
	sol->sld = gsl_matrix_calloc(Nl,Noccs);
	sol->ald = gsl_matrix_calloc(Nl,Noccs);

	gsl_vector * zdist_hist = gsl_vector_alloc(maxestiter);

	struct shock_mats mats1;
	struct sys_sol * sol_hist;
	sol_hist = malloc(sizeof(struct sys_sol)*simT);

	if(fix_fac!=1){
		mats1.Gamma	= gsl_matrix_alloc(Nfac,Nfac*Nglag);
		mats1.var_eta	= gsl_matrix_alloc(Nfac,Nfac);
	}
	else{
		mats1.facs = gsl_matrix_alloc(simT-1,Nfac);
		mats1.Gamma = GammaCoef;
		mats1.var_eta = var_eta;
	}
	mats1.Lambda	= gsl_matrix_alloc(Noccs,Nfac+1);
	mats1.var_zeta	= gsl_matrix_alloc(Noccs,Noccs);

	// initialize Zz_hist with shocks that are actually the finding rate deviations
	gsl_matrix_set_zero(Zz_hist);
	for(di=0;di<simT;di++){
		for(d=0;d<Noccs;d++)
 			gsl_matrix_set(Zz_hist,di,d+Notz,gsl_matrix_get(simdat->fd_hist_dat,di,d));
	//	gsl_matrix_set(Zz_hist,di,0,gsl_vector_get(fd_mats->ag,di));
	//	for(fi=0;fi<Nfac;fi++){
	//		gsl_matrix_set(Zz_hist,di,fi+1,gsl_matrix_get(fd_mats->facs,di,fi));
	//	}
	}
	// re-norm so that the mean of e(z) is always 1:
	double Zmean_esti = 0.;
	double zzmean_esti[Noccs];
	for(d=0;d<Noccs;d++) zzmean_esti[d] = 0.;
	for(di=0;di<simT;di++){
		Zmean_esti += gsl_matrix_get(Zz_hist,di,0)/(double)simT;
		for(d=0;d<Noccs;d++)
			zzmean_esti[d] += gsl_matrix_get(Zz_hist,di,Notz+d)/(double)simT;
	}
	double Zvar_esti = 0.;
	double zzvar_esti[Noccs];
	for(d=0;d<Noccs;d++) zzvar_esti[d] = 0.;
	for(di=0;di<simT;di++){
		Zvar_esti += pow(gsl_matrix_get(Zz_hist,di,0) - Zmean_esti,2)/(double)simT;
		for(d=0;d<Noccs;d++)
			zzvar_esti[d] += pow(gsl_matrix_get(Zz_hist,di,Notz+d)-zzvar_esti[d],2)/(double)simT;
	}
	for(di=0;di<simT;di++){
		gsl_matrix_set(Zz_hist,di,0,gsl_matrix_get(Zz_hist,di,0)-Zmean_esti-Zvar_esti/2 );
		for(d=0;d<Noccs;d++)
			gsl_matrix_set(Zz_hist,di,Notz+d,gsl_matrix_get(Zz_hist,di,Notz+d)-zzmean_esti[d]-zzvar_esti[d]/2);
	}

	if(printlev>=2)
		printmat("Zz_hist0.csv",Zz_hist);

	// compute the standard deviation of finding for each occupation
	double fd_sim_sd[Noccs];
	double fd_sim_mean[Noccs];
	double fd_dat_sd[Noccs];
	double fd_datsim_sd[Noccs];
	for(d=0;d<Noccs;d++){
		double fdhere_hist[simT];
		for(di=0;di<simT;di++) fdhere_hist[di] = gsl_matrix_get(simdat->fd_hist_dat,di,d);
		fd_dat_sd[d] = gsl_stats_sd(fdhere_hist,1,simT);
	}

	if(printlev >= 3){ // punch out one set of policies
		gsl_vector_set_zero(Zz);
		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);
		status += xprime(xp,sol->ald,ss,sys,sol,x,Zz);
		printmat("tld_z0.csv",sol->tld);
		printmat("gld_z0.csv",sol->gld);
		printmat("sld_z0.csv",sol->sld);
		printmat("ald_z0.csv",sol->ald);
		printmat("xp_z0.csv",xp);
	}

	int printlev_old 	= printlev;
	/////////////////////////////////////////////////////////////
	printlev = printlev_old>1 ? 1: printlev_old;
	int verbose_old 	= verbose;
	verbose  = verbose_old > 1 ? 1 : verbose_old;


	for(estiter=0;estiter<maxestiter;estiter++){
		status =0;

		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,0);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0, Zz);
		double s_fnd = 0.;
		double s_urt = 0.;
		for(di=0;di<init_T;di++){

			status += theta(sol->tld, ss, sys, sol, Zz);
			status += gpol(sol->gld, ss, sys, sol, Zz);
			status += spol(sol->sld, ss, sys, sol, Zz);
			status += xprime(xp,sol->ald,ss,sys,sol,x,Zz);

			gsl_matrix_memcpy(x,xp);

			gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

			gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
			gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
			gsl_vector_add(Zzl,shocks);
			gsl_vector_memcpy(Zz,Zzl);
		}
		gsl_matrix_set_zero(xhist);

		// make x0 even distribution, just for a check.
		if(sym_occs == 1){
			for(l=0;l<Noccs+1;l++){
				for(d=0;d<Noccs+2;d++)
					gsl_matrix_set(x,l,d,1./(double)( (Noccs+1)*(Noccs+2) ));
			}
		}

		for(di=0;di<simT;di++){

			gsl_vector_view Zz_hist_di = gsl_matrix_row(Zz_hist,di);
			gsl_vector_memcpy(Zz,&Zz_hist_di.vector);
		//	gsl_vector_set(Zz,1,1);
		//	gsl_vector_set_zero(Zz);
			status += theta(sol->tld, ss, sys, sol, Zz);
			status += gpol(sol->gld, ss, sys, sol, Zz);
			status += spol(sol->sld, ss, sys, sol, Zz);
			status += xprime(xp,sol->ald,ss,sys,sol,x,Zz);

			//store the solution
			status += sol_calloc(&sol_hist[di]);
			status += sol_memcpy(&sol_hist[di],sol);

			for(d=0;d<Noccs+2;d++){
				// store x vecotrized in rows
				gsl_matrix_set(xd_hist,di,d,0.);
				for(l=0;l<Noccs+1;l++){
					gsl_matrix_set(xhist,di,l*(Noccs+2)+d,gsl_matrix_get(x,l,d));
					gsl_matrix_set(xd_hist,di,d, gsl_matrix_get(x,l,d) + gsl_matrix_get(xd_hist,di,d));
				}
			}

			// calculate the unemployment rate:
			double urtp = 0.0;
			for(l=0;l<Noccs+1;l++)
				urtp += gsl_matrix_get(xp,l,0)+gsl_matrix_get(xp,l,Noccs+1);
			s_urt += urtp/((double) simT);
			gsl_matrix_set(uf_hist,di,1,urtp);


			for(l=0;l<Noccs+1;l++){
				for(ll=0;ll<2;ll++){
					if(urtp>0.0)
						gsl_vector_set(x_u,l+ll*(Noccs+1),gsl_matrix_get(xp,l,ll*(Noccs+1))/urtp);
					else
						gsl_vector_set(x_u,l,0.0);
				}
			}

			// compute finding rates
			for(l=0;l<Nl;l++){
				for(d=0;d<Noccs;d++)
					pld[l][d] = pmatch(gsl_matrix_get(sol->tld,l,d) );
			}
			double d_fnd =0.0;
			double x_d[Noccs];
			double d_x = 0;
			for(d=0;d<Noccs;d++){
				int goodl = 0;
				fnd_d->data[d] = 0.0;
				x_d[d] = 0.;
				for(l=0;l<Noccs+1;l++){
					for(ll=0;ll<2;ll++){
						if(gsl_matrix_get(sol->ald,l+ll*(Noccs+1),d) > 0. && pld[l+ll*(Noccs+1)][d] > 0.){
							fnd_d->data[d] += gsl_matrix_get(sol->ald,l+ll*(Noccs+1),d)*pld[l+ll*(Noccs+1)][d];
							x_d[d] += gsl_matrix_get(sol->ald,l+ll*(Noccs+1),d);
							goodl ++;
						}
					}
				}
				if(goodl>0){
					d_fnd += fnd_d->data[d];
					d_x   += x_d[d];
					fnd_d->data[d] /= x_d[d];
					gsl_matrix_set(fnd_d_hist,di,d,fnd_d->data[d]);
				}
			}
			d_fnd /=d_x;
			d_fnd = gsl_finite(d_fnd)==1? d_fnd : s_fnd*( (double)simT )/( (double) di);
			// this will norm the finding rate for estimation below
			s_fnd += d_fnd/(double)simT;

			for(d=0;d<Noccs;d++){
				pd_hist[di][d] = 0.;
				for(l=0;l<Nl;l++)
					pd_hist[di][d] += pld[l][d];
			}

			// advance x on period, put xp into x
			gsl_matrix_memcpy(x,xp);

		}// end di = 1:simT
		int zz_edges = 0;
		/////////////////////////////////////////////////////////
		// Solve for the implied z at each date
		#pragma omp parallel for private(di,l,d)
		for(di=0;di<simT;di++){
			// make private objects so it can be parallelized
			double Zz_in[Nx];
			double zz_invth[Noccs];
			double fd_dat[Noccs];

			for(d=0;d<Noccs;d++) fd_dat[d] = gsl_matrix_get(simdat->fd_hist_dat,di,d) + s_fnd;
			//setup x:
			for(d=0;d<Noccs+2;d++){
				for(l=0;l<Noccs+1;l++)
						gsl_matrix_set(x,l,d,gsl_matrix_get(xhist,di,l*(Noccs+2)+d));
			}
			for(d=0;d<Nx;d++) Zz_in[d] = gsl_matrix_get(Zz_hist,di,d);

			// compute finding rates
			for(l=0;l<Nl;l++){
				for(d=0;d<Noccs;d++)
					pld[l][d] = pmatch(gsl_matrix_get(sol_hist[di].tld,l,d) );
			}

			for(d=0;d<Noccs;d++) zz_invth[d] = 0.;
			zz_edges += invtheta_z(zz_invth,fd_dat,ss,sys,&sol_hist[di],Zz_in);

			//store Z (weighted) average
			double xd[Noccs];
			for( d=0;d<Noccs;d++ ){
				xd[d] = 0.;
				for( l=0;l<Noccs+1;l++ ){
					if(gsl_matrix_get(x,l,d+1)>0 && gsl_finite(gsl_matrix_get(x,l,d+1)))
						xd[d] += gsl_matrix_get(x,l,d+1);
				}
			}
			double xd_sum =0.;
			for( d=0;d<Noccs;d++ ) xd_sum += xd[d];
			for( d=0;d<Noccs;d++ ) xd[d] /= xd_sum;
			double Z_fd = 0.;
			for(d=0;d<Noccs;d++) Z_fd +=   zz_invth[d]/(double)Noccs;//*xd[d];
			gsl_vector_set(Zhist,di,Z_fd);

			// store zz_invth
			for(d=0;d<Noccs;d++) gsl_matrix_set(zhist,di,d,zz_invth[d]);
			sol_free(&sol_hist[di]);
		}
		if(zz_edges/simT > Noccs/2){ // more than half the time we're at the edge
			if(verbose>=3) printf("hit the edge %d times\n",zz_edges);
		}
		// copy the history of implied shocks in to the feed-in shocks (with a weighting)
		double zlb = log(b[1]); // can't be so low that even experienced workers don't want to work there
		double zub = -zub_fac*zlb;
		for(di=0;di<simT;di++){
		//	if(gsl_finite(gsl_vector_get(Zhist,di))==1 && gsl_vector_get(Zhist,di) > zlb && gsl_vector_get(Zhist,di) <zub)
			if(gsl_finite(gsl_vector_get(Zhist,di))==1)		
		gsl_matrix_set(Zz_hist,di,0,zmt_upd*gsl_vector_get(Zhist,di)
											+ (1.-zmt_upd)*gsl_matrix_get(Zz_hist,di,0));
			else
				gsl_matrix_set(Zz_hist,di,0,gsl_matrix_get(Zz_hist,di,0));
			for(d=0;d<Noccs;d++){
				double xpd = 0.; double xd =0.;
				for(l=0;l<Noccs+1;l++)
						xd += gsl_matrix_get(xhist,di,l*(Noccs+2)+d+1);
				if(di<simT-1){
					for(l=0;l<Noccs+1;l++)
						xpd += gsl_matrix_get(xhist,di,l*(Noccs+2)+d+1);
				}
				//if ( (xd>0 && xpd>0) && //not zero in this occuption
				//	(gsl_finite(gsl_matrix_get(zhist,di,d))==1 && gsl_matrix_get(zhist,di,d) > zlb && gsl_matrix_get(zhist,di,d) <zub)) // finite z
				if(gsl_finite(gsl_matrix_get(zhist,di,d))==1)
					gsl_matrix_set(Zz_hist,di,d+Notz,zmt_upd*gsl_matrix_get(zhist,di,d)
												 +(1.-zmt_upd)*gsl_matrix_get(Zz_hist,di,d+Notz));
				else
					gsl_matrix_set(Zz_hist,di,d+Notz,gsl_matrix_get(Zz_hist,di,d+Notz));
			}
		}
		// re-norm so that the mean of e(z) is always 1:
		Zmean_esti = 0.;
		for(d=0;d<Noccs;d++) zzmean_esti[d] = 0.;
		for(di=0;di<simT;di++){
			Zmean_esti += gsl_matrix_get(Zz_hist,di,0)/(double)simT;
			for(d=0;d<Noccs;d++)
				zzmean_esti[d] += gsl_matrix_get(Zz_hist,di,d+Notz)/(double)simT;
		}
		for(d=0;d<Noccs;d++){
			if(gsl_finite(zzmean_esti[d]) != 1)
				zzmean_esti[d] = 0.;
		}
		double Zvar_esti = 0.;
		double zzvar_esti[Noccs];
		for(d=0;d<Noccs;d++) zzvar_esti[d] = 0.;
		for(di=0;di<simT;di++){
			Zvar_esti += pow(gsl_matrix_get(Zz_hist,di,0) - Zmean_esti,2)/(double)simT;
			for(d=0;d<Noccs;d++)
				zzvar_esti[d] += pow(gsl_matrix_get(Zz_hist,di,d+Notz)-zzvar_esti[d],2)/(double)simT;
		}
		for(d=0;d<Noccs;d++){
			if(gsl_finite(zzvar_esti[d]) != 1)
				zzvar_esti[d] = 0.;
		}

		for(di=0;di<simT;di++){
			gsl_matrix_set(Zz_hist,di,0,gsl_matrix_get(Zz_hist,di,0)-Zmean_esti-Zvar_esti/2 );
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(Zz_hist,di,Notz+d,gsl_matrix_get(Zz_hist,di,Notz+d)-zzmean_esti[d]-zzvar_esti[d]/2);
		}

		/////////////////////////////////////////////////////////
		// Estimate on new data and update!!
		// now have to estimate the process given zhist (idiosync prod) and Zhist (ag prod)
		gsl_vector_view Zhist_upd = gsl_matrix_column(Zz_hist,0);
		gsl_vector_memcpy(Zhist,&Zhist_upd.vector);
		for(d=0;d<Noccs;d++){
			gsl_vector_view zhist_upd = gsl_matrix_column(Zz_hist,d+Notz);
			gsl_vector_view zhist_d = gsl_matrix_column(zhist,d);
			gsl_vector_memcpy(&zhist_d.vector ,&zhist_upd.vector);
		}
		// this destroys zhist and Zhist in the process
		status += est_fac_pro(zhist,Zhist,&mats1);
		// make sure everything is still hunk dory
		if(gsl_finite(mats1.rhozz)!= 1)     status++;
		if(gsl_finite(mats1.rhoZ)!= 1)      status++;
		if(gsl_finite(mats1.sig_eps2)!= 1)  status++;
		status += isfinmat(mats1.var_zeta);


		// update process matrices
		if(status == 0){
			mats0->rhozz   = zmt_upd*mats1.rhozz + (1.-zmt_upd)*mats0->rhozz;
			mats0->rhoZ	   = zmt_upd*mats1.rhoZ + (1.-zmt_upd)*mats0->rhoZ;
			mats0->sig_eps2 = zmt_upd*mats1.sig_eps2 + (1.-zmt_upd)*mats0->sig_eps2;
			for(d=0;d<Noccs;d++) {
				for (l = 0; l < Nfac + 1; l++){
					gsl_matrix_set(mats0->Lambda, d, l,gsl_matrix_get(mats1.Lambda, d, l)*zmt_upd+
								   gsl_matrix_get(mats0->Lambda, d, l)*(1.-zmt_upd));
				}
			}
			for(d=0;d<Nfac;d++){
				for(l=0;l<Nfac+1;l++) {
					gsl_matrix_set(mats0->Gamma, d, l,gsl_matrix_get(mats1.Gamma, d, l)*zmt_upd +
							gsl_matrix_get(mats0->Gamma, d, l)*(1.-zmt_upd));
					gsl_matrix_set(mats0->var_eta, d, l,gsl_matrix_get(mats1.var_eta, d, l)*zmt_upd+
							gsl_matrix_get(mats0->var_eta, d, l)*(1.-zmt_upd));

				}
			}
			// resolve decision rules around the process
			// essentially this is updating the matrices
			status += sys_ex_set(sys->N,sys->S,mats0);
			status += sol_dyn(ss,sol, sys,1);
		}else{
			status++;
			zbad ++;
			printmat("zhist_bad.csv",zhist);
			printvec("Zhist_bad.csv",Zhist);
			if(zbad>10) break;
		}

		//if(printlev>=4 || (status >0 && printlev>=2) || (estiter%10 ==0 && printlev>=2) ){
			printf("rhozz, rhoZ,s_urt,e^Z, status = %f,%f,%f,%f,%d \n",mats0->rhozz,mats0->rhoZ,s_urt,exp(Zmean_esti),status);
			printmat("zhist_in.csv",zhist);
			printmat("xd_hist.csv",xd_hist);
			printmat("zhist_dist.csv",zhist0);
			printmat("Zz_hist.csv",Zz_hist);
			if(fix_fac!=1){
				printmat("GammaSol.csv",mats1.Gamma);
				printmat("var_etaSol.csv",mats1.var_eta);
			}
			printmat("LambdaSol.csv",mats1.Lambda);
			printmat("var_zetaSol.csv",mats1.var_zeta);

			double pd_sim_mean[Noccs];
			double pd_sim_sd[Noccs];
			for(d=0;d<Noccs;d++){
				fd_sim_sd[d] = 0;
				fd_sim_mean[d] = 0.;
				for(di=0;di<simT;di++) fd_sim_mean[d] += gsl_matrix_get(fnd_d_hist,di,d)/(double)simT;
				for(di=0;di<simT;di++) fd_sim_sd[d] += pow(gsl_matrix_get(fnd_d_hist,di,d) -fd_sim_mean[d],2)/(double)simT;
				fd_sim_sd[d] = pow(fd_sim_sd[d],0.5);
				// difference in standard deviation:
				fd_datsim_sd[d] = fd_sim_sd[d]/fd_dat_sd[d];
				pd_sim_sd[d] = 0;
				pd_sim_mean[d] = 0.;
				for(di=0;di<simT;di++) pd_sim_mean[d] += pd_hist[di][d]/(double)simT;
				for(di=0;di<simT;di++) pd_sim_sd[d] += pow(pd_hist[di][d] -pd_sim_mean[d],2)/(double)simT;
				pd_sim_sd[d] = pow(pd_sim_sd[d],0.5);
			}
			gsl_vector_view fd_datsim_sd_vec = gsl_vector_view_array(fd_datsim_sd,Noccs);
			printvec("fd_datsim_sd.csv",&fd_datsim_sd_vec.vector);
		//}
		double zdist_i = 0.0;
		gsl_matrix_sub(zhist0,zhist);
		int NTz = zhist0->size1*zhist0->size2;
		for(d=0;d<NTz;d++)
			zdist_i += pow(zhist0->data[d],2)/(double)NTz;
		zdist_i = sqrt(zdist_i);
		gsl_vector_set(zdist_hist,estiter,zdist_i);
		double zdist_av = 0;
		if(estiter>19)
			for(ll=estiter-19;ll<estiter+1;ll++) zdist_av += gsl_vector_get(zdist_hist,ll)/20.;
		else
			zdist_av =10.;
		double zdist_av_l = 0;
		if (estiter>40)
			for(ll=estiter-39;ll<estiter-20;ll++) zdist_av_l += gsl_vector_get(zdist_hist,ll)/20.;
		else
			zdist_av_l = zdist_av+10.;

		if(verbose_old>1){
			if(estiter==0) printf("stoch proc, iter: ");
			else printf(".");
		}

		// converged on the right z process
		if(zdist_i<5e-3 || (zdist_av >zdist_av_l+5e-3 && zdist_i<1e-2) ){
			printf("rhozz, rhoZ,s_urt,e^Z, status = %f,%f,%f,%f,%d \n",mats0->rhozz,mats0->rhoZ,s_urt,exp(Zmean_esti),status);
			break;
		}
		gsl_matrix_memcpy(zhist0,zhist);
	}// end estiter
	if(printlev_old>=2) {
		printvec("zdist_hist.csv", zdist_hist);
		printmat("Zz_hist.csv", Zz_hist);
	}
	if(verbose_old>1)
		printf("\n");
	verbose 	= verbose_old;
	printlev	= printlev_old;

	if(printlev>=2){
		printmat("GammaSol.csv",mats1.Gamma);
		printmat("LambdaSol.csv",mats1.Lambda);
		printmat("var_zetaSol.csv",mats1.var_zeta);
		printmat("var_etaSol.csv",mats1.var_eta);
	}

	gsl_matrix_memcpy(mats0->Zz_hist,Zz_hist);



	gsl_matrix_free(sol->sld);
	gsl_matrix_free(sol->gld);
	gsl_matrix_free(sol->tld);
	gsl_matrix_free(sol->ald);
	gsl_matrix_free(xp);gsl_matrix_free(x);
	gsl_vector_free(Zz);gsl_vector_free(Zzl);gsl_vector_free(shocks);


	free(sol_hist);
	gsl_matrix_free(xhist);gsl_matrix_free(xd_hist);
	gsl_matrix_free(zhist);gsl_matrix_free(zhist0);
	gsl_vector_free(Zhist);
	gsl_vector_free(zdist_hist);
	gsl_vector_free(fnd_d);
	gsl_vector_free(x_u);
	gsl_matrix_free(uf_hist);
	for(l=0;l<Nl;l++)
		free(pld[l]);
	free(pld);
	for(di=0;di<simT;di++)
		free(pd_hist[di]);
	free(pd_hist);
	if(fix_fac==1){
		gsl_matrix_free(mats1.facs);
	}
	else{
		gsl_matrix_free(mats1.Gamma);
		gsl_matrix_free(mats1.var_eta);
	}

	gsl_matrix_free(mats1.Lambda);
	gsl_matrix_free(mats1.var_zeta);


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

	int status,l,d,ll,di,si,Ndraw;
	int Nl = 2*(Noccs+1);
	int JJ1 = Noccs*(Noccs+1);
	double s_fnd,s_urt,s_wg,s_chng,m_Zz, s_sdlu,s_sdu,s_sdZz,s_elpt,s_sdsep,s_sep;
	int ss_gld_i, ss_Wld_i,ss_Wl0_i;
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Nl;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Noccs+1;
	gsl_vector * Es = gsl_vector_calloc(Ns);

	gsl_matrix * xp 	= gsl_matrix_alloc(Noccs+1,Noccs+2);
	gsl_matrix * x 		= gsl_matrix_calloc(Noccs+1,Noccs+2);
	//gsl_matrix * wld 	= gsl_matrix_calloc(Noccs+1,Noccs);
	gsl_vector * Zz 	= gsl_vector_calloc(Nx);
	gsl_vector * fnd_l 	= gsl_vector_calloc(Nl);
	gsl_vector * x_u 	= gsl_vector_calloc(Nl);


	double ** pld 		= malloc(sizeof(double*)*Nl);
	for(l=0;l<Nl;l++)
		pld[l]=malloc(sizeof(double)*Noccs);
	gsl_matrix * wld = gsl_matrix_calloc(Nl,Noccs);
	struct sys_sol * sol 	= st->sol;
	struct sys_coef * sys 	= st->sys;
	struct aux_coef * simdat= st->sim;
	status = 0;
	int bad_coeffs=0;
	int Xrow,Xrow0 =0;
	// take sequence from st->mats->Zz_hist.  Just feed it in, no changes
	gsl_matrix * Zz_hist = st->mats->Zz_hist;

	Ndraw = Zz_hist->size1;
	gsl_vector * shocks 	= gsl_vector_calloc(Nx);
	gsl_vector * Zzl		= gsl_vector_calloc(Nx);
	gsl_matrix * uf_hist	= gsl_matrix_calloc(Ndraw,4);
	gsl_matrix * urt_l, * urt_l_wt, * fnd_l_hist, * x_u_hist, * Zzl_hist, * s_wld,*tll_hist,*tld_hist,*gld_hist,*pld_hist,*fac_hist;
	if(printlev>=2){
		urt_l		= gsl_matrix_calloc(Ndraw,Noccs);
		fac_hist	= gsl_matrix_calloc(Ndraw,Nfac);
		fnd_l_hist	= gsl_matrix_calloc(Ndraw,Nl);
		tll_hist	= gsl_matrix_calloc(Ndraw,Noccs);
		tld_hist	= gsl_matrix_calloc(Ndraw,Noccs*Nl);
		gld_hist	= gsl_matrix_calloc(Ndraw,Noccs*Nl);
		pld_hist	= gsl_matrix_calloc(Ndraw,Noccs*Nl);

		x_u_hist	= gsl_matrix_calloc(Ndraw,Nl);
		Zzl_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
		s_wld		= gsl_matrix_calloc(Nl,Noccs);
		urt_l_wt 	= gsl_matrix_calloc(Ndraw,Noccs);
	}
	struct dur_moments s_mom;
	if(printlev>=2){
		s_mom.Ft = gsl_vector_calloc(5);
		s_mom.Ft_occ = gsl_vector_calloc(5);
		s_mom.xFt = gsl_vector_calloc(5);
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
	gsl_matrix * XX = gsl_matrix_calloc(Ndraw*(Nl-4)*Noccs,Nskill); //max number I'll need
	gsl_vector_view X0 = gsl_matrix_column(XX,Nskill-1);
	gsl_vector_set_all(&X0.vector,1.0);

	//gsl_matrix * Wt	= gsl_matrix_calloc((Noccs-2)*Nl+1,(Noccs-2)*Nl+1);
	gsl_vector * yloss = gsl_vector_alloc(XX->size1);
	gsl_vector * coefs	= gsl_vector_calloc(XX->size2);
	gsl_vector * er = gsl_vector_alloc(yloss->size);

	// Take a draw for Zz and initialize things for init_T periods
	int init_T = 200;
	gsl_vector_view Zdraw = gsl_matrix_row(Zz_hist,0);
	gsl_vector_memcpy(Zz,&Zdraw.vector);
	if(printlev>=3)
		printvec("Zdraws.csv",&Zdraw.vector);

	//run through a few draws without setting anything to get rid of the initial state
	gsl_matrix_memcpy(x,xss);
	sol->tld = gsl_matrix_calloc(Nl,Noccs);
	sol->gld = gsl_matrix_calloc(Nl,Noccs);
	sol->ald = gsl_matrix_calloc(Nl,Noccs);
	sol->sld = gsl_matrix_calloc(Noccs+1,Noccs);
	FILE * zzhist_init;
	if(printlev>=3) zzhist_init = fopen("zzhist.csv","a+");
	for(di=0;di<init_T;di++){

		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);
		status += xprime(xp,sol->ald,ss,sys,sol,x,Zz);

		gsl_matrix_memcpy(x,xp);

		gsl_vector_view Zdraw = gsl_matrix_row(Zz_hist,di);

		gsl_vector_memcpy(Zz,&Zdraw.vector);
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

		double fac_ave = (double)Ndraw/(double)(di-1);

		/* advance a period in the shock:
		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);
		*/
		// just pull the current shock from Zz_hist
		gsl_vector_view Zdraw = gsl_matrix_row(Zz_hist,di);
		gsl_vector_memcpy(Zz,&Zdraw.vector);

		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);

		if( printlev>=1 && (gpol_zeros>0 || t_zeros>0)){
			simerr = fopen(simer_f,"a+");
			fprintf(simerr,"%d times g^ld = 0 in sim # %d \n",gpol_zeros,di);
			fprintf(simerr,"%d times t^ld = 0 in sim # %d \n",t_zeros,di);
			fclose(simerr);
		}
		status += xprime(xp,sol->ald,ss,sys,sol,x,Zz);
		for(l=0;l<Nl;l++){
			for(d=0;d<Noccs;d++)
				pld[l][d] = pmatch(gsl_matrix_get(sol->tld,l,d));
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
				for(l=0;l<Nl;l++){
					tld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = gsl_matrix_get(sol->tld,l,d-1);
					gld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = gsl_matrix_get(sol->gld,l,d-1);
					pld_hist->data[di*tld_hist->tda+l*Noccs+d-1] = pld[l][d-1];
				}

			}
		}

		// mean Z this period
		double m_Zz_t = gsl_vector_get(Zz,0);
		//double m_Zz_t = 0.0;
		//for(d=0;d<Noccs;d++)
		//	m_Zz_t += gsl_vector_get(Zz,Notz+d)/(double)Noccs;
		//m_Zz_t 	/= 2.0;
		//m_Zz_t 	+= gsl_vector_get(Zz,0)/2.0;
		m_Zz	+= m_Zz_t/(double)Ndraw;
		gsl_matrix_set(uf_hist,di,0,m_Zz_t);

		// calculate the unemployment rate:

		double urtp = 0.0;
		for(l=0;l<Noccs+1;l++)
			urtp += gsl_matrix_get(xp,l,0)+gsl_matrix_get(xp,l,Noccs+1);
		s_urt += urtp/((double) Ndraw);
		gsl_matrix_set(uf_hist,di,1,urtp);

		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++){
				if(urtp>0.0)
					gsl_vector_set(x_u,l+ll*(Noccs+1),gsl_matrix_get(xp,l,ll*(Noccs+1))/urtp);
				else
					gsl_vector_set(x_u,l,0.0);
			}
		}
		if(printlev>=2){
			gsl_vector_view x_u_d = gsl_matrix_row(x_u_hist,di);
			gsl_vector_memcpy(&x_u_d.vector,x_u);
		}

		// average tightness
		double m_tld =0.;
		double m_ald =0.;
		for(l=0;l<Nl;l++){
			for(d=0;d<Noccs;d++) {
				m_tld += gsl_matrix_get(sol->ald, l, d) * gsl_matrix_get(sol->tld, l, d);
				m_ald += gsl_matrix_get(sol->ald, l, d);
			}
		}
		gsl_matrix_set(uf_hist,di,3,m_tld/m_ald);

		fr00+= (x_u->data[0] + x_u->data[Noccs+1])/(double)Ndraw;

		for(d=1;d<Noccs+1;d++)
			frdd += gsl_matrix_get(xp,d,d)/(double)Ndraw;

		// get the finding rate by origin:
		double d_fnd =0.0;
		for(l=0;l<Nl;l++){
			fnd_l->data[l] = 0.0;
			double ald_l = 0.;
			for(d=0;d<Noccs;d++){
				fnd_l->data[l] += gsl_matrix_get(sol->ald,l,d)*pld[l][d];
				ald_l += gsl_matrix_get(sol->ald,l,d);
			}
			fnd_l->data[l] /= ald_l;
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
		for(l=0;l<Nl;l++){
			for(d=0;d<Noccs;d++){
				d_elpt += gsl_matrix_get(sol->ald,l,d)*
						1.0/(1.0+ pow(gsl_matrix_get(sol->tld,l,d),phi));
				x_elpt += gsl_matrix_get(sol->ald,l,d);
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
					d_sep += gsl_matrix_get(x,l,d)*( (1.0-tau)*gsl_matrix_get(sol->sld,l,d-1)
							+ tau*gsl_matrix_get(sol->sld,d,d-1) );
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
			for(ll=0;ll<2;ll++){
				double d_chng_l = 0.;
				for(d=0;d<Noccs;d++){
					if(d!=l-1) {
						d_chng_l += gsl_matrix_get(sol->ald,l+ll*(Noccs+1),d)*pld[l+ll*(Noccs+1)][d];
					}
				}
				d_chng_l /= (d_chng_l+gsl_matrix_get(sol->ald,l+ll*(Noccs+1),l-1)*pld[l+ll*(Noccs+1)][l-1]);
				d_chng += d_chng_l*gsl_vector_get(x_u,l+ll*(Noccs+1));
			}
		}
		// now get the inexperienced
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
					inexp_d += gsl_matrix_get(x,l,d+1)/inexper;
					//d_chng_0x += gsl_matrix_get(x,l,d)/gsl_vector_get(x_u,0);
			}
			d_chng_0x = (1.0-inexp_d);
			d_chng_0 += gsl_matrix_get(sol->gld,0,d)*pld[0][d]*d_chng_0x / fnd_l->data[0];

		}
		d_chng += d_chng_0*x_u->data[0];

		d_chng = gsl_finite(d_chng)==0 ? s_chng*fac_ave : d_chng;
		s_chng +=d_chng/(double)Ndraw;

		////////////////////////////////////////////////////////////////////
		//	Wage growth

		// compute wages at each ld
		gsl_vector_set_all(Es,0.);
		gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
		status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

		for(l=0;l<Noccs+1;l++) {
			double bl = l > 0 ? b[1] : b[0];
			for (ll = 0; ll < 2; ll++) {
				for (d = 0; d < Noccs; d++) {
					double nud = l == d + 1 ? 0. : nu;
					double zd = Zz->data[d + Notz];

					double cont = Es->data[Wld_i + l * Noccs + d] -
								  (1. - 1. / bdur) * Es->data[Wl0_i + l + ll * (Noccs + 1)] -
								  1. / bdur * Es->data[Wl0_i + l + Noccs + 1];

					double wld_ld = (1. - fm_shr) *
							   (chi[l][d] * exp(zd) - bl * (1. - (double) ll) - privn * (double) ll - nud)
						+ bl*(1.-(double)ll) + privn*(double)ll ;
					if (wld_ld > chi[l][d])
						gsl_matrix_set(wld,l+ll*(Noccs+1),d,chi[l][d]);
					else if(wld_ld < bl*(1.-(double)ll) + privn*(double)ll)
						gsl_matrix_set(wld,l+ll*(Noccs+1),d,bl*(1.-(double)ll) + privn*(double)ll );
					else
						gsl_matrix_set(wld,l+ll*(Noccs+1),d,wld_ld);
				}
			}
		}
		// take out occupation fixed effects and re-norm to the average, just in case:
		double Ewg_d[Noccs];
		double Ewg = 0.;
		double xwg = 0.;
		for(d=0;d<Noccs;d++){
			Ewg_d[d] =0;
			double xwg_d =0;
			for(l=0;l<Noccs+1;l++){
				Ewg_d[d] += log(gsl_matrix_get(wld,l,d))*gsl_matrix_get(xp,l,d+1);
				xwg_d += gsl_matrix_get(xp,l,d+1);
				Ewg += Ewg_d[d];
				xwg += xwg_d;
			}
			Ewg_d[d] /= xwg_d;
			Ewg_d[d]  = exp(Ewg_d[d]);
		}
		Ewg /= xwg;
		Ewg  = exp(Ewg);
		double ** wldFE = malloc(sizeof(double*)*Nl);
		for(l=0;l<Nl;l++)
			wldFE[l] = malloc(sizeof(double)*Noccs);
		for(d=0;d<Noccs;d++){
			for(l=0;l<Nl;l++)
				wldFE[l][d] = gsl_matrix_get(wld,l,d)/Ewg_d[d]*Ewg;
		}


		double d_wg = 0.0;
		double x_wg = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					d_wg += (gsl_matrix_get(x,l,d+1))*
							log(gsl_matrix_get(wld,d+1,d) / gsl_matrix_get(wld,l,d));
					x_wg += (gsl_matrix_get(x,l,d+1));
				}
			}
		}
		d_wg *= tau; // dividing by expected time for that growth: 1.0/tau
		if(x_wg>0.)
			d_wg /= x_wg;
		else
			d_wg = 0.;
		d_wg = gsl_finite(d_wg) == 0  ? s_wg*fac_ave : d_wg;
		s_wg += d_wg/(double)Ndraw;

		// wage loss due to change
		double d_wl = 0.0;
		double sx_wl= 0.0;
		for(l=1;l<Nl;l++){
			for(d=0;d<Noccs;d++){
				if(l%(Noccs+1)!=d+1 && l%(Noccs+1)!=0){
					//d_wl +=gsl_matrix_get(sol->ald,l,d)*pld[l][d] *log(gsl_matrix_get(wld,l,d)/gsl_matrix_get(wld,l,l%(Noccs+1)-1) );
					d_wl  += gsl_matrix_get(sol->ald,l,d)*pld[l][d] *log(wldFE[l][d]/wldFE[l%(Noccs+1)][l%(Noccs+1)-1]);
					sx_wl += gsl_matrix_get(sol->ald,l,d)*pld[l][d] ;
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
		// only look at experienced changers
			Xrow = Xrow0; // where I start for this iteration
			double wgt[XX->size1/Ndraw]; //max I'll need
			double totwgt =0;
			for(l=1;l<Nl;l++){
				for(d=0;d<Noccs;d++){
					if(l % (Noccs+1)!=d+1 && l % (Noccs+1)>0){
						// the loss is from when they were a stayer to now... actually using current stayers
						//double wlast = gsl_matrix_get(wld,l%(Noccs+1),l%(Noccs+1)-1);
						//double wnext = gsl_matrix_get(wld,l,d);
						double wlast = wldFE[l%(Noccs+1)][l%(Noccs+1)-1];
						double wnext = wldFE[l][d];
						double ylossld = log(wnext/wlast);
						// the fraction doing it
						double wgt_here = pld[l][d]*gsl_matrix_get(sol->ald,l,d);
						if(gsl_finite(ylossld) && wgt_here>1e-5 && gsl_finite(wgt_here)){
							wgt[Xrow - Xrow0] = wgt_here;
							gsl_vector_set(yloss,Xrow,ylossld);
							for(si=0;si<Nskill-1;si++)
								gsl_matrix_set(XX,Xrow,si,
									(gsl_matrix_get(f_skills,d,si+1)-gsl_matrix_get(f_skills,l-1,si+1) ));
							gsl_matrix_set(XX,Xrow,Nskill-1, 1.);
							Xrow++;
							totwgt += wgt_here;
						}
					}
				}
			}
			for(l=0;l<Xrow-Xrow0;l++) wgt[l]/=totwgt;
			for(l=0;l<Xrow-Xrow0;l++){
				gsl_vector_view X_r = gsl_matrix_row(XX,l+Xrow0);
				gsl_vector_scale(&X_r.vector,wgt[l]);
				gsl_vector_set(yloss,l+Xrow0, gsl_vector_get(yloss,l+Xrow0)*wgt[l]);
			}
			Xrow0 = Xrow;
		}

		for(l=0;l<Nl;l++)
			free(wldFE[l]);
		free(wldFE);
		/*if(printlev>=2){
		// compute wages in  this period
			gsl_vector * Es = gsl_vector_calloc(Ns);
			gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
			status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

			for(l=0;l<Nl;l++){
				double bb = l==0 ? b[0] : b[1];
				double bl = l>= Noccs+1 ? privn : bb;
				for(d=0;d<Noccs;d++){
					// need to use spol to get \bar \xi^{ld}, then evaluate the mean, \int_{-\bar \xi^{ld}}^0\xi sh e^{sh*\xi}d\xi
					//and then invert (1-scale_s)*0 + scale_s*log(sld/shape_s)/scale_s
					double barxi = -log(gsl_matrix_get(sol->sld,l,d)/shape_s )/shape_s;
					double Exi = scale_s*((1.0/shape_s+barxi)*exp(-shape_s*barxi)-1.0/shape_s);
					double W0_ld = l>=Noccs+1 ? Es->data[Wl0_i+l] : (1.0-1.0/bdur)*Es->data[Wl0_i+l]-1.0/bdur *Es->data[Wl0_i+l+Noccs+1] ;
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					double bEJ = beta*fm_shr*(Es->data[Wld_i + l*Noccs+d] -beta*W0_ld);
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////


			//		double wld_ld = (1.0-fm_shr)*chi[l][d]*exp(Zz->data[0]+Zz->data[Notz+d]) +
			//				bEJ - fm_shr*beta*(Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i+l]) - fm_shr*(bl+Exi);
					double wld_ld = (1.0-fm_shr)*chi[l*(Noccs+1)][d]*exp(Zz->data[Notz+d]) +
							bEJ - fm_shr*beta*(Es->data[Wld_i + l*Noccs+d] -W0_ld) - fm_shr*(bl+Exi);

					wld_ld = wld_ld >0.0 ? wld_ld : 0.0;
					gsl_matrix_set(wld,l,d,wld_ld);
				}
			}
			s_wld->data[l*wld->tda+d] += wld->data[l*wld->tda+d]/(double) Noccs;

		}*/

		gsl_matrix_memcpy(x,xp);
	}// for d<Ndraw

	if((st->cal_set<=0 || st->cal_set==2) ) {
		gsl_vector_view y_v = gsl_vector_subvector(yloss, 0, Xrow);
		gsl_matrix_view X_v = gsl_matrix_submatrix(XX, 0, 0, Xrow, XX->size2);
		if (printlev >= 2) {
			printmat("XXloss.csv", &X_v.matrix);
			printvec("yloss.csv", &y_v.vector);
		}

		status += OLS(&y_v.vector, &X_v.matrix, coefs, er);

	}

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
	for(di=0;di<coefs->size;di++)
		fincoefs *= gsl_finite(gsl_vector_get(coefs,di));
	if(fincoefs!=0){
		for(di=0;di<coefs->size;di++)
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
		printvec("xFt.csv",s_mom.xFt);
		printvec("Ft_occ.csv",s_mom.Ft_occ);
		printvec("dur_hist.csv",s_mom.dur_hist);
		printmat("dur_l_hist.csv",s_mom.dur_l_hist);
		FILE * durstats = fopen("durstats.csv","w+");
		printf("Printing durstats.csv\n");
		fprintf(durstats,"E[dur],Pr[>=6mo],sd(dur),D(wg)\n");
		fprintf(durstats,"%f,%f,%f,%f\n",s_mom.E_dur,s_mom.pr6mo,s_mom.sd_dur,s_mom.chng_wg);

		s_sdlu	= 0.0;
		s_sdZz	= 0.0;
		for(di=0;di<uf_hist->size1;di++){
			s_sdlu += pow(log(gsl_matrix_get(uf_hist,di,1)) - log(s_urt),2);
			s_sdZz+= pow(gsl_matrix_get(uf_hist,di,0) - m_Zz,2);
		}
		s_sdlu 	/=(double)uf_hist->size1;
		s_sdZz	/=(double)uf_hist->size1;
		s_sdlu	= sqrt(s_sdlu);
		s_sdZz	= sqrt(s_sdZz);

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
		int off_nohc = fabs(st->cal_set)==3 ? 2 : 0;
		if(fabs(st->cal_set) !=3){
			gsl_matrix_set(simdat->data_moments,0,0,(avg_wg  - s_wg)/avg_wg);
			gsl_matrix_set(simdat->data_moments,0,1,(avg_fnd - s_fnd)/avg_fnd);
			gsl_matrix_set(simdat->data_moments,0,2,(chng_pr - s_chng)/chng_pr);
		}
		else{
			gsl_matrix_set(simdat->data_moments,0,0,(avg_fnd - s_fnd)/avg_fnd);
		}
		gsl_matrix_set(simdat->data_moments,0,3 - off_nohc,(avg_urt - s_urt)/avg_urt);
		gsl_matrix_set(simdat->data_moments,0,4 - off_nohc,(avg_elpt - s_elpt)/avg_elpt);
		gsl_matrix_set(simdat->data_moments,0,5 - off_nohc,(avg_sdsep - s_sdsep)/avg_sdsep);
	}
	else{// penalize it for going in a region that can't be solved
		gsl_matrix_scale(simdat->data_moments,5.0);
		status ++;
	}

	if(verbose>=3){
		printf("(obj-sim)/obj : ");
		if(fabs(st->cal_set) != 3){
			printf("wg=%f, ",gsl_matrix_get(simdat->data_moments,0,0));
			printf("fnd=%f, ",gsl_matrix_get(simdat->data_moments,0,1));
			printf("chng=%f, ",gsl_matrix_get(simdat->data_moments,0,2));
		}
		else{
			printf("fnd=%f, ",gsl_matrix_get(simdat->data_moments,0,1));
		}
		int off_nohc = fabs(st->cal_set)==3 ? 2 : 0;
		printf("urt=%f, ",gsl_matrix_get(simdat->data_moments,0,3-off_nohc));
		printf("elpt=%f, ",gsl_matrix_get(simdat->data_moments,0,4-off_nohc));
		printf("sdsep=%f, ",gsl_matrix_get(simdat->data_moments,0,5-off_nohc));
		if(printlev>=2)printf("sd(log u)=%f, ",s_sdlu);
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
	gsl_matrix_free(sol->ald);

	gsl_vector_free(Es);
	for(l=0;l<Nl;l++)
		free(pld[l]);
	free(pld);
	gsl_matrix_free(XX);
	gsl_vector_free(coefs);
	gsl_vector_free(yloss);
	gsl_vector_free(er);
	if(printlev>=2){
		gsl_vector_free(s_mom.Ft);
		gsl_vector_free(s_mom.Ft_occ);
		gsl_vector_free(s_mom.xFt);
		gsl_vector_free(s_mom.dur_hist);
		gsl_matrix_free(s_mom.dur_l_hist);
		free(s_mom.dur_l);
	}
	gsl_vector_free(fnd_l);
	gsl_vector_free(Zzl);
	gsl_vector_free(shocks);
	gsl_matrix_free(uf_hist);
	gsl_vector_free(x_u);

	if(printlev>=2){
		gsl_matrix_free(Zzl_hist);
		gsl_matrix_free(fac_hist);
		gsl_matrix_free(s_wld);
		gsl_matrix_free(urt_l);
		gsl_matrix_free(fnd_l_hist);
		gsl_matrix_free(tll_hist);
		gsl_matrix_free(tld_hist);
		gsl_matrix_free(x_u_hist);
		gsl_matrix_free(urt_l_wt);
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

	for(duri =0;duri <5;duri ++){ // loop over each value of vector durs
		double Ft_ld = 0.0; // finding rate in each direction
		double xt_ld = 0.0; // number in each direction
		double Ft_occ = 0.0; // overall finding rate
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
		gsl_vector_set(s_mom->xFt,duri,xt_ld);
		gsl_vector_set(s_mom->Ft_occ,duri,Ft_occ);
	}// for duri in durs

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
		d_sddur  = 0.0;
		double x_sddur = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(gsl_matrix_get(pld_hist,t,l*Noccs+d)>0.0){
					d_sddur += gsl_matrix_get(gld_hist,t,l*Noccs+d)*gsl_matrix_get(x_u_hist,t,l)*pow(gsl_matrix_get(pld_hist,t,l*Noccs+d),-2);
					x_sddur+= gsl_matrix_get(gld_hist,t,l*Noccs+d)*gsl_matrix_get(x_u_hist,t,l);
				}
			}
		}
		d_sddur /= x_sddur;
		d_sddur  = sqrt(d_sddur);


		s_mom->sd_dur += d_sddur/(double)Ndraw;
		double d_6mo = 0.0;

		for(l=0;l<Noccs+1;l++){
			double d_6mo_l = 1.0;//gsl_matrix_get(x_u_hist,t,l)*gsl_matrix_get(gld_hist,t,l*Noccs+d);
			double x_6mo_l = 0.0;
			for(tp=6;tp>0;tp--){
				int tidx = t+tp>Ndraw-1 ? t - Ndraw + tp : t + tp;
				for(d=0;d<Noccs;d++){
					d_6mo_l += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d))
									*(1.0-gsl_matrix_get(pld_hist,tidx,l*Noccs+d))*d_6mo_l;
					x_6mo_l += gsl_matrix_get(gld_hist,tidx,l*Noccs+d)*exp(- gsl_matrix_get(pld_hist,tidx,l*Noccs+d));
				}
				d_6mo_l /=x_6mo_l;
			}
			d_6mo += d_6mo_l*gsl_matrix_get(x_u_hist,t,l);
		}
		s_mom->pr6mo += d_6mo/(double)Ndraw;
	}
	s_mom->sd_dur =pow(s_mom->sd_dur,0.5);
	free(d_dur_l);


	return status;
}



int TGR(struct st_wr * st){

	int status,l,d,ll,t,Tmo,di,Nl= 2*(Noccs+1);
	status	= 0;
	Tmo 	= 24;
	int Ndraw = (st->sim->draws->size1)/Tmo;
	gsl_matrix * xp = gsl_matrix_calloc(Noccs+1,Noccs+2);
	gsl_matrix * x = gsl_matrix_calloc(Noccs+1,Noccs+2);
	gsl_matrix * r_occ_wr = gsl_matrix_alloc(9416,7);
	gsl_vector * Zz_2008 	= gsl_vector_alloc(Nx);
	gsl_vector * Zz_2009 	= gsl_vector_alloc(Nx);
	gsl_vector * Zz_2010 	= gsl_vector_alloc(Nx);
	gsl_vector * Zzl	= gsl_vector_alloc(Nx);
	gsl_matrix * Zhist	= gsl_matrix_alloc(Tmo,Nx);
	gsl_matrix * Zave	= gsl_matrix_calloc(Tmo,Nx);

	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);

	gsl_vector * x_u 		= gsl_vector_calloc(Noccs+1);
	struct sys_sol * sol 	= st->sol;
	struct sys_coef * sys 	= st->sys;
	gsl_vector * ss 	= st->ss;

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
	gsl_matrix * Zpool		= gsl_matrix_calloc(Tmo*Ndraw,Nx);
	readmat("occ_wr.csv",r_occ_wr);
	// skim until peak unemployment in June 2008
	for(l=0;l<r_occ_wr->size1;l++){
		int date = (int) rint(gsl_matrix_get(r_occ_wr,l,0));
		if( date == 200808 ){
			d = gsl_matrix_get(r_occ_wr,l,1);
			gsl_matrix_set(x,d,0,gsl_matrix_get(r_occ_wr,l,4));
			gsl_matrix_set(x,d,d,gsl_matrix_get(r_occ_wr,l,3));
		}
	}

	double usum = 0.0;
	double uread = 0.0;
	for(l=1;l<Noccs+1;l++){
		usum += gsl_matrix_get(x,l,0);
		uread+= gsl_matrix_get(x,l,0);
		usum += gsl_matrix_get(x,l,l);
	}
	gsl_matrix_set(x,0,0,uread/usum*fr00);
	for(l=1;l<Noccs+1;l++){
		gsl_matrix_set(x,l,0,(1.0-fr00)*gsl_matrix_get(x,l,0)/usum);
		double xll_t = gsl_matrix_get(x,l,l)/usum;
		gsl_matrix_set(x,l,l,xll_t);
		gsl_matrix_set(x,l,l,frdd*xll_t);
		double xld_av = 0.0;
		for(ll=0;ll<Noccs+1;ll++){
			if(ll!=l)
				xld_av += gsl_matrix_get(st->xss,ll,l);
		}
		for(ll=0;ll<Noccs+1;ll++){
			if(ll!=l)
				gsl_matrix_set(x,ll,l,(1.0-frdd)*xll_t*gsl_matrix_get(st->xss,ll,l)/xld_av);
		}

	}
	// check the sum:
	if(printlev>=2)
		printmat("xTGR0.csv",x);
	double sx0 = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs+1;d++)
			sx0 += gsl_matrix_get(x,l,d);
	}
/*	*urt = 0.0;
	for(l=0;l<Noccs+1;l++)
		*urt+=gsl_matrix_get(x,l,0);
*/	//readvec("outzz.csv",Zz_TGR);
	readvec("outzz2008.csv",Zz_2008);
	readvec("outzz2009.csv",Zz_2009);
	readvec("outzz2010.csv",Zz_2010);

	if(sol->gld!=NULL)
		gsl_matrix_free(sol->gld);
	if(sol->sld!=NULL)
			gsl_matrix_free(sol->sld);
	if(sol->tld!=NULL)
			gsl_matrix_free(sol->tld);

	sol->tld = gsl_matrix_calloc(Nl,Noccs);
	sol->gld = gsl_matrix_calloc(Nl,Noccs);
	sol->ald = gsl_matrix_calloc(Nl,Noccs);
	sol->sld = gsl_matrix_calloc(Noccs+1,Noccs);

	for(di=0;di<Ndraw;di++){
		double d_wl		= 0.0;
		double d_sw 	= 0.0;

		// Zz_TGR "stochastic" impulse response
		gsl_vector_view Zt = gsl_matrix_row(Zhist,0);
		// set this at 2008 levels
		gsl_vector_memcpy(&Zt.vector,Zz_2008);
		//printlev = 4;
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
				printmat("Zhist_orig.csv",Zhist);

		// adjust to 2008-2010 and trajectory
		for(l=0;l<Nx;l++){
		//	Zave->data[t*Nx +l] = Zhist->data[l];
		//	gsl_matrix_set(Zpool,di*Tmo + t,l,Zhist->data[l]);
			double zd2008	= 0.0;
			for(t=0;t<4;t++){
				zd2008 += gsl_matrix_get(Zhist,t,l)/4.0;
			}
			double zd2009	= 0.0;
			for(t=4;t<16;t++){
				zd2009 += gsl_matrix_get(Zhist,t,l)/12;
			}
			double zd2010	= 0.0;
			for(t=16;t<24;t++){
				zd2010 += gsl_matrix_get(Zhist,t,l)/8.0;
			}
			double zd2008_dat = (1.0-1.0/3.0)*gsl_vector_get(Zz_2009,l) + 1.0/3.0*gsl_vector_get(Zz_2008,l);
			double zd2009_dat = gsl_vector_get(Zz_2009,l);
			double zd2010_dat = (1.0-1.0/3.0)*gsl_vector_get(Zz_2009,l) + 1.0/3.0*gsl_vector_get(Zz_2010,l);
			// change the means
			for(t=0;t<4;t++){
				Zhist->data[t*Nx + l] += zd2008_dat - zd2008;
			}
			for(t=4;t<16;t++){
				Zhist->data[t*Nx + l] += zd2009_dat - zd2009;
			}
			for(t=16;t<24;t++){
				Zhist->data[t*Nx + l] += zd2010_dat - zd2010;
			}
			// now pivot
			/*double slope_0809 = (zd2009_dat-zd2008_dat)/12.0;
			double slope_0910 = (zd2010_dat-zd2009_dat)/12.0;
			for(t=0;t<12;t++)
				Zhist->data[t*Nx+l] += slope_0809*((double)t+1.0 - 6.0)
			for(t=0;t<12;t++)
				Zhist->data[t*Nx+l] += slope_0910*((double)t+1.0 - 6.0)
			*/
/*			double offset2009	= gsl_vector_get(Zz_2009,l) - zd2009;
			gsl_matrix_set(Zhist,11,l,gsl_vector_get(Zz_2009,l));
			double slope_dat1 	= (gsl_vector_get(Zz_2010,l)-gsl_vector_get(Zz_2009,l))/12;
			for(t=1;t<12;t++){
				double Znew = gsl_matrix_get(Zhist,t,l) - (slope_sim0 - slope_dat0)*((double)t);
				gsl_matrix_set(Zhist,t,l,Znew);
			}
			gsl_matrix_set(Zhist,11,l,gsl_vector_get(Zz_2009,l));
			for(t=12;t<24;t++){
				double Znew = gsl_matrix_get(Zhist,t,l) - (slope_sim1 - slope_dat1)*((double)t)+ offset2009;
				gsl_matrix_set(Zhist,t,l,Znew);
			}
*/			for(t=0;t<24;t++){
				Zave->data[t*Nx+l]+= gsl_matrix_get(Zhist,t,l)/((double)(Ndraw));
				gsl_matrix_set(Zpool,di*Tmo + t,l,gsl_matrix_get(Zhist,t,l));
			}


		}
		if(printlev>=4)
			printmat("Zhist_TGR.csv",Zhist);
		for(t=0;t<Tmo;t++){// this is going to loop through 2 years of recession
			gsl_vector_view Zz_t = gsl_matrix_row(Zhist,t);

			status += theta(sol->tld,ss, sys, sol, &Zz_t.vector);
			status += gpol(sol->gld,ss, sys,  sol, &Zz_t.vector);
			status += spol(sol->sld,ss, sys,  sol, &Zz_t.vector);
			status += xprime(xp, sol->ald,ss, sys,  sol, x, &Zz_t.vector);

			// store the policies for the dur_dist later
			for(l=0;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					gsl_matrix_set(pld_hist,di*Tmo+t,l*Noccs+d,pmatch(gsl_matrix_get(sol->tld,l,d)) );
					gsl_matrix_set(gld_hist,di*Tmo+t,l*Noccs+d,gsl_matrix_get(sol->gld,l,d) );
				}
			}


			urt_hist->data[t] =0.0;
			for(l=0;l<Noccs+1;l++)
				urt_hist->data[t+di*Tmo] += gsl_matrix_get(xp,l,0);
			//*urt += urt_hist->data[t+di*Tmo]/(double)Tmo;
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
	if(verbose>=2)
		printf("\n experiment done!\n");


	struct dur_moments dur_TGR;
	dur_TGR.Ft = gsl_vector_alloc(5);
	dur_TGR.xFt = gsl_vector_alloc(5);
	dur_TGR.Ft_occ = gsl_vector_alloc(5);
	dur_TGR.dur_l_hist = gsl_matrix_calloc(Tmo*Ndraw,Noccs+1);
	dur_TGR.dur_hist = gsl_vector_calloc(Tmo*Ndraw);
	dur_TGR.dur_l = malloc(sizeof(double)*Noccs+1);
	if(verbose>=2)
		printf("Calculating duration stats\n");

	for(l=0;l<Noccs+1;l++){
		dur_TGR.dur_l[l] = 0.0;
	}
	status += dur_dist(&dur_TGR,gld_hist,pld_hist,x_l_hist);
	// other duration stats
	printf("printing durstats_TGR.csv\n");
	FILE * durstatsTGR = fopen("durstats_TGR.csv","w+");
	fprintf(durstatsTGR,"E[dur],Pr[>=6mo],sd(dur),D(wg),ave fnd\n");
	fprintf(durstatsTGR,"%f,%f,%f,%f,%f\n",dur_TGR.E_dur,dur_TGR.pr6mo,dur_TGR.sd_dur,dur_TGR.chng_wg,s_fnd);
	fclose(durstatsTGR);

	printvec("Ft_TGR.csv",dur_TGR.Ft);
	printvec("xFt_TGR.csv",dur_TGR.xFt);
	printvec("Ft_occ_TGR.csv",dur_TGR.Ft_occ);
	gsl_vector_view dur_v = gsl_vector_view_array(dur_TGR.dur_l,Noccs+1);
	printvec("dur_l_TGR.csv",&dur_v.vector);
	printvec("urt_TGR.csv",urt_hist);
	printvec("sw_TGR.csv",sw_hist);
	printvec("wl_TGR.csv",wl_hist);
	printmat("Zave_TGR.csv",Zave);
	printmat("fnd_l_TGR.csv",fnd_l_hist);
	printmat("x_l_TGR.csv",x_l_hist);
	printmat("Zpool.csv",Zpool);

	gsl_matrix_free(sol->gld);gsl_matrix_free(sol->tld);gsl_matrix_free(sol->sld);
	sol->gld = NULL;
	sol->sld = NULL;
	sol->tld = NULL;

	/*

	gsl_vector_free(dur_TGR.Ft);gsl_vector_free(dur_TGR.Ft_occ);
	gsl_matrix_free(dur_TGR.dur_l_hist);free(dur_TGR.dur_l);
	gsl_vector_free(dur_TGR.dur_hist);

	gsl_matrix_free(Zpool);
	gsl_vector_free(fnd_l);
	gsl_vector_free(av_fnd_l);
	gsl_matrix_free(fnd_l_hist);
	gsl_vector_free(x_u);
	gsl_matrix_free(r_occ_wr);
	gsl_matrix_free(xp);
	gsl_matrix_free(x);
	gsl_vector_free(Zzl);
	gsl_matrix_free(x_l_hist);
	gsl_matrix_free(Zhist);
	gsl_matrix_free(Zave);
	gsl_vector_free(Zz_2008);
	gsl_vector_free(Zz_2009);
	gsl_vector_free(Zz_2010);
	gsl_vector_free(sw_hist);
	gsl_vector_free(wl_hist);
	gsl_matrix_free(pld_hist);
	gsl_matrix_free(gld_hist);
*/
	return status;
}

int fd_dat_sim(gsl_matrix * fd_hist_dat, struct shock_mats * fd_mats){
	int status,simT, t,d,dd,fi;
	status =0;
	gsl_matrix *zeta_draw, *eta_draw;
	gsl_matrix *chol_varzeta,*chol_vareta;
	gsl_matrix *fd_fac;
	gsl_vector *fd_ag, *epsilon_draw,*fd_fac_tm1;
	simT = fd_hist_dat->size1;

	zeta_draw    = gsl_matrix_calloc(simT,Noccs);
	epsilon_draw = gsl_vector_calloc(simT);
	eta_draw     = gsl_matrix_calloc(simT,Nfac);
	fd_fac_tm1   = gsl_vector_calloc(Nfac);
	chol_varzeta = gsl_matrix_calloc(Noccs,Noccs);
	chol_vareta  = gsl_matrix_calloc(Nfac,Nfac);
	fd_fac 		 = fd_mats->facs;
	fd_ag		 = fd_mats->ag;

	randn(zeta_draw,9041987);
	gsl_matrix_view epsilon_draw_mat = gsl_matrix_view_vector(epsilon_draw,1,simT);
	randn( &(epsilon_draw_mat.matrix),12281951);
	randn(eta_draw, 9241951);

	double rhoZ = fd_mats->rhoZ;
	double rhozz = fd_mats->rhozz;
	double sig_eps = pow(fd_mats->sig_eps2,0.5);
	if(sym_occs ==1){
		double avg_vzeta = 0.;
		for(d=0;d<Noccs;d++) avg_vzeta += gsl_matrix_get(fd_mats->var_zeta,d,d);
		avg_vzeta /= (double)Noccs;
		gsl_matrix_set_identity(fd_mats->var_zeta);
		for(d=0;d<Noccs;d++) gsl_matrix_set(fd_mats->var_zeta,d,d, avg_vzeta);
		for(d=0;d<Noccs;d++) {
			gsl_matrix_set(fd_mats->Lambda,d,0,0.1);
			gsl_matrix_set(fd_mats->Lambda,d,1,0.1);
			gsl_matrix_set(fd_mats->Lambda,d,2,1.0);
		}
	}

	if(fd_dat_read==1){
		gsl_matrix * Zz_fd_read = gsl_matrix_calloc(simT, Noccs+1 + Nfac);
		status += readmat("Zz_fd_dat.csv", Zz_fd_read);
		// esnure everything is de-meaned
		double read_mean=0;
		//for(d=0;d<Noccs;d++) read_mean[d] = 0.;
		for(t=0;t<simT;t++){
			for(d=0;d<Noccs;d++) read_mean += gsl_matrix_get(Zz_fd_read,t,d)/(double)Noccs;
		}
		read_mean /= (double)simT;
		for(t=0;t<simT;t++){
			for(d=0;d<Noccs;d++) gsl_matrix_set(Zz_fd_read,t,d, gsl_matrix_get(Zz_fd_read,t,d)-read_mean );
		}

		for(t=0;t<simT;t++){
			for(d=0;d<Noccs;d++) gsl_matrix_set(fd_hist_dat,t,d,gsl_matrix_get(Zz_fd_read,t,d) );
			gsl_vector_set(fd_ag,t,gsl_matrix_get(Zz_fd_read,t,Noccs));
			for(fi=0;fi<Nfac;fi++){
				gsl_matrix_set(fd_fac,t,fi,gsl_matrix_get(Zz_fd_read,t,Noccs+1+fi));
			}
		}
		gsl_matrix_free(Zz_fd_read);
	}
	else{
		if(diag_zeta ==1 || homosk_zeta ==1){
			for(d=0;d<Noccs;d++){
				gsl_matrix_set(chol_varzeta,d,d,gsl_matrix_get(fd_mats->var_zeta,d,d) );
			}
		}else{
			gsl_matrix_memcpy(chol_varzeta,fd_mats->var_zeta);
			status+= gsl_linalg_cholesky_decomp (chol_varzeta);
			for(d=0;d<Noccs;d++){//set the upper triangle to 0.
				for(dd=d+1;dd<Noccs;dd++)
					gsl_matrix_set(chol_varzeta,d,dd,0.);
			}
		}
		gsl_matrix_memcpy(chol_vareta,fd_mats->var_eta);
		status += gsl_linalg_cholesky_decomp(chol_vareta);
		for(d=0;d<Nfac;d++){
			if(Nfac>1){
				for(dd=d+1;dd<Nfac;dd++)
					gsl_matrix_set(chol_vareta,d,dd,0.);
			}
		}
		gsl_vector_set(fd_ag,0,sig_eps*gsl_vector_get(epsilon_draw,0));
		gsl_vector_view eta_draw_i	= gsl_matrix_row(eta_draw,0);
		gsl_vector_view fd_fac_i	= gsl_matrix_row(fd_fac,0);
		gsl_vector_view zeta_draw_i	= gsl_matrix_row(zeta_draw,0);
		gsl_vector_view fd_dat_i	= gsl_matrix_row(fd_hist_dat,0);
		status += gsl_blas_dgemv(CblasNoTrans, 1., chol_vareta, & eta_draw_i.vector, 0., &fd_fac_i.vector);
		status += gsl_blas_dgemv(CblasNoTrans, 1., chol_varzeta, & zeta_draw_i.vector , 0., & fd_dat_i.vector);
		for(t=1;t<simT;t++){
			eta_draw_i 	= gsl_matrix_row(eta_draw,t);
			fd_fac_i 	= gsl_matrix_row(fd_fac,t);
			zeta_draw_i	= gsl_matrix_row(zeta_draw,t);
			fd_dat_i	= gsl_matrix_row(fd_hist_dat,t);
			gsl_vector_view fd_fac_im1 	= gsl_matrix_row(fd_fac,t-1);

			gsl_vector_set(fd_ag,t,
						   rhoZ* gsl_vector_get(fd_ag,t-1) +sig_eps*gsl_vector_get(epsilon_draw,t));
			//advance facs
			status += gsl_blas_dgemv(CblasNoTrans, 1., chol_vareta, & eta_draw_i.vector, 0., &fd_fac_i.vector);
			status += gsl_blas_dgemv(CblasNoTrans, 1., fd_mats->Gamma, &fd_fac_im1.vector, 0., fd_fac_tm1 );
			status += gsl_vector_add( & fd_fac_i.vector , fd_fac_tm1);
			//advance fd_hist
			status += gsl_blas_dgemv(CblasNoTrans, 1., chol_varzeta, & zeta_draw_i.vector, 0., &fd_dat_i.vector);
			for(d=0;d<Noccs;d++){
				double fd_here = gsl_matrix_get(fd_hist_dat, t,d);
				fd_here += gsl_matrix_get(fd_mats->Lambda,d,Nfac)*gsl_vector_get(fd_ag,t)
						   + rhozz * gsl_matrix_get(fd_hist_dat,t-1,d);
				for(fi=0;fi<Nfac;fi++)
					fd_here += gsl_matrix_get(fd_mats->Lambda,d,fi)*gsl_matrix_get(fd_fac,t,fi);
				gsl_matrix_set(fd_hist_dat,t,d,fd_here);
			}
		}
		double simdat_mean = 0.;
		for(t=0;t<simT;t++){
			for(d=0;d<Noccs;d++) simdat_mean += gsl_matrix_get(fd_hist_dat,t,d)/(double)Noccs;
		}
		simdat_mean /= (double)simT;
		for(t=0;t<simT;t++){
			for(d=0;d<Noccs;d++) gsl_matrix_set(fd_hist_dat,t,d, gsl_matrix_get(fd_hist_dat,t,d)-simdat_mean );
		}
	}
	if(printlev >=2){
		printmat("fd_facs.csv",fd_fac);
		printmat("fd_hist_dat.csv",fd_hist_dat);
		printvec("fd_ag.csv",fd_ag);
	}

	gsl_matrix_free(zeta_draw);
	gsl_matrix_free(eta_draw);
	gsl_vector_free(epsilon_draw);
	//gsl_matrix_free(fd_fac);
	//gsl_vector_free(fd_ag);
	gsl_vector_free(fd_fac_tm1);
	gsl_matrix_free(chol_varzeta);
	gsl_matrix_free(chol_vareta);
	return status;
}



/*
 * Policy functions
 */
int gpol(gsl_matrix * gld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d,dd,ll;
	status=0;
	int JJ1 = Noccs*(Noccs+1);
	int Nl 	= 2*(Noccs+1);
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Nl;
	int ss_gld_i,ss_Wld_i,ss_Wl0_i;
	double ret_d[Noccs];
	double *R_dP_ld	= malloc(sizeof(double)*Noccs*2+1);

		// indices where these things are
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Nl;
		ss_gld_i	= ss_Wld_i + JJ1;

	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);

	status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

	gsl_matrix * pld = gsl_matrix_calloc(Nl,Noccs);

	// make sure not negative value anywhere:
	/*for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			Es->data[Wld_i + l*Noccs+d] = Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l]<0 ?
				Es->data[Wl0_i + l] : Es->data[Wld_i + l*Noccs+d];
	}
	*/
	/*double avgcont = 0.;
	double Es_d[Noccs];
	for(d=0;d<Noccs;d++){
		Es_d[d]=0;
		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++)
				Es_d[d]+= (Es->data[Wld_i+l*Noccs+d] -
						   (1.0-1.0/bdur)*Es->data[Wl0_i+l + ll*(Noccs+1)] - 1.0/bdur*Es->data[Wl0_i+l + Noccs+1] );
		}
		Es_d[d] /= (double)((Noccs+1)*2);
		avgcont += Es_d[d]/(double)Noccs;
	}*/

	if(sol->tld==0)
		status += theta(pld,ss,sys,sol,Zz);
	else
		gsl_matrix_memcpy(pld,sol->tld);
	for(l=0;l<Nl;l++){
		for(d=0;d<Noccs;d++){
			double tld = gsl_matrix_get(pld,l,d);
			double pld_ld = pmatch(tld);
			gsl_matrix_set(pld,l,d,pld_ld);
		}
	}
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		for(ll=0;ll<2;ll++){
			for(d=0;d<Noccs;d++){
				double nud = l == d+1? 0. : nu;
				double zd 	= Zz->data[d+Notz];

				double cont	= Es->data[Wld_i + l*Noccs+d] - (1.-1./bdur)*Es->data[Wl0_i + l+ll*(Noccs+1)] - 1./bdur*Es->data[Wl0_i + l+Noccs+1];

				ret_d[d]	= (1.-fm_shr)*(chi[l][d]*exp(zd) - bl*(1.-(double)ll) - privn*(double)ll - nud + beta*cont);
				//	+ bl*(1.-(double)ll) + privn*(double)ll
				//	+ (1.-1./bdur)*Es->data[Wl0_i + l+ll*(Noccs+1)] + 1./bdur*Es->data[Wl0_i + l+Noccs+1];
				ret_d[d]	= ret_d[d] < 0 ? 0. : ret_d[d];
				ret_d[d]	*= gsl_matrix_get(pld,l+ll*(Noccs+1),d);

			}
			double sexpret_dd =0.;
			if(homosk_psi != 1){
				for(d=0;d<Noccs;d++){ // values in any possible direction
					double ret_d_dd = ret_d[d];
					R_dP_ld[d+1] =  ret_d_dd > 0. &&  ret_d_dd < 1.e10 ? ret_d_dd : 0.;
					R_dP_ld[d+1+Noccs] = sig_psi*gsl_matrix_get(pld,l+ll*(Noccs+1),d);
				}

				gsl_function gprob;
				gprob.function = &hetero_ev;
				gprob.params = R_dP_ld;
				//#pragma omp parallel for private(gprob, d) reduction(+:sexpret_dd)
				for(d=0;d<Noccs;d++){ // choice probabilities
					//gsl_integration_workspace * integw = gsl_integration_workspace_alloc(1000);
					if(ret_d[d]>0 && gsl_matrix_get(pld,l+ll*(Noccs+1),d) > 0.){
						R_dP_ld[0] = (double)d;
						double gg,ggerr;
						size_t neval;
					//	gsl_integration_qags(&gprob,1.e-5,20, 1.e-5, 1.e-5, 1000, integw, &gg, &ggerr);
						gsl_integration_qng(&gprob,1.e-5,20, 1.e-5, 1.e-5, &gg, &ggerr,&neval);
						sexpret_dd += gg;
						gsl_matrix_set(gld,l+ll*(Noccs+1),d,gg );
					}
					else
						gsl_matrix_set(gld, l + ll * (Noccs + 1), d, 0.);
					//gsl_integration_workspace_free(integw);
				}
			} // check homosk_psi (this is much faster)
			else{
				double gld_l = 0;
				for(d=0;d<Noccs;d++)
					gld_l += exp(ret_d[d]*sig_psi);
				for(d=0;d<Noccs;d++){
					gsl_matrix_set(gld,l+ll*(Noccs+1),d,
								   exp(ret_d[d]*sig_psi)/gld_l
					);
					sexpret_dd += gsl_matrix_get(gld,l+ll*(Noccs+1),d);
				}
			}
			// check it was pretty close to 1
			if (sexpret_dd>0.){
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(gld,l+ll*(Noccs+1),d,gsl_matrix_get(gld,l+ll*(Noccs+1),d)/sexpret_dd);
			}
			if(fabs(sexpret_dd -1)>1.e-2){
				solerr = fopen(soler_f,"a+");
				fprintf(solerr,"In gpol, non-SS choice probabilities add to %f != 1 at l=%d\n",sexpret_dd,l);
				fclose(solerr);
				if(verbose>=2)
					printf("In gpol, non-SS choice probabilities add to %f != 1 at l=%d\n",sexpret_dd,l);
			}
			if (sexpret_dd<1e-2){
				if(l>0){
					for(d=0;d<Noccs;d++)
						gsl_matrix_set(gld,l+ll*(Noccs+1),d,0.0);
					gsl_matrix_set(gld,l+ll*(Noccs+1),l-1,1.0);
					gpol_zeros ++;
				}
				else{
					for(d=0;d<Noccs;d++)
						gsl_matrix_set(gld,l+ll*(Noccs+1),d,1.0/(double)Noccs);
					gpol_zeros ++;
				}
			}
		}
	}
	/*
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double * ret_d = malloc(Noccs*sizeof(double));
		for(ll=0;ll<2;ll++){
			double gdenom =0;
			for(d=0;d<Noccs;d++){
				double nud = l == d+1? 0.0 : nu;
				double zd 	= Zz->data[d+Notz];
				//double post = -kappa*gsl_matrix_get(sol->tld,l,d)/gsl_matrix_get(pld,l,d);
				double cont	= Es->data[Wld_i + l*Noccs+d] - (1.0-1.0/bdur)*Es->data[Wl0_i + l+ll*(Noccs+1)] - 1.0/bdur*Es->data[Wl0_i + l+Noccs+1];

				ret_d[d]	= (1.0-fm_shr)*(chi[l][d]*(1.0 + zd) - bl*(1.0-(double)ll) - privn*(double)ll - nud + beta*cont)
										+ bl*(1.0-(double)ll) + privn*(double)ll
										+ (1.0-1.0/bdur)*Es->data[Wl0_i + l+ll*(Noccs+1)] + 1.0/bdur*Es->data[Wl0_i + l+Noccs+1];



				double gld_ld 	= exp(sig_psi*gsl_matrix_get(pld,l,d)*ret_d[d]);
				if(gsl_matrix_get(pld,l,d)>0  && ret_d[d]>0 && gsl_finite(gld_ld))//  && ss->data[ss_tld_i+l*Noccs+d]>0.0)
					gld_l[d]= gld_ld;
				else
					gld_l[d] = 0.0;
				gdenom += gld_l[d];
			}
			if(gdenom>0){
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(gld,l+ll*(Noccs+1),d,gld_l[d]/gdenom);
			}
			else if(l>0){
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(gld,l+ll*(Noccs+1),d,0.0);
				gsl_matrix_set(gld,l+ll*(Noccs+1),l-1,1.0);
				gpol_zeros ++;

			}
			else{
				// should never end up here.  If so, what's up?
				for(d=0;d<Noccs;d++)
					gsl_matrix_set(gld,l+ll*(Noccs+1),d,1.0/(double)Noccs);
				gpol_zeros ++;
			}
		}

	}
	*/

	gsl_vector_free(Es);
	gsl_matrix_free(pld);

	free(R_dP_ld);


	return status;
}
int spol(gsl_matrix * sld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d;
	int JJ1 = Noccs*(Noccs+1);
	int Nl	= 2*(Noccs+1);
	status =0;
	gsl_matrix * gld;
	gsl_matrix * pld = gsl_matrix_calloc(Nl,Noccs);

	int Wl0_i = 0;
		int Wld_i = Wl0_i + Noccs+1;
		int ss_gld_i, ss_tld_i,ss_Wld_i,ss_Wl0_i;
			//ss_x_i		= 0;
			ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
			ss_Wld_i	= ss_Wl0_i + Nl;
			ss_gld_i	= ss_Wld_i + JJ1;//x_i + pow(Noccs+1,2);
			ss_tld_i	= ss_gld_i + 2*JJ1;

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
			gsl_matrix_set(pld,l,d,pmatch(gsl_matrix_get(pld,l,d)));
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

			double ret	= chi[d][d]*exp(zd) - nud
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

			double cutoff= - (chi[l][d]*exp(zd)+ Wdiff);
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
	int status,l,d,ll;
	status=0;
	int JJ1	= Noccs*(Noccs+1);
	int Nl 	= 2*(Noccs+1);
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Nl;
	int ss_gld_i, ss_tld_i, ss_Wld_i,ss_Wl0_i;

		ss_Wl0_i	= 0;
		ss_Wld_i	= ss_Wl0_i + Nl;
		ss_gld_i	= ss_Wld_i + JJ1;
		ss_tld_i	= ss_gld_i + 2*JJ1;

	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
	//printvec("ss_W.csv",&ss_W.vector);
	status += Es_cal(Es, &ss_W.vector, sol->PP, Zz);

	/*double avgcont = 0.;
	double Es_d[Noccs];
	for(d=0;d<Noccs;d++){
		Es_d[d]=0;
		for(l=0;l<Noccs+1;l++){
			for(ll=0;ll<2;ll++)
				Es_d[d]+= (Es->data[Wld_i+l*Noccs+d] -
						   (1.0-1.0/bdur)*Es->data[Wl0_i+l + ll*(Noccs+1)] - 1.0/bdur*Es->data[Wl0_i+l + Noccs+1] );
		}
		Es_d[d] /= (double)((Noccs+1)*2);
		avgcont += Es_d[d]/(double)Noccs;
		printf("Es_d = %f,",Es_d[d]);
		printf("\n");
	}
	 */

	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];		
		for(ll=0;ll<2;ll++){
			double tld_s = 0.0;
			for(d=0;d<Noccs;d++){
				double nud = l == d+1? 0:nu;
				double zd = Zz->data[d+Notz];
				double cont	= (Es->data[Wld_i+l*Noccs+d] -
						(1.0-1.0/bdur)*Es->data[Wl0_i+l + ll*(Noccs+1)] - 1.0/bdur*Es->data[Wl0_i+l + Noccs+1] );

				double surp =chi[l][d]*exp(zd) - bl*(1.-(double)ll) - (double)ll*privn - nud + beta*cont ;

				double qhere = kappa/(fm_shr*surp);
				double tldhere = invq(qhere);

				if(gsl_finite(tldhere) && surp > 0.0 && ss->data[ss_tld_i+l*Noccs+d +ll*JJ1]>0.0){
					gsl_matrix_set(tld,l+ll*(Noccs+1),d,tldhere);
					tld_s += tldhere;
				}
				else
					gsl_matrix_set(tld,l+ll*(Noccs+1),d,0.0);
			} // end for d=0:Noccs-1
			if(tld_s <= 0.0){
				t_zeros ++;
				if(l>0)
					gsl_matrix_set(tld,l+ll*(Noccs+1),l-1,ss->data[ss_tld_i+ll*JJ1 + l*Noccs + (l-1)]);
				else{
					for(d=0;d<Noccs;d++)
						gsl_matrix_set(tld,l+ll*(Noccs+1),d,ss->data[ss_tld_i+ll*JJ1 + l*Noccs +d]);
				}
			}
		}// end for ll=0:1
	}

	gsl_vector_free(Es);
	return status;
}


double zsol_resid(double zdhere, void *params){
	double  resid;
	struct zsol_params * zp =(struct zsol_params *) params;
	double *fd_xg_surp = zp->fd_xg_surp;
	int d = zp->d;
	int Nl 	= 2*(Noccs+1);
	int l,ll;
	double fd_here = fd_xg_surp[0];
	int goodl;
	double avgp,avga;

	avgp  = 0.;
	avga  = 0.;
	goodl = 0 ;

	for(l=0;l<Noccs+1;l++){
		for(ll=0;ll<2;ll++){
			double surp = exp(zdhere) * chi[l][d] + fd_xg_surp[1+Nl+ll*(Noccs+1)+l];
			if(surp>0){
				goodl++;
				avgp += fd_xg_surp[1+ll*(Noccs+1)+l]*pmatch(invq(kappa/(fm_shr*surp)));
				avga += fd_xg_surp[1+ll*(Noccs+1)+l];
			}
		}
	}
	if(goodl >0){
		resid = fd_here - avgp/avga;
	} else
		resid = fd_here + exp(-zdhere); // put in a penalty for how negative was the surplus

	return resid;
}


int invtheta_z(double * zz_fd, const double * fd_dat, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const double * Zz_in){
	// also requires tld and gld, but these are both in struct sys_sol * sol
	// inverts theta to find a set of zld that are consistent.  Then takes the average across these to get the zd that will be realized

	int status,l,d,ll,zlde_m1,iter,seed;
	status=0;
	clock_t seedt;
	int maxiter = 1;
	int JJ1	= Noccs*(Noccs+1);
	int Nl 	= 2*(Noccs+1);
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Nl;
	int ss_gld_i, ss_tld_i, ss_Wld_i,ss_Wl0_i;
	gsl_matrix * gld, *tld, *ald;
	gsl_matrix * zlde = gsl_matrix_calloc(Nl,Noccs);
	double ** pld_dat = malloc(sizeof(double*)*Nl);
	for(l=0;l<Nl;l++)
		pld_dat[l] = malloc(sizeof(double)*Noccs);
	double fd_sim[Noccs],meanzd;
	gsl_vector * Es = gsl_vector_calloc(Ns);
	struct zsol_params zp;
	status = 0;

	ss_Wl0_i	= 0;
	ss_Wld_i	= ss_Wl0_i + Nl;
	ss_gld_i	= ss_Wld_i + JJ1;
	ss_tld_i	= ss_gld_i + 2*JJ1;

	gsl_vector * Zz_here = gsl_vector_calloc(Nx);
	for(d=0;d<Nx;d++)
		gsl_vector_set(Zz_here,d,Zz_in[d]);

	if(sol->tld==0){
		tld = gsl_matrix_calloc(Nl,Noccs);
		status += theta(tld, ss, sys, sol, Zz_here);
	}
	else
		tld = sol->tld;
	if(sol->gld==0){
		gld = gsl_matrix_calloc(Nl,Noccs);
		status += gpol(gld, ss, sys, sol, Zz_here);
	}
	else
		gld = sol->gld;
	ald = sol->ald;



	for(iter=0;iter<maxiter;iter++){
		if(iter>0) seedt = clock(); // may need to initialize a seed
		status=0;
		meanzd = 0.;
		gsl_vector_set_zero(Es);
		gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-ss_Wl0_i);
		status += Es_cal(Es, &ss_W.vector, sol->PP, Zz_here);

		for(d=0;d<Noccs;d++){

			double fd_xg_surp[1+2*Nl];
			// the fd part
			fd_xg_surp[0] = fd_dat[d];

			for(l=0;l<Noccs+1;l++){
				for(ll=0;ll<2;ll++){
					// the x * g part
					fd_xg_surp[1+ll*(Noccs+1)+l] = gsl_matrix_get(ald,l+ll*(Noccs+1),d);

					// i'll worry about it being non-negative surplus in zsol_resid
					double bl = l>0 ? b[1]:b[0];

					double nud = l == d + 1 ? 0 : nu;
					double surp_minprod = -bl * (1. - (double) ll) - (double) ll * privn - nud;
					// if this is not the first iteration, add in the continuation value
					//if(iter>1) {
						double cont = (Es->data[Wld_i + l * Noccs + d] -
									   (1.0 - 1.0 / bdur) * Es->data[Wl0_i + l + ll * (Noccs + 1)] -
									   1.0 / bdur * Es->data[Wl0_i + l + Noccs + 1]);

						surp_minprod += beta * cont;
					//}
					fd_xg_surp[1+Nl+ll*(Noccs+1)+l] = surp_minprod;
				}// end for ll=0:1
			}

			zp.fd_xg_surp = malloc(sizeof(double)*(2*Nl+1));
			for(l=0;l<1+2*Nl;l++) zp.fd_xg_surp[l] = fd_xg_surp[l];
			gsl_function F;
			F.function = &zsol_resid;
			gsl_root_fsolver *zsolver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

			zp.d = d;
			F.params = &zp;
			double zdhere =0;
			double zdmin = log(b[1]); // can't be so low that no exerpienced worker would want to be there
			double zdmin0 = zdmin;
			double zdmax = -zdmin0*zub_fac; // this is ad hoc - rough symmetry
			double zdmax0 = zdmax;
			/* check bounds and otherwise bring them in:
			double residhere_min = zsol_resid(zdmin,(void*)&zp);
			if(residhere_min > fd_dat[d])
				zdmin = 0.5*(zdmin0+zdhere);
			double residhere_max = zsol_resid(zdmax,(void*)&zp);
			if(residhere_max > fd_dat[d])
				zdmax = 0.5*(zdmax0+zdhere);
			*/
			double residhere_min = zsol_resid(zdmin,(void*)&zp);
			double residhere_max = zsol_resid(zdmax,(void*)&zp);
			if(residhere_min*residhere_max<0.) { // straddle zero
				if(szt_gsl == 1){
					gsl_root_fsolver_set(zsolver, &F, zdmin, zdmax);
					int zditer = 0;
					int zdstatus;
					do {
						zditer++;
						zdstatus = gsl_root_fsolver_iterate(zsolver);
						zdhere = gsl_root_fsolver_root(zsolver);
						zdmin = gsl_root_fsolver_x_lower(zsolver);
						zdmax = gsl_root_fsolver_x_upper(zsolver);
						double tol0 = 1e-5;
						zdstatus = gsl_root_test_interval(zdmin, zdmax, tol0, tol0);
						if (zdstatus == GSL_SUCCESS)
							break;

					} while (zdstatus == GSL_CONTINUE && zditer < 100);
					double residhere = zsol_resid(zdhere, (void *) &zp);
					if (zdmin <= zdmin0 || zdmax >= zdmax0 || residhere > fd_dat[d]) {
						status++;
					}
				}else
					zdhere = zero_brent(zdmin, zdmax, zdhere, (void *) &zp, &zsol_resid);
			}
			else if( fabs(residhere_min)>fabs(residhere_max) ){
				zdhere = zdmax;
				status ++;
			}
			else{
				zdhere = zdmin;
				status ++;
			}
			// check no NaNs
			if(gsl_finite(zdhere)!= 1){
				if(fabs(residhere_min)>fabs(residhere_max)){
					zdhere = zdmax;
					status ++;
				}else{
					zdhere = zdmin;
					status ++;
				}

			}


			zz_fd[d] = zdhere;
			meanzd += zdhere/(double)Noccs;
			gsl_root_fsolver_free(zsolver);
			free(zp.fd_xg_surp);
			if(iter<maxiter-1){
				gsl_vector_set(Zz_here,d+Notz,zmt_upd*zdhere + (1.-zmt_upd)*gsl_vector_get(Zz_here,d+Notz));
			}
			else
				gsl_vector_set(Zz_here,d+Notz,zdhere );
		}
		if(status == Noccs){ // this means hit the boundary on every one, I need to do something to keep it full rank.
			if(maxiter<=1)
				maxiter ++; // first try running it again which the higher Zz_here
			else{ // need to throw in white-noise so the row doesn't make the matrix singular

				gsl_rng * rnghere = gsl_rng_alloc(gsl_rng_default);
				time_t timenow;
				time(&timenow);
				seedt = clock() - seedt;
				seed  = (int) seedt + (int) timenow;
				gsl_rng_set (rnghere, seed);
				for(d=0;d<Noccs;d++)
					zz_fd[d] += (gsl_rng_uniform(rnghere)-0.5)/100.;

				gsl_rng_free(rnghere);
			}

		}

		gsl_vector_set(Zz_here,0,meanzd);
		for(d=0;d<Nfac;d++)gsl_vector_set(Zz_here,1+d,0.);
	}
	

	if(sol->tld==0)
		gsl_matrix_free(tld);
	if(sol->gld==0)
		gsl_matrix_free(gld);

	gsl_vector_free(Zz_here);
	gsl_matrix_free(zlde);
	gsl_vector_free(Es);
	for(l=0;l<Nl;l++)
		free(pld_dat[l]);
	free(pld_dat);


	return status;
}


int xprime(gsl_matrix * xp, gsl_matrix * ald, gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_matrix * x, const gsl_vector * Zz){
	// writes xp and also ald, which is the number of applicants to each d conditional on l (2*(Noccs+1) X Noccs)

	int status,j,k,l,d,ll;
	int Nl	= 2*(Noccs+1);
	int JJ1	= Noccs*(Noccs+1);
	double urt,urtp;
	double * findrt;
	status=0;
	double ** pld	= malloc(sizeof(double*)*Nl);
	gsl_matrix * tld,*gld,*sld;
	for(l=0;l<Nl;l++)
		pld[l] = malloc(sizeof(double)*Noccs);

	// define theta and g
	if(sol->tld==0){
		tld = gsl_matrix_calloc(Nl,Noccs);
		status += theta(tld, ss, sys, sol, Zz);
	}
	else
		tld = sol->tld;
	if(sol->gld==0){
		gld = gsl_matrix_calloc(Nl,Noccs);
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
	// zero out ald
	gsl_matrix_set_zero(ald);



	if(printlev>=2){
		urt = 0.0;
		for(l=0;l<Noccs+1;l++){
			// short-term and long-term unemployed
			urt+=gsl_matrix_get(x,l,0)+gsl_matrix_get(x,l,Noccs+1);
		}
	}


	double newdisp = 0.0;
	for(k=0;k<Noccs+1;k++){
		for(j=1;j<Noccs+1;j++){
			if(j!=k)
				newdisp += gsl_matrix_get(sld,k,j-1)*(1.0-tau)*gsl_matrix_get(x,k,j);
		}
	}

	//ald  =  malloc(sizeof(double*)*Nl);
	//for(l=0;l<Nl;l++)
	//	ald[l] = malloc(sizeof(double)*Noccs);

	l = 0;
	for(ll=0;ll<2;ll++){
		for(d=0;d<Noccs;d++)
			gsl_matrix_set(ald,ll*(Noccs+1),d,
						   gsl_matrix_get(gld,ll*(Noccs+1),d)*(gsl_matrix_get(x,0,ll*(Noccs+1)) + (1.0-(double)ll)*newdisp) );
	}
	for(l=1;l<Noccs+1;l++){
		double newexp = 0.0;
		for(j=0;j<Noccs+1;j++)
			newexp =j!=l? tau*gsl_matrix_get(x,j,l)+newexp : newexp;
		for(ll=0;ll<2;ll++){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(ald,ll*(Noccs+1)+l,d,
							   gsl_matrix_get(gld,l+ll*(Noccs+1),d)*( gsl_matrix_get(x,l,ll*(Noccs+1)) +
									   (1.0-(double)ll)*gsl_matrix_get(sld,l,l-1)*(gsl_matrix_get(x,l,l)+newexp) ) );
		}
	}
	for(l=0;l<Nl;l++){
		for(d=0;d<Noccs;d++)
			pld[l][d]= pmatch(gsl_matrix_get(tld,l,d));
	}

	findrt = malloc(sizeof(double)*Nl);
	for(l=0;l<Nl;l++){
		findrt[l]=0.0;
		for(d=0;d<Noccs;d++){
			double pld_ld =pld[l][d];
			double gld_ld =gsl_matrix_get(gld,l,d);
			findrt[l] += pld_ld*gld_ld;
		}
	}

	if(gsl_vector_get(Zz,1)==1){
		printmat("gld.csv",gld);
		printmat("ald.csv",ald);
	}

	//x00

	gsl_matrix_set(xp,0,0,
		(1.0-findrt[0])*(gsl_matrix_get(x,0,0)*(1.0-1.0/bdur) + newdisp)
		);
	gsl_matrix_set(xp,0,Noccs+1,
		(1.0-findrt[Noccs+1])*gsl_matrix_get(x,0,Noccs+1) +  (1.0-findrt[0])*gsl_matrix_get(x,0,0)*1.0/bdur
		);
	//xl0
	for(l=1;l<Noccs+1;l++){
		double newexp = 0.0;
		for(j=0;j<Noccs+1;j++)
			newexp =j!=l? tau*gsl_matrix_get(x,j,l)+newexp : newexp;
		gsl_matrix_set(xp,l,0,
				(1.0-findrt[l])*(gsl_matrix_get(x,l,0)*(1.0-1.0/bdur) + gsl_matrix_get(sld,l,l-1)*(gsl_matrix_get(x,l,l) + newexp))
				);
		gsl_matrix_set(xp,l,Noccs+1,
				(1.0-findrt[l+Noccs+1])*gsl_matrix_get(x,l,Noccs+1) + 1.0/bdur*(1.0-findrt[l])*gsl_matrix_get(x,l,0)
				);
	}


	//xld : d>0
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			if(l!=d+1)
				gsl_matrix_set(xp,l,d+1,
					(1.0-tau)*(1.0-gsl_matrix_get(sld,l,d))*gsl_matrix_get(x,l,d+1) +
							pld[l][d]*gsl_matrix_get(ald,l,d) + pld[l+Noccs+1][d]*gsl_matrix_get(ald,l+Noccs+1,d)
					);
			else{
				double newexp = 0.0;
				for(j=0;j<Noccs+1;j++)
					newexp = j!=d+1 ? tau*gsl_matrix_get(x,j,d+1)+newexp : newexp;
				gsl_matrix_set(xp,l,d+1,
					(1.0-gsl_matrix_get(sld,l,d))*(gsl_matrix_get(x,l,d+1)+newexp )+
							pld[l][d]*gsl_matrix_get(ald,l,d)+pld[l+Noccs+1][d]*gsl_matrix_get(ald,l+Noccs+1,d)
					);
			}
		}
	}
	if(printlev>=2){
		urtp = 0.0;
		for(l=0;l<Noccs+1;l++){
			urtp+=gsl_matrix_get(xp,l,0)+gsl_matrix_get(xp,l,Noccs+1);
		}
	}

	// normalize to sum to 1:
	double xsum = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs+2;d++){
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

	free(findrt);
	for(l=0;l<Nl;l++)
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

int interp_shock_seq(const gsl_matrix * obs_prod, gsl_matrix * mon_prod){
	gsl_interp_accel *acc = gsl_interp_accel_alloc ();
	gsl_spline * prod_spline = gsl_spline_alloc (gsl_interp_cspline, obs_prod->size1);
	gsl_vector * t_dat = gsl_vector_calloc(obs_prod->size1);
	int ti=0,ii,status=0;
	for(ti = 0; ti<obs_prod->size1; ti ++)
		gsl_vector_set(t_dat,ti,(double)ti*12.0);

	for(ii =0;ii<obs_prod->size2;ii++){
		ti=0;
		gsl_vector_const_view obs_prod_i = gsl_matrix_const_column(obs_prod,ii);
		gsl_vector_view mon_prod_i = gsl_matrix_column(mon_prod,ii);

		status += gsl_spline_init (prod_spline, t_dat->data, (&obs_prod_i.vector)->data, obs_prod->size1);
		for(ti=1;ti<t_dat->data[obs_prod->size1-1]; ti++)
			gsl_vector_set( &mon_prod_i.vector, ti, gsl_spline_eval(prod_spline, (double)ti, acc));
	}

	gsl_spline_free (prod_spline);
	gsl_vector_free(t_dat);
	gsl_interp_accel_free (acc);

	return status;
}



int est_fac_pro(gsl_matrix* occ_prod,gsl_vector* mon_Z, struct shock_mats * mats){

	gsl_vector *vzztm1,*vzz,*vresidzz,*coefregs,*lvresidzz;
	gsl_matrix *residzz, * facs, *loadings,*xreg;
	int t,c,fi,d,status = 0;
	int T = mon_Z->size-1;
	int iter,maxiter=1;
	facs 	= gsl_matrix_alloc(T,Nfac);
	loadings = gsl_matrix_alloc(Noccs,Nfac);
	xreg	= gsl_matrix_calloc(T*Noccs,Noccs*(Nfac+1) +1);
	vzztm1 	= gsl_vector_alloc(T*Noccs );
	vzz 	= gsl_vector_alloc(T*Noccs );
	residzz	= gsl_matrix_alloc(T, Noccs);
	vresidzz= gsl_vector_alloc(T*Noccs);
	lvresidzz	= gsl_vector_alloc(T*Noccs);
	coefregs= gsl_vector_alloc(Noccs*(Nfac+1)+1);
	status =0;

	// de-mean everything
	double meanZ = 0.;
	for(t=0;t<T;t++)
		meanZ += gsl_vector_get(mon_Z,t)/(double)T;
	gsl_vector_add_constant(mon_Z,-meanZ);
	for(c=0;c<Noccs;c++){
		double meanzz = 0.;
		for(t=1;t<T+1;t++)
			meanzz += gsl_matrix_get(occ_prod,t,c)/(double)T;
		for(t=1;t<T+1;t++)
			gsl_matrix_set(occ_prod,t,Noccs,gsl_matrix_get(occ_prod,t,c) - meanzz);
	}

	// stack everything
	gsl_matrix_view occ_prod_t = gsl_matrix_submatrix(occ_prod,1,0,T,Noccs);
	vec(&occ_prod_t.matrix,vzz);
	gsl_matrix_view occ_prod_tm1 = gsl_matrix_submatrix(occ_prod,0,0,T,Noccs);
	vec(&occ_prod_tm1.matrix,vzztm1);
	// correlation
	double rhozz = gsl_stats_correlation (vzz->data, vzz->stride, vzztm1->data, vzztm1->stride, vzztm1->size)/(double)Noccs;
	for(t=0;t<T;t++){
		for(c=0;c<Noccs;c++){
			gsl_matrix_set(residzz,t,c,gsl_vector_get(vzz,c*T+t) - rhozz*gsl_vector_get(vzztm1,c*T+t) );
		}
	}
	// Not needed: initial estimate of the factor process
	//status += pca(residzz, Nfac, 0, loadings, facs);

	double rhoZ 	= gsl_stats_lag1_autocorrelation(mon_Z->data,mon_Z->stride,mon_Z->size);
	double sig_eps2_unc = gsl_stats_variance(mon_Z->data,mon_Z->stride,mon_Z->size);
	double sig_eps2 = sig_eps2_unc*(1.-rhoZ*rhoZ);
	status += gsl_matrix_set_col(xreg, Noccs*(Nfac+1), vzztm1);

	double fd_fac_mean[Nfac],fd_fac_scale[Nfac];
	if(fix_fac!=1){
		for(fi=0;fi<Nfac;fi++){
			fd_fac_mean [fi] = 0.;
			fd_fac_scale[fi] = 0.;
			for(c=0;c<simT;c++)
				fd_fac_mean[fi] += gsl_matrix_get(fd_mats->facs,c,fi)/(double)simT;
			for(c=0;c<simT;c++)
				fd_fac_scale[fi] += pow(gsl_matrix_get(fd_mats->facs,c,fi)-fd_fac_mean[fi],2)/(double)simT;
			fd_fac_scale[fi] = sqrt(fd_fac_scale[fi]);
		}
	}
	//if(fix_fac==1)
	// unclear if this is the right thing: now I will extract the factors once... can I just set them to zero?
			maxiter=1;
	for(iter=0;iter<maxiter;iter++){
		if(fix_fac!=1){
			status += pca(residzz,Nfac,1,1,loadings,facs);
			double fac_scale[Nfac],fac_mean[Nfac];
			for(fi=0;fi<Nfac;fi++){
				fac_scale[fi] = 0.;
				fac_mean [fi] = 0.;
				for(c=0;c<simT;c++)
					fac_mean[fi] += gsl_matrix_get(facs,c,fi)/(double)simT;
				for(c=0;c<simT;c++)
					fac_scale[fi] += pow(gsl_matrix_get(facs,c,fi)-fac_mean[fi],2)/(double)simT;
				fac_scale[fi] = sqrt(fac_scale[fi]);
				gsl_vector_view facs_i = gsl_matrix_column(facs,fi);
				gsl_vector_scale(&facs_i.vector,fd_fac_scale[fi]/fac_scale[fi]);
			}
		}
		else{
			gsl_matrix_memcpy(facs,mats->facs);
		}
		for(c=0;c<Noccs;c++){
			gsl_matrix_view xreg_c = gsl_matrix_submatrix(xreg,c*T,c*(Nfac+1),T,Nfac+1);
			for(fi=0;fi<Nfac;fi++){
				gsl_vector_view facs_i = gsl_matrix_column(facs,fi);
				status += gsl_matrix_set_col(&xreg_c.matrix,fi,&facs_i.vector);
			}
			gsl_vector_view Z_sub = gsl_vector_subvector(mon_Z,1,T);
			status += gsl_matrix_set_col(&xreg_c.matrix, Nfac, &Z_sub.vector);
		}

		status += OLS(vzz, xreg, coefregs, vresidzz);
		gsl_vector_sub(vresidzz,vzz);

		if(printlev>=4){
			printmat("xreg.csv",xreg);
			printvec("vzz.csv",vzz);
		}

		gsl_vector_view rZ_iter =  gsl_vector_subvector_with_stride(coefregs, 2, 3, Noccs);
		for(c=0;c<Noccs;c++){
			for(t=0;t<T;t++)
				gsl_matrix_set(residzz,t,c, gsl_vector_get(vzz,c*T+t)-gsl_vector_get(coefregs,Noccs*(Nfac+1))*gsl_vector_get(vzztm1,c*T+t)
						- gsl_vector_get(&rZ_iter.vector,c)*gsl_vector_get(mon_Z,t+1));
		}
		//test distance on vresidzz

		gsl_vector_sub(lvresidzz,vresidzz);

		double dmin,dmax;
		gsl_vector_minmax (lvresidzz, &dmin, &dmax);
		dmax = fabs(dmax); dmin = fabs(dmin);
		dmax = dmax>dmin ? dmax : dmin;
		if(dmax<1e-5)
			break;
		gsl_vector_memcpy(lvresidzz,vresidzz);
				
	}
	// estimate process for factors:
	if(fix_fac!=1)
		status += VARest(facs,mats->Gamma, mats->var_eta);

	// assign values:
	mats->sig_eps2 	= sig_eps2;
	mats->rhozz 	= gsl_vector_get(coefregs,Noccs*(Nfac+1));
	mats->rhoZ		= rhoZ;
	gsl_vector_view vLambda = gsl_vector_subvector(coefregs,0,Noccs*(Nfac+1));
	gsl_matrix_view mLambda = gsl_matrix_view_vector(&vLambda.vector, Noccs, Nfac+1);
	gsl_matrix_memcpy(mats->Lambda,&mLambda.matrix);
	// impose diagonality on the var_zeta matrix?
	if(diag_zeta==1){
		gsl_matrix_set_zero(mats->var_zeta);
		for(c=0;c<Noccs;c++){
			double var_zeta_ct=0.0;
			for(t=0;t<T;t++)
				var_zeta_ct += pow(vresidzz->data[c*T+t],2);
			gsl_matrix_set(mats->var_zeta,c,c,var_zeta_ct/(double)T);
			if(gsl_finite(var_zeta_ct) == 0) status++;
		}
	}
	else if(homosk_zeta==1){
		gsl_matrix_set_zero(mats->var_zeta);
		double var_zeta_i = 0.0;
		for(c=0;c<vresidzz->size;c++){
			var_zeta_i += pow(vresidzz->data[c],2);
		}
		var_zeta_i /= (double)vresidzz->size;
		gsl_vector_view vz_diag = gsl_matrix_diagonal (mats->var_zeta);
		gsl_vector_set_all(&vz_diag.vector,var_zeta_i);
	}
	else{
		for(c=0;c<Noccs;c++){
		for(d=0;d<Noccs;d++){
			double var_zeta_dct=0.0;
			for(t=0;t<T;t++)
				var_zeta_dct += vresidzz->data[c*T+t]*vresidzz->data[d*T+t];
			gsl_matrix_set(mats->var_zeta,c,d,var_zeta_dct/(double)T);
		}
		}
	}
	if(gsl_finite(mats->rhoZ) == 0) status ++;
	if(gsl_finite(mats->rhozz) == 0) status ++;
	for(d=0;d<Nfac;d++){
		for(c=0;c<Nfac;c++){
			if(gsl_finite(gsl_matrix_get(mats->Gamma,d,c) )==0) status++;
			if(gsl_finite(gsl_matrix_get(mats->var_eta,d,c) )==0) status++;
		}
	}
	for(c=0;c<coefregs->size;c++)
		if(gsl_finite(gsl_vector_get(coefregs,c)) == 0) status++;



	// free stuff!
	gsl_matrix_free(facs);
	gsl_matrix_free(loadings);
	gsl_matrix_free(xreg);
	gsl_vector_free(vzztm1);
	gsl_vector_free(vzz);
	gsl_matrix_free(residzz);
	gsl_vector_free(vresidzz);
	gsl_vector_free(lvresidzz);
	gsl_vector_free(coefregs);

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
	int Nl = 2*(Noccs+1);

	Ns = Nl + Noccs*(Noccs+1);
	//		sld			+  gld + tld
	Nc = Noccs*(Noccs+1) + 2*Noccs*Nl;
	Nx = Noccs + Nagf + Nfac*(Nllag+1);
	Notz 	= Nagf +Nfac*(Nllag+1);

	st->ss = gsl_vector_alloc(Ns + Nc);
	st->xss = gsl_matrix_calloc(Noccs+1,Noccs+2);

	// allocate system coefficients and transform:
	st->sys  = malloc(sizeof(struct sys_coef));
	st->sol  = malloc(sizeof(struct sys_sol));
	st->sim  = malloc(sizeof(struct aux_coef));
	st->mats = malloc(sizeof(struct shock_mats));
	//(st->sys)->COV = malloc(sizeof(double)*(Ns+Nc));

	//setup shock matrices and initialize
	(st->mats)->Lambda   = gsl_matrix_calloc(Noccs,Nfac+1);
	(st->mats)->Gamma    = gsl_matrix_calloc(Nfac,Nfac);
	(st->mats)->var_eta  = gsl_matrix_calloc(Nfac,Nfac);
	(st->mats)->var_zeta = gsl_matrix_calloc(Noccs,Noccs);

	//initialize the shock matrices with the value that corresponds to occupation finding rates
	gsl_matrix_memcpy((st->mats)->Gamma,fd_mats->Gamma);
	gsl_matrix_memcpy(st->mats->var_eta , fd_mats->var_eta);
	gsl_matrix_memcpy(st->mats->var_zeta , fd_mats->var_zeta);
	st->mats->rhoZ	= fd_mats->rhoZ;
	st->mats->rhozz	= fd_mats->rhozz;
	st->mats->sig_eps2 = fd_mats->sig_eps2;
	gsl_matrix_memcpy(st->mats->Lambda,fd_mats->Lambda);



/*
 * System is:
 * 	A0 s =  A1 Es'+ A2 c + A3 Z
 * 	F0 c  = F1 Es'+ F2 s'+ F3 Z
 *
 */

	//for(l=0;l<Ns + Nc;l++)
	//	(st->sys)->COV[l] = 1.0;
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
	(st->sol)->ss_wld = gsl_matrix_calloc(Nl,Noccs);
	//(st->sol)->gld	= NULL;
	//(st->sol)->tld	= NULL;
	//(st->sol)->sld 	= NULL;

	(st->sim)->data_moments 	= gsl_matrix_calloc(2,6); // pick some moments to match
	(st->sim)->moment_weights 	= gsl_matrix_calloc(2,6); // pick some moments to match
	(st->sim)->draws 			= gsl_matrix_calloc(simT,Nx); // do we want to make this nsim x simT
	(st->sim)->fd_hist_dat		= gsl_matrix_calloc(simT,Noccs);
	(st->mats)->Zz_hist 		= gsl_matrix_calloc(simT,Nx);

	//simulate the data
	st->sim->fd_hist_dat = gsl_matrix_alloc(simT,Noccs);
	fd_mats->facs    = gsl_matrix_calloc(simT,Nfac);
	fd_mats->ag = gsl_vector_calloc(simT);
	fd_dat_sim(st->sim->fd_hist_dat,fd_mats);


	randn((st->sim)->draws,6071984);
	if(printlev>2)
		printmat("draws.csv",(st->sim)->draws);

	gsl_matrix_set((st->sim)->data_moments,0,1,avg_fnd);
	gsl_matrix_set((st->sim)->data_moments,0,2,chng_pr);
	gsl_matrix_set((st->sim)->data_moments,0,0,avg_wg);
	gsl_matrix_set((st->sim)->data_moments,0,3,avg_urt);
	gsl_matrix_set((st->sim)->data_moments,0,4,avg_elpt);
	gsl_matrix_set((st->sim)->data_moments,0,5,avg_sdsep);
	// row 2 are the coefficients on 3 skills and a constant
	for(i=0;i<Nskill;i++)
		gsl_matrix_set((st->sim)->data_moments,1,i,sk_wg[i]);

	gsl_matrix_set((st->sim)->moment_weights,0,0,1.0);
	gsl_matrix_set((st->sim)->moment_weights,0,1,10.0);
	gsl_matrix_set((st->sim)->moment_weights,0,2,0.50);
	gsl_matrix_set((st->sim)->moment_weights,0,3,5.0);
	gsl_matrix_set((st->sim)->moment_weights,0,4,1.0);
	gsl_matrix_set((st->sim)->moment_weights,0,5,1.0);
	for(i=0;i<Nskill-1;i++)
		gsl_matrix_set((st->sim)->moment_weights,1,i,1.0);
	gsl_matrix_set((st->sim)->moment_weights,1,Nskill-1,2.0);



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
	gsl_matrix_free( (st->sim)->moment_weights);
	gsl_matrix_free( (st->sim)->draws );
	gsl_matrix_free( (st->sim)->fd_hist_dat);
	gsl_matrix_free( (st->mats->Gamma));
	gsl_matrix_free( (st->mats->Lambda));
	gsl_matrix_free( (st->mats->var_eta));
	gsl_matrix_free( (st->mats->var_zeta));
	gsl_matrix_free( (st->mats->Zz_hist));


	gsl_matrix_free(fd_mats->facs);
	gsl_vector_free(fd_mats->  ag);

	free(st->sys);
	free(st->sol);
	free(st->sim);
	free(st->mats);
	//free((st->sys)->COV);

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

	gsl_matrix_set_zero( (st->mats->Zz_hist)); 
	gsl_matrix_set_zero( (st->mats->var_zeta) );
	gsl_matrix_set_zero( (st->mats->var_eta) );
	gsl_matrix_set_zero( (st->mats->Lambda) );
	gsl_matrix_set_zero( (st->mats->Gamma) );

//initialize the shock matrices with the value that corresponds to occupation finding rates
	gsl_matrix_memcpy((st->mats)->Gamma,fd_mats->Gamma);
	gsl_matrix_memcpy(st->mats->var_eta , fd_mats->var_eta);
	gsl_matrix_memcpy(st->mats->var_zeta , fd_mats->var_zeta);
	st->mats->rhoZ	= fd_mats->rhoZ;
	st->mats->rhozz	= fd_mats->rhozz;
	st->mats->sig_eps2 = fd_mats->sig_eps2;
	gsl_matrix_memcpy(st->mats->Lambda,fd_mats->Lambda);


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


double hetero_ev(double eps, void * Vp_in){
	// this is the probability of going to i, and we have to integrate over the values if go to any other j

	int i,j;
	// Vp has Noccs*2 + 1 values.  The first value is the index of the probability we're choosing (i) and the rest are the Values and then the scale parameters
	double intval, Vsum,Vj,Vi, pj, pi;
	double * Vp = (double*) Vp_in;

	Vsum= 0;intval =0.;
	i = (int) Vp[0];
	pi = Vp[i+1+Noccs];
	Vi = Vp[i + 1];
	if(pi > 0. && Vi>0.) {
		for (j = 0; j < Noccs; j++) {
			pj = Vp[j + 1 + Noccs]>0. ? Vp[j + 1 + Noccs]: 1.e-5;
			Vj = Vp[j + 1];
			Vsum = i != j ? exp(-(Vi - Vj - log(eps) * pi) / pj) + Vsum : Vsum;
		}
		//			Lambda					lambda <- Following language of Bhat (1995)
		intval = exp(-Vsum) * exp(-eps);
	}else
		intval = 0.;
	return intval;
}


double dhetero_ev(double eps, void * Vp_in){
	int i,j,k;
	// Vp has Noccs*2 + 2 values.  The first value is the index of the probability we're choosing and the rest are the Values and then the scale parameters
	double intval,pi, pj,pk, Vi,Vj,Vk;
	double * Vp = (double*) Vp_in;
	
	intval = 0;
	i = (int) Vp[0];
	k = (int) Vp[1];
	pi = Vp[i+2+Noccs] >0. ? Vp[i+2+Noccs] : 1e-5;
	Vi = Vp[i+2];
	for(j=0;j<Noccs;j++){
		pj = Vp[j+2+Noccs] >0 ? Vp[j+2+Noccs] : 1e-5;
		Vj = Vp[j+2];
		intval = i != j ? exp( -(Vi - Vj - log(eps)*pi)/pj ) + intval : intval;
	}
	intval = exp(-intval)*exp(-eps);
	pk = Vp[k+2+Noccs]>0.?Vp[k+2+Noccs] : 1e-5;
	Vk = Vp[k+2];
	intval *= -1./pk*exp(-(Vi-Vk-log(eps)*pi)/pk);
	return intval;
}
int sol_memcpy(struct sys_sol * cpy, const struct sys_sol * base){
	int status;
	status =0;
	status += gsl_matrix_memcpy(cpy->gld,base->gld);
	status += gsl_matrix_memcpy(cpy->tld,base->tld);
	status += gsl_matrix_memcpy(cpy->sld,base->sld);
	status += gsl_matrix_memcpy(cpy->ald,base->ald);
//	status += gsl_matrix_memcpy(cpy->ss_wld,base->ss_wld);

//	status += gsl_matrix_memcpy(cpy->P0,base->P0);
//	status += gsl_matrix_memcpy(cpy->P1,base->P1);
//	status += gsl_matrix_memcpy(cpy->P2,base->P2);
	status += gsl_matrix_memcpy(cpy->PP,base->PP);

	return status;
}

int sol_calloc(struct sys_sol * new_sol){
	int status;
	int Nl = 2*(Noccs+1);
	status =0;

//	new_sol->P0	= gsl_matrix_calloc(Ns,Ns);
//	new_sol->P1	= gsl_matrix_calloc(Ns,Ns);
//	new_sol->P2	= gsl_matrix_calloc(Ns,Nx);
	new_sol->PP	= gsl_matrix_calloc(Ns,Nx);
//	new_sol->ss_wld = gsl_matrix_calloc(Nl,Noccs);

	new_sol->gld = gsl_matrix_calloc(Nl,Noccs);
	new_sol->tld = gsl_matrix_calloc(Nl,Noccs);
	new_sol->sld = gsl_matrix_calloc(Nl,Noccs);
	new_sol->ald = gsl_matrix_calloc(Nl,Noccs);

	return status;
}

int sol_free(struct sys_sol * new_sol){
	int status;
	int Nl = 2*(Noccs+1);
	status =0;

//	new_sol->P0	= gsl_matrix_calloc(Ns,Ns);
//	new_sol->P1	= gsl_matrix_calloc(Ns,Ns);
//	new_sol->P2	= gsl_matrix_calloc(Ns,Nx);
	gsl_matrix_free(new_sol->PP);
//	new_sol->ss_wld = gsl_matrix_calloc(Nl,Noccs);

	gsl_matrix_free(new_sol->gld);
	gsl_matrix_free(new_sol->tld);
	gsl_matrix_free(new_sol->sld);
	gsl_matrix_free(new_sol->ald);
	return status;
}



int VARest(gsl_matrix * X,gsl_matrix * coef, gsl_matrix * varcov){
	//gsl_matrix * Ydat = gsl_matrix_alloc(Xdat->size1,Xdat->size2);
	//gsl_matrix_memcpy(Ydat,Xdat);
	gsl_vector * Y = gsl_vector_alloc(X->size1-1);
	gsl_matrix * E = gsl_matrix_alloc(X->size1-1,X->size2);
	int i,t,status=0;
	int incconst = coef->size2 == X->size2+1 ? 1 : 0;

	int Nregors = incconst==1 ? X->size2+1 : X->size2;
	gsl_matrix * Xreg = gsl_matrix_alloc(X->size1-1,Nregors);
	for (t=0;t<X->size1-1;t++){
		for(i=0;i<X->size2;i++)
			gsl_matrix_set(Xreg,t,i,gsl_matrix_get(X,t,i));
		if(incconst==1)
		gsl_matrix_set(Xreg,t,X->size2,1.0);
	}

	for(i=0;i<coef->size2;i++){
		for(t=0;t<X->size1-1;t++)
			Y->data[t] = gsl_matrix_get(X,t+1,i);//Y leads 1
		gsl_vector_view bi = gsl_matrix_row(coef,i);
		gsl_vector_view ei = gsl_matrix_column(E,i);
		status += OLS(Y,Xreg,&bi.vector,&ei.vector);
		gsl_vector_sub(&ei.vector,Y);
		gsl_vector_scale(&ei.vector,-1.0);
	}
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,
			(1.0/(double)E->size1),E,E,0.0,varcov);
	gsl_matrix_free(E);
	gsl_vector_free(Y);
	gsl_matrix_free(Xreg);

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
	nlopt_set_maxeval(opt0, 200*pow(n,2));
	//nlopt_set_maxeval(opt0, 1);
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
		fprintf(f_startf,"Flag = %d\n",status);
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
		fprintf(f_startf,"Flag = %d\n",status);
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
		sig_psi = x[1];
		tau 	= x[2];
		scale_s	= x[3];
		shape_s = x[4];
		effic	= x[5];
		param_offset = 6;
	}
	if(fabs(cal_set)==3){
		phi 	= x[0];
		//sig_psi = x[1];
		//tau 	= x[2];
		scale_s	= x[1];
		shape_s = x[2];
		effic	= x[3];
		param_offset = 4;

		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				chi[l][d] = 1.0;
		}
	}
	if(cal_set==0 || cal_set==2){
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1) {
					chi[l][d] = 0.0;
					for (i = 0; i < Nskill - 1; i++) {
						chi[l][d] += x[param_offset + i] *
									 (gsl_matrix_get(f_skills, d, i + 1) - gsl_matrix_get(f_skills, l - 1, i + 1));
					}
					chi[l][d] += x[param_offset + Nskill - 1];
					chi[l][d] = exp(chi[l][d]);
					//chi[l][d] += x[param_offset+Nskill];
				}
				else{
					chi[l][d] = 1.;
				}
			}
		}
		double chi_lb = 0.25;
		double chi_ub = 1.2;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					if(chi[l][d]<=chi_lb) chi[l][d] = chi_lb;
					if(chi[l][d]>=chi_ub) chi[l][d] = chi_ub;
				}
			}
		}
		double mean_chi = 0.0;
		for(d=0;d<Noccs;d++){
			for(l=1;l<Noccs+1;l++)
				mean_chi = l!=d+1 ? chi[l][d] + mean_chi : mean_chi;
		}
		for(d=0;d<Noccs;d++)
			chi[0][d] = mean_chi/(double)(Noccs*(Noccs-1));
		for(d=0;d<Noccs;d++)
			chi[d+1][d] = 1.0;
	}

	b[1] = brt;
	b[0] = 0.0;
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
	 		chi[l][d] = chi[l][d]<b[1] ? b[1] : chi[l][d];
		chi[0][d] =chi[0][d]<b[0]? b[0] :  chi[0][d];
	}

	// check on whether to make all occupations symmetric for debugging purposes
	if(sym_occs == 1){
		for(d=0;d<Noccs;d++){
			for(l=1;l<Noccs+1;l++)
				chi[l][d] = l-1 != d ? 0.8 : 1.0;
			l =0 ;
			chi[l][d] = 0.75;
		}
	}

	if(printlev>=1){
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
	// clear the z-process and everything else
	clear_econ(st);

	struct aux_coef  * simdat	=  st->sim;
	struct sys_coef  * sys		= st->sys;
	struct sys_sol   * sol		= st->sol;
	struct shock_mats* mats0	= st->mats;
	gsl_matrix * xss = st->xss;
	gsl_vector * ss = st->ss;
	double dist;

	set_params(x, st->cal_set);


	int param_offset = st->cal_set == 1 || st->cal_set == 0 ? ((st->sim)->data_moments)->size2 : 0;


	if(printlev>=1){
		// print where I am searching
		calhist = fopen(calhi_f,"a+");
		for(i=0;i<n;i++)
			fprintf(calhist,"%f,",x[i]);
		fprintf(calhist,"\n");
		fclose(calhist);
	}
	//printlev = 3;
	//verbose = 3;
	status 	= sol_ss(ss,0,xss,sol);
	if(printlev>=0 && status!=0) printf("Steady state not solved \n");
	status += sys_def(ss,sys,mats0);
	if(printlev>=0 && status!=0) printf("System not defined\n");
	status += sol_dyn(ss,sol,sys,0);
	if(printlev>=0 && status!=0) printf("System not solved\n");

	if(status == 0){
		status += sol_zproc(st,ss,xss,mats0);
		status += sim_moments(st,ss,xss);
		Nsolerr =0;
	}
	else{
		++ Nsolerr;
		gsl_matrix_scale(simdat->data_moments,5.0);
		gsl_matrix_set_all(simdat->data_moments,st->cal_worst/(double)st->n);
		solerr = fopen(soler_f,"a+");
		fprintf(solerr,"System did not solve at: (%f,%f,%f,%f,%f,%f)\n",x[0],x[1],x[2],x[3],x[4],x[5]);
		fclose(solerr);
	}
	int Nmo = (simdat->data_moments)->size2;

	if(status == 0){
		dist = 0.0;
		if(st->cal_set != 2){
			for(i=0;i<Nmo;i++)
				dist += gsl_matrix_get(simdat->moment_weights,0,i)*pow(gsl_matrix_get(simdat->data_moments,0,i),2);
			if(verbose>=1){
				if(st->cal_set!=3) printf("At phi=%f,sig_psi=%f,tau=%f,scale_s=%f,shape_s=%f,effic=%f\n",x[0],x[1],x[2],x[3],x[4],x[5]);
				if(st->cal_set==3) printf("At phi=%f,scale_s=%f,shape_s=%f,effic=%f\n",x[0],x[1],x[2],x[3]);
				if(status>0) printf("System did not solve \n");
				printf("Calibration distance is %f\n",dist);
			}
		}
		if(st->cal_set!=1 && st->cal_set != 3){
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
		if(dist>st->cal_worst)
			st->cal_worst = dist;
		if(dist<st->cal_best)
			st->cal_best = dist;
	}
	else{
		dist = 1.5*st->cal_worst;
	}

	if(printlev>=1){
		// print the result of what I found here
		calhist = fopen(calhi_f,"a+");
		fprintf(calhist,"%f,",dist);
		if(st->cal_set!=2){
			for(i=0;i<n;i++)
				fprintf(calhist,"%f,",(simdat->data_moments)->data[i]);
		}
		if(st->cal_set!=1 && st->cal_set !=3){
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
	int alg = nlopt_get_algorithm(st->opt0);
	if(exploding>=5 && alg<100)
		nlopt_force_stop(st->opt0);
	if(Nsolerr>2 || ((dist>1000 && Nsolerr>1 )  && alg<100))
		nlopt_force_stop(st->opt0);


	return dist;
}

