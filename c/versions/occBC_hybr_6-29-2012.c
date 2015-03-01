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

#define _MKL_USE // this changes LAPACK options to use faster MKL versions of LAPACK
//#define _MPI_USE

#ifdef _MKL_USE
#include "mkl_lapacke.h"
#endif
#ifdef _MPI_USE
#include <mpi.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "utils.h"
#include <math.h>
#include <time.h>
#include <nlopt.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_qrng.h>


int Nthread 	= 8;
int Nobs,simT, nsim, Nskill;
int nshock;
int neq;
int verbose 	= 1;
int printlev 	= 3;
int use_anal	= 1;
int opt_alg		= 1; // 10x => Nelder-Meade, 5x => Subplex, o.w => BOBYQA
int polish_alg	= 0;
double ss_tol	= 1e-7; // tighten this with iterations?
double dyn_tol	= 1e-7; // ditto?
int homosk_zeta	= 0;
int diag_zeta	= 1;
int gsl_fin_diffs=0;
int csize		= 8;
FILE * calhist, *solerr,*simerr ;


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

double 	beta= 0.9967;	// monthly discount
double	nu 	= 0.0; 		// switching cost --- I don't need this!
double 	sig_psi	= 1.865434;//3.671249; 	// higher reduces the switching prob

double *	b; // unemployment benefit


double ** chi;
double 	tau 	= 0.026561;//0.020659;
double 	kappa	= 0.45;//.1306; corresponds to HM number
double 	phi		= .320825;//0.455997;	// : log(1/UE elasticity -1) = phi*log(theta)= log(1/.38-1)/log(5)
double 	fm_shr	= 0.9;	// firm's surplus share
double 	effic	= 1.074104;//1.162658;
double 	sbar	= 0.026949;	// will endogenize this a la Cheremukhin (2011)

double 	scale_s	= 0.03;
double 	shape_s	= 0.010;


double rhoZ		= 0.9732;
double rhozz	= 0.99;

double sig_eps	= 0.00002557;//7.2445e-7;  (using the FRED Number, rather than mine)
double sig_zet	= 1e-5;//1.0e-6; (adjustment for the FRED number)

gsl_vector * cov_ze;
gsl_matrix * GammaCoef, *var_eta; 	//dynamics of the factors
gsl_matrix * LambdaCoef,*var_zeta;	//factors' effect on z
gsl_matrix * LambdaZCoef;			//agg prod effect on z
gsl_matrix * f_skills;				//skills attached to occupations

// data moments that are not directly observable
double avg_sep	= 0.019;
double avg_fnd	= 0.3323252;
double avg_urt	= 0.055;
double avg_wg	= 0.006;	// this is the 5 year wage growth rate: make tau fit this
double avg_elpt	= 0.38;		// elasticity of p wrt theta, Shimer 2005 estimate
double avg_sdsep= 0.01256;	//the average standard deviation of sld across occupations
double med_dr	= 13;
double chng_pr	= 0.45;
double sk_wg[]	= {0.0173749,.0462705,-0.03424,0.2024};//{0.0117762,-0.0351453,0.01041,0.2074417}; //abs 1 - 3
double Zfcoef[] = {0.0233,-0.0117};

			// old : {-0.0028,-0.0089,0.0355,0.0083};

struct aux_coef{
	gsl_matrix * data_moments;
	gsl_matrix * moment_weights;
	gsl_matrix * draws;
	gsl_matrix * Xdat;
};


struct dur_moments{
	double pr6mo;
	double sd_dur;
	double E_dur;
	double chng_wg;
	gsl_matrix * dur_qtl;
	gsl_vector * Ft;
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
	gsl_matrix *gld,*tld,*sld;
};
struct st_wr{
	struct sys_coef * sys;
	struct sys_sol * sol;
	struct aux_coef * sim;
	unsigned cal_set,n;
	const double * x;
};

// Solve the steady state
int sol_ss(gsl_vector * ss, gsl_matrix * xss, struct sys_coef *sys);


// Solving the dynamic model
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
int ss_moments(struct aux_coef * ssdat, gsl_vector * ss, gsl_matrix * xss);
int sim_moments(struct st_wr * st, gsl_vector * ss,gsl_matrix * xss);
int dur_dist(struct dur_moments * s_mom, const gsl_matrix * fnd_l_hist, const gsl_matrix * x_u_hist);

int TGR(gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		double *urt, double *opt_urt,
		gsl_vector * ss,struct sys_sol * sol,struct sys_coef * sys);

// Utilities
int Es_cal(gsl_vector * Es, const gsl_vector * ss_W, const gsl_matrix * PP, const gsl_vector * Zz);
int set_params(const double * x,int cal_set);
int QZwrap(gsl_matrix * A, gsl_matrix *B, //gsl_matrix * Omega, gsl_matrix * Lambda,
		gsl_matrix * Q, gsl_matrix * Z, gsl_matrix * S, gsl_matrix * T);
int QZdiv(double stake, gsl_matrix * S,gsl_matrix * T,gsl_matrix * Q,gsl_matrix * Z);
int QZswitch(int i, gsl_matrix * A,gsl_matrix * B,gsl_matrix * Q,gsl_matrix * Z);

void VARest(gsl_matrix * Xdat,gsl_matrix * coef, gsl_matrix * varcov);

double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg0, int alg1);
double cal_dist_df(unsigned n, const double *x, double *grad, void* params);
double cal_dist(unsigned n, const double *x, double *grad, void* params);
int cal_dist_wrap(const gsl_vector * x, void* params, gsl_vector * f);

double quad_solve(unsigned n, const double *x, double *grad, void*params){
	// to test the multi-start
	int i;
	double obj=0.0;
	for(i=0;i<n;i++)
		obj+= (double)(i+1) * pow(x[i],2);
	return obj;
}

int main(){
	int i,l,d,status;

	status = 0;

	omp_set_num_threads(Nthread);
	//initialize parameters
	Nskill = 3;

	f_skills = gsl_matrix_alloc(Noccs,6);
	readmat("f_skills.csv",f_skills);


	b	= (double*)malloc(sizeof(double)*2);
	chi = (double**)malloc(sizeof(double*)*Noccs+1);
	for(l=0;l<Noccs+1;l++)
		chi[l] = (double*)malloc(sizeof(double*)*Noccs);
	double chi_coef0[] = {0.02,0.015,0.01};
	set_params(chi_coef0, 2);


	cov_ze = gsl_vector_calloc(Noccs);
	//readvec("cov_ze.csv",cov_ze);
	GammaCoef	= gsl_matrix_calloc(Nfac,Nfac*Nglag);
	LambdaCoef	= gsl_matrix_calloc(Noccs,Nfac*(Nllag+1));
	LambdaZCoef = gsl_matrix_calloc(Noccs,Nagf);
	gsl_matrix * LamRead		= gsl_matrix_calloc(Noccs,Nagf+Nfac*(Nllag+1));
	var_eta		= gsl_matrix_calloc(Nfac,Nfac);
	var_zeta	= gsl_matrix_calloc(Noccs,Noccs);
	readmat("Gamma.csv",GammaCoef);
	readmat("Lambda.csv",LamRead); // this one file has both Lambda nad coefs on Z
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
	}
	else
		readmat("var_zeta.csv",var_zeta);

	if(printlev>=3){
		printmat("readGamma.csv",GammaCoef);
		printmat("readLambda.csv",LambdaCoef);
	}

	Ns = Noccs+1 + Noccs*(Noccs+1);//if including x: pow(Noccs+1,2) + Noccs+1 + 2*Noccs*(Noccs+1);
	Nc = 3*Noccs*(Noccs+1);
	Nx = Noccs + Nagf + Nfac*(Nllag+1);
	Notz 	= Nagf +Nfac*(Nllag+1);

	gsl_vector* ss = gsl_vector_alloc(Ns + Nc);

	// allocate system coefficients and transform:

	struct sys_coef sys;

	sys.COV = (double*)malloc(sizeof(double)*(Ns+Nc));
/*
 * System is:
 * 	A0 s =  A1 Es'+ A2 c + A3 Z
 * 	F0 c  = F1 Es'+ F2 s'+ F3 Z
 *
 */

	for(l=0;l<Ns + Nc;l++)
		sys.COV[l] = 1.0;
	sys.A0	= gsl_matrix_calloc(Ns,Ns);
	sys.A1	= gsl_matrix_calloc(Ns,Ns);
	sys.A2	= gsl_matrix_calloc(Ns,Nc);
	sys.A3	= gsl_matrix_calloc(Ns,Nx);

	sys.F0	= gsl_matrix_calloc(Nc,Nc);
	sys.F1	= gsl_matrix_calloc(Nc,Ns);
	sys.F2	= gsl_matrix_calloc(Nc,Ns);
	sys.F3	= gsl_matrix_calloc(Nc,Nx);

	sys.N  = gsl_matrix_calloc(Nx,Nx);
	sys.S	= gsl_matrix_calloc(Nx,Nx);

	gsl_matrix * xss = gsl_matrix_calloc(Noccs+1,Noccs+1);

	struct sys_sol sol;
/*
	P0*s' = P2*x + P1*s
*/
	sol.P0	= gsl_matrix_calloc(Ns,Ns);
	sol.P1	= gsl_matrix_calloc(Ns,Ns);
	sol.P2	= gsl_matrix_calloc(Ns,Nx);
	sol.PP	= gsl_matrix_calloc(Ns,Nx);
	sol.gld	= 0;
	sol.tld	= 0;
	simT = 2000;
	struct aux_coef simdat;
	simdat.data_moments 	= gsl_matrix_calloc(2,6); // pick some moments to match
	simdat.moment_weights 	= gsl_matrix_calloc(2,6); // pick some moments to match
	simdat.draws 			= gsl_matrix_calloc(simT,Nx); // do we want to make this nsim x simT

	randn(simdat.draws,6071984);
	if(printlev>2)
		printmat("draws.csv",simdat.draws);

	gsl_matrix_set(simdat.data_moments,0,1,avg_fnd);
	gsl_matrix_set(simdat.data_moments,0,2,chng_pr);
	gsl_matrix_set(simdat.data_moments,0,0,avg_wg);
	gsl_matrix_set(simdat.data_moments,0,3,avg_urt);
	gsl_matrix_set(simdat.data_moments,0,4,avg_elpt);
	gsl_matrix_set(simdat.data_moments,0,5,avg_sdsep);
	// row 2 are the coefficients on 4 skills and a constant
	for(i=0;i<Nskill+1;i++)
		gsl_matrix_set(simdat.data_moments,1,i,sk_wg[i]);

	gsl_matrix_set(simdat.moment_weights,0,0,1.0);
	gsl_matrix_set(simdat.moment_weights,0,1,10.0);
	gsl_matrix_set(simdat.moment_weights,0,2,0.50);
	gsl_matrix_set(simdat.moment_weights,0,3,5.0);
	gsl_matrix_set(simdat.moment_weights,0,4,1.0);
	gsl_matrix_set(simdat.moment_weights,0,5,1.0);

	gsl_matrix_set(simdat.moment_weights,1,0,1.0);
	gsl_matrix_set(simdat.moment_weights,1,1,1.0);
	gsl_matrix_set(simdat.moment_weights,1,2,1.0);
	gsl_matrix_set(simdat.moment_weights,1,3,10.0);

	struct st_wr st;
	st.sim 	= &simdat;
	st.sol 	= &sol;
	st.sys 	= &sys;
	st.cal_set = 1;

	calhist = fopen("calhist.txt","w+");
	solerr = fopen("solerr.log","w+");
	simerr = fopen("simerr.log","w+");
	/* Calibration Loop!
	*/
	double x0_0[]	= {phi	,sig_psi	,tau	,scale_s	,shape_s	,effic
			,sk_wg[0]		,sk_wg[1]		,sk_wg[2]		,sk_wg[3]};
	double lb_0[]	= {0.15	,0.005		,0.001	,0.015		,0.001		,0.1
			,sk_wg[0]*(0.7)	,sk_wg[1]*(0.7)	,sk_wg[2]*(0.7)	,sk_wg[3]*(0.7)}; // what to do about sig_psi?  Skipping that right now
	double ub_0[]	= {1.0	,7.5		,0.05	,0.04		,0.25		,1.5
			,sk_wg[0]*(1.3)	,sk_wg[1]*(1.3)	,sk_wg[2]*(1.3)	,sk_wg[3]*(1.3)};

	for(i=0;i<5+Nskill+1;i++){
		if (lb_0[i]>=ub_0[i]){
			double lbi = lb_0[i];
			lb_0[i] = ub_0[i];
			ub_0[i] = lbi;
		}
	}
	gsl_vector_view x0vw = gsl_vector_view_array (x0_0, 10);
	gsl_vector_view lbvw = gsl_vector_view_array (lb_0, 10);
	gsl_vector_view ubvw = gsl_vector_view_array (ub_0, 10);


	double objval = 100.0;
	int printlev_old = printlev;
	printlev =1;
	//first calibrate the parameters, fix those and then find chi
	int param_off = 6;
	st.cal_set = 1;
	st.n	= param_off;

	gsl_vector_view x0_1 = gsl_vector_subvector(&x0vw.vector, 0, param_off);//{x0_0[0],x0_0[1],x0_0[2],x0_0[3],x0_0[4]};
	gsl_vector_view lb_1 = gsl_vector_subvector(&lbvw.vector, 0, param_off);//{lb_0[0],lb_0[1],lb_0[2],lb_0[3],lb_0[4]};
	gsl_vector_view ub_1 = gsl_vector_subvector(&ubvw.vector, 0, param_off);//{ub_0[0],ub_0[1],ub_0[2],ub_0[3],ub_0[4]};

	calhist = fopen("calhist1.txt","w+");
	fprintf(calhist,"phi,sig_psi,tau,sbar,effic,b1,b2,b3,b0\n");
	fprintf(calhist,"dist,wg,fnd,chng,urt,elpt,b1,b2,b3,b0\n");

	fprintf(calhist,"***Beginning Calibration of param set 1***\n");


	solerr = fopen("solerr1.log","w+");
	fprintf(solerr,"Errors while solving model with param set 1\n");

	simerr = fopen("simerr1.log","w+");
	fprintf(simerr,"Errors in policies while simulating param set 1\n");

	if(verbose >= 1)printf("Starting to calibrate on variable subset 1\n");
	//							f		lb ub n x0 param, ftol xtol, ctol , algo
	//							quad_solve


	objval = bndCalMinwrap(&cal_dist,(lb_1.vector).data,(ub_1.vector).data,param_off ,(x0_1.vector).data,&st,1e-7,1e-7,0.0,opt_alg,0);
	if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st.cal_set);
	fclose(calhist);
	fclose(solerr);
	fclose(simerr);

	// FIND CHI
	/*
	st.n	= Nskill;
	gsl_vector_view x0_2 = gsl_vector_subvector(&x0vw.vector, param_off,st.n);
	gsl_vector_view lb_2 = gsl_vector_subvector(&lbvw.vector, param_off,st.n);
	gsl_vector_view ub_2 = gsl_vector_subvector(&ubvw.vector, param_off,st.n);
	st.cal_set = 2;

	calhist = fopen("calhist2.txt","w+");
	fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3,b0\n");
	fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3,b0\n");
	fprintf(calhist,"***Beginning Calibration of param set 2***\n");

	solerr = fopen("solerr2.log","w+");
	fprintf(solerr,"Errors while solving model with param set 2\n");

	simerr = fopen("simerr2.log","w+");
	fprintf(simerr,"Errors in policies while simulating param set 2\n");

	if(verbose >= 1)printf("Starting to calibrate on variable subset 2\n");
	//							f		lb ub n x0 param, ftol xtol, ctol , algo
	objval = bndCalMinwrap(&cal_dist,(lb_2.vector).data,(ub_2.vector).data,Nskill+1,(x0_2.vector).data,&st,1e-7,1e-7,0.0,opt_alg,polish_alg);
	if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st.cal_set);
	fclose(calhist);
	fclose(solerr);
	fclose(simerr);

	// THE WHOLE SHEBANG
	st.cal_set = 0;
	st.n	= param_off+4;
	calhist = fopen("calhist0.txt","w+");
	fprintf(calhist,"phi,sig_psi,tau,effic,b1,b2,b3,b0\n");
	fprintf(calhist,"dist,wg,fnd,chng,elpt,b1,b2,b3,b0\n");
	fprintf(calhist,"***Beginning Calibration of param set 0***\n");

	solerr = fopen("solerr0.log","w+");
	fprintf(solerr,"Errors while solving model with param set 0\n");

	simerr = fopen("simerr0.log","w+");
	fprintf(simerr,"Errors in policies while simulating param set 0\n");

	if(verbose >= 1)printf("Starting to calibrate on variable subset 0\n");
	//							f		lb ub n x0 param, ftol xtol, ctol , algo
	objval = bndCalMinwrap(&cal_dist,(lbvw.vector).data,(ubvw.vector).data,st.n,(x0vw.vector).data,&st,1e-7,1e-7,0.0,opt_alg,polish_alg);
	if (verbose >= 1)printf("Final distance : %f from param set %d\n",objval, st.cal_set);


	printlev =printlev_old;

	*/
	status += sol_ss(ss,xss,&sys);
	if(verbose>=1 && status ==0) printf("Successfully computed the steady state\n");
	if(verbose>=0 && status ==1) printf("Broke while computing steady state\n");
	status += sys_def(ss,&sys);
	if(verbose>=1 && status >=1) printf("System not defined\n");
	if(verbose>=0 && status ==0) printf("System successfully defined\n");



	if(verbose>=2) printf("Now defining the 1st order solution to the dynamic model\n");
	int t;
	if(verbose>=1) t = clock();
	status += sol_dyn(ss, &sol, &sys);
	if(verbose>=1 && status ==0){
		printf("System successfully solved\n");
		printf ("It took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	}
	if(verbose>=0 && status >=1) printf("System not solved\n");



	//status += ss_moments(&simdat, ss, xss);
	status += sim_moments(&st,ss,xss);

	//	TGR experiment
	gsl_vector * u_dur_dist = gsl_vector_alloc(5);
	gsl_vector * opt_dur_dist = gsl_vector_alloc(5);
	gsl_vector * fnd_dist	= gsl_vector_alloc(Noccs+1);
	gsl_vector * opt_fnd_dist = gsl_vector_alloc(Noccs+1);
	double urt,opt_urt;
	status +=  TGR(u_dur_dist,opt_dur_dist,fnd_dist,opt_fnd_dist,&urt,&opt_urt,ss,&sol,&sys);



	//status += sim_moments(&simdat,ss,xss,&sol,&sys);


	gsl_matrix_free(simdat.draws);
	gsl_matrix_free(simdat.data_moments);
	gsl_matrix_free(simdat.moment_weights);

	gsl_matrix_free(sys.A0);gsl_matrix_free(sys.A1);gsl_matrix_free(sys.A2);gsl_matrix_free(sys.A3);
	gsl_matrix_free(sys.F0);gsl_matrix_free(sys.F1);gsl_matrix_free(sys.F2);gsl_matrix_free(sys.F3);
	gsl_matrix_free(sys.N);gsl_matrix_free(sys.S);free(sys.COV);


	gsl_matrix_free(sol.P1);gsl_matrix_free(sol.P2);gsl_matrix_free(sol.P0);gsl_matrix_free(sol.PP);

	gsl_vector_free(ss);gsl_matrix_free(xss);

	gsl_matrix_free(GammaCoef);	gsl_matrix_free(LambdaCoef);
	gsl_matrix_free(var_eta); gsl_matrix_free(var_zeta);

	gsl_vector_free(u_dur_dist);gsl_vector_free(opt_dur_dist);	gsl_matrix_free(f_skills);


	fclose(calhist);
	fclose(solerr);
	fclose(simerr);

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

	
	int Ns,Nc,Nx, i, maxrecur=50000; // these might be different from the globals of the same name.... but should not be
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
	int * ipiv = (int*)malloc(sizeof(int)* ((sys->F0)->size2) );
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
	gsl_matrix_sub(sol->P2,A2invF0F3);
	gsl_matrix_free(A2invF0F3);gsl_matrix_free(invF0F3);

	if(printlev>=2){
		printmat("P0.csv",sol->P0);
		printmat("P1.csv",sol->P1);
		printmat("P2.csv",sol->P2);
	}
	nonfin += isfinmat(sol->P0);
	nonfin += isfinmat(sol->P1);
	nonfin += isfinmat(sol->P2);
	if(printlev>=2)
		fprintf(solerr,"Non-finite %d times in P0-P2\n",nonfin);


	// and forward looking for PP

	gsl_matrix * P1invP0= gsl_matrix_calloc(Ns,Ns);
#ifdef _MKL_USE
	// will have to solve P0^T X^T = P1 instead of  X P0 = P1

	gsl_matrix * P0tLU =gsl_matrix_calloc((sol->P0)->size1,(sol->P0)->size2);
	// this is now the transpose
	 gsl_matrix_transpose_memcpy(P0tLU,sol->P0);
	int * ipiv_P0 = (int*)malloc(sizeof(int)* ((sol->P0)->size2) );
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
	gsl_matrix * invP1PP 		= gsl_matrix_calloc(Ns,Nx);
#ifndef _MKL_USE
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0,sol->P2,0.0,P1invP0P2N);
#endif
#ifdef _MKL_USE
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,P1invP0,sol->P2,0.0,P1invP0P2N);
#endif

	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0P2N,sys->N,0.0,P1invP0P2NN);
	gsl_matrix_memcpy(P1invP0P2N,P1invP0P2NN);
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
		gsl_matrix_add(invP1PP,P1invP0P2N);
		mnorml = mnorm;
	}
	//status = i<maxrecur-1? status : status+1;
	if(i>=maxrecur-1){
		fprintf(solerr,"Forward expectations stopped at %f, did not converge\n",mnorm);;
	}
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,invP1PP,0.0,sol->PP);
	gsl_matrix_add(sol->PP,sol->P2);


	gsl_matrix_free(invP1PP);

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

int sol_ss(gsl_vector * ss, gsl_matrix * xss, struct sys_coef * sys){
	// will solve the ss with
	//vars = [ x U_l0 U_ld J_ld gld wld thetald sld] in row-major order, i.e. {(0,1),(0,2),...,(J,J)}

	int status,l,d,dd,ll,iter,itermax,i;
	
	gsl_vector * W_l0,*W_ld;
	gsl_vector * gld,*lgld,*wld,*thetald,*findrt, *sld;
	gsl_vector * ret_d;
	gsl_matrix * pld,*x;//,*lx;

	x		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	//lx		= gsl_matrix_calloc(Noccs+1,Noccs+1);
	W_l0 	= gsl_vector_calloc(Noccs+1);
	W_ld 	= gsl_vector_calloc(Noccs+Noccs*Noccs);
	wld 	= gsl_vector_calloc(Noccs+Noccs*Noccs);
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
			gsl_vector_set(gld,l*Noccs+d,1.0/(double)Noccs);
			gsl_vector_set(wld,l*Noccs+d, phi*chi[l][d]);
			gsl_vector_set(thetald,l*Noccs+d,pow(pow(0.25/effic,-phi)-1.0,1.0/phi));  //pow(kappa/(effic*(1.0-phi)*(chi[l][d] - gsl_vector_get(wld,l*Noccs+d))),-1.0/phi));
			gsl_vector_set(sld,l*Noccs+d,sbar);
			gsl_matrix_set(pld,l,d,effic*thetald->data[l*Noccs+d]/pow(1.0+pow(thetald->data[l*Noccs+d],phi),1.0/phi) );
			if(gsl_matrix_get(pld,l,d)>1.0)
				gsl_matrix_set(pld,l,d,1.0);
			W_l0->data[l] += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*(chi[l][d] - bl);
		}
		W_l0->data[l] += bl;
	}
	itermax = 10000;
	double maxdist=0;
	if(printlev>=2){
		printvec("wld0.csv",wld);
		printvec("gld0.csv",gld);
		printvec("thetald0.csv",thetald);
		printvec("W_l0_0.csv",W_l0);
	}

	for(iter=0;iter<itermax;iter++){
		for(d=0;d<Noccs;d++){
			l = d+1;
			double sepld = sld->data[l*Noccs+d];
			gsl_vector_set(W_ld,l*Noccs+d,((1.0-sepld)*chi[l][d] + sepld*W_l0->data[l])/(1.0-(1.0-sepld)*beta));
		}
		// W_ld
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
			double sepld = sld->data[l*Noccs+d];
			if(l-1!=d){
				gsl_vector_set(W_ld,l*Noccs+d,
					((1.0-tau)*((1.0-sepld )*chi[l][d]+sepld *W_l0->data[l] ) + tau*W_ld->data[(d+1)*Noccs+d]  )
						/(1.0-(1.0-sepld)*(1.0-tau)*beta));
			}
			}
		}

		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				W_ld->data[l*Noccs+d] =W_ld->data[l*Noccs+d]< W_l0->data[l] ? W_l0->data[l] : W_ld->data[l*Noccs+d];
		}

		if(printlev>=2 && iter==0) printvec("W_ld_0.csv",W_ld);
		// update policies
		gsl_vector_memcpy(lgld, gld); // for convergence check later

		//either make ret_d local, private or do not parallelize
		//#pragma omp parallel for default(shared) private(d,l,dd,ret_d)
		for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			for(d=0;d<Noccs;d++){
				double nud = l == d+1? 0:nu;
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l];
				Wdif = Wdif<0.0 ?  0.0 : Wdif;
				double pld_ld =gsl_matrix_get(pld,l,d);
				double post = pld_ld > 0 ? -kappa*gsl_vector_get(thetald,l*Noccs+d)/pld_ld : 0.0;
				double arr_d = (chi[l][d] -nud - bl + post + beta*Wdif);
				arr_d = arr_d > 0.0 ? arr_d : 0.0;
				gsl_vector_set(ret_d,d,
					pld_ld*arr_d
				);

			}
			double sexpret_dd =0.0;
			for(dd=0;dd<Noccs;dd++)
				sexpret_dd = gsl_vector_get(ret_d,dd)>0 ?
						exp(sig_psi*gsl_vector_get(ret_d,dd)) + sexpret_dd:sexpret_dd;

			for(d=0;d<Noccs;d++){
				//gld

				// known return in occ d
				double gg = gsl_vector_get(ret_d,d)>0 ? exp(sig_psi*gsl_vector_get(ret_d,d))/sexpret_dd : 0.0;
				gsl_vector_set(gld,l*Noccs+d,gg);

				//theta
				double nud 	= l-1 !=d ? nu : 0.0;
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l];
				Wdif = Wdif<0.0 ? 0.0 : Wdif;
				double tinner = chi[l][d] - nud - bl +beta*Wdif;
				//tinner = tinner > 0.0 ? tinner : 0.0;

				// theta is actually theta^phi
				double theta =pow(kappa
						/(effic*fm_shr*tinner),-phi)-1.0 ;
				if(effic>1.0)
					theta = theta>1.0/(pow(effic,phi)-1.0)? 1.0/(pow(effic,phi)-1.0) : theta;
				if(tinner > 0.0 && gsl_finite(theta) && theta>0.0 ){
						gsl_vector_set(thetald,l*Noccs+d,pow(theta,1.0/phi));}
				else
					gsl_vector_set(thetald,l*Noccs+d,0.0);
				//theta = theta > maxtss ? maxtss: theta;

				// sld
				double cutoff = -(chi[l][d] + beta*W_ld->data[l*Noccs+d] - W_l0->data[l]);
				cutoff = cutoff >0.0 ? 0.0 : cutoff;
				double sep_ld = scale_s*exp(shape_s*cutoff);
				gsl_vector_set(sld,l*Noccs+d,sep_ld);

			}
			// adds to 1
			double gsum = 0.0;
			for(d =0 ;d<Noccs;d++)	gsum += gld->data[l*Noccs+d];
			for(d =0 ;d<Noccs;d++)	gld->data[l*Noccs+d] /= gsum;
		}

		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				gsl_matrix_set(pld,l,d,effic*thetald->data[l*Noccs+d]/pow(1.0+pow(thetald->data[l*Noccs+d],phi),1.0/phi) );

			}
		}
		for(l=0;l<Noccs+1;l++){
			double findrt_ld =0.0;
			for(d=0;d<Noccs;d++)
				findrt_ld += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d);
			gsl_vector_set(findrt,l,findrt_ld);
		}
		// W_l0
		gsl_vector * lW_l0 = gsl_vector_calloc(Noccs+1);
		gsl_vector_memcpy(lW_l0,W_l0);
		maxdist = 0.0;
		for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			double W_0 =0.0;
			for(d=0;d<Noccs;d++){
				double nud = l==d+1? 0.0:nu;
				double post = gsl_matrix_get(pld,l,d)>0? - kappa*thetald->data[l*Noccs+d]/gsl_matrix_get(pld,l,d) : 0.0;
				W_0 +=gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*(chi[l][d] - nud
						+ post
						+ beta*gsl_vector_get(W_ld,l*Noccs+d));
			}
			W_0 += (1.0-gsl_vector_get(findrt,l))*bl;
			W_0 /= (1.0 - beta*(1.0-gsl_vector_get(findrt,l)));
			W_l0->data[l] = W_0;
			if(fabs(W_0 - lW_l0->data[l])>maxdist)
				maxdist = fabs(W_0 - lW_l0->data[l])/lW_l0->data[l];
		}
		if(maxdist < ss_tol)
			break;

		if(printlev>=4){
			printvec("W_l0_i.csv",W_l0);
			printvec("W_ld_i.csv",W_ld);
			printvec("sld_i.csv",sld);
			printvec("gld_i.csv",gld);
			printvec("thetald_i.csv",thetald);
			printvec("findrt_i.csv",findrt);
		}

	}// end iter=0:maxiter
	status = iter<itermax? 0 : 1;
	if(printlev>=2){
		printvec("W_l0_i.csv",W_l0);
		printvec("W_ld_i.csv",W_ld);
		printvec("gld_i.csv",gld);
		printvec("sld_i.csv",sld);
		printvec("thetald_i.csv",thetald);
		printvec("findrt_i.csv",findrt);
	}

	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			if(W_ld->data[l*Noccs+d]<= W_l0->data[l]){
				W_ld->data[l*Noccs+d] =W_l0->data[l];
	//			thetald->data[l*Noccs+d] = 0.0;
			}
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

	gsl_matrix_free(Pxx0);gsl_matrix_free(Pxx1);


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
		if(thetald->data[i]>1)
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
	gsl_vector_free(W_l0);
	gsl_vector_free(W_ld);
	gsl_matrix_free(x);
	gsl_vector_free(gld);
	gsl_vector_free(ret_d);
	gsl_vector_free(lgld);
	gsl_vector_free(wld);
	gsl_vector_free(thetald);
	gsl_matrix_free(pld);
	gsl_vector_free(findrt);

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

	if(printlev>=1)
		fprintf(solerr,"Defining the system had %d \n",nonfin);

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

	//1st order derivatives:
	for(l=0;l<Noccs+1;l++){
			double bl = l>0 ? b[1]:b[0];
			double findrtl = 0.0;
			for(d=0;d<Noccs;d++)
				findrtl += effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi)
					*ss->data[ss_gld_i+l*Noccs+d];
			double gdenom = 0.0;
			for(d=0;d<Noccs;d++){
				double nud = l ==d+1? 0.0 : nu;
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				double pld 	= effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
				double post = pld > 0.0? -kappa*ss->data[ss_tld_i+l*Noccs+d]/pld: 0.0;
				gdenom +=exp(sig_psi*pld*
						(chi[l][d]-bl+nud+ post + beta*contval));
			}
			/*// xl0
			gsl_matrix_set(Dvar,x_i+l*(Noccs+1),x_i+l*(Noccs+1),1.0-findrtl);
			gsl_matrix_set(Dvar,x_i+l*(Noccs+1),x_i+l*(Noccs+1)+l,(1.0-findrtl)*sbar);
			gsl_matrix_set(Dvar_tp1,x_i+l*(Noccs+1),x_i+l*(Noccs+1),1.0);
			 */
			// Wl0
			double Wl0_ss = ss->data[ss_Wl0_i + l];
			gsl_matrix_set(Dst_tp1,Wl0_i+l,Wl0_i+l,	beta*(1.0-findrtl) );// /Wl0_ss*Wl0_ss
			gsl_matrix_set(Dst,Wl0_i+l,Wl0_i+l,1.0);
			for(d=0;d<Noccs;d++){
				double zd = xx->data[d+1];
				// Wld
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,1.0);
				double Wld_ss = ss->data[ss_Wld_i + l*Noccs+d];
				if(d+1 ==l){
					gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d, Wld_i+l*Noccs+d,(1.0-ss->data[ss_sld_i + l*Noccs+d])*beta);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l, - (ss->data[ss_sld_i + l*Noccs+d])
							*ss->data[ss_Wl0_i+l]/Wld_ss);
					double cont_dd =chi[l][d]*exp(Z+zd) -bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
					double dWds = -cont_dd + ss->data[ss_Wl0_i+l];
					gsl_matrix_set(Dco,Wld_i + l*Noccs+d, sld_i+l*Noccs+d,
							 dWds/Wld_ss*ss->data[ss_sld_i+l*Noccs+d]);
				}
				else{
					gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,beta*(1.0-tau)*(1.0-ss->data[ss_sld_i + l*Noccs+d]) );
					//gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d,Wld_i+(d+1)*Noccs+d,beta*tau*(1.0-sbar));
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+(d+1)*Noccs+d,-tau
							*ss->data[ss_Wld_i+(d+1)*Noccs+d]/Wld_ss);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l,-(1-tau)*ss->data[ss_sld_i + l*Noccs+d]*ss->data[ss_Wl0_i+l]/Wld_ss);
					double cont_ld =chi[l][d]*exp(Z+zd) - bl + beta*ss->data[ss_Wld_i+l*Noccs+d];
					double dWds =-(1.0-tau)*(cont_ld - ss->data[ss_Wl0_i+0]);
					gsl_matrix_set(Dco,Wld_i + l*Noccs+d, sld_i+l*Noccs+d,
							dWds/Wld_ss*ss->data[ss_sld_i+l*Noccs+d]);
				}

				// Wl0

				// ss_ret = -nu - kappa/theta^ld/p^ld + mu^ld - mu^l0
				double pld		= effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
				double post 	=  - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld;
				double ss_ret 	= d+1==l ? 0.0 : -nu;
				ss_ret += chi[l][d]*exp(Z+zd) +post -bl;
				double contval = beta*(ss->data[ss_Wld_i+l*Noccs+d] -ss->data[ss_Wl0_i+l] );
				ss_ret += contval;//>0? ss_ret+contval : ss_ret;
				//ss_ret = ss_ret>1e-5? ss_ret : 1e-5;
				double dtld = ss->data[ss_gld_i+l*Noccs+d]*(
								ss_ret*effic/
								pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi+1)
								- kappa*pow(ss->data[ss_tld_i+l*Noccs+d],phi)/(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi)) );
							//-effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)
							//*kappa/effic*phi*pow(ss->data[ss_tld_i+l*Noccs+d],phi-1.0));

				if(gsl_finite(dtld) && pld>0.0)
					gsl_matrix_set(Dco,Wl0_i+l,tld_i+l*Noccs+d,dtld*ss->data[ss_tld_i+l*Noccs+d]/Wl0_ss);
				else
					gsl_matrix_set(Dco,Wl0_i+l,tld_i+l*Noccs+d,0.0);

				if(pld>0.0 && gsl_finite(pld*ss_ret))
					gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,
						pld*ss_ret*ss->data[ss_gld_i+l*Noccs+d]/Wl0_ss);
				else
					gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,0.0);

				double disc_cont =beta*pld*ss->data[ss_gld_i+l*Noccs+d];
				if(pld>0.0 && gsl_finite(disc_cont))
					gsl_matrix_set(Dst_tp1,Wl0_i+l,Wld_i+l*Noccs+d,disc_cont/Wl0_ss*Wld_ss);
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
			zd = xx->data[d+1];
			double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			//contval = contval <0.0 ? 0.0 : contval;
			double pld = ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
			gdenom +=exp(sig_psi*effic*pld*
					(chi[l][d]*exp(Z+zd)-bl+nud+beta*contval));
		}

		for(d=0;d<Noccs;d++){
			zd = xx->data[d+1];
			double nud = l==d+1 ? 0: nu;
			double Wld_ss = ss->data[ss_Wld_i +l*Noccs+d];
			// tld
			double tld_ss = ss->data[ss_tld_i+l*Noccs+d];
			gsl_matrix_set(Dco,tld_i + l*Noccs+d,tld_i + l*Noccs+d, 1.0);
			if(tld_ss>0.0){
				double surp = chi[l][d] - nud - bl + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				//dt/dWld

				double dtld =beta*pow(tld_ss*kappa/(fm_shr*effic*surp),1.0-phi)*(fm_shr*effic)/kappa;
//				dtld = surp<0 ? 0.0 : dtld;

				// COME BACK AND FIGURE THIS OUT!!!!!!!!!!!! IT'S WAY FUBAR
				if(gsl_finite(dtld) && surp>0.0)
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, dtld*Wld_ss/tld_ss);
				else
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, 0.0);
				//dt/dWl0
				dtld *= -1.0;
				if(gsl_finite(dtld) && surp>0.0)
					gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wl0_i + l,dtld*Wl0_ss/tld_ss);
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

			if(gld_ss>0){
				// exp(sig_psi*effic*pow(ss->data[tld_i+l*Noccs+d],1.0-phi)*(ss->data[wld_i+l*Noccs+d]-b+nud+beta*(ss->data[Uld_i+l*Noccs+d]-ss->data[Ul0_i+l])))
				double pld		= effic*tld_ss/pow(1.0+pow(tld_ss,phi),1.0/phi);
				double post		= pld>0 ? - kappa*tld_ss/pld : 0.0;
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				//contval = contval <0.0 ? 0.0 : contval;
				double arr_d 	= (chi[l][d]*exp(Z+zd) -bl + post -nud + beta*contval);
				//double ret_d 	= pld*arr_d;
				double dgdWld = beta*sig_psi*pld*gld_ss*(1.0-gld_ss);
				if(tld_ss>0.0 && arr_d>0)
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,
							dgdWld*Wld_ss/gld_ss);
				else
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,0.0);
				/*
				 * dg/dWl0 = 0
				*/

				// dg/dt

				double dpdt = effic/pow(1.0 + pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi +1.0);
				double dgdt =sig_psi*(dpdt*arr_d
						- kappa*pow(tld_ss,phi)/(1.0 + pow(tld_ss,phi)) )
						*gld_ss*(1.0-gld_ss);
				if(tld_ss>0.0 && arr_d>0.0)
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+d,
						-1.0*dgdt/gld_ss*tld_ss
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
				if(gld_ss>0.0){
					double nudd = l==dd+1 ? 0: nu;
					double zdd = xx->data[dd+1];
					double contval = (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
					double pldd		= effic*ss->data[ss_tld_i+l*Noccs+dd]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+dd],phi),1.0/phi);
					double postdd	=  pldd>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+dd]/pldd : 0.0;
					double arr_dd 	= (chi[l][dd]*exp(Z+zdd)-bl + postdd - nudd+beta*contval);
					//double ret_dd 	= pldd*arr_dd;

					double dpdt		= effic/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+dd],phi),1.0/phi+1);
					double dgdWld	= -pldd*beta*sig_psi*ss->data[ss_gld_i+l*Noccs+dd]*ss->data[ss_gld_i+l*Noccs+d];
					if(arr_dd>0.0)
						gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,
							dgdWld/gld_ss*ss->data[ss_Wld_i+l*Noccs+dd]
							);
					else
						gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,0.0);
					double dgdtdd = -sig_psi*(dpdt*arr_dd-
							kappa*pow(ss->data[ss_tld_i+l*Noccs+dd],phi)/(1.0 + pow(ss->data[ss_tld_i+l*Noccs+dd],phi)))
							*gld_ss*ss->data[ss_gld_i+l*Noccs+dd];
					if(arr_dd>0.0)
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,-1.0*
							dgdtdd*ss->data[ss_tld_i +l*Noccs+dd]/gld_ss
							);
					else
						gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,0.0);
				} //gld_ss >0
				else{
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,0.0);
					gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,0.0);
				}
			}
			}

			// sld
			gsl_matrix_set(Dco,sld_i +l*Noccs+d,sld_i +l*Noccs+d,1.0);
			double cut 	= ss->data[ss_Wl0_i+l]-exp(Z+zd)*chi[l][d]-beta*ss->data[ss_Wld_i +l*Noccs+d];
			cut = cut > 0.0 ? 0.0 : cut;
			double dF 	= scale_s*shape_s*exp(shape_s*cut);
			// ds/dWld
			double dsdWld = -beta*dF;
			gsl_matrix_set(Dst_tp1,sld_i+l*Noccs+d,Wld_i+l*Noccs+d,
					dsdWld*ss->data[ss_Wld_i+l*Noccs+d]/ss->data[ss_sld_i+l*Noccs+d]);
			// ds/dWl0
			double dsdWl0 =  dF;
			gsl_matrix_set(Dst,sld_i+l*Noccs+d,Wl0_i+l,dsdWl0*ss->data[ss_Wl0_i+l]/ss->data[ss_sld_i+l*Noccs+d]);

		}
	}
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
			double pld 	= effic* ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
			double post	= pld>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld : 0.0;
			double nud 	= l ==d+1? 0.0 : nu;
			double ret_d=(chi[l][d]-bl- nud + post + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]));
			ret_d 		= ret_d <0.0 ? 0.0 : ret_d;
			gdenom +=exp(sig_psi*pld*ret_d);
		}
		double dWl0dZ =0.0;
		for(d=0;d<Noccs;d++)
			dWl0dZ += ss->data[ss_gld_i+l*Noccs+d]*chi[l][d]*
			          effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0 + pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
		gsl_matrix_set(Dst,Wl0_i+l,0,dWl0dZ/ss->data[ss_Wl0_i+l]);
		for(d=0;d<Noccs;d++){

			double nud = l ==d+1? 0.0 : nu;
			double pld 	= effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
			double post	= pld>0 ? - kappa*ss->data[ss_tld_i+l*Noccs+d]/pld : 0.0;
								// dWl0/dz
			double dWdz =ss->data[ss_gld_i+l*Noccs+d]*chi[l][d]*
					effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+ pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
			gsl_matrix_set(Dst,Wl0_i+l,d+Notz,dWdz/ss->data[ss_Wl0_i+l]);

			if(ss->data[ss_tld_i+l*Noccs+d]>0.0){
				double surp = (chi[l][d]-nud- bl +beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]));
				//surp= surp<0 ? 0.0 : surp;

				double dtdz =chi[l][d]*pow(ss->data[ss_tld_i+l*Noccs+d]*kappa/(fm_shr*effic*surp),1.0-phi)*fm_shr*effic/kappa;
				if( surp>0 && gsl_finite(dtdz)){
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,   // dtheta/dZ
						dtdz/ss->data[ss_tld_i+l*Noccs+d]);
					gsl_matrix_set(Dco,tld_i + l*Noccs+d,d+Notz,// dtheta/dz
							dtdz/ss->data[ss_tld_i+l*Noccs+d]);
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
			if(l!=d+1){
				// dWld/dZ
				double dWlddZ =(1.0- ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d];
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,dWlddZ/ss->data[ss_Wld_i+l*Noccs+d]);
				double dWlddzd =((1.0-ss->data[ss_sld_i + l*Noccs+d])*(1.0-tau)*chi[l][d]+
						(1.0-ss->data[ss_sld_i + (d+1)*Noccs+d])*tau*chi[d+1][d]) ;
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,dWlddzd/ss->data[ss_Wld_i+l*Noccs+d]);
			}
			else{
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,
						(1.0-ss->data[ss_sld_i + l*Noccs+d])*chi[l][d]/ss->data[ss_Wld_i+l*Noccs+d]);
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,
						// dWld/dzd
						(1.0-ss->data[ss_sld_i + l*Noccs+d])*chi[l][d]/ss->data[ss_Wld_i+l*Noccs+d]);
			}

			// gld
			if(ss->data[ss_gld_i+l*Noccs+d]>0){
				double contval  = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				//contval	= contval>0.0? contval : 0.0;
				double arr_d 	= (chi[l][d] -bl - nud + post +beta*contval);
				double ret_d 	= pld*arr_d;
				ret_d = ret_d<0 ? 0.0: ret_d;
				double dgdzd =sig_psi*pld*chi[l][d]*ss->data[ss_gld_i+l*Noccs+d]*(1.0-ss->data[ss_gld_i+l*Noccs+d]);
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,
					dgdzd/ss->data[ss_gld_i+l*Noccs+d]);
			}
			else{
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,0.0);
			}
			for(dd=0;dd<Noccs;dd++){
			if(dd!=d){
				if(ss->data[ss_gld_i+l*Noccs+d]>0){
					double nudd = l==dd+1 ? 0: nu;
					double p_dd		= effic*ss->data[ss_tld_i+l*Noccs+d]/pow(1.0+pow(ss->data[ss_tld_i+l*Noccs+d],phi),1.0/phi);
					double postdd	= p_dd > 0 ? - kappa*ss->data[ss_tld_i+l*Noccs+dd]/p_dd : 0.0;
					double contval	= (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
					//contval = contval<0.0 ? 0.0 : contval;
					double arr_dd 	= (chi[l][dd]-bl - nudd + postdd +beta*contval);

					double ret_dd 	= p_dd*arr_dd;
					ret_dd = ret_dd< 0 ? 0.0 : ret_dd;
					double dgdzdd =-1.0*sig_psi*chi[l][dd]*ss->data[ss_gld_i+l*Noccs+d]*ss->data[ss_gld_i+l*Noccs+dd];
					gsl_matrix_set(Dco,gld_i+l*Noccs+d,dd+Notz,
							dgdzdd/ss->data[ss_gld_i+l*Noccs+d]);
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
			gsl_matrix_set(Dco,sld_i+l*Noccs+d,0,dsdZ/ss->data[ss_sld_i+l*Noccs+d]);
			// ds/dzd
			gsl_matrix_set(Dco,sld_i+l*Noccs+d,Notz+d,dsdZ/ss->data[ss_sld_i+l*Noccs+d]);
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
	double ** pld = (double**)malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		pld[l]= (double*)malloc(sizeof(double)*Noccs);


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
	double ** pld 		= (double**)malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		pld[l]=(double*)malloc(sizeof(double)*Noccs);
	struct sys_sol * sol 	= st->sol;
	struct sys_coef * sys 	= st->sys;
	struct aux_coef * simdat= st->sim;
	status = 0;


	Ndraw = (simdat->draws)->size1;
	gsl_vector * shocks 	= gsl_vector_calloc(Nx);
	gsl_vector * Zzl		= gsl_vector_calloc(Nx);
	gsl_matrix * uf_hist	= gsl_matrix_calloc(Ndraw,3);
	gsl_matrix * urt_l		= gsl_matrix_calloc(Ndraw,Noccs);
	gsl_matrix * fnd_l_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
	gsl_matrix * x_u_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
	gsl_matrix * Zzl_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);

	struct dur_moments s_mom;
	s_mom.Ft = gsl_vector_calloc(5);
	s_mom.dur_qtl = gsl_matrix_calloc(Ndraw,6);

	double urtss = 0.0;
	for(l=0;l<Noccs+1;l++)
		urtss += gsl_matrix_get(xss,l,0);
	if(urtss>0.01)
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
	gsl_matrix * XX = gsl_matrix_calloc(Noccs*Noccs,Nskill+1);
	gsl_vector_view X0 = gsl_matrix_column(XX,Nskill);
	gsl_vector_set_all(&X0.vector,1.0);

	gsl_matrix * Wt	= gsl_matrix_alloc(Noccs*Noccs,Noccs*Noccs);
	gsl_vector * yloss = gsl_vector_alloc(Noccs*Noccs);
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

		status += theta(sol->tld, ss, sys, sol, Zz);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		status += spol(sol->sld, ss, sys, sol, Zz);

		if( printlev>=1 && (gpol_zeros>0 || t_zeros>0)){
			fprintf(simerr,"%d times g^ld = 0 in sim # %d \n",gpol_zeros,di);
			fprintf(simerr,"%d times t^ld = 0 in sim # %d \n",t_zeros,di);
		}

		status += xprime(xp,ss,sys,sol,x,Zz);
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				pld[l][d] =
					effic*
					gsl_matrix_get(sol->tld,l,d)/
					pow(1.0+pow(gsl_matrix_get(sol->tld,l,d),phi),1.0/phi);
		}
		if(printlev>=4){
			printmat("sol->tld.csv",sol->tld);
			printmat("sol->gld.csv",sol->gld);
			printmat("xp.csv",xp);
		}

		for(d=1;d<Noccs+1;d++){
			gsl_matrix_set(urt_l,di,d-1,gsl_matrix_get(xp,d,0)/(gsl_matrix_get(xp,d,0) + gsl_matrix_get(xp,d,d)));
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
		gsl_vector_view x_u_d = gsl_matrix_row(x_u_hist,di);
		gsl_vector_memcpy(&x_u_d.vector,x_u);

		// get the finding rate:
		double d_fnd =0.0;
		for(l=0;l<Noccs+1;l++){
			fnd_l->data[l] = 0.0;
			for(d=0;d<Noccs;d++){
				fnd_l->data[l] += gsl_matrix_get(sol->gld,l,d)*pld[l][d];
			}
			d_fnd += fnd_l->data[l]*gsl_vector_get(x_u,l);
		}
		s_fnd += d_fnd/(double)Ndraw;

		gsl_matrix_set(uf_hist,di,2,d_fnd);
		gsl_vector_view fnd_l_di = gsl_matrix_row(fnd_l_hist,di);
		gsl_vector_memcpy(&fnd_l_di.vector,fnd_l);

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
		s_elpt += d_elpt/(double)Ndraw;

		//separation rate and sd of it
		double d_sep=0,d_sdsep=0;
		for(l=0;l<Noccs+1;l++){
			for(d=1;d<Noccs;d++){
				if(l!=d)
					d_sep += gsl_matrix_get(xp,l,d)*((1.0-tau)*gsl_matrix_get(sol->sld,l,d-1)
							+ tau*gsl_matrix_get(sol->sld,d,d-1));
				else
					d_sep += gsl_matrix_get(xp,l,d)*gsl_matrix_get(sol->sld,l,d-1);
			}
		}
		s_sep += d_sep/(double)Ndraw;
		for(l=0;l<Noccs+1;l++){
			for(d=1;d<Noccs;d++){
				if(l!=d)
					d_sdsep += pow(gsl_matrix_get(xp,l,d)*((1.0-tau)*gsl_matrix_get(sol->sld,l,d-1)
						+ tau*gsl_matrix_get(sol->sld,d,d-1)) - d_sep, 2);
				else
					d_sdsep += pow(gsl_matrix_get(xp,l,d)*gsl_matrix_get(sol->sld,l,d-1) -d_sep,2);
			}
		}
		s_sdsep += d_sdsep/(double)Ndraw;


		//generate the change probability:
		double d_chng = 0.0;
		for(l=1;l<Noccs+1;l++){
			double d_chng_l = 1.0 - (gsl_matrix_get(sol->gld,l,l-1)*pld[l][l-1] )/
				fnd_l->data[l];
			d_chng += d_chng_l*x_u->data[l];
		}
		double max0 = 0.0;
		for(d=0;d<Noccs;d++)
			max0 = (gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi))>max0 ?
					(gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi)):max0;
		d_chng += (1.0-max0/fnd_l->data[0])*x_u->data[0];

		d_chng /= (1.0 - x_u->data[0]);
		d_chng = gsl_finite(d_chng)==0 ? s_chng*((double)Ndraw/((double)di+1.0) ) : d_chng;
		s_chng +=d_chng/(double)Ndraw;


		//	Wage growth
		//  Compute wages for each occupation here.  Requires computation of forward terms
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
				gsl_matrix_set(wld,l,d,wld_ld);
			}
		}
		double d_wg = 0.0;
		double x_wg = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l-1!=d){
					d_wg += gsl_vector_get(x_u,l)*(gsl_matrix_get(sol->gld,l,d))*pld[l][d]*
							(gsl_matrix_get(wld,d+1,d) - gsl_matrix_get(wld,l,d));
					x_wg += gsl_vector_get(x_u,l)*(gsl_matrix_get(sol->gld,l,d))*pld[l][d];
				}
			}
		}
		d_wg *= tau; // dividing by expected time for that growth: 1.0/tau
		if(x_wg>0.0)
			d_wg /= x_wg;
		else
			d_wg = 0;
		d_wg = gsl_finite(d_wg) == 0  ? s_wg*((double)Ndraw )/((double)di +1.0) : d_wg;
		s_wg += d_wg/(double)Ndraw;

		// wage loss due to change
		double d_wl = 0.0;
		double sx_wl= 0.0;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					d_wl +=gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d)*pld[l][d] *(gsl_matrix_get(wld,l,l-1) - gsl_matrix_get(wld,l,d));;
					sx_wl +=gsl_vector_get(x_u,l)*gsl_matrix_get(sol->gld,l,d)*pld[l][d];
				}
			}
		}
		if(sx_wl>0.0)
			d_wl /=sx_wl;
		else
			d_wl = 0.0;
		s_mom.chng_wg += d_wl/(double)Ndraw;
	//	gsl_matrix_set(simdat->data_moments,0,3,s_mom.chng_wg);
	//	gsl_matrix_set(simdat->moment_weights,0,3,0.0);


		if(st->cal_set==0 || st->cal_set==2){
		// do a regression on the changes in wages:
		// only look at experienced changers, and set chi[0][d] s.t. makes the avg wage loss match
			for(l=1;l<Noccs+1;l++){
				for(d=0;d<Noccs;d++){
					gsl_matrix_set(Wt,(l-1)*Noccs+d,(l-1)*Noccs+d,
							pld[l][d]*
							gsl_matrix_get(sol->gld,l,d)*gsl_vector_get(x_u,l));
					gsl_vector_set(yloss,(l-1)*Noccs+d,
							log(gsl_matrix_get(wld,l,d)/gsl_matrix_get(wld,d+1,d)));
					for(si=0;si<Nskill;si++)
						gsl_matrix_set(XX,(l-1)*Noccs+d,si,
								fabs(gsl_matrix_get(f_skills,l-1,si+1)-gsl_matrix_get(f_skills,d,si+1) ));

				}
			}
			status += WOLS( yloss, XX ,Wt, coefs_di, er);
			gsl_vector_scale(coefs_di,1.0/(double)Ndraw);
			gsl_vector_add(coefs,coefs_di);
		}


		// advance a period:
		gsl_matrix_memcpy(x,xp);
		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);
		gsl_matrix_set(Zzl_hist,di,0,Zz->data[0]);
		for(l=0;l<Noccs;l++)
			gsl_matrix_set(Zzl_hist,di,l,Zz->data[Notz+l]);

		gsl_vector_free(x_u);
	}// for d<Ndraw

	if(printlev>=2){
		printmat("sol->tld.csv",sol->tld);
		printmat("sol->gld.csv",sol->gld);
		printmat("xp.csv",xp);
		printmat("wld.csv",wld);
		printmat("Zzl_hist.csv",Zzl_hist);
		printmat("x_u_hist.csv",x_u_hist);
		printmat("fnd_l_hist.csv",fnd_l_hist);
		printmat("uf_hist.csv",uf_hist);
	}


	int fincoefs = 1;
	for(di=0;di<coefs->size;di++)
		fincoefs *= gsl_finite(gsl_vector_get(coefs,di));
	if(fincoefs!=0){
		for(di=0;di<coefs->size;di++)
			gsl_matrix_set(simdat->data_moments,1,di,coefs->data[di]- sk_wg[di]);
	}
	else{
		gsl_vector_view momcoef = gsl_matrix_row(simdat->data_moments,1);
		gsl_vector_scale(&momcoef.vector,1.5);
	}

	status += dur_dist(&s_mom,fnd_l_hist,x_u_hist);

	if(printlev>=2){
		printvec("Ft.csv",s_mom.Ft);
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
	if(verbose >=2) printf("The average wage-loss was %f\n",s_mom.chng_wg);
	int finmoments = 1;
	finmoments *= gsl_finite(s_wg);
	finmoments *= gsl_finite(s_fnd);
	finmoments *= gsl_finite(s_chng);
	finmoments *= gsl_finite(s_urt);
	finmoments *= gsl_finite(s_elpt);
	finmoments *= gsl_finite(s_sdsep);

	if(status == 0 && finmoments!=0){
		gsl_matrix_set(simdat->data_moments,0,0,(avg_wg  - s_wg)/avg_wg);
		gsl_matrix_set(simdat->data_moments,0,1,(avg_fnd - s_fnd)/avg_fnd); // this one is much more important (arbitrary)
		gsl_matrix_set(simdat->data_moments,0,2,(chng_pr - s_chng)/chng_pr);
		gsl_matrix_set(simdat->data_moments,0,3,(avg_urt - s_urt)/avg_urt);
		gsl_matrix_set(simdat->data_moments,0,4,(avg_elpt - s_elpt)/avg_elpt);
		gsl_matrix_set(simdat->data_moments,0,5,(avg_sdsep - s_sdsep)/avg_sdsep);
	}
	else// penalize it for going in a region that can't be solved
		gsl_matrix_scale(simdat->data_moments,1.5);

	// Cleaning up!
	// !!!!!!!!!!!!
	gsl_matrix_free(sol->tld);
	sol->tld=0;
	gsl_matrix_free(sol->gld);
	sol->gld=0;
	gsl_matrix_free(sol->sld);
		sol->sld=0;

	for(l=0;l<Noccs+1;l++)
		free(pld[l]);
	free(pld);
	gsl_matrix_free(Wt);gsl_matrix_free(XX);gsl_vector_free(coefs);gsl_vector_free(coefs_di);
	gsl_vector_free(yloss);	gsl_vector_free(er);
	gsl_vector_free(s_mom.Ft);
	gsl_matrix_free(s_mom.dur_qtl);gsl_matrix_free(Zzl_hist);
	gsl_vector_free(fnd_l);gsl_vector_free(Zzl);gsl_vector_free(shocks);
	gsl_matrix_free(uf_hist);
	gsl_matrix_free(urt_l);gsl_matrix_free(fnd_l_hist);gsl_matrix_free(x_u_hist);
	gsl_matrix_free(x);gsl_matrix_free(xp);gsl_vector_free(Zz);
	return status;
}


int dur_dist(struct dur_moments * s_mom, const gsl_matrix * fnd_l_hist, const gsl_matrix * x_u_hist){
	int status,l,d,t;
	s_mom->E_dur	= 0.0;
	s_mom->sd_dur	= 0.0;
	s_mom->pr6mo	= 0.0;

	status =0;

	int Ndraw = x_u_hist->size1;

	gsl_vector * sort_fnd = gsl_vector_alloc(Noccs+1);
	gsl_vector * sort_x	  = gsl_vector_alloc(Noccs+1);
	gsl_permutation * sort_p = gsl_permutation_calloc(Noccs+1);

	gsl_vector * Ft = gsl_vector_calloc(5); // 1 month, 3 months, 6 months, 1 yr, 2yrs
	double durs[] = {1.0,3.0,6.0,12.0,24.0};

	gsl_vector_set_zero(s_mom->Ft);
	for(t=0;t<Ndraw;t++){
		//Get the 25 pct, 50 pct, 75 pct and 90 pct duration

		for(l=0;l<Noccs+1;l++){
			gsl_vector_set(sort_x,l,gsl_matrix_get(x_u_hist,t,l));
			gsl_vector_set(sort_fnd,l,gsl_matrix_get(fnd_l_hist,t,l));
		}
		for(d =0;d<5;d++){
			double Ft_l = 0.0;
			double xt_l = 0.0;
			for(l=0;l<Noccs+1;l++){
				Ft_l += sort_x->data[l]*exp(- gsl_vector_get(sort_fnd,l)*durs[d])*gsl_vector_get(sort_fnd,l);
				xt_l += sort_x->data[l]*exp(- gsl_vector_get(sort_fnd,l)*durs[d]);
			}
			Ft_l /= xt_l;
			gsl_vector_set(s_mom->Ft,d,gsl_vector_get(s_mom->Ft,d) +Ft_l/(double)Ndraw);
		}
	}

	gsl_vector_free(sort_x);gsl_vector_free(sort_fnd);
	gsl_permutation_free(sort_p);
	gsl_vector_free(Ft);
	for(t=0;t<Ndraw;t++){
		double d_dur = 0.0;
		for(l=0;l<Noccs;l++){
			if(gsl_matrix_get(fnd_l_hist,t,l)>1e-8)
				d_dur += 1/gsl_matrix_get(fnd_l_hist,t,l)*gsl_matrix_get(x_u_hist,t,l);
		}
		s_mom->E_dur += d_dur/((double) Ndraw);
		double d_sddur = 0.0;
		for(l=0;l<Noccs;l++){
			if(gsl_matrix_get(fnd_l_hist,t,l)>1e-8)
				d_sddur += pow(gsl_matrix_get(fnd_l_hist,t,l),-2)*gsl_matrix_get(x_u_hist,t,l);
		}
		s_mom->sd_dur += d_sddur/(double)Ndraw;
		double d_6mo = 0.0;
		for(l=0;l<Noccs;l++){
			d_6mo += pow(1 - gsl_matrix_get(fnd_l_hist,t,l),6)*gsl_matrix_get(x_u_hist,t,l);
		}
		s_mom->pr6mo += d_6mo/(double)Ndraw;
	}
	s_mom->sd_dur =pow(s_mom->sd_dur,0.5);

	return status;
}



int TGR(gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		double * urt, double * opt_urt,
		gsl_vector * ss, struct sys_sol * sol,struct sys_coef * sys){

	int status,l,d,t,Tmo;
	gsl_matrix * xp = gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_matrix * x = gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_matrix * r_occ_wr = gsl_matrix_alloc(9416,7);
	gsl_vector * Zz_TGR = gsl_vector_alloc(Nx);
	gsl_vector * Zzl = gsl_vector_alloc(Nx);
	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);
	gsl_vector * av_fnd_l = gsl_vector_calloc(Noccs+1);
	gsl_vector * x_u 	= gsl_vector_calloc(Noccs+1);

	status	= 0;
	Tmo 	= 24;
	gsl_vector * sw_hist = gsl_vector_calloc(Tmo);
	gsl_vector * wl_hist = gsl_vector_calloc(Tmo);
	gsl_matrix * dur_dist_hist = gsl_matrix_calloc(5,Tmo);
	gsl_matrix * fnd_l_hist	= gsl_matrix_calloc(Noccs+1,Tmo);
	gsl_matrix * x_l_hist	= gsl_matrix_calloc(Noccs+1,Tmo);
	gsl_vector * urt_hist	= gsl_vector_calloc(Tmo);
	readmat("occ_wr.csv",r_occ_wr);
	// skim until peak unemployment in June 2008
	for(l=0;l<r_occ_wr->size1;l++){
		int date = (int) rint(gsl_matrix_get(r_occ_wr,l,0));
		if( date == 200906 ){
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
	if(printlev>=1)
		printmat("xTGR0.csv",x);
	double sx0 = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs+1;d++)
			sx0 += gsl_matrix_get(x,l,d);
	}
	*urt = 0.0;
	for(l=0;l<Noccs+1;l++)
		*urt+=gsl_matrix_get(x,l,0);
	readvec("outzz.csv",Zz_TGR);
	sol->gld = 0; // just to make sure (this might cause a memory leak)
	sol->tld = 0;
	sol->sld = 0;
	*urt = 0.0;
	for(t=0;t<Tmo;t++){// this is going to loop through 2 years of recession
		Zz_TGR->data[0] *= -1.0;
//		for(d=0;d<Noccs;d++)
//			Zz_TGR->data[1+d+Nfac*Nflag] *= -1.0;
		sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += theta(sol->tld,ss, sys,  sol, Zz_TGR);
		sol->gld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += gpol(sol->gld,ss, sys,  sol, Zz_TGR);
		sol->sld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += spol(sol->sld,ss, sys,  sol, Zz_TGR);
		status += xprime(xp, ss, sys,  sol, x, Zz_TGR);
		// now should have tld, gld in sol
		urt_hist->data[t] =0.0;
		for(l=0;l<Noccs+1;l++)
			urt_hist->data[t] += gsl_matrix_get(xp,l,0);
		*urt += urt_hist->data[t]/(double)Tmo;
		for(l=0;l<Noccs+1;l++)
			gsl_vector_set(x_u,l,gsl_matrix_get(xp,l,0)/urt_hist->data[t]);
		gsl_vector_view x_l_t = gsl_matrix_column(x_l_hist,t);
		gsl_vector_memcpy(&x_l_t.vector,x_u);
		for(l=0;l<Noccs+1;l++){
			double fnd_l_t_l = 0.0;
			for(d=0;d<Noccs;d++)
				fnd_l_t_l += gsl_matrix_get(sol->gld,l,d)
				*effic*gsl_matrix_get(sol->tld,l,d)/pow(1.0 + pow(gsl_matrix_get(sol->tld,l,d),phi) ,1.0/phi);
			gsl_vector_set(fnd_l,l,fnd_l_t_l);
		}
		double d_fnd = 0.0;
		for(l=0;l<Noccs+1;l++)
			d_fnd += fnd_l->data[l]*x_u->data[l];

		//gsl_vector_view dur_dist_t = gsl_matrix_column(dur_dist_hist,t);
		gsl_vector_view fnd_l_t = gsl_matrix_column(fnd_l_hist,t);
		gsl_vector_memcpy(&fnd_l_t.vector,fnd_l);
		//status += dur_dist(&fnd_l_t.vector,&dur_dist_t.vector,fnd_l,x_u);

		// calculate the number of switches:
		sw_hist->data[t] = 0.0;
		for(l=0;l<Noccs+1;l++){
			double sw_l = 0.0;
			for(d=0;d<Noccs;d++){
				if(l!=d+1)
					sw_l+=gsl_matrix_get(sol->gld,l,d)
					*effic*gsl_matrix_get(sol->tld,l,d)/pow(1.0+ pow(gsl_matrix_get(sol->tld,l,d),phi),1.0/phi);
			}
			sw_l /= fnd_l->data[l];
			sw_hist->data[t]+= sw_l*gsl_vector_get(x_u,l);
		}
		//wage loss (compare to tau* wg in normal times)
		double s_wl = 0.0;
		double x_wl = 0.0;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					s_wl += (chi[l][l-1] - chi[l][d])*gsl_vector_get(x_u,l)
							*effic*gsl_matrix_get(sol->tld,l,d)/pow(1.0+pow(gsl_matrix_get(sol->tld,l,d),phi),1.0/phi);
					x_wl += gsl_vector_get(x_u,l)
							*effic*gsl_matrix_get(sol->tld,l,d)/pow(1.0+pow(gsl_matrix_get(sol->tld,l,d),phi),1.0/phi);
				}
			}
		}
		s_wl /= x_wl;
		wl_hist->data[t] = s_wl;
		// advance to next step
		gsl_matrix_memcpy(x,xp);

		// Zz_TGR impulse response
		Zz_TGR->data[0] *= -1.0;
//		for(d=0;d<Noccs;d++)
//			Zz_TGR->data[1+d+Nfac*Nflag] *= -1.0;
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz_TGR, 0.0,Zzl);
		gsl_vector_memcpy(Zz_TGR,Zzl);
		gsl_matrix_free(sol->gld);gsl_matrix_free(sol->tld);
		sol->gld = 0;sol->tld = 0;
	}
	printvec("urt_hist.csv",urt_hist);
	printvec("sw_hist.csv",sw_hist);
	printvec("wl_hist.csv",wl_hist);
	printmat("fnd_l_hist.csv",fnd_l_hist);
	printmat("x_l_hist.csv",x_l_hist);
	printmat("dur_dist_hist.csv",dur_dist_hist);

	gsl_vector_free(fnd_l);gsl_vector_free(av_fnd_l);gsl_matrix_free(fnd_l_hist);
	gsl_vector_free(x_u);gsl_matrix_free(dur_dist_hist);
	gsl_matrix_free(r_occ_wr);gsl_vector_free(Zz_TGR);gsl_matrix_free(xp);gsl_matrix_free(x);
	gsl_vector_free(Zzl);gsl_matrix_free(x_l_hist);
	gsl_vector_free(sw_hist);gsl_vector_free(wl_hist);
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
	int ss_gld_i, ss_Wld_i,ss_Wl0_i;

		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
	double *gld_l 	= (double*) malloc(sizeof(double)*Noccs);

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
			double pld_ld = effic*tld/pow(1.0+pow(tld,phi) ,1.0/phi);
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
			double cont	= Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l] ;
			//cont = cont<0.0 ? 0.0 : cont; // I want to take this out?
			double ret_d 	= chi[l][d]*exp(Z + zd) - bl - nud +post + beta*cont ;
			double gld_ld 	= exp(sig_psi*gsl_matrix_get(pld,l,d)*ret_d);
			if(gsl_matrix_get(pld,l,d)>0  && ret_d>0 && gsl_finite(gld_ld))
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
	#pragma omp parallel for private(l,d)
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double W0 = 0.0;
		for(d=0;d<Noccs;d++){
			double zd = Zz->data[d+Notz];
			double nud = l == d+1? 0:nu;
			double post = -kappa*gsl_matrix_get(sol->tld,l,d)/gsl_matrix_get(pld,l,d);
			double cont	= Es->data[Wld_i + l*Noccs+d] - Es->data[Wl0_i + l] ;
			W0+= gsl_matrix_get(gld,l,d)*gsl_matrix_get(pld,l,d)*(exp(Z+zd)*chi[l][d] -bl -nud +post + beta*cont);
		}
		W0 += bl + beta*Es->data[Wl0_i + l];
		for(d=0;d<Noccs;d++){
			double zd = Zz->data[d+Notz];

			double Wdiff	= beta*Es->data[Wld_i+l*Noccs+d] - W0;

			double cutoff= - (chi[l][d]*exp(Z+zd)+ beta*Wdiff);
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
	/*gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_blas_dgemv(CblasNoTrans, 1.0, sol->PP, Zz, 0.0, Es);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-Wl0_i);
	for(l=0;l<Ns;l++)
		Es->data[l] =exp(Es->data[l]);
	status += gsl_vector_mul(Es,&ss_W.vector);
	*/
	for(l=0;l<Noccs+1;l++){
		double bl = l>0 ? b[1]:b[0];
		double tld_s = 0.0;
		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0:nu;
			double zd = Zz->data[d+Notz];
			double cont	= (Es->data[Wld_i+l*Noccs+d] - Es->data[Wl0_i+l]);
			//cont = cont<0.0 ? 0.0 : cont;
			double surp =chi[l][d]*exp(Z+zd) - bl - nud + beta*cont;
			//surp = surp>=0.0 ? surp:0.0;
			double inside = pow(effic*fm_shr*surp/kappa,phi);
			double tld_i =  pow(inside-1.0,1.0/phi);
			if(effic>1.0)
				tld_i = tld_i > pow(pow(effic,phi)-1.0,-1.0/phi) ? pow(pow(effic,phi)-1.0,-1.0/phi) : tld_i;
			if(gsl_finite(tld_i) && surp > 0.0 && inside >=1.0){
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
	double ** pld	= (double**)malloc(sizeof(double*)*(Noccs+1));
	gsl_matrix * tld,*gld,*sld;
	for(l=0;l<Noccs+1;l++)
		pld[l] = (double*)malloc(sizeof(double)*Noccs);

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

	ald  = (double**) malloc(sizeof(double*)*(Noccs+1));
	for(l=0;l<Noccs+1;l++)
		ald[l] = (double*)malloc(sizeof(double)*Noccs);
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

	findrt = (double*)malloc(sizeof(double)*(Noccs+1));
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
		if(printlev >=1) fprintf(simerr,"xsum = %f\n",xsum);
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



/*
 * Utilities
 *
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

int Es_cal(gsl_vector * Es, const gsl_vector * ss_W, const gsl_matrix * PP, const gsl_vector * Zz){
	int status =0,l;
	gsl_blas_dgemv(CblasNoTrans, 1.0, PP, Zz, 0.0, Es);

	for(l=0;l<Ns;l++)
		Es->data[l] =exp(Es->data[l]);
	status += gsl_vector_mul(Es,ss_W);
	return status;
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
	xh	= (double *) malloc(sizeof(double)*n);
	xl	= (double *) malloc(sizeof(double)*n);
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
			xh[i] *= (1.0-diff_eps);
			xl[i] *= (1.0+diff_eps);
		}
		else{
			xh[i] -= diff_eps;
			xl[i] += diff_eps;
		}
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

	int status,resi,starts,i,locstatus;
	status 	= 0;
	// csize 	= 8;  = how many nodes it's running on.
	starts 	= 8*csize; // this is how many will draw to begin, but will only start from a subset of these

	nlopt_opt opt0;
	gsl_multiroot_fdfsolver *sloc;//gsl_multiroot_solver *sloc;
	gsl_multiroot_function_fdf obj_wrap;//gsl_multiroot_function obj_wrap;
	if(alg0%10==0)
		opt0 = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	else if(alg0%5 ==0)
		opt0 = nlopt_create(NLOPT_LN_SBPLX,n);
	else
		opt0 = nlopt_create(NLOPT_LN_BOBYQA,n);

//  Non- deriv based
	double xtol_abs[n];
	for(i=0;i<n;i++)
		xtol_abs[i] = x_tol*x0[i];
	nlopt_set_xtol_rel(opt0, x_tol);
	nlopt_set_xtol_abs(opt0, xtol_abs);
	nlopt_set_ftol_rel(opt0, f_tol);
	nlopt_set_ftol_abs(opt0, f_tol); // because function value should be approaching zero, need to set this
	nlopt_set_maxeval(opt0, 50+pow(10,n));
	nlopt_set_stopval(opt0, 0.0);
	nlopt_set_lower_bounds(opt0,lb);
	nlopt_set_upper_bounds(opt0,ub);
	nlopt_set_min_objective(opt0, Vobj, (void*) ss);

	if(alg1>0){
		sloc = gsl_multiroot_fdfsolver_alloc(gsl_multiroot_fdfsolver_hybridsj,n);//sloc = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids,n);
		obj_wrap.f = &cal_dist_wrap;
		obj_wrap.df = &cal_df_wrap;
		obj_wrap.fdf= &cal_fdf_wrap;
		obj_wrap.n = n;
		obj_wrap.params = (void*)ss;
	}

	double opt_f=100;

	if(alg0>=10){//random restarts with MPI -- here's where MPI branching begins.
		double * dim_scale = malloc(sizeof(double)*n);
		for(i=0;i<n;i++){
			dim_scale[i] = ub[i] - lb[i];
		}
		// the queue of places to start
		double ** x0queue = (double**)malloc(sizeof(double*)*starts);
		double *xmin;
		int nmax=0;
		xmin = (double*)malloc(sizeof(double)*n);
		gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol,n);
		double xqr[n];
		double ftol_ms = f_tol/10.0;
		double minf = 1000;
		// set the restart locations before hand
		for(resi=0;resi<starts;resi++){
			x0queue[resi] = (double*)malloc(sizeof(double)*n);
			gsl_qrng_get(q,xqr);
			for(i=0;i<n;i++)
				x0queue[resi][i] = xqr[i]*(ub[i]-lb[i]) + lb[i];
		}
		// alloc space for the list of optimal points
		double ** x0_loc=(double**)malloc(sizeof(double*)*starts); // this is the set of local minima
		for(resi=0;resi<starts;resi++)
			x0_loc[resi] = (double*)malloc(sizeof(double)*n);
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
				if(verbose >=1) printf("Starting an optimize\n");
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
				if(li>=nmax){// close enough to call the same as one of these points
					for(i=0;i<n;i++)
						x0_loc[nmax][i]=x0queue[resi][i];
					nmax++;
				}
				// time to stop the multi-start?
				if(nmax>=stop_crit_N || opt_f< ftol_ms)
					break;
				starti ++;
			}//if cluster
		}// for restarts
		gsl_qrng_free(q);
		free(xmin);
		for(resi=0;resi<starts;resi++)
			free(x0queue[resi]);
		free(x0queue);
		for(resi=0;resi<starts;resi++)
			free(x0_loc[resi]);
		free(x0_loc);
		free(dim_scale);
	}// end multi start conditional
	else{
		status = nlopt_optimize(opt0, x0, &opt_f);
		if((status == 4 || status == 3) && opt_f>0.1){
			// start over from the last stop point
			status = nlopt_optimize(opt0, x0, &opt_f);
		}
		set_params( x0,ss->cal_set);
	}

	// POLISH IT OFF!
	if(polish_alg>0 && opt_f>f_tol/10.0){
		fprintf(calhist,"**** Now solving by hybrid method equation solver ****\n");
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

		gsl_multiroot_fdfsolver_free(sloc);
		gsl_vector_free(x_vec);
	}

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
		xh	= (double *) malloc(sizeof(double)*n);
		xl	= (double *) malloc(sizeof(double)*n);
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
		sig_psi = x[1];
		tau 	= x[2];
		scale_s	= x[3];
		shape_s = x[4];
		effic	= x[5];
		param_offset = 6;
	}

	if(cal_set==0 || cal_set==2){
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				chi[l][d] = 0.0;
				for(i=0;i<Nskill;i++){
					chi[l][d] += x[param_offset+i]*fabs(gsl_matrix_get(f_skills,d,i) - gsl_matrix_get(f_skills,l-1,i));
				}
				//chi[l][d] += x[param_offset+Nskill];
			}
		}
		double chi_lb = 0.01;
		double chi_ub = 0.99;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double child = chi[l][d];
				if(chi[l][d]<=chi_lb)
					child = chi_lb;
				if(chi[l][d]>=chi_ub)
					child = chi_ub;
				chi[l][d] = 1.0 - child;
			}
		}

		for(d=0;d<Noccs;d++){
			double mean_chi = 0.0;
			for(l=1;l<Noccs+1;l++)
				mean_chi = l!=d+1 ? chi[l][d] + mean_chi : mean_chi;
			chi[0][d] = mean_chi/((double)Noccs-1.0);

		}
		for(d=0;d<Noccs;d++)
			chi[d+1][d] = 1.0;
	}

	// recompute b[0]
	//b[1]=0.71;
	b[1] = 0.5;
	b[0]=0.0;
	for(d=0;d<Noccs;d++){
		for(l=0;l<Noccs+1;l++)
			b[0] += chi[l][d];
		b[0] -= chi[d+1][d];
	}
	b[0] /= (double)(Noccs*Noccs);
	b[0] *= 0.5;
	//b[0] += 0.21; // 50% replacement ratio plus 0.21 flat for home production: Hall & Milgrom (2008)

	//do something about chi below b --- this might screw everything up?
	//for(d=0;d<Noccs;d++){
	//	for(l=1;l<Noccs+1;l++)
	// 		chi[l][d] = chi[l][d]<b[1]? b[1] :chi[l][d];
	//	chi[0][d] =chi[0][d]<b[0]? b[0] :  chi[0][d];
	//}

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

	int status,i,l,d;
	struct st_wr * st = (struct st_wr * )params;
	struct aux_coef  * simdat	=  st->sim;
	struct sys_coef  * sys		= st->sys;
	struct sys_sol   * sol		= st->sol;
	gsl_matrix * xss = gsl_matrix_calloc(Noccs+1,Noccs+1);
	gsl_vector * ss  = gsl_vector_calloc(Ns+Nc);
	double dist;

	set_params(x, st->cal_set);
	int param_offset = st->cal_set == 1 || st->cal_set == 0 ? 5 : 0;

	status 	= sol_ss(ss,xss,sys);
	if(printlev>=0 && status>=1) printf("Steady state not solved \n");
	status += sys_def(ss,sys);
	if(printlev>=0 && status>=1) printf("System not defined\n");
	status += sol_dyn(ss,sol,sys);
	if(printlev>=0 && status>=1) printf("System not solved\n");

	if(status == 0)
		status += sim_moments(st,ss,xss);
	else{
		gsl_matrix_scale(simdat->data_moments,1.5);
		fprintf(solerr,"System did not solve at: (%f,%f,%f,%f)\n",x[0],x[1],x[2],x[3]);
	}	
	int Nmo = (simdat->data_moments)->size2;

	dist = 0.0;
	if(st->cal_set == 0 || st->cal_set ==1){
		for(i=0;i<Nmo;i++)
			dist += gsl_matrix_get(simdat->moment_weights,0,i)*pow(gsl_matrix_get(simdat->data_moments,0,i),2);
		if(verbose>=1){
			printf("At phi=%f,sig_psi=%f,tau=%f,scale_s=%f,shape_s=%f,effic=%f\n",x[0],x[1],x[2],x[3],x[4],x[5]);
			printf("Calibration distance is %f\n",dist);
		}
	}
	if(st->cal_set==0 || st->cal_set==2){
		// regression distance
		for(i=0;i<Nskill+1;i++)
			dist += gsl_matrix_get(simdat->moment_weights,1,i)*pow(gsl_matrix_get(simdat->data_moments,1,i),2);
		if(verbose>=1){
			printf("At b1=%f,b2=%f,b3=%f,b0=%f\n",x[0+param_offset],x[1+param_offset],x[2+param_offset],x[3+param_offset]);
			printf("Calibration distance is %f\n",dist);
		}
	}
	if(printlev>=1){
		for(i=0;i<n;i++)
			fprintf(calhist,"%f,",x[i]);
		fprintf(calhist,"\n");
		fprintf(calhist,"%f,",dist);
		if(st->cal_set==0 || st->cal_set==1){
			for(i=0;i<(simdat->data_moments)->size2;i++)
				fprintf(calhist,"%f,",(simdat->data_moments)->data[i]);
		}
		if(st->cal_set==0 || st->cal_set==2){
			for(i=0;i<(simdat->data_moments)->size2;i++)
					fprintf(calhist,"%f,",gsl_matrix_get((simdat->data_moments),1,i));
		}
		fprintf(calhist,"\n");
	}

	gsl_matrix_free(xss);
	gsl_vector_free(ss);
	return dist;
}
