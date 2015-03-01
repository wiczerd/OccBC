/* This program solves my dissertation by a hybrid k=order perturbation with an optimal COV and exact nonlinear policies
*  First, it solves the first order approximation by some strightforward linear algebra.  This can be used then for the 2nd, too.
*  For more on the optimal COV see J F-V J R-R (JEDC 2006)
*  The hybrid portion involves using the exactly computed decision rules given approximated state-dynamics: see Maliar, Maliar and Villemot (Dynare WP 6)
*
*  au : David Wiczer
*  mdy: 2-20-2011
*  rev: 5-06-2012
*
* compile line: icc occBC_hybr.c -lgsl -lgslcblas -lnlopt -lm -I/home/david/Computation -I/home/david/Computation/gsl/gsl-1.14 -o occBC_hybr.out
*/


#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <nlopt.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_qrng.h>


int Nobs,simT, nsim, Nskill;
int nshock;
int neq;
int verbose 	= 1;
int printlev 	= 2;
int use_anal	= 1;
int opt_alg		= 1;

//declare some parameters
int const Noccs	= 22;
int const Nfac	= 2;// how many factors in the z_i model?
int const Nflag	= 1;
int const Ntfpl	= 0;
int Ns,Nc,Nx;

double 	beta= 0.9967;	// monthly discount
double	nu 	= 0.0; // switching cost --- I don't need this!
double 	sig_psi	=2.837577; //2.477130;	// sd of taste shocks

double 	b	= 0.0; 	// value of unemployment


//At ef=0.850702,sig_psi=2.850816,tau=0.004187
//At ef=0.677420,sig_psi=2.500327,tau=0.005802
//At ef=0.780289,sig_psi=2.852924,tau=0.002010

double ** chi;
double 	tau 	= 0.0401;//0.003857;
double 	kappa	= 1.36;
double 	phi		= 0.5;	// match elasticity
double 	effic	= 0.280289;//0.747769;
double 	sbar	= 0.0192;	// will endogenize this a la Cheremukhin (2011)

double rhoZ		= 0.961;
double rhozz	= 0.961;
double sig_eps	= 6.9168e-05;

double * corrZ;
gsl_matrix * GammaCoef, *var_eta; 	//dynamics of the factors
gsl_matrix * LambdaCoef,*var_zeta;	//factors' effect on z
gsl_matrix * f_skills;				//skills attached to occupations

// data moments that are not directly observable
double avg_sep	= 0.008;
double avg_fnd	= 0.3323252;
double avg_urt	= 0.05;
double avg_wg	= 0.06;  // this is the 5 year wage growth rate: make tau fit this
double med_dr	= 13;
double chng_pr	= 0.45;
double sk_wg[] = {.01774233, .02444173,-.04125535, .01427091,0.1753615};

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
	gsl_matrix *gld,*tld;
};
struct st_wr{
	struct sys_coef * sys;
	struct sys_sol * sol;
	struct aux_coef * sim;
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
int theta(gsl_matrix * tld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz);
int xprime(gsl_matrix * xp, gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_matrix * x, const gsl_vector * Zz);

// results:
int ss_moments(struct aux_coef * ssdat, gsl_vector * ss, gsl_matrix * xss);
int sim_moments(struct aux_coef* simdat,gsl_vector * ss,gsl_matrix * xss,struct sys_sol * sol,struct sys_coef * sys);
int dur_dist(struct dur_moments * s_mom, const gsl_matrix * fnd_l_hist, const gsl_matrix * x_u_hist);

int TGR(gsl_vector* u_dur_dist, gsl_vector* opt_dur_dist, gsl_vector * fnd_dist, gsl_vector * opt_fnd_dist,
		double *urt, double *opt_urt,
		gsl_vector * ss,struct sys_sol * sol,struct sys_coef * sys);

// Utilities
int QZwrap(gsl_matrix * A, gsl_matrix *B, //gsl_matrix * Omega, gsl_matrix * Lambda,
		gsl_matrix * Q, gsl_matrix * Z, gsl_matrix * S, gsl_matrix * T);
int QZdiv(double stake, gsl_matrix * S,gsl_matrix * T,gsl_matrix * Q,gsl_matrix * Z);
int QZswitch(int i, gsl_matrix * A,gsl_matrix * B,gsl_matrix * Q,gsl_matrix * Z);

void VARest(gsl_matrix * Xdat,gsl_matrix * coef, gsl_matrix * varcov);

double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg);
double cal_dist(unsigned n, const double *x, double *grad, void* params);

int main(){
	int i,s,t,n,l,d,Nvaren,Nvarex,status;
	double chi_ld;
	status = 0;
	//initialize parameters

	// this is from 1996-2007

	//double sk_wg[] = {-.02097734 ,   .04767405  , -.05522915  ,  .0669940,  .17053784};
	Nskill = 4;
	// with a mean:
	//double sk_wg[] = {.0433766,.0106902,-.0102159,-.0618296,.5426171};
	// this is from 2008 - 2011
	// double sk_wg[] ={.04099066    .05934782   -.05383748    .05959541,  .92449145}
	// double sk_wg[] = {.05619771,.01231506,.00093825,-.05426961, .50018398}; //skills 1-4 and constant
	/*
	 *      Variable |       Obs        Mean    Std. Dev.       Min        Max
		-------------+--------------------------------------------------------
      	  	  chi_ld |       391    .5009007    .2361208  -.1590407   1.159409
	 *
	*/
	f_skills = gsl_matrix_alloc(Noccs,6);
	readmat("f_skills.csv",f_skills);
	//take out the rowmean from skills
	for(d=0;d<Noccs;d++){
		double rmean = 0.0;
		for(i=0;i<Nskill;i++)
			rmean += gsl_matrix_get(f_skills,d,i+1)/4.0;
	}

	chi = (double**)malloc(sizeof(double*)*Noccs+1);
	for(l=0;l<Noccs+1;l++){
		chi[l] = (double*)malloc(sizeof(double*)*Noccs);
	}
	for(d=0;d<Noccs;d++)
		chi[0][d] = 0.5;
	for(l=1;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			chi[l][d] = 0.0;
			for(i=0;i<Nskill;i++){
				chi[l][d] += fabs(gsl_matrix_get(f_skills,d,i) - gsl_matrix_get(f_skills,l-1,i) )*sk_wg[i];
			}
			chi[l][d] += sk_wg[Nskill];
			chi[l][d] = 0.99; /// THIS IS DIAGNOSTIC
		}
	}
	double chi_lb = 0.01;
	double chi_ub = 0.99;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			double child = chi[l][d];
			if(chi[l][d]<=chi_lb)
				child = chi_lb;
			if(chi[l][d]>=chi_ub)
				child = chi_ub;
			chi[l][d] = 1.0 - child;
		}
	}
	for(d=0;d<Noccs;d++)
		chi[d+1][d] = 1.0;

	corrZ = (double*)malloc(sizeof(double)*Noccs);
	for(l=0;l<Noccs;l++)
		corrZ[l] = 0.0;//0.25 + ((double)l/(double) Noccs/2.0 );
	GammaCoef	= gsl_matrix_calloc(Nfac,(Nflag+1));
	LambdaCoef	= gsl_matrix_calloc(Noccs,Nfac*(Nflag+1));
	var_eta		= gsl_matrix_calloc(Nfac,Nfac);
	var_zeta		= gsl_matrix_calloc(Noccs,Noccs);
	readmat("Gamma.csv",GammaCoef);
	readmat("Lambda.csv",LambdaCoef);
	readmat("var_eta.csv",var_eta);
	readmat("var_zeta.csv",var_zeta);
	if(printlev>=2){
		printmat("readGamma.csv",GammaCoef);
		printmat("readLambda.csv",LambdaCoef);
	}

	Ns = Noccs+1 + Noccs*(Noccs+1);//if including x: pow(Noccs+1,2) + Noccs+1 + 2*Noccs*(Noccs+1);
	Nc = 2*Noccs*(Noccs+1);
	Nx = Noccs+1 + Ntfpl + Nfac*(Nflag+1);


	Nvaren = Ns + Nc;
	Nvarex = Nx;


	gsl_vector* ss = gsl_vector_alloc(Nvaren);

	// allocate system coefficients and transform:

	struct sys_coef sys;

	sys.COV = (double*)malloc(sizeof(double)*Nvaren);
/*
 * System is:
 * 	A0 s =  A1 Es'+ A2 c + A3 Z
 * 	F0 c  = F1 Es'+ F2 s'+ F3 Z
 *
 */

	for(l=0;l<Nvaren;l++)
		sys.COV[l] = 1.0;
	sys.A0	= gsl_matrix_calloc(Ns,Ns);
	sys.A1	= gsl_matrix_calloc(Ns,Ns);
	sys.A2	= gsl_matrix_calloc(Ns,Nc);
	sys.A3	= gsl_matrix_calloc(Ns,Nvarex);

	sys.F0	= gsl_matrix_calloc(Nc,Nc);
	sys.F1	= gsl_matrix_calloc(Nc,Ns);
	sys.F2	= gsl_matrix_calloc(Nc,Ns);
	sys.F3	= gsl_matrix_calloc(Nc,Nvarex);

	sys.N  = gsl_matrix_calloc(Nvarex,Nvarex);
	sys.S	= gsl_matrix_calloc(Nvarex,Nvarex);

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
	simT = 1000;
	struct aux_coef simdat;
	simdat.data_moments = gsl_matrix_alloc(2,5); // pick some moments to match
	simdat.moment_weights = gsl_matrix_alloc(2,5); // pick some moments to match
	simdat.draws = gsl_matrix_calloc(simT,Nvarex); // do we want to make this nsim x simT

	randn(simdat.draws,6071984);

	simdat.data_moments->data[1] = avg_fnd;
/*	simdat.data_moments->data[0] = avg_sep;
	simdat.data_moments->data[2] = avg_urt;*/
	simdat.data_moments->data[2] = chng_pr;
	simdat.data_moments->data[0] = avg_wg;
	// row 2 are the coefficients on 4 skills and a constant
	for(i=0;i<Nskill+1;i++)
		gsl_matrix_set(simdat.data_moments,1,i,sk_wg[i]);


	simdat.moment_weights->data[0] = 1.0;
	simdat.moment_weights->data[1] = 10.0;
	simdat.moment_weights->data[2] = 0.50;

	/* Calibration Loop!
	effic 	= x[0];
	sig_psi = x[3];
	nu		= x[1];
	tau 	= x[2]; */
	double x0[]	= {effic,sig_psi,tau,sk_wg[0],sk_wg[1],sk_wg[2],sk_wg[3],sk_wg[4]};
	double lb[]	= {0.005,0.0,0.0001,x0[3]*(0.7),x0[4]*(0.7),x0[5]*(0.7),x0[3]*(0.7)}; // what to do about sig_psi?  Skipping that right now
	double ub[]	= {1.75,3.0,0.5,x0[3]*(1.3),x0[4]*(1.3),x0[5]*(1.3),x0[3]*(1.3)};
	for(i=0;i<7;i++){
		if (lb[i]>=ub[i]){
			double lbi = lb[i];
			lb[i] = ub[i];
			ub[i] = lbi;
		}
	}


	struct st_wr st;
	st.sim = &simdat;
	st.sol = &sol;
	st.sys = &sys;

	int printlev_old = printlev;
	printlev =1;
	//			f		lb ub n x0 param, ftol xtol, ctol , algo
	//double objval = bndCalMinwrap(&cal_dist,lb,ub,3,x0,&st,1e-7,1e-7,0.0,opt_alg);
	printlev =printlev_old;

	status += sol_ss(ss,xss,&sys);
	if(verbose>=1 && status ==0) printf("Successfully computed the steady state\n");
	if(verbose>=1 && status ==1) printf("Broke while computing steady state\n");
	status += sys_def(ss,&sys);
	if(verbose>=1 && status >=1) printf("System not defined\n");
	if(verbose>=1 && status ==0) printf("System successfully defined\n");



	if(verbose>=2) printf("Now defining the 1st order solution to the dynamic model\n");
	status += sol_dyn(ss, &sol, &sys);
	if(verbose>=1 && status ==0) printf("System successfully solved\n");
	if(verbose>=1 && status >=1) printf("System not solved\n");
	status += sim_moments(&simdat,ss,xss,&sol,&sys);

	//	TGR experiment
	gsl_vector * u_dur_dist = gsl_vector_alloc(5);
	gsl_vector * opt_dur_dist = gsl_vector_alloc(5);
	gsl_vector * fnd_dist	= gsl_vector_alloc(Noccs+1);
	gsl_vector * opt_fnd_dist = gsl_vector_alloc(Noccs+1);
	double urt,opt_urt;
	status +=  TGR(u_dur_dist,opt_dur_dist,fnd_dist,opt_fnd_dist,&urt,&opt_urt,ss,&sol,&sys);



	//status += sim_moments(&simdat,ss,xss,&sol,&sys);


	gsl_matrix_free(simdat.draws);
	gsl_matrix_free(sys.A0);gsl_matrix_free(sys.A1);gsl_matrix_free(sys.A2);gsl_matrix_free(sys.A3);
	gsl_matrix_free(sys.F0);gsl_matrix_free(sys.F1);gsl_matrix_free(sys.F2);gsl_matrix_free(sys.F3);
	gsl_matrix_free(sys.N);gsl_matrix_free(sys.S);free(sys.COV);


	gsl_matrix_free(sol.P1);gsl_matrix_free(sol.P2);gsl_matrix_free(sol.P0);gsl_matrix_free(sol.PP);
	gsl_vector_free(ss);gsl_matrix_free(xss);

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

	
	int Ns,Nc,Nx, i, maxrecur=1000; // these might be different from the globals of the same name.... but should not be
	int status = 0;
	Ns = (sys->F1)->size2;
	Nc = (sys->F1)->size1;
	Nx = (sys->F3)->size2;

	if(verbose>=2) printf("Ns=%d,Nc=%d,Nx=%d\n",Ns,Nc,Nx);

	gsl_matrix * invF0 	= gsl_matrix_calloc(Nc,Nc);
	gsl_matrix * A2invF0= gsl_matrix_calloc(Ns,Nc);
	status += inv_wrap(invF0,sys->F0);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sys->A2,invF0,0.0,A2invF0);

	gsl_matrix* A2F0F1	= gsl_matrix_calloc(Ns,Ns);

	// Find P1
	gsl_matrix * A2invF0F1 = gsl_matrix_calloc(Ns,Ns);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,A2invF0,sys->F1,0.0,A2invF0F1);
	gsl_matrix_memcpy(sol->P1,sys->A1);
	gsl_matrix_add(sol->P1,A2invF0F1);
	gsl_matrix_free(A2F0F1);

	// Find P0
	gsl_matrix * A2invF0F2 = gsl_matrix_calloc(Ns,Ns);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,A2invF0,sys->F2,0.0,A2invF0F2);
	gsl_matrix_memcpy(sol->P0,sys->A0);
	gsl_matrix_sub(sol->P0,A2invF0F2);
	gsl_matrix_free(A2invF0F2);

	//Find P2
	gsl_matrix * A2invF0F3 = gsl_matrix_calloc(Ns,Nx);
	status += gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,A2invF0,sys->F3,0.0,A2invF0F3);
	gsl_matrix_memcpy(sol->P2,sys->A3);
	gsl_matrix_sub(sol->P2,A2invF0F3);
	gsl_matrix_free(A2invF0F3);

	if(printlev>=2){
		printmat("P0.csv",sol->P0);
		printmat("P1.csv",sol->P1);
		printmat("P2.csv",sol->P2);
	}

	// and forward looking for PP

	gsl_matrix * invP0	= gsl_matrix_calloc(Ns,Ns);
	gsl_matrix * P1invP0= gsl_matrix_calloc(Ns,Ns);
	status += inv_wrap(invP0,sol->P0);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,invP0,0.0,P1invP0);


	gsl_matrix * P1invP0P2N 	= gsl_matrix_calloc(Ns,Nx);
	gsl_matrix * P1invP0P2NN 	= gsl_matrix_calloc(Ns,Nx);
	gsl_matrix * invP1PP 		= gsl_matrix_calloc(Ns,Nx);

	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0,sol->P2,0.0,P1invP0P2N);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0P2N,sys->N,0.0,P1invP0P2NN);
	gsl_matrix_memcpy(P1invP0P2N,P1invP0P2NN);
	double mnorm = norm1mat(P1invP0P2N);
	double mnorml=1.0;
	for(i = 0;i<maxrecur;i++){
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0P2N,sys->N,0.0,P1invP0P2NN);
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,P1invP0,P1invP0P2NN,0.0,P1invP0P2N);
		// check convergence in the 1 norm
		mnorm = norm1mat(P1invP0P2N);
		if(mnorm>=1e20){
			status++;
			break;
		}
		if(fabs(mnorm)<1e-6)
			break;
		mnorml = mnorm;
		gsl_matrix_add(invP1PP,P1invP0P2N);
	}
	status = i<maxrecur-1? status : status+1;
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,sol->P1,invP1PP,0.0,sol->PP);
	gsl_matrix_add(sol->PP,sol->P2);


	gsl_matrix_free(invP1PP);
	gsl_matrix_free(invP0);gsl_matrix_free(P1invP0);gsl_matrix_free(P1invP0P2NN);gsl_matrix_free(P1invP0P2N);


	gsl_matrix_free(A2invF0);gsl_matrix_free(invF0);

	if(printlev>=2)
		printmat("PP.csv",sol->PP);

	return status;
}

/*
 * Solve for the steady state
 */

int sol_ss(gsl_vector * ss, gsl_matrix * xss, struct sys_coef * sys){
	// will solve the ss with
	//vars = [ x U_l0 U_ld J_ld gld wld thetald] in row-major order, i.e. {(0,1),(0,2),...,(J,J)}

	int status,l,d,dd,ll,iter,itermax,i,breakflag=0;
	
	gsl_vector * W_l0,*W_ld;
	gsl_vector * gld,*lgld,*wld,*thetald,*findrt;
	gsl_vector * ret_d;
	gsl_matrix * pld,*x;//,*lx;

	double max_tss = 0.5;
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
	
	pld	= gsl_matrix_alloc(Noccs+1,Noccs);
	
	// initialize the policies:
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			gsl_vector_set(gld,l*Noccs+d,1.0/(double)Noccs);
			gsl_vector_set(wld,l*Noccs+d, phi*chi[l][d]);
			gsl_vector_set(thetald,l*Noccs+d,pow(0.5/effic,1.0/(1.0-phi)));  //pow(kappa/(effic*(1.0-phi)*(chi[l][d] - gsl_vector_get(wld,l*Noccs+d))),-1.0/phi));
			gsl_matrix_set(pld,l,d,effic*pow(thetald->data[l*Noccs+d],1.0-phi));
			W_l0->data[l] += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*(chi[l][d] - b);
		}
		W_l0->data[l] += b;
	}
	itermax = 1000;
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
			gsl_vector_set(W_ld,l*Noccs+d,((1.0-sbar)*chi[l][d] + sbar*W_l0->data[l])/(1.0-(1.0-sbar)*beta));
		}
		// W_ld
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
			if(l-1!=d){
				gsl_vector_set(W_ld,l*Noccs+d,
					((1.0-tau)*((1.0-sbar)*chi[l][d]+sbar*W_l0->data[l] ) + tau*W_ld->data[(d+1)*Noccs+d]  )
						/(1.0-(1.0-sbar)*(1.0-tau)*beta));
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
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double nud = l == d+1? 0:nu;
				double post = -kappa/effic*pow(gsl_vector_get(thetald,l*Noccs+d),phi);
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l];
				Wdif = Wdif<0.0 ?  0.0 : Wdif;
				gsl_vector_set(ret_d,d,
					gsl_matrix_get(pld,l,d)*(chi[l][d] -nud - b + post + beta*Wdif)
				);
			}
			double sexpret_dd =0.0;
			for(dd=0;dd<Noccs;dd++)
				sexpret_dd += exp(sig_psi*gsl_vector_get(ret_d,dd));

			for(d=0;d<Noccs;d++){
				//gld

				// known return in occ d
				double gg = exp(sig_psi*gsl_vector_get(ret_d,d))/sexpret_dd;
				gsl_vector_set(gld,l*Noccs+d,gg);

				//theta
				double nud 	= l-1 !=d ? nu : 0.0;
				double Wdif	= W_ld->data[l*Noccs+d] - W_l0->data[l];
				Wdif = Wdif<0.0 ?  0.0 : Wdif;
				double theta =	pow(effic*(1.0-phi)/kappa,1.0/phi)*
						pow(chi[l][d] - nud -b +beta*Wdif,1.0/phi) ;
				theta = theta<0? 1e-5 :  theta;
				theta = theta > pow(max_tss/effic,1.0/(1.0-phi)) ? pow(max_tss/effic,1.0/(1.0-phi)): theta;
				if(verbose>=2) printf("t0=%f \n",theta);
				gsl_vector_set(thetald,l*Noccs+d,theta);
			}
			// adds to 1
			double gsum = 0.0;
			for(d =0 ;d<Noccs;d++)	gsum += gld->data[l*Noccs+d];
			for(d =0 ;d<Noccs;d++)	gld->data[l*Noccs+d] /= gsum;
		}
		// check convergence in g_ld
		maxdist=0;
		int ii_max =0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double glddist = fabs(gsl_vector_get(gld,l*Noccs+d) - gsl_vector_get(lgld,l*Noccs+d));
				ii_max = glddist> maxdist ? l : ii_max;
				maxdist = glddist> maxdist ? glddist : maxdist;
			}
		}
		if(verbose >=2) printf("g_ld dist = %f\n",maxdist);
		if(breakflag == 1)
			break;
		if(maxdist<1e-9 && iter>5){
			breakflag=1;}
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++)
				gsl_matrix_set(pld,l,d,effic*pow(thetald->data[l*Noccs+d],1.0-phi));
		}
		for(l=0;l<Noccs+1;l++){
			double findrt_ld =0.0;
			for(d=0;d<Noccs;d++)
				findrt_ld += gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d);
			gsl_vector_set(findrt,l,findrt_ld);
		}
		// W_l0

		for(l=0;l<Noccs+1;l++){
			double W_0 =0.0;
			for(d=0;d<Noccs;d++){
				double nud = l==d+1? 0.0:nu;
				double post = - kappa/effic*pow(gsl_matrix_get(pld,l,d),phi);
				W_0 +=gld->data[l*Noccs+d]*gsl_matrix_get(pld,l,d)*(chi[l][d] - nud
						+ post
						+ beta*gsl_vector_get(W_ld,l*Noccs+d));
			}
			W_0 += (1.0-gsl_vector_get(findrt,l))*b;
			W_0 /= (1.0 - beta*(1.0-gsl_vector_get(findrt,l)));
			W_l0->data[l] = W_0;
		}

	}
	status = iter<itermax? 0 : 1;
	if(printlev>=2){
		printvec("W_l0_i.csv",W_l0);
		printvec("W_ld_i.csv",W_ld);
		printvec("gld_i.csv",gld);
		printvec("thetald_i.csv",thetald);
		printvec("findrt_i.csv",findrt);
	}

	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			W_ld->data[l*Noccs+d] =W_ld->data[l*Noccs+d]< W_l0->data[l] ? W_l0->data[l] : W_ld->data[l*Noccs+d];
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
			gsl_matrix_set(Pxx0,0,0*(Noccs+1)+d,
					-(1.0-tau)*sbar);
		}
	}
	//x_l0 : l>0
	for(l=1;l<Noccs+1;l++){
		gsl_matrix_set(Pxx1,l*(Noccs+1),l*(Noccs+1),
				(1.0-findrt->data[l]) );
		gsl_matrix_set(Pxx0,l*(Noccs+1),l*(Noccs+1)+l, -sbar );
	}
	//x_0d : d>0
	for(d=1;d<Noccs+1;d++){
		gsl_matrix_set(Pxx1,d,d,
				(1.0-tau)*(1.0-sbar));
		gsl_matrix_set(Pxx1,d,0,
				gsl_vector_get(gld,d-1)*gsl_matrix_get(pld,0,d-1));
	}
	//x_ld : l,d>0
	for(l=1;l<Noccs+1;l++){
		for(d=1;d<Noccs+1;d++){
			if(l!=d){
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,
						(1.0-tau)*(1.0-sbar));
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+0,
						gsl_vector_get(gld,l*Noccs + d-1)*gsl_matrix_get(pld,l,d-1));
			}else{
				gsl_matrix_set(Pxx1,l*(Noccs+1)+d,l*(Noccs+1)+d,(1.0-sbar));
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

	for(l=0;l<Nxx;l++){
		for(d=0;d<Nxx;d++)
			gsl_matrix_set(xtransT,d,l,gsl_matrix_get(xtrans,l,d));
	}
	gsl_eigen_nonsymmv(xtransT, eval, xxmat, w);

	gsl_eigen_nonsymmv_sort (eval, xxmat, GSL_EIGEN_SORT_ABS_DESC);

	int ii =0;
	gsl_complex eval_l = gsl_vector_complex_get(eval, ii);
	if(eval_l.dat[0]>1.0001){
		ii =0;
		double dist1 = 1.0;
		for(l=0;l<Nxx;l++){
			gsl_complex eval_l = gsl_vector_complex_get(eval, l);
			if(fabs(eval_l.dat[0]-1.0 )<dist1 ){
				ii=l;
				dist1 = fabs(eval_l.dat[0]-1.0 );
			}
		}
	}

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

	/*gsl_vector * xvec = gsl_vector_alloc((Noccs+1)*(Noccs+1));
	vec(x,xvec);
	for(i=0;i<(Noccs+1)*(Noccs+1);i++){
		gsl_vector_set(ss,i,xvec->data[i]);
		ssi++;
	}
	gsl_vector_free(xvec);
	*/

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
		gsl_vector_set(ss,ssi,thetald->data[i]);
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
	int l,d,status,dd,f;
//	int wld_i, tld_i, x_i,gld_i;
	int Nvarex 	= (sys->A3)->size2;
	int Nvaren 	= ss->size;
	int Notz 	= 1+ Ntfpl +Nfac*(Nflag+1);


	gsl_matrix *N0,*N1,*invN0;
	if(Nvarex != Noccs+1+ Ntfpl +Nfac*(Nflag+1))
		printf("Size of N matrix is wrong!");

	N0 		= gsl_matrix_calloc(Nvarex,Nvarex);
	invN0 	= gsl_matrix_calloc(Nvarex,Nvarex);
	N1 		= gsl_matrix_calloc(Nvarex,Nvarex);
	gsl_matrix_set_identity(N0);

	for(l=0;l<Noccs;l++)
		gsl_matrix_set(N0,l+ Notz,0,-corrZ[l]);
	for(f=0;f<Nfac;f++){
	for(d=0;d<Noccs;d++)
		gsl_matrix_set(N0,d+Notz,f+1+Ntfpl,
				-gsl_matrix_get(LambdaCoef,d,f));
	}
	/* First partition, N1_11 =
	 *
	 * (rhoZ  0 )
	 * ( I    0 )
	 */
	gsl_matrix_set(N1,0,0,rhoZ);
	for(l=0;l<Ntfpl;l++)
		gsl_matrix_set(N1,l,l,1.0);

	/* Second partition, N1_22 =
	 * (Gamma  0 )
	 * ( I     0 )
	 */
	set_matrix_block(N1,GammaCoef,1+Ntfpl,1+Ntfpl);
	for(f=0;f<Nfac*(Nflag);f++)
		gsl_matrix_set(N1,1+Ntfpl+Nfac+f,1+Ntfpl+f,1.0);
	/*
	 * Fourth partition, N1_32 =
	 * Lambda_{1:(Nflag-1)}
	 */
	gsl_matrix_view N1_Lambda = gsl_matrix_submatrix(LambdaCoef, 0, Nfac, LambdaCoef->size1, LambdaCoef->size2-Nfac);
	set_matrix_block(N1,&N1_Lambda.matrix,Notz,1+Ntfpl);

	/*
	 * Final partition, N1_33 =
	 *  rhozz*I
	 */
	for(l=0;l<Noccs;l++)
		gsl_matrix_set(N1,l+Notz,l+Notz,rhozz);

	for(l=0;l<Noccs;l++)
		gsl_matrix_set(N0,l+1,0,0.5*corrZ[l]);
	inv_wrap(invN0,N0);
	status = gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,invN0,N1,0.0,sys->N);

	gsl_matrix_set_identity(sys->S);
	gsl_matrix_set(sys->S,0,0,
			sig_eps);
	set_matrix_block(sys->S,var_eta,1+Ntfpl,1+Ntfpl);
	for(l=0;l<Noccs;l++){
		for(d=0;d<Noccs;d++){
			double Sld = gsl_matrix_get(var_zeta,l,d);
			Sld = l!=d && (Sld>-5e-5 && Sld<5e-5) ? 0.0 : Sld;
			gsl_matrix_set(sys->S,Notz+l,Notz+d,Sld);

		}
	}

	status += gsl_linalg_cholesky_decomp(sys->S);
	// set as zero innovations to carried lag terms
	for(l=0;l<Nfac*Nflag;l++)
		gsl_matrix_set(sys->S,1+Ntfpl+Nfac+l,1+Ntfpl+Nfac+l,0.0);
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
	sys_st_diff(ss,sys->A0,sys->A2,sys->A1,Zz);
	sys_co_diff(ss,sys->F2,sys->F0,sys->F1,Zz);
	sys_ex_diff(ss,sys->A3,sys->F3);

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


	gsl_vector_free(Zz);
	return status;
}

int sys_st_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	int l,ll,d,dd,status;
	int tld_i,x_i,gld_i,Wld_i,Wl0_i;
	x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);

	int ss_tld_i, ss_x_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;
	ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	status =0;
	double Z = xx->data[0];

	//1st order derivatives:
	for(l=0;l<Noccs+1;l++){
			double findrtl = 0.0;
			for(d=0;d<Noccs;d++)
				findrtl += effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*ss->data[ss_gld_i+l*Noccs+d];
			double gdenom = 0.0;
			for(d=0;d<Noccs;d++){
				double nud = l ==d+1? 0.0 : nu;
				double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
				contval <0.0 ? 0.0 : contval;
				double post = -kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+d],phi);
				gdenom +=exp(sig_psi*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*
						(chi[l][d]-b+nud+ post + beta*contval));
			}
	/*		// xl0
			gsl_matrix_set(Dvar,x_i+l*(Noccs+1),x_i+l*(Noccs+1),1.0-findrtl);
			gsl_matrix_set(Dvar,x_i+l*(Noccs+1),x_i+l*(Noccs+1)+l,(1.0-findrtl)*sbar);
			gsl_matrix_set(Dvar_tp1,x_i+l*(Noccs+1),x_i+l*(Noccs+1),1.0);
	*/
			// Wl0
			gsl_matrix_set(Dst_tp1,Wl0_i+l,Wl0_i+l,	beta*(1.0-findrtl));
			gsl_matrix_set(Dst,Wl0_i+l,Wl0_i+l,1.0);
			for(d=0;d<Noccs;d++){
				double zd = xx->data[d+1];
				// Wld
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,1.0);
				if(d+1 ==l){
					gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d, Wld_i+l*Noccs+d,(1.0-sbar)*beta);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l, -sbar);
				}
				else{
					gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d,Wld_i+l*Noccs+d,beta*(1.0-tau)*(1.0-sbar));
					//gsl_matrix_set(Dst_tp1,Wld_i + l*Noccs+d,Wld_i+(d+1)*Noccs+d,beta*tau*(1.0-sbar));
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d,Wld_i+(d+1)*Noccs+d,-tau);
					gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l,-(1-tau)*sbar);
				}
				//gsl_matrix_set(Dst,Wld_i + l*Noccs+d, Wl0_i+l,-(1-tau)*sbar - tau*sbar); // these will be different separation rates
				// Wl0

				// ss_ret = -nu - kappa/theta^ld/p^ld + mu^ld - mu^l0
				double post 	= - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+d],phi);
				double ss_ret 	= d+1==l ? 0.0 : -nu;
				ss_ret += chi[l][d]*exp(Z+zd) -b + post;
				double contval = beta*(ss->data[ss_Wld_i+l*Noccs+d] -ss->data[ss_Wl0_i+l] );
				ss_ret = contval>0? ss_ret+contval : ss_ret;
				ss_ret = ss_ret>0? ss_ret : 0.0;
				gsl_matrix_set(Dco,Wl0_i+l,tld_i+l*Noccs+d,
						effic*(1.0-phi)*pow(ss->data[ss_tld_i+l*Noccs+d],-phi)
						*ss->data[ss_gld_i+l*Noccs+d]*ss_ret
						- ss->data[ss_gld_i+l*Noccs+d]*kappa*phi);
						//-effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)
						//*kappa/effic*phi*pow(ss->data[ss_tld_i+l*Noccs+d],phi-1.0));
				double fnd_ld =effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi);
				gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,
						fnd_ld*ss_ret);
				//if(fnd_ld*ss_ret>1.0) /// TAKE THIS OUT???????????????????????
				//	gsl_matrix_set(Dco,Wl0_i+l,gld_i+l*Noccs+d,0.99);

				gsl_matrix_set(Dst_tp1,Wl0_i+l,Wld_i+l*Noccs+d,
						beta*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*ss->data[ss_gld_i+l*Noccs+d]);
			}
	}
	return status;
}
int sys_co_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco, gsl_matrix* Dst_tp1, gsl_vector * xx){
	int l,d,dd,status;
	int tld_i,gld_i,Wld_i,Wl0_i;//x_i,
	double Z,zd;
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);

	int ss_tld_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i
	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	status =0;
	Z = xx->data[0];

	// 1st order derivatives
	for(l=0;l<Noccs+1;l++){
		double findrtl = 0.0;
		for(d=0;d<Noccs;d++)
			findrtl += effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*ss->data[ss_gld_i+l*Noccs+d];
		double gdenom = 0.0;
		for(d=0;d<Noccs;d++){
			double nud = l ==d+1? 0.0 : nu;
			zd = xx->data[d+1];
			double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			contval <0.0 ? 0.0 : contval;
			gdenom +=exp(sig_psi*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*
					(chi[l][d]*exp(Z+zd)-b+nud+beta*contval));
		}

		for(d=0;d<Noccs;d++){
			zd = xx->data[d+1];
			double nud = l==d+1 ? 0: nu;
			// tld
			double tinside = (chi[l][d] - nud - b + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]))
					*effic*(1.0-phi)/kappa;
			tinside = tinside <0 ? (chi[l][d])*effic*(1.0-phi)/kappa: tinside;
			gsl_matrix_set(Dco,tld_i + l*Noccs+d,tld_i + l*Noccs+d, 1.0);
			gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wld_i + l*Noccs+d, //dt/dWld
				(beta*(1.0-phi)*effic/phi/kappa)*pow(tinside,1.0/phi-1.0)
				);

			gsl_matrix_set(Dst_tp1,tld_i + l*Noccs+d,Wl0_i + l, //dt/dWl0
				-1.0*(beta*(1.0-phi)*effic/phi/kappa)*pow(tinside,1.0/phi-1.0)
				);


			// gld
			gsl_matrix_set(Dco,gld_i+l*Noccs+d,gld_i+l*Noccs+d,1.0);
			// exp(sig_psi*effic*pow(ss->data[tld_i+l*Noccs+d],1.0-phi)*(ss->data[wld_i+l*Noccs+d]-b+nud+beta*(ss->data[Uld_i+l*Noccs+d]-ss->data[Ul0_i+l])))
			double post=  - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+d],phi);
			double contval = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			contval <0.0 ? 0.0 : contval;
			double arr_d 	= (chi[l][d]*exp(Z+zd) -b + post +nud+beta*contval);
			double pld		= effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi);
			double ret_d 	= pld*arr_d;
			gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+d,
				(beta*sig_psi*pld*exp(sig_psi*ret_d)*gdenom - beta*pld*sig_psi*exp(sig_psi*ret_d)*exp(sig_psi*ret_d))/pow(gdenom,2)
				//	-(gdenom + (beta*sig_psi*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*ss->data[ss_gld_i+l*Noccs+d] -1.0)*
				//		exp(sig_psi*ret_d) )*exp(sig_psi*ret_d) )/pow(gdenom,2)
				);

			/*
			 * dg/dWl0 = 0
			 *
			*/
			gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+d,-1.0*
				(sig_psi*(1.0-phi)*effic*pow(ss->data[ss_tld_i+l*Noccs+d],-phi)*(arr_d)*gdenom*exp(sig_psi*ret_d)
					-(sig_psi*(1.0-phi)*effic*pow(ss->data[ss_tld_i+l*Noccs+d],-phi)*(arr_d)*
						exp(sig_psi*ret_d) )*exp(sig_psi*ret_d) )/pow(gdenom,2)
				);
			for(dd=0;dd<Noccs;dd++){
			if(dd!=d){
				double nudd = l==dd+1 ? 0: nu;
				double zdd = xx->data[dd+1];
				double contval = (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
				contval <0.0 ? 0.0 : contval;
				double postdd	=  - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+dd],phi);
				double arr_dd 	= (chi[l][dd]*exp(Z+zdd)-b + postdd+nudd+beta*contval);
				double pldd		= effic*pow(ss->data[ss_tld_i+l*Noccs+dd],1.0-phi);
				double ret_dd 	= pldd*arr_dd;

				gsl_matrix_set(Dst_tp1,gld_i+l*Noccs+d,Wld_i+l*Noccs+dd,
					( - pldd*beta*sig_psi*exp(sig_psi*ret_d)*exp(sig_psi*ret_d))/pow(gdenom,2)
					);

				gsl_matrix_set(Dco,gld_i+l*Noccs+d,tld_i+l*Noccs+dd,-1.0*
					( -(sig_psi*(1.0-phi)*effic*pow(ss->data[ss_tld_i+l*Noccs+dd],-phi)*(arr_dd)*//-(gdenom + (sig_psi*(1.0-phi)*effic*pow(ss->data[ss_tld_i+l*Noccs+dd],-phi)*arr_dd -1.0)*
							exp(sig_psi*ret_dd) )*exp(sig_psi*ret_dd) )/pow(gdenom,2)
					);
			}
			}


		}
	}
	return status;
}


int sys_ex_diff(gsl_vector * ss, gsl_matrix * Dst, gsl_matrix * Dco){
// derivatives for exog
	int tld_i,gld_i,Wld_i,Wl0_i;//x_i,
	int Notz = 1 + Ntfpl + Nfac*(Nflag+1);
	//x_i 	= 0;
	Wl0_i	= 0;
	Wld_i	= Wl0_i + Noccs+1;
	// for Dco:
	gld_i	= 0;
	tld_i	= gld_i + Noccs*(Noccs+1);

	int ss_tld_i, ss_gld_i, ss_Wld_i,ss_Wl0_i;
	int l,d,dd,status;
	//ss_x_i	= 0;
	ss_Wl0_i	= 0; //x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	status =0;

	for (l=0;l<Noccs+1;l++){
		double gdenom = 0.0;
		for(d=0;d<Noccs;d++){
			double post	=  - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+d],phi);
			double nud 	= l ==d+1? 0.0 : nu;
			double ret_d=(chi[l][d]-b- nud + post + beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]));
			ret_d 		= ret_d <0.0 ? 0.0 : ret_d;
			gdenom +=exp(sig_psi*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*ret_d);
		}
		double dWl0dZ =0.0;
		for(d=0;d<Noccs;d++)
			dWl0dZ += ss->data[ss_gld_i+l*Noccs+d]*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*chi[l][d];
		gsl_matrix_set(Dst,Wl0_i+l,0,dWl0dZ);
		for(d=0;d<Noccs;d++){
			double nud = l ==d+1? 0.0 : nu;
			double post=  - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+d],phi);
								// dWl0/dz
			gsl_matrix_set(Dst,Wl0_i+l,d+Notz,
					ss->data[ss_gld_i+l*Noccs+d]*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi)*chi[l][d]);

			double tinside = (chi[l][d]-nud- b +beta*(ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]))*effic*(1.0-phi)/kappa;
			tinside = tinside<0 ? 0.0 : tinside;
			gsl_matrix_set(Dco,tld_i + l*Noccs+d,0,   // dtheta/dZ
				(1.0/phi)*pow(tinside,1.0/phi-1.0)*chi[l][d]*effic*(1.0-phi)/kappa
				);
			gsl_matrix_set(Dco,tld_i + l*Noccs+d,d+Notz,
					// dtheta/dz
				(1.0/phi)*pow(tinside,1.0/phi-1.0)*chi[l][d]*effic*(1.0-phi)/kappa
				);
			if(l!=d+1){
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,
					// dWld/dZ
					(1.0-sbar)*(1.0-tau)*chi[l][d]+(1.0-sbar)*tau*chi[d+1][d] );
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,
					// dWld/dzd
					((1.0-sbar)*(1.0-tau)*chi[l][d]+ // these differ when separation is endogeneous
							(1.0-sbar)*tau*chi[d+1][d]) );
			}
			else{
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,0,
						// dWld/dZ
						(1.0-sbar)*chi[l][d]);
				gsl_matrix_set(Dst,Wld_i + l*Noccs+d,d+Notz,
						// dWld/dzd
						(1.0-sbar)*chi[l][d]);
			}
			// gld

			// exp(sig_psi*effic*pow(ss->data[tld_i+l*Noccs+d],1.0-phi)*(ss->data[wld_i+l*Noccs+d]-b+nud+beta*(ss->data[Uld_i+l*Noccs+d]-ss->data[Ul0_i+l])))
			double contval  = (ss->data[ss_Wld_i+l*Noccs+d]-ss->data[ss_Wl0_i+l]);
			contval	= contval>0.0? contval : 0.0;

			double arr_d 	= (chi[l][d] -b - nud + post +beta*contval);
			double p_d		= effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi);
			double ret_d 	= p_d*arr_d;
			ret_d = ret_d<0 ? 0.0: ret_d;
			double dnum 	= sig_psi*p_d*exp(sig_psi*ret_d);
			double ddenom	= sig_psi*p_d*exp(sig_psi*ret_d);
			gsl_matrix_set(Dco,gld_i+l*Noccs+d,d+Notz,
				(dnum*gdenom - ddenom*exp(sig_psi*ret_d) )/
						pow(gdenom,2)
				);
			for(dd=0;dd<Noccs;dd++){
			if(dd!=d){
				double nudd = l==dd+1 ? 0: nu;
				double postdd=  - kappa/effic*pow(ss->data[ss_tld_i+l*Noccs+dd],phi);
				double contval	= (ss->data[ss_Wld_i+l*Noccs+dd]-ss->data[ss_Wl0_i+l]);
				contval = contval<0.0 ? 0.0 : contval;
				double arr_dd 	= (chi[l][dd]-b - nudd + postdd +beta*contval);
				double p_dd		= effic*pow(ss->data[ss_tld_i+l*Noccs+dd],1.0-phi);
				double ret_dd 	= p_dd*arr_dd;
				ret_dd = ret_dd< 0 ? 0.0 : ret_dd;
				double ddenom	= sig_psi*p_dd*exp(sig_psi*ret_dd);
				gsl_matrix_set(Dco,gld_i+l*Noccs+d,dd+Notz,
					(-ddenom*exp(sig_psi*ret_dd))/
						pow(gdenom,2)
					);
			}
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
	double s_fnd,s_urt,s_wg, s_chng;
	int ss_tld_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;//, ss_x_i
	//ss_x_i	= 0;
	ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
	ss_Wld_i	= ss_Wl0_i + Noccs+1;
	ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
	ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);


	gsl_vector * Zz = gsl_vector_calloc(Nx);
	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);
	status = 0;

	// calculate the unemployment rate:
	double urtss = 0.0;
	for(l=0;l<Noccs+1;l++)
		urtss += gsl_matrix_get(xss,l,0);
	gsl_vector * x_u = gsl_vector_calloc(Noccs+1);
	for(l=0;l<Noccs+1;l++)
		gsl_vector_set(x_u,l,gsl_matrix_get(xss,l,0)/urtss);


	s_fnd = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			fnd_l->data[l] += ss->data[ss_gld_i+l*Noccs+d]*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi);
		}
		s_fnd += fnd_l->data[l]*gsl_vector_get(x_u,l);
	}

	//generate the wage growth rate and subtract from tau

	s_wg = 0.0;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			s_wg += (chi[d+1][d] - chi[l][d])*gsl_vector_get(x_u,l)*effic*pow(ss->data[ss_tld_i+l*Noccs+d],1.0-phi);

		}
	}
	s_wg *= tau; // dividing by expected time for that growth: 1.0/tau
	s_wg += 1.0;
	//generate the change probability:
	for(l=1;l<Noccs+1;l++){
		double s_chng_l = 1.0 - (ss->data[ss_gld_i+l*Noccs+l-1]*effic*pow(ss->data[ss_tld_i+l*Noccs+l-1],1.0-phi))/
				fnd_l->data[l];
		s_chng += s_chng_l*x_u->data[l];
	}

	gsl_matrix_set(ssdat->data_moments,0,0,avg_wg  - s_wg);
	gsl_matrix_set(ssdat->data_moments,1,0,avg_urt - urtss);
	gsl_matrix_set(ssdat->data_moments,2,0,chng_pr - s_chng);

	gsl_vector_free(fnd_l);
	return status;
}


int sim_moments(struct aux_coef* simdat,gsl_vector * ss,gsl_matrix * xss,struct sys_sol * sol,struct sys_coef * sys){
	/* will compute moments that match
	 * avg_sep
	 * avg_fnd
	 * avg_urt
	 * avg_wg
	 *
	 */
	int status,l,d,di,si,Ndraw;
	double s_fnd,s_urt,s_wg,s_wl,s_chng,m_Zz;
	gsl_matrix * tld = gsl_matrix_alloc(Noccs+1,Noccs);
	gsl_matrix * gld = gsl_matrix_alloc(Noccs+1,Noccs);
	gsl_matrix * xp = gsl_matrix_alloc(Noccs+1,Noccs+1);
	gsl_matrix * x = gsl_matrix_alloc(Noccs+1,Noccs+1);
	gsl_vector * Zz = gsl_vector_calloc(Nx);
	gsl_vector * fnd_l = gsl_vector_calloc(Noccs+1);

	status = 0;


	Ndraw = (simdat->draws)->size1;
	gsl_vector * shocks = gsl_vector_calloc(Nx);
	gsl_vector * Zzl	= gsl_vector_calloc(Nx);
	gsl_matrix * uf_hist	= gsl_matrix_calloc(Ndraw,3);
	gsl_matrix * urt_l	= gsl_matrix_calloc(Ndraw,Noccs);
	gsl_matrix * fnd_l_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
	gsl_matrix * x_u_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);
	gsl_matrix * Zzl_hist	= gsl_matrix_calloc(Ndraw,Noccs+1);

	struct dur_moments s_mom;
	s_mom.Ft = gsl_vector_calloc(5);
	s_mom.dur_qtl = gsl_matrix_calloc(Ndraw,6);
	double urtss = 0.0;
	for(l=0;l<Noccs+1;l++)
		urtss += gsl_matrix_get(xss,l,0);
	gsl_matrix_memcpy(x,xss);

	s_fnd 	= 0.0;
	s_mom.chng_wg	= 0.0;
	s_wl	= 0.0;
	s_chng	= 0.0;
	s_urt	= 0.0;
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

	// Take a draw for Zz
	gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,0);
	gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0, Zz);

	/*run through a few draws without setting anything
	for(di=0;di<50;di++){
		status += xprime(xp,ss,sys,sol,x,Zz);
		gsl_matrix_memcpy(x,xp);
		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);

	}

*/
	for(di = 0;di<Ndraw;di++){
		gsl_vector_set_zero(fnd_l);

		sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += theta(sol->tld, ss, sys, sol, Zz);
		sol->gld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += gpol(sol->gld, ss, sys, sol, Zz);
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double gld_i = gsl_matrix_get(sol->gld,l,d);
				double tld_i = gsl_matrix_get(sol->tld,l,d);
				gld_i = gld_i!=gld_i ? 0 : gld_i;
				tld_i = tld_i!=tld_i ? 0 : tld_i;
				gsl_matrix_set(sol->tld,l,d,tld_i);
				gsl_matrix_set(sol->gld,l,d,gld_i);
			}
		}
		status += xprime(xp,ss,sys,sol,x,Zz);

		for(d=1;d<Noccs+1;d++){
			gsl_matrix_set(urt_l,di,d-1,gsl_matrix_get(xp,d,0)/(gsl_matrix_get(xp,d,0) + gsl_matrix_get(xp,d,d)));
		}

		// mean Z this period
		double m_Zz_t = 0.0;
		for(d=0;d<Noccs;d++)
			m_Zz_t += gsl_vector_get(Zz,1+Ntfpl + Nfac*(Nflag+1)+d)/(double)Noccs;
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
		for(l=0;l<Noccs+1;l++)
			gsl_vector_set(x_u,l,gsl_matrix_get(xp,l,0)/urtp);
		gsl_vector_view x_u_d = gsl_matrix_row(x_u_hist,di);
		gsl_vector_memcpy(&x_u_d.vector,x_u);

		// get the finding rate:
		double d_fnd =0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				double pld_i = effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi);
				fnd_l->data[l] += gsl_matrix_get(sol->gld,l,d)*pld_i;
			}
			d_fnd += fnd_l->data[l]*gsl_vector_get(x_u,l);
		}
		s_fnd += d_fnd/(double)Ndraw;

		gsl_matrix_set(uf_hist,di,2,d_fnd);
		gsl_vector_view fnd_l_di = gsl_matrix_row(fnd_l_hist,di);
		gsl_vector_memcpy(&fnd_l_di.vector,fnd_l);

		//generate the change probability:
		double d_chng = 0.0;
		for(l=1;l<Noccs+1;l++){
			double d_chng_l = 1.0 - (gsl_matrix_get(sol->gld,l,l-1)*effic*pow(gsl_matrix_get(sol->tld,l,l-1),1.0-phi))/
				fnd_l->data[l];
			d_chng += d_chng_l*x_u->data[l];
		}
		//	double max0 = 0.0;
		//	for(d=0;d<Noccs;d++)
		//		max0 = (gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi))>max0 ?
		//				(gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi)):max0;
		//	d_chng += (1.0-max0/fnd_l->data[0])*x_u->data[0];
		d_chng /= x_u->data[0];
		d_chng = d_chng!=d_chng ? s_chng*((double)Ndraw/(double)di ) : d_chng;
		s_chng +=d_chng/(double)Ndraw;

//	Wage growth
		double d_wg = 0.0;
		for(l=0;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				d_wg += gsl_vector_get(x_u,l)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi)*(chi[d+1][d] - chi[l][d]);
			}
		}
		d_wg *= tau; // dividing by expected time for that growth: 1.0/tau
		s_wg += d_wg/(double)Ndraw;

		// wage loss due to change
		double d_wl = 0.0;
		double sx_wl= 0.0;
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				if(l!=d+1){
					d_wl +=gsl_vector_get(x_u,l)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi)*(chi[l][l-1] - chi[l][d]);
					sx_wl +=gsl_vector_get(x_u,l)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi);
				}
			}
		}
		d_wl /=sx_wl;
		s_mom.chng_wg += d_wl/(double)Ndraw;
		gsl_matrix_set(simdat->data_moments,0,3,s_mom.chng_wg);
		gsl_matrix_set(simdat->moment_weights,0,3,0.0);

		// do a regression on the changes in wages:
		// only look at experienced changers, and set chi[0][d] s.t. makes the avg wage loss match
		for(l=1;l<Noccs+1;l++){
			for(d=0;d<Noccs;d++){
				gsl_matrix_set(Wt,(l-1)*Noccs+d,(l-1)*Noccs+d,
						effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi)*
						gsl_matrix_get(sol->gld,l,d)*gsl_vector_get(x_u,l));
				gsl_vector_set(yloss,(l-1)*Noccs+d,
						chi[d+1][d] - chi[l][d] );
				for(si=0;si<Nskill;si++)
					gsl_matrix_set(XX,(l-1)*Noccs+d,si,
							fabs(gsl_matrix_get(f_skills,l-1,si)-gsl_matrix_get(f_skills,d,si) ));

			}
		}
		status += WOLS( yloss, XX ,Wt, coefs_di, er);
		gsl_vector_scale(coefs_di,1.0/(double)Ndraw);
		gsl_vector_add(coefs,coefs_di);

		// advance a period:
		gsl_matrix_memcpy(x,xp);
		gsl_vector_view Zdraw = gsl_matrix_row(simdat->draws,di);

		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->S, &Zdraw.vector, 0.0,shocks);
		gsl_blas_dgemv (CblasNoTrans, 1.0, sys->N, Zz, 0.0,Zzl);
		gsl_vector_add(Zzl,shocks);
		gsl_vector_memcpy(Zz,Zzl);
		gsl_matrix_set(Zzl_hist,di,0,Zz->data[0]);
		for(l=1;l<Noccs;l++)
			gsl_matrix_set(Zzl_hist,di,l,Zz->data[Ntfpl+Nfac*(Nflag+1)+l]);

		if(printlev>=3){
			printmat("sol->tld.csv",sol->tld);
			printmat("sol->gld.csv",sol->gld);
		}
		gsl_vector_free(x_u);
		gsl_matrix_free(sol->tld);
		sol->tld=0;
		gsl_matrix_free(sol->gld);
		sol->gld=0;

	}
	printmat("Zzl_hist.csv",Zzl_hist);
	printmat("x_u_hist.csv",x_u_hist);
	printmat("fnd_l_hist.csv",fnd_l_hist);

	gsl_vector_view momcoef = gsl_matrix_row(simdat->data_moments,1);
	gsl_vector_memcpy(&momcoef.vector,coefs);
	status += dur_dist(&s_mom,fnd_l_hist,x_u_hist);

	if(printlev>=1){
		printmat("uf_hist.csv",uf_hist);
		printvec("Ft.csv",s_mom.Ft);
	}
	if(verbose >=1) printf("The average wage-loss was %f\n",s_wl);

	if(status ==0){
		gsl_matrix_set(simdat->data_moments,0,0,(avg_wg  - s_wg)/avg_wg);
		gsl_matrix_set(simdat->data_moments,1,0,(avg_fnd - s_fnd)/avg_fnd); // this one is twice as important (arbitrary)
		gsl_matrix_set(simdat->data_moments,2,0,(chng_pr - s_chng)/chng_pr);
	}
	else{// penalize it for going in a region that can't be solved
		gsl_matrix_set(simdat->data_moments,0,0,1e6);
		gsl_matrix_set(simdat->data_moments,1,0,1e6);
		gsl_matrix_set(simdat->data_moments,2,0,1e6);
	}

	gsl_matrix_free(Wt);gsl_matrix_free(XX);gsl_vector_free(coefs);gsl_vector_free(coefs_di);
	gsl_vector_free(yloss);	gsl_vector_free(er);
	gsl_vector_free(s_mom.Ft);
	gsl_matrix_free(s_mom.dur_qtl);
	gsl_vector_free(fnd_l);gsl_vector_free(Zzl);gsl_vector_free(shocks);
	gsl_matrix_free(gld);gsl_matrix_free(tld);gsl_matrix_free(uf_hist);
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
	gsl_vector * fnd_dur = gsl_vector_calloc(5);
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
/*
		gsl_sort_vector_index(sort_p, sort_fnd);
		gsl_permutation_reverse(sort_p); // find rate sorted in descending order
		gsl_permute_vector(sort_p, sort_fnd);
		gsl_permute_vector(sort_p, sort_x);

		double cdf_x = 0.0;
		double dur0,dur1;
		double pctls[] = {0.1, 0.25, 0.5, 0.75, 0.9, 1.0};
		int i=0;
		dur0 = 1.0/gsl_vector_get(sort_fnd,0);
		for(l=0;l<Noccs+1;l++){
			cdf_x += sort_x->data[l];
			dur1 = 1.0/sort_fnd->data[l];
			if(cdf_x>pctls[i]){
				gsl_matrix_set(s_mom->dur_qtl,t,i,0.5*dur0 + 0.5*dur1);
				i++;
			}
			dur0 = dur1;
		}
		// compute distribution of finding rates at different durations.
		// Need to compute cum survival for this to get density at various durations
*/
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

	int status,l,d,i,t,Tmo;
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
	*urt = 0.0;
	for(t=0;t<Tmo;t++){// this is going to loop through 2 years of recession
		Zz_TGR->data[0] *= -1.0;
//		for(d=0;d<Noccs;d++)
//			Zz_TGR->data[1+d+Nfac*Nflag] *= -1.0;
		sol->gld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += gpol(sol->gld,ss, sys,  sol, Zz_TGR);
		sol->tld = gsl_matrix_calloc(Noccs+1,Noccs);
		status += theta(sol->tld,ss, sys,  sol, Zz_TGR);
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
				fnd_l_t_l += gsl_matrix_get(sol->gld,l,d)*effic*pow( gsl_matrix_get(sol->tld,l,d) ,1.0-phi);
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
					sw_l+=gsl_matrix_get(sol->gld,l,d)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi);
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
					s_wl += (chi[l][l-1] - chi[l][d])*gsl_vector_get(x_u,l)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi);
					x_wl += gsl_vector_get(x_u,l)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi);
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
	int ss_tld_i, ss_x_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;
		ss_x_i		= 0;
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);
		ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	double Z = Zz->data[0];
	gsl_vector * Es = gsl_vector_calloc(Ns);
	status += gsl_blas_dgemv(CblasNoTrans, 1.0, sol->PP, Zz, 0.0, Es);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-Wl0_i);
	status += gsl_vector_add(Es,&ss_W.vector);
	gsl_matrix * pld = gsl_matrix_calloc(Noccs+1,Noccs);

	// make sure not negative value anywhere:
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			Es->data[Wld_i + l*Noccs+d] = Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l]<0 ?
				Es->data[Wl0_i + l] : Es->data[Wld_i + l*Noccs+d];
	}
	if(sol->tld==0)
		status += theta(pld,ss,sys,sol,Zz);
	else
		gsl_matrix_memcpy(pld,sol->tld);
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			double tld = gsl_matrix_get(pld,l,d);
			gsl_matrix_set(pld,l,d,effic*pow(tld,1.0-phi));
		}
	}

	for(l=0;l<Noccs+1;l++){
		double gdenom =0;
		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0:nu;
			double zd = Zz->data[d+1+Ntfpl+Nfac*(Nflag)];
			gdenom += exp(sig_psi*gsl_matrix_get(pld,l,d)*
					(chi[l][d]*exp(Z + zd) - b - nud + beta*(Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l])) );
		}
		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0:nu;
			double zd = Zz->data[d+1+Ntfpl+Nfac*(Nflag)];
			double gld_ld =exp(sig_psi*gsl_matrix_get(pld,l,d)*
					(chi[l][d]*exp(Z + zd) - b - nud + beta*(Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l])) )/gdenom;
			gld_ld = gld_ld<=1.0 && gld_ld>=0.0 ? gld_ld : 0.0;
			gsl_matrix_set(gld, l,d,gld_ld);
		}
	}


	gsl_vector_free(Es);
	gsl_matrix_free(pld);
	return status;
}
int theta(gsl_matrix * tld, const gsl_vector * ss, const struct sys_coef * sys, const struct sys_sol * sol, const gsl_vector * Zz){
	int status,l,d;
	status=0;
	int Wl0_i = 0;
	int Wld_i = Wl0_i + Noccs+1;
	int ss_tld_i, ss_x_i,ss_gld_i, ss_Wld_i,ss_Wl0_i;
		ss_x_i		= 0;
		ss_Wl0_i	= 0;//ss_x_i + pow(Noccs+1,2);
		ss_Wld_i	= ss_Wl0_i + Noccs+1;
		ss_gld_i	= ss_Wld_i + Noccs*(Noccs+1);//x_i + pow(Noccs+1,2);
		ss_tld_i	= ss_gld_i + Noccs*(Noccs+1);

	double Z = Zz->data[0];
	gsl_vector * Es = gsl_vector_calloc(Ns);
	gsl_blas_dgemv(CblasNoTrans, 1.0, sol->PP, Zz, 0.0, Es);
	gsl_vector_const_view ss_W = gsl_vector_const_subvector(ss,ss_Wl0_i,ss_gld_i-Wl0_i);
	gsl_vector_add(Es,&ss_W.vector);
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++)
			Es->data[Wld_i + l*Noccs+d] = Es->data[Wld_i + l*Noccs+d] -Es->data[Wl0_i + l]<0 ?
				Es->data[Wl0_i + l] : Es->data[Wld_i + l*Noccs+d];
	}

	double mul = pow(effic*(1.0-phi)/kappa,1.0/phi);
	for(l=0;l<Noccs+1;l++){

		for(d=0;d<Noccs;d++){
			double nud = l == d+1? 0:nu;
			double zd = Zz->data[d+1+Ntfpl+Nfac*(Nflag)];
			double tld_i =mul*pow(chi[l][d]*exp(Z+zd) - b - nud + beta*(Es->data[Wld_i+l*Noccs+d] - Es->data[Wl0_i+l]),1.0/phi);
			tld_i = tld_i>=0 ? tld_i:1e-5;
			tld_i = tld_i<= pow(effic,1.0/(phi-1.0)) ? tld_i : pow(effic,1.0/(phi-1.0))-0.001;
			gsl_matrix_set(tld,l,d,
					tld_i);

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

	gsl_matrix * tld = gsl_matrix_alloc(Noccs+1,Noccs);
	gsl_matrix * gld = gsl_matrix_alloc(Noccs+1,Noccs);
	// define theta and g
	if(sol->tld==0)
		status += theta(tld, ss, sys, sol, Zz);
	else
		gsl_matrix_memcpy(tld,sol->tld);
	if(sol->gld==0)
		status += gpol(gld, ss, sys, sol, Zz);
	else
		gsl_matrix_memcpy(gld,sol->gld);

	double newdisp = 0.0;
	for(k=0;k<Noccs+1;k++){
		for(j=1;j<Noccs+1;j++){
			if(j!=k)
				newdisp += sbar*(1.0-tau)*gsl_matrix_get(x,k,j);
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
			ald[l][d] = gsl_matrix_get(gld,l,d)*(gsl_matrix_get(x,l,0) + sbar*gsl_matrix_get(x,l,l));
			for(k=0;k<Noccs+1;k++){
				if(k!=l)
					ald[l][d]+= gsl_matrix_get(gld,l,d)*tau*sbar*gsl_matrix_get(x,k,l);
			}
		}
	}


	findrt = (double*)malloc(sizeof(double)*(Noccs+1));
	for(l=0;l<Noccs+1;l++){
		findrt[l]=0.0;
		for(d=0;d<Noccs;d++)
			findrt[l] += effic*pow(gsl_matrix_get(tld,l,d),1.0-phi)*gsl_matrix_get(gld,l,d);
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
			(1.0-findrt[l])*(gsl_matrix_get(x,l,0) + sbar*gsl_matrix_get(x,l,l) + tau*sbar*sxpjl)
			);
	}


	//xld : d>0
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			if(l!=d+1)
				gsl_matrix_set(xp,l,d+1,
					(1.0-tau)*(1.0-sbar)*gsl_matrix_get(x,l,d+1) + effic*pow(gsl_matrix_get(tld,l,d),1.0-phi)*ald[l][d]
					);
			else if(l==d+1){
				double newexp = 0.0;
				for(j=0;j<Noccs+1;j++)
					newexp = j!=d+1 ? tau*gsl_matrix_get(x,j,d+1)+newexp : newexp;
				gsl_matrix_set(xp,l,d+1,
					(1.0-sbar)*(gsl_matrix_get(x,l,d+1)+newexp )+ effic*pow(gsl_matrix_get(tld,l,d),1.0-phi)*ald[l][d]
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
	if(verbose >=1 && (xsum>1.00001 || xsum<0.99999)) printf("xsum = %f",xsum);

	gsl_matrix_free(tld);gsl_matrix_free(gld);
	for(l=0;l<Noccs+1;l++)
		free(ald[l]);
	free(ald);
	free(findrt);
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
	int i,j,status;
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
	int status,cc;
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
double bndCalMinwrap(double (*Vobj)(unsigned n, const double *x, double *grad, void* params),
		double* lb, double * ub, int n, double* x0, struct st_wr * ss,
		double x_tol, double f_tol, double constr_tol, int alg){

	int status,resi,starts,i;
	starts = 15*n;
	nlopt_opt opt;
	if(alg%10==0)
		opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	else
		opt = nlopt_create(NLOPT_LN_BOBYQA,n);

	nlopt_set_xtol_rel(opt, x_tol);
	nlopt_set_ftol_rel(opt, f_tol);
	nlopt_set_maxeval(opt, 50+pow(10,n));
	nlopt_set_stopval(opt, 0.0);
	nlopt_set_lower_bounds(opt,lb);
	nlopt_set_upper_bounds(opt,ub);
	nlopt_set_min_objective(opt, Vobj, (void*) ss);
	double opt_f;
	if(alg>=10){//random restarts
		gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol,n);
		double xqr[n];
		double xmin[n];
		double minf = 1000;
		for(resi=0;resi<starts;resi++){
			gsl_qrng_get(q,xqr);
			for(i=0;i<n;i++){
				x0[i] = xqr[i]*(ub[i]-lb[i]) + lb[i];
			}
			status = nlopt_optimize(opt, x0, &opt_f);
			if(opt_f<minf){
				minf = opt_f;
				for(i=0;i<n;i++)
					xmin[i] = x0[i];
			}

		}
		opt_f = minf;
		for(i=0;i<n;i++)
			x0[i] = xmin[i];
		gsl_qrng_free(q);
	}else{
		status = nlopt_optimize(opt, x0, &opt_f);
	}
	nlopt_destroy(opt);
	return opt_f;
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

	effic 	= x[0];
	sig_psi = x[1];
//	nu		= x[1];
	tau 	= x[2];
//	kappa	= x[4];
//	phi		= x[5];
//	b		= x[6];

	for(l=1;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			chi[l][d] = 0.0;
			for(i=0;i<Nskill;i++){
				chi[l][d] += fabs(gsl_matrix_get(f_skills,d,i) - gsl_matrix_get(f_skills,l-1,i) )*x[3+i];
			}
			chi[l][d] += x[3+Nskill];
		}
	}
	double chi_lb = 0.01;
	double chi_ub = 0.99;
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			double child = chi[l][d];
			if(chi[l][d]<=chi_lb)
				child = chi_lb;
			if(chi[l][d]>=chi_ub)
				child = chi_ub;
			chi[l][d] = 1.0 - child;
		}
	}
	for(d=0;d<Noccs;d++)
		chi[d+1][d] = 1.0;
/*
		// set chi[0][d] to make the average wage loss match
	double fnd_l0 = 0.0;
	for(d=0;d<Noccs;d++)
		fnd_l0 += gsl_matrix_get(sol->gld,0,d)*effic*pow(gsl_matrix_get(sol->tld,0,d),1.0-phi);
	double fnd_ldchi= 0.0;
	double childxld	= 0.0;
	double xld		= 0.0;
	for(l=1;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			fnd_ldchi += gsl_matrix_get(sol->gld,l,d)*effic*pow(gsl_matrix_get(sol->tld,l,d),1.0-phi)
				*(chi[l][0] - chi[l][d]);
		}
	}
	for(l=0;l<Noccs+1;l++){
		for(d=0;d<Noccs;d++){
			if(l!=d+1){
				xld	+= gsl_matrix_get(xss,l,d+1);
				childxld += gsl_matrix_get(xss,l,d+1)*chi[l][d];
			}
		}
	}
	childxld /= xld;
	// check that simdat->data_moments[3], o.w. flip it
	double chi0d = -(simdat->data_moments[3]-fnd_ldchi-childxld)/fnd_l0;
	if(chi0d<0.01)
		chi0d=0.01;
	if(chi0d>0.5)
		chi0d =0.5;
	*/
	status =  sol_ss(ss,xss,sys);
	status += sys_def(ss,sys);
	status += sol_dyn(ss,sol,sys);

	status += sim_moments(simdat,ss,xss,sol,sys);
	int Nmo = (simdat->data_moments)->size2;
	dist = 0.0;
	for(i=0;i<Nmo;i++)
		dist += gsl_matrix_get(simdat->moment_weights,0,i)*gsl_matrix_get(simdat->data_moments,0,i)*gsl_matrix_get(simdat->data_moments,0,i);
	if(verbose>=1){
		printf("At ef=%f,sig_psi=%f,tau=%f\n",x[0],x[1],x[2]);
		printf("Calibration distance is %f\n",dist);
	}
	FILE * calhist;
	if(printlev>=1){
		calhist = fopen("calhist.txt","a");
		fprintf(calhist,"dist=%f, ef=%f,sig_psi=%f,tau=%f\n",dist,x[0],x[1],x[2]);
		fprintf(calhist,"d_1 = %f, d_2=%f, d_3= %f\n",
				(simdat->data_moments)->data[0],(simdat->data_moments)->data[1],(simdat->data_moments)->data[2]);
		fclose(calhist);
	}
	// regression distance
	for(i=0;i<Nskill+1;i++)
		dist += pow(gsl_matrix_get(simdat->data_moments,1,i) - sk_wg[i] ,2);

	gsl_matrix_free(xss);
	gsl_vector_free(ss);
	return dist;
}
