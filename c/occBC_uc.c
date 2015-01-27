/* This program solves my disertation by a hybrid k=order perturbation with an optimal COV and exact nonlinear policies
*  First, it solves the first order approximation as in Uhlig's toolbox.  Higher orders are then computed 
*  following Judd (1999)
*  For more on the optimal COV see J F-V J R-R (JEDC 2006)
*  The hybrid portion involves using the exactly computed decision rules given approximated state-dynamics: see Maliar, Maliar and Villemot (Dynare WP 6)
*
*  au : David Wiczer
*  mdy: 2-20-2011
*
*  How to use this: put the FOCs in the form g(x,y,w)=0 in the function 'NL_Sys' and, if you
*  set use_anal>0, then put the analytically derived linearization in there.
*
*  With the current dimensions, this program is designed to solve DSGE problems with one state
*  and one choice variable, i.e. capital and endogenous hours, and any number of shocks.  In
*  the example given, there is a capital investment shock and a TFP shock.
*
*/
#include  "minime.h"
#include <math.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <stdio.h>

int Nobs,simT, nsim;
int nshock;
int neq;
int verbose =0;
int use_anal=0;

//declare some parameters
int const Noccs	= 22;

double 	beta	= 0.99;	// discount
double	nu 	= 0.01; // switching cost
double 	sig_psi	= 6.0;	// variance of taste shocks

double 	b	= 0.4; 	// value of unemployment

double ** chi;
double 	tau 	= 0.33;
double 	kappa	= 1.35;
double 	phi	= 0.5;	// match elasticity
double 	effic	= 0.2/(double)Noccs;
double 	xi	= 0.04;	// will endogenize this a la Cheremukhin (2011)

struct aux_coef{
	gsl_matrix * data_coefs;
	gsl_matrix ** draws;
	gsl_matrix * Xdat;
};

struct sys_coef{
	gsl_matrix *A1, *A2, *A3, *A4, *A5;
	gsl_matrix * N;
};

struct sys_sol{
	gsl_matrix *F1, *F2, *F3, *F4;
	gsl_matrix *P1, *P2, *P3, *P4;
	gsl_matrix *tilPhi,*tilGamma,*tilPsi;
};

// Solve the steady state
int sol_ss();
int ss_sys(const gsl_vector* ss, void * p, gsl_vector *diff);
int ss_sys_df(const gsl_vector* ss, void * p, gsl_matrix * Jac);
int ss_sys_fdf(const gsl_vector* ss, void * p, gsl_vector * diff, gsl_matrix * Jac);

// Solving the dynamic model
int sol_dyn(double * ss, struct sys_sol * sol, struct sys_coef *sys);

void nl_sys(double* vars, double * off);

void sys_dif(void (*fun)(double*, double* ), double * x, gsl_matrix * Jac);


// Utilities

void QZeig(gsl_matrix * Xi, gsl_matrix * Delta, gsl_matrix * Omega, gsl_matrix* Lambda,int save_ws);
void invmat(gsl_matrix* A, gsl_matrix* invA);
void VARest(gsl_matrix * Xdat,gsl_matrix * coef, gsl_matrix * varcov);
void OLS(gsl_vector * y, gsl_matrix * X, gsl_vector * coefs, gsl_vector *e);
void fprintf_mat(FILE * f, gsl_matrix * mat);

int main(){
	int i,s,t,n;

	return 0;
}

/*
 * Solve the dynamic system, by QZ decomp
 * The technique follows Lombardo & Sutherland
 *
 */
int sol_dyn(double * ss, struct sys_sol * sol, struct sys_coef *sys){
	//set up matrices according to Lombardo & Sutherland
	/*
	A1 (s';E[c']) = A2 (s;c) + A3*x + A4*LL + A5*E[LL']
	LL = vech((w;s;c)'*(w;s;c) )
	w' =   Nw  + e'
	
	solved by
	s' = F1*w + F2*s + F3*V + F4*Sig
	c  = P1*w + P2*s + P3*V + P4*Sig
	V' = tilPhi V + tilGamma*tileps + tilPsi*tilxi
	w' = N*w + eps
	*/

	int status = 0;

	return status;
}

/*
 * Solve for the steady state
 */

int sol_ss(){
	gsl_multiroot_fdfsolver *s;
	int status;
	size_t iter	= 0;
	size_t n	= 3;
	void * p=0;
	gsl_multiroot_function_fdf f = {&ss_sys,&ss_sys_df,&ss_sys_fdf,	n, p};
	gsl_vector *ss = gsl_vector_alloc (3);
	gsl_vector_set (ss, 0, c_ss);
	gsl_vector_set (ss, 1, k_ss);
	gsl_vector_set (ss, 2, chi);
	s = gsl_multiroot_fdfsolver_alloc (gsl_multiroot_fdfsolver_gnewton, n);
	gsl_multiroot_fdfsolver_set (s, &f, ss);
	do{
		iter++;
		status = gsl_multiroot_fdfsolver_iterate (s);
		if (status && iter>5) {
			printf("Failed finding ss");
			break;}
		status = gsl_multiroot_test_residual (s->f, 1e-7);
	}while (status == GSL_CONTINUE && iter < 1000);
	c_ss = gsl_vector_get(s->x,0);
	k_ss = gsl_vector_get(s->x,1);
	chi  = gsl_vector_get(s->x,2);
	gsl_multiroot_fdfsolver_free (s);
	gsl_vector_free (ss);
	return 0;
}


int ss_sys(const gsl_vector* ss, void * p, gsl_vector *diff){
	// c = ss[0], k = ss[1], chi=ss[2]
	diff->data[0] = beta*(alpha*pow(ss->data[1]/h_ss,alpha-1)+1-delta)-1;
	diff->data[1] =	pow(
				(ss->data[0]/ss->data[2])*(1-alpha)*pow(ss->data[1],alpha),
				phi/(1+alpha*phi)) - h_ss;
	diff->data[2] = pow(ss->data[1],alpha)*pow(h_ss,1-alpha)-delta*ss->data[1]
	                   -ss->data[0];
	return 0;
}
int ss_sys_df(const gsl_vector* ss, void * p, gsl_matrix * Jac){
	gsl_matrix_set(Jac,0,0,0);
	gsl_matrix_set(Jac,0,1,
			beta*alpha*(alpha-1)*pow(ss->data[1],alpha-2)*pow(h_ss,1-alpha) );
	gsl_matrix_set(Jac,0,2,0);
	gsl_matrix_set(Jac,1,0,
			(phi/(1+alpha*phi))*pow(ss->data[0]/ss->data[2]*(1-alpha)*pow(ss->data[1],alpha),
					phi/(1+alpha*phi))/ss->data[0] );
	gsl_matrix_set(Jac,1,1,
			(phi*alpha/(1+alpha*phi))*(ss->data[0]/ss->data[2])*(1-alpha)*
			pow(ss->data[1],alpha-1) );
	gsl_matrix_set(Jac,1,2,
			-(phi/(1+alpha*phi))*pow(ss->data[0]/ss->data[2]*(1-alpha)*pow(ss->data[1],alpha),
								phi/(1+alpha*phi))/pow(ss->data[0],2));
	gsl_matrix_set(Jac,2,0,-1);
	gsl_matrix_set(Jac,2,1,alpha*pow(ss->data[1]/h_ss,alpha-1)-delta);
	gsl_matrix_set(Jac,2,2,0);
	return 0;
}
int ss_sys_fdf(const gsl_vector* ss, void * p, gsl_vector * diff, gsl_matrix * Jac){
	//being lazy
	ss_sys(ss,p,diff);
	ss_sys_df(ss,p,Jac);
	return 0;
}
/*
 *
 * The nonlinear system that will be solved
 */

void nl_sys(double* vars, double * off){
	//vars = [	k h z b  kp kpp hp zp bp ]
	//			0 1 2 3  4   5   6  7  8
	
	double c = vars[2]*pow(vars[0]/vars[1],alpha)*vars[1]+(1.0-delta)/vars[3]*vars[0]-vars[4]/vars[3];
	//	c'= e^zp                  k'^alpha     * h'^(1-alpha)
	double cp = vars[7]*pow(vars[4],alpha)*pow(vars[6],1-alpha)//
	//		        e^(-b')        *k'  -    e^(-b')      * k''
		+(1.0-delta)*(1.0/vars[8])*vars[4]-(1.0/vars[8])*vars[5];
	// static FOC
	// 		(c^_cra/chi  * (1-alpha)*e^z(k/h)^alpha)^phi - h
	off[0] = pow(c,-cra)*(1-alpha)*vars[2]*pow(vars[0],alpha)-pow(vars[1],(1.0+alpha*phi)/phi)*chi;
	//Euler Eqn
	off[1] = beta*(pow(cp,-cra)*vars[3]*pow(c,cra)*
		(alpha*vars[7]*pow(vars[4]/vars[6],alpha-1)+(1.0/vars[8])*(1.0-delta)) ) -1;
}
/*
 *
 * Finite differences to the system
 */

void sys_dif(void (*fun)(double*, double* ), double * x, gsl_matrix * Jac){
	//differentiation assuming evaluation point is in the interior.  provided function should f(invector,outvector), and this spits a Jacobian
	int nin = (int)Jac->size2;
	int nout= (int)Jac->size1;
	int r,c;
	double *fxph= (double*)malloc(nout*sizeof(double));
	double *fxmh= (double*)malloc(nout*sizeof(double));
	double *xph = (double*)malloc(nin*sizeof(double));
	double *xmh = (double*)malloc(nin*sizeof(double));
	for(c=0;c<nin;c++){
		xph[c]=x[c];
		xmh[c]=x[c];
	}
	double step = 5.0e-5;
	for(c=0;c<nin;c++){
		double prstep = x[c]*step;
		prstep = prstep <step/10.0 && prstep>-step/10.0 ? step : prstep;
		xph[c] = x[c]+prstep;
		xmh[c] = x[c]-prstep;
		fun(xph,fxph);
		fun(xmh,fxmh);
		for(r=0;r<nout;r++){
			double Jrc = (fxph[r]-fxmh[r])/(2*prstep);
			gsl_matrix_set(Jac,r,c,	Jrc);
		}
		xph[c] =x[c];
		xmh[c] =x[c];
	}
	free(fxph);free(fxmh);free(xph);free(xmh);
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
	FILE * fh = fopen("Xdata_VAR.txt","w");
	fprintf_mat(fh,Xreg);
	fclose(fh);
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


void OLS(gsl_vector * y, gsl_matrix * X, gsl_vector * coefs, gsl_vector * e){
	gsl_matrix * XpX = gsl_matrix_alloc(X->size2,X->size2);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans,1.0, X, X,0.0, XpX);
	gsl_matrix * invXpX = gsl_matrix_alloc(X->size2,X->size2);
	invmat(XpX,invXpX);
	gsl_vector * Xpy = gsl_vector_alloc(X->size2);
	gsl_blas_dgemv(CblasTrans,1.0,X,y,0.0,Xpy);
	gsl_blas_dgemv(CblasNoTrans,1.0,invXpX,Xpy,0.0,coefs);
	gsl_blas_dgemv(CblasNoTrans,1.0,X,coefs,0.0,e);
	gsl_vector_scale(e,-1.0);
	gsl_vector_add(e,y);
}

void QZeig(gsl_matrix * Xi, gsl_matrix * Delta, gsl_matrix * Omega, gsl_matrix* Lambda,int save_ws){
	size_t n = Lambda->size1; gsl_matrix* Xi_s, * Delta_s;
	double lam_i;
	if (Lambda->size1*2 != Delta->size1)
		printf("Matrices Xi and Delta must be 2x Omega, Lambda");
	int i,j;
	if(save_ws>0){
		Xi_s = gsl_matrix_alloc(n,n);
		Delta_s= gsl_matrix_alloc(n,n);
		gsl_matrix_memcpy(Xi_s, Xi);
		gsl_matrix_memcpy(Delta_s, Delta);
	}
	gsl_vector * lambda = gsl_vector_alloc(2*n);
	gsl_vector_complex * ilambda = gsl_vector_complex_alloc(2*n);
	gsl_matrix_complex * Omega_comp = gsl_matrix_complex_alloc(2*n,2*n);
	gsl_eigen_genv_workspace * w = gsl_eigen_genv_alloc (2*n);
	gsl_eigen_genv(Xi,Delta, ilambda, lambda, Omega_comp, w);
	if(save_ws>0){
		gsl_matrix_memcpy(Xi, Xi_s);
		gsl_matrix_memcpy(Delta, Delta_s);
		gsl_matrix_free(Xi_s);
		gsl_matrix_free(Delta_s);
	}
	for(i=0;i<n;i++){
		double alpha = GSL_REAL(gsl_vector_complex_get(ilambda,i));
		double beta = gsl_vector_get(lambda,i);
		lam_i = beta/alpha;
		gsl_matrix_set(Lambda,i,i,lam_i);
	}
	for(j=0;j<n;j++){
		for(i=0;i<n;i++)
			// by construction, eigen vector is s = [x, \lambda x]', put x in Omega
			gsl_matrix_set(Omega,i,j,GSL_REAL(gsl_matrix_complex_get(Omega_comp,i,j))/lam_i );
	}
	gsl_eigen_genv_free (w);
	gsl_vector_complex_free(ilambda);
	gsl_vector_free(lambda);
}


void invmat(gsl_matrix* A, gsl_matrix* invA){
	gsl_matrix_memcpy(invA,A);
	gsl_permutation * p = gsl_permutation_alloc(A->size1);
	int signum,i,r;
	gsl_linalg_LU_decomp (invA, p, &signum);
	gsl_vector * x = gsl_vector_alloc(invA-> size1);
	gsl_vector * b = gsl_vector_calloc(invA-> size1);
	gsl_matrix* Aaux = gsl_matrix_alloc(invA->size1,invA->size2);
	
	for(i=0;i<(int)invA->size2;i++){
		b->data[i] = 1.0;
		gsl_linalg_LU_solve (invA, p, b, x);
		for(r=0; r < (int)invA->size1; r++)
			gsl_matrix_set(Aaux,r,i,x->data[i]);
		b->data[i] = 0.0;
	}
	gsl_matrix_memcpy(invA,Aaux);
	gsl_vector_free(x);gsl_vector_free(b);gsl_matrix_free(Aaux);
}
void fprintf_mat(FILE * f, gsl_matrix * mat){
	int rows = mat->size1;
	int cols = mat->size2;
	int r,c;
	for(r=0;r<rows;r++){
		for(c=0;c<cols-1;c++)
			fprintf(f,"%f,",gsl_matrix_get(mat,r,c));
		fprintf(f,"%f\n",gsl_matrix_get(mat,r,cols-1));
	}
}


//BC data, this is not a good way to do this, but I don't want to mess with reads
void setData(double * dalpha, double * dldiff, double * dkdiff, double * dydiff, double * dz){
	double rkshare []= {0.66946784,0.67252425,0.67346078,0.67653436,0.66833477,0.66740051,0.66865099,0.66921740,0.66948095,0.67395320,0.67890326,0.67849191,0.68162425,0.68267831,0.68381963,0.68393855,0.68701355,0.68692767,0.69048857,0.69170409,0.69219275,0.69634493,0.70150682,0.70684155,0.71412368,0.71022202,0.70884504,0.71061959,0.70345847,0.70214823,0.70188608,0.70107275,0.70142581,0.70401850,0.69919509,0.69734530,0.69925754,0.70334132,0.70365602,0.70257227,0.70508725,0.70440436,0.70567639,0.70433430,0.69922052,0.69240488,0.68516688,0.68594676,0.68522637,0.68958471,0.69146326,0.69413744,0.69204790,0.68483588,0.68045875,0.68509919,0.69003112,0.68417512,0.68411763,0.68296482,0.68710662,0.68682263,0.68755668,0.68627361,0.68413414,0.69231226,0.69022374,0.68161061,0.68119488,0.67780338,0.66786624,0.66997848,0.67254435,0.66720349,0.66964936,0.67203428,0.66833644,0.66459369,0.66133918,0.66104939,0.66124620,0.65927997,0.65951268,0.65906357,0.66157685,0.66337183,0.66310701,0.66680898,0.66699571,0.66870665,0.67200251,0.67592616,0.67619338,0.67182342,0.66792726,0.66926892,0.66929376,0.67047872,0.66936118,0.66454128,0.66612572,0.66671779,0.66684232,0.67100172,0.67220253,0.67022466,0.67457922,0.67221845,0.67035013,0.67284534,0.67429698,0.67625733,0.67621051,0.67732958,0.68173585,0.67847832,0.67897686,0.67722676,0.67860506,0.67449664,0.67549111,0.67349888,0.66849289,0.66606693,0.66734980,0.66615643,0.66424656,0.66303361,0.65987601,0.65972386,0.65969695,0.65770591,0.65630104,0.65538898,0.65425152,0.65794385,0.66486985,0.66531222,0.66527449,0.66924043,0.66778677,0.66731387,0.66885801,0.67017292,0.67414786,0.67392883,0.67812772,0.68002902,0.68254127,0.67772115,0.67894252,0.67850998,0.67697797,0.67832620,0.67908389,0.67566943,0.67721786,0.67759639,0.67836405,0.67430991,0.66737420,0.66748191,0.66603627,0.66732891,0.65769293,0.65679552,0.65632731,0.65081859,0.64902916,0.64691881,0.64453650,0.65106975,0.65610128,0.65338885,0.65425806,0.65541245,0.66100711,0.66056005,0.65592217,0.66753392,0.65450258,0.65773485,0.65192349,0.64514907};
	int ri;
	for (ri = 0; ri<Nobs;ri++)
		dalpha[ri] = 1-rkshare[ri];
	double ldiff []= {-0.01843,-0.014271,-0.01258,-0.010327,-0.0036959,0.00018618,0.0015094,0.0066535,0.014527,0.018541,0.016462,0.014666,0.0054155,-0.0075024,-0.0070271,-0.0034201,-0.0042711,0.0001294,0.0038598,0.0076135,0.014993,0.020742,0.022805,0.021022,0.012436,-0.0044135,-0.011988,-0.030254,-0.029907,-0.026641,-0.028354,-0.022457,-0.0094967,-0.0024289,-0.00061834,0.012139,0.025338,0.030823,0.031514,0.034732,0.029519,0.0254,0.021092,-0.0020546,-0.038859,-0.053162,-0.042827,-0.033732,-0.019546,-0.02042,-0.021082,-0.025547,-0.020137,-0.0052331,-0.001964,0.0037228,-0.0011674,0.020486,0.026254,0.030421,0.034479,0.031878,0.0347,0.0342,0.028221,0.0059473,-0.0034088,0.010262,0.015106,0.016679,0.01452,0.0030229,-0.018134,-0.024732,-0.037185,-0.053734,-0.05266,-0.040155,-0.029224,-0.010396,-0.00057845,0.0054461,0.0078968,0.0093474,0.0074444,0.0091618,0.00499,0.0065748,0.00557,-0.0015355,-0.0048985,-0.005013,-0.00024462,0.0015913,0.0049144,0.0080523,0.006708,0.010597,0.012968,0.017818,0.021841,0.01985,0.018705,0.01983,0.022247,0.019023,0.011572,0.0026668,-0.0090671,-0.018194,-0.018958,-0.021281,-0.024952,-0.022215,-0.022916,-0.02258,-0.018335,-0.014428,-0.01213,-0.0084817,-0.0066729,0.0021011,0.0057967,0.0085124,0.0082007,0.00092319,-8.0436e-05,-0.00287,-0.0094082,-0.0036027,-0.001504,-0.00068249,0.00048947,0.0049709,0.0055433,0.0072672,0.0073392,0.0052302,0.0055398,0.008359,0.0063784,0.010172,0.012711,0.017608,0.020415,0.022518,0.021678,0.021065,0.018384,0.011012,0.0030895,-0.0082715,-0.010043,-0.011391,-0.015706,-0.018426,-0.024064,-0.031181,-0.031692,-0.029112,-0.025797,-0.020856,-0.017708,-0.01268,-0.010416,-0.0051051,-0.0010374,0.0042048,0.013182,0.017202,0.018753,0.022652,0.024508,0.030787,0.029856,0.0327,0.032833,0.030736,0.02121,0.0024645,-0.021047,-0.038643,-0.043154,-0.042825};
	for (ri = 0; ri<Nobs;ri++)
		dldiff[ri] = ldiff[ri];
	double kdiff []={-0.0036677,-0.0036985,-0.0035275,-0.0032018,-0.0027643,-0.0020665,-0.0011186,-0.00018836,0.00046933,0.00078611,0.00094624,0.0010786,0.0013068,0.0017927,0.0024766,0.003152,0.0036193,0.0039864,0.0043448,0.0044799,0.0041813,0.0033468,0.0021733,0.00089452,-0.0002638,-0.0012847,-0.002223,-0.0029683,-0.0034122,-0.003497,-0.0032699,-0.0027792,-0.0020699,-0.00084366,0.00083041,0.0023735,0.0032268,0.0034017,0.0032781,0.0028599,0.0021484,0.0010019,-0.00048487,-0.0019633,-0.0030928,-0.0037203,-0.0040053,-0.0040763,-0.0040564,-0.0039198,-0.0035764,-0.0030465,-0.0023475,-0.0013461,-4.5782e-05,0.0013389,0.0026008,0.0039712,0.0055047,0.0067431,0.0072391,0.0069819,0.0063861,0.0056421,0.0049316,0.0043494,0.0037589,0.0029887,0.0018685,0.00019162,-0.0018968,-0.0040436,-0.0059009,-0.0072095,-0.0080278,-0.0084831,-0.0086961,-0.0084159,-0.0075585,-0.0064781,-0.0055128,-0.0046306,-0.0036476,-0.0026575,-0.0017474,-0.00079456,0.00025151,0.0012054,0.0018895,0.0022898,0.0025488,0.002726,0.002878,0.0030526,0.00325,0.0034519,0.0036394,0.0038604,0.0041361,0.0044058,0.0046086,0.0049382,0.0053924,0.0056203,0.0052768,0.0042658,0.002928,0.0015696,0.00048885,-0.00025854,-0.00083465,-0.0012727,-0.001605,-0.0017882,-0.0018089,-0.001742,-0.0016605,-0.0015552,-0.0013987,-0.0012332,-0.0010993,-0.0010457,-0.0010521,-0.001056,-0.00099564,-0.0006303,-7.2092e-05,0.00021988,-0.00020077,-0.001355,-0.0027806,-0.0041954,-0.0053225,-0.005742,-0.0054651,-0.004856,-0.0042636,-0.0035946,-0.002664,-0.0016647,-0.00078005,0.0001029,0.0010625,0.0018776,0.0023362,0.00239,0.0022065,0.0018994,0.0015772,0.0012942,0.0010254,0.00075656,0.00047371,0.00017155,-0.0001243,-0.00038176,-0.00056951,-0.00063418,-0.00058735,-0.00049328,-0.00041327,-0.0003745,-0.00031526,-0.00016284,0.00015366,0.00072589,0.0015178,0.0024105,0.0032881,0.004373,0.005664,0.0067204,0.0071114,0.0070591,0.0068661,0.0062286,0.004844,0.0029601,0.00091617,-0.0014544,-0.0043175,-0.0078377,-0.012177,-0.017495};
	for (ri = 0; ri<Nobs;ri++)
		dkdiff[ri] = kdiff[ri];
	double ydiff []= {-0.01272,-0.014206,-0.011988,-0.020225,-0.0022995,-0.00041587,0.0037667,0.011638,0.021589,0.013048,0.0077364,0.0085992,0.0054322,-0.002858,-0.0034463,-0.0036279,0.0042587,0.012568,0.010515,0.0092494,0.016274,0.010307,0.0066401,-0.0062502,-0.017772,-0.016543,-0.013235,-0.026508,-0.0038626,-0.0038048,-0.0045396,-0.008396,-0.00019272,0.0079796,0.012424,0.019274,0.029409,0.025243,0.011073,0.0093512,-0.0058059,-0.0079434,-0.019854,-0.024452,-0.034279,-0.028446,-0.013899,-0.0077118,0.0037671,-0.00093031,-0.0057241,-0.0086174,-0.0044922,0.0083282,0.017034,0.0058827,-0.0025737,0.02271,0.022408,0.027252,0.017918,0.014488,0.015919,0.013608,0.012913,-0.014223,-0.018827,-0.0032219,0.010187,0.0010503,0.013492,-0.0014407,-0.023307,-0.021192,-0.034503,-0.043483,-0.038825,-0.025167,-0.013166,-0.0029425,0.0057339,0.014508,0.015908,0.015217,0.013622,0.01155,0.014553,0.0098722,0.0088422,0.001885,0.00055651,-0.0065483,-0.0086929,-0.004304,-0.0026276,0.0020826,-0.00067865,0.0018745,0.00092984,0.0099501,0.011413,0.012032,0.013341,0.0075023,0.011732,0.014259,0.0070941,0.0019839,-0.0047382,-0.0069372,-0.0097639,-0.012569,-0.0090021,-0.0063443,-0.0073355,-0.00098384,-0.0058001,-0.0070718,-0.010741,-0.005749,-0.0050147,0.00017857,0.0024016,0.006032,0.0004939,-0.0041244,-0.004476,-0.0038695,-0.0025016,0.0021363,-0.000387,0.0021829,0.00040645,0.0052346,0.0076865,0.0024805,-0.0027395,-0.0039772,-0.0018707,0.0016443,0.0039122,0.0037589,0.0055574,0.013146,0.0055832,0.01613,0.0071264,0.006843,-0.0042669,0.0014358,-0.0091536,-0.0056775,-0.0055147,-0.008685,-0.0091169,-0.0097665,-0.014058,-0.013719,-0.0076885,-0.0028503,0.0026693,-0.0011676,-0.001361,-0.0050272,0.0064686,0.0026026,0.0035807,0.0039304,0.01131,0.011444,0.0088905,0.0063441,0.0013925,0.0082889,0.013101,0.017605,0.0089281,0.009039,0.0071371,-0.023649,-0.021914,-0.023512,-0.010102,0.007116};
	for (ri = 0; ri<Nobs;ri++)
		dydiff[ri] = ydiff[ri];
	double z []= {0.0163298218651748,0.0136273079475417,0.0161670831353113,0.00536823503883799,0.0153211422103068,0.0152311636903058,0.0213456757780524,0.0289314919096935,0.0359866428700286,0.0252702504905704,0.0207487347562436,0.0193163789809914,0.0225493353302788,0.0190924766494902,0.0174175714413618,0.0138879676381798,0.0243345819590397,0.0294982957549363,0.0250576616578044,0.0192161705213429,0.0223528755796831,0.0133076868046107,0.00994621016380748,-0.00244356605868568,-0.00778866347143836,-0.00421155216906222,0.00293761837000872,-0.00732256008205479,0.0137748629968932,0.00976651345257462,0.0109834491996024,0.0021064701358009,0.00617714546246884,0.0158019910311111,0.0170396636288688,0.0172784453318293,0.0276734011365267,0.0268733420750547,0.0135439476051138,0.012684208232419,0.00139738638546394,-0.00232637604021146,-0.0186597718382249,-0.0202059143864508,-0.0222018399421495,-0.0147889027597374,-0.00779143322796649,-0.00564393636233707,0.00115206689619463,0.00131222666253628,-0.0000910480408533587,-0.000340117662981143,0.00225535242208519,0.00454171011864268,0.0103760965870121,-0.00402104652945656,-0.00628435687754703,0.00860364606830721,0.00679717817471559,0.0101903515512012,0.000807502483825218,-0.00405790951394902,-0.00610575897183274,-0.0106869374669847,-0.010714712215389,-0.0279592460835749,-0.0302395405471323,-0.0295736060530061,-0.0176700082005103,-0.0316733082167344,-0.0215851898609181,-0.0324828757445594,-0.0435478891516512,-0.0394463160400322,-0.0446998985602409,-0.0416618705190448,-0.0374028418420798,-0.0285238780755916,-0.0210973320300916,-0.0184118427087823,-0.0161727800985183,-0.00979937861429114,-0.0115950717651012,-0.0132382927696471,-0.0136184787876132,-0.0143271468540052,-0.0071373527345866,-0.00781042500846851,-0.00810056302005435,-0.00872033822000073,-0.00501172997426824,-0.0103116035510711,-0.016412168220576,-0.0124704679052785,-0.014545118806506,-0.00673067746150569,-0.00812893573325102,-0.00636382342093622,-0.0108672313387741,-0.00702006058903493,-0.00767910769177593,-0.00645836975227709,-0.00444093749086516,-0.0070035414837184,-0.00633441557728442,-0.00702793504671195,-0.00981219265933841,-0.0158493199882761,-0.0215496988928088,-0.0162653687913696,-0.0182692886230313,-0.0172819997220053,-0.0110855759988713,-0.00875907163598288,-0.00548899711270767,-0.00155690527477237,-0.00647978036474406,-0.010359536224966,-0.0142451221248843,-0.0119916879225466,-0.0107837812315861,-0.0100146117429176,-0.0134308668337448,-0.0121096592446079,-0.015792697489081,-0.0181825829935933,-0.0189006962554643,-0.0182726859780589,-0.0146559814561416,-0.0122060073205752,-0.0148728760957821,-0.0124792152078408,-0.0148398614376806,-0.0108209932244439,-0.00994429679856879,-0.0135397829132584,-0.0130276171154975,-0.0122011437494987,-0.0101220425310782,-0.00320592409692733,-0.000796758324835167,-0.00215077176635337,0.000454923783793859,0.00849340082589301,0.00101088418386031,0.0123718174006804,0.00781683139099076,0.00910411462412064,0.00038510387324564,0.00611786184510699,0.000414788794815379,0.0103091861135791,0.0124863174950667,0.00998853793690202,0.0117272591574702,0.0106881640220422,0.0109609379861721,0.0178557962902879,0.0283102946326679,0.0311232703352511,0.0331194405478863,0.0296629078726518,0.0313296686857223,0.0298841933615472,0.035545810081504,0.0305478072744396,0.0298544636478315,0.0240711677630694,0.0262245057672503,0.0220467606967691,0.014877246814514,0.015831020407342,0.0119513766574508,0.0121634450128161,0.0190878286099982,0.0242578489448499,0.0194184279184846,0.0174339894954469,0.0114246766094661,-0.0023980167317772,-0.0138287317054742,-0.00462383986360493,0.00471952106632312,0.0164732724434353};
	for (ri = 0; ri<Nobs;ri++)
		dz[ri] = z[ri];
}
