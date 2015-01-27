FC 	= ifort
IC	= icc
FF77	= ifort

SOLDIR	= /home/david/Computation/DFBOLS-Web/
CDIR	= /home/david/Documents/CurrResearch/OccBC/program/c/
FDIR	= /home/david/Documents/CurrResearch/OccBC/program/fortran/
MKLROOT = /opt/intel/mkl/

ILIBS	= -lgsl -lnlopt -lifcore -lifport -mkl=parallel
IADV	= -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread 
ILOC	= -L/usr/local/lib/ -L/home/david/Computation/NLopt/lib -L/opt/intel/lib/intel64/
IINC	= -I/home/david/Computation -I/home/david/Computation/gsl/gsl-1.15 -I/home/david/Computation/NLopt/include


ICFLAGS =  -openmp -static -g -D_MKL_USE=1 -D_DFBOLS_USE=1
FCFLAGS = -c
F77FLAGS= -c

SOLOBJS = altmov.o rescue_h.o update.o prelim_h.o bobyqa_h.o bobyqb_h.o trsbox_h.o dfovec.o


occBC: $ $(CDIR)occBC_hybr.c $(SOLOBJS) 
	$(IC)  $(ICFLAGS)  $(CDIR)occBC_hybr.c $(SOLOBJS)  $(IINC) $(ILOC) $(ILIBS) -o occBC_hybr.out

occBC_HM: $ $(CDIR)occBC_HM.c $(SOLOBJS) 
	$(IC)  $(ICFLAGS)  $(CDIR)occBC_HM.c $(SOLOBJS)  $(IINC) $(ILOC) $(ILIBS) -o occBC_HM.out

occBC_noexp: $ $(CDIR)occBC_noexp.c $(SOLOBJS) 
	$(IC)  $(ICFLAGS)  $(CDIR)occBC_noexp.c $(SOLOBJS)  $(IINC) $(ILOC) $(ILIBS) -o occBC_noexp.out
	
# All the solver objects:
altmov.o :  $(SOLDIR)altmov.f
	$(FF77) $(F77FLAGS) $(SOLDIR)altmov.f

rescue_h.o :  $(SOLDIR)rescue_h.f
	$(FF77) $(F77FLAGS) $(SOLDIR)rescue_h.f

update.o :  $(SOLDIR)update.f
	$(FF77) $(F77FLAGS) $(SOLDIR)update.f
	
prelim_h.o :  $(SOLDIR)prelim_h.f
	$(FF77) $(F77FLAGS) $(SOLDIR)prelim_h.f	
	
bobyqa_h.o :  $(SOLDIR)bobyqa_h.f
	$(FF77) $(F77FLAGS) $(SOLDIR)bobyqa_h.f

bobyqb_h.o :  $(SOLDIR)bobyqb_h.f
	$(FF77) $(F77FLAGS) $(SOLDIR)bobyqb_h.f

trsbox_h.o :  $(SOLDIR)trsbox_h.f
	$(FF77) $(F77FLAGS) $(SOLDIR)trsbox_h.f
	
dfovec.o : $(FDIR)dfovec.f
	$(FC) $(FCFLAGS) $(FDIR)dfovec.f
