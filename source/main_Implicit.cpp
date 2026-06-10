#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include "errorfunc.h"
#include "log.h"
#include <mutex>
#include <openacc.h>
#ifdef _CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#endif

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
#define TWO_DIMENSIONAL //If you want to perform 3D simulation, please comment out //
#define DIM 3


#define FLUID         // this is for fluid calculation coupling. please comment out if you want to perform only elastoplastic simulation //
#define ELASTOPLASTIC
#define BIGHAM         // this is for the Bingham fluid calculation coupling. please comment out if you want to perform only elastoplastic simulation //
#define TEMPERATURE
#define CORE_NUMBER   12 //please change the core numbers of your CPU for OpenMP


// Property definition
#define TYPE_COUNT   6
#define FLUID_BEGIN  0
#define FLUID_END    2
#define STRUCTURE_BEGIN 2
#define STRUCTURE_END   4
#define WALL_BEGIN   4
#define WALL_END     6

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double Elastic_Dt=1.0e100;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 

#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];
static double (*InitialPosition)[DIM];
static double (*PositionOld)[DIM] = NULL;
static double (*PositionNewtonBase)[DIM] = NULL;
static double (*VelocityOld)[DIM] = NULL;
static double (*StressOld)[DIM][DIM] = NULL;
static double (*StrainTmp)[DIM][DIM] = NULL;
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total explicit force acting on the particle
static int *NeighborCount;                   // [ParticleCount]
static int (*Neighbor)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static double (*NeighborCalculatedPosition)[DIM];
#define MARGIN (0.1*ParticleSpacing)
#pragma acc declare create(ParticleCount,Property,Mass,Position,InitialPosition,PositionNewtonBase,Velocity,Force,NeighborCount,Neighbor,NeighborCalculatedPosition)
#pragma acc declare create(PositionOld,VelocityOld,StressOld,StrainTmp)	

static double (*VelocityNewtonBase)[DIM] = NULL;
static double (*DeltaVelocity)[DIM]      = NULL;
static double (*ResidualV)[DIM]          = NULL;
static double (*InternalForceBase)[DIM]  = NULL;
#pragma acc declare create(VelocityNewtonBase,DeltaVelocity,ResidualV,InternalForceBase)

static double (*Residual)[DIM] = NULL;
static double (*DeltaPosition)[DIM] = NULL;
static double (*InternalForceBuf)[DIM] = NULL;
static double (*InternalForceOldBuf)[DIM] = NULL;
static double (*ExternalForceBuf)[DIM] = NULL;
static double (*NewtonDiagVec)[DIM] = NULL;
static double *NewtonDiag = NULL;
#pragma acc declare create(Residual,DeltaPosition,InternalForceBuf,InternalForceOldBuf,ExternalForceBuf,NewtonDiagVec,NewtonDiag)



// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth = 0.0;
static int CellCount[DIM];
static int CellCounts = 0;
static int *CellParticleBegin;  // beginning of particles in the cell
static int *CellParticleEnd;    // number of particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,CellCounts,CellParticleBegin,CellParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mu,Lambda,Kappa)

static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallCenter,WallVelocity,WallOmega,WallRotation)

//Structure
static double YoungModulus[TYPE_COUNT];
static double PoissonRatio[TYPE_COUNT];
static double Cohesion[TYPE_COUNT];
static double InternalFrictionAngle[TYPE_COUNT];
static double DilatancyFrictionAngle[TYPE_COUNT];
static int StructureParticleBegin;
static int StructureParticleEnd;
static double *LambdaLames;
static double *MuLames;
static double (*Strain)[DIM][DIM];
static double (*Spin)[DIM][DIM];
static double (*PlasticStrainRate)[DIM][DIM];  //added for the elastic
static double (*Stress)[DIM][DIM];
static double (*Acceleration)[DIM];
static double (*DiffusiveCoefficient);
static double (*ShearRate);
#pragma acc declare create(YoungModulus,PoissonRatio,Cohesion,InternalFrictionAngle,DilatancyFrictionAngle,LambdaLames,MuLames)
#pragma acc declare create(Strain,Spin,PlasticStrainRate,Stress,Acceleration,DiffusiveCoefficient,ShearRate)

static double (*EquivalentStrain);
static double (*Damage);
static double (*MaxEquivalentStrain);
static double (*DamageOld);
static double (*MaxEquivalentStrainOld);
static double FractureEnergy[TYPE_COUNT];
static double (*KappaTmp);
#pragma acc declare create(EquivalentStrain,Damage,MaxEquivalentStrain,DamageOld,MaxEquivalentStrainOld)
#pragma acc declare create(FractureEnergy,KappaTmp)

static double GranularSize[TYPE_COUNT];
static double (*alpha);
static double (*DragC);
#pragma acc declare create(GranularSize,DragC,alpha)

static double (*KrylovR)[DIM];
static double (*KrylovP)[DIM];
static double (*KrylovZ)[DIM];
static double (*KrylovAp)[DIM];
#pragma acc declare create(KrylovR,KrylovP,KrylovZ,KrylovAp)

static double (*TangentStress)[DIM][DIM];
static double (*TangentForce)[DIM];
// 新しいグローバル配列（宣言部に追加）
static int    (*YieldActive);
static double (*PlasticDf)[DIM][DIM];  // df/dsigma
static double (*PlasticDg)[DIM][DIM];  // dg/dsigma
static double (*PlasticCepDen);         // df:C:dg (denominator)
#pragma acc declare create(TangentStress,TangentForce,YieldActive,PlasticDf,PlasticDg,PlasticCepDen)



static double (*BcgR)[DIM];
static double (*BcgRhat)[DIM];
static double (*BcgP)[DIM];
static double (*BcgV)[DIM];
static double (*BcgS)[DIM];
static double (*BcgT)[DIM];
static double (*BcgPhat)[DIM];
static double (*BcgShat)[DIM];
#pragma acc declare create(BcgR,BcgRhat,BcgP,BcgV,BcgS,BcgT,BcgPhat,BcgShat)

//The temperature estimation with the Energy conservation equation
static double InitialTemperature[TYPE_COUNT];
static double ThermalConductivity[TYPE_COUNT];
static double SpecificHeat[TYPE_COUNT];
static double SolidusTemperature[TYPE_COUNT];
static double LiquidusTemperature[TYPE_COUNT];
static double LatentHeat[TYPE_COUNT];
static double CriticalSolidFraction[TYPE_COUNT];
#pragma acc declare create(InitialTemperature,ThermalConductivity,SpecificHeat,SolidusTemperature,LiquidusTemperature,LatentHeat,CriticalSolidFraction)


//Parameters for melting and solidifcation //
static double *Temperature;
static double *TemperatureOld;
static double *SolidFraction;   //How amount of solid is contained in one particle
static double *Conductivity;
static double *Cp;
static double *SolidusTemp;
static double *LiquidusTemp;
static double *LatentH;
#pragma acc declare create(Temperature,TemperatureOld,SolidFraction)
#pragma acc declare create(Conductivity,Cp,SolidusTemp,LiquidusTemp,LatentH)






// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );
static void initializeStructure( void );
static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static int neighborCalculation();
static void calculateNeighbor();
static void calculatePhysicalCoefficients();
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateVirialPressureAtParticle();
static void calculateVirialStressAtParticle();

//Elastoplastic
static void calculateLamesconstant();
static void resetAcceleration();
static void calculateStressForce();
static void calculateInterfaceForce();
static void calculateInterfaceViscosity();
static void selectFreeGPU();
static void calculateStrainRateTensor();
static void calculateSpinTensor();
static void calculateStressImplicitLocal();
static void updateElasticForce();
static void updateElasticPosition();
static void implicitElasticStepPicard();
static void resetStructureForce();
static void buildExternalForce();
static void addStructureWallContactForce();
static void copyStructureForceToInternalBuffer();
static void calculateNewtonDiag();
static void calculateResidual();
static void implicitElasticStepVelocityBased();

//Damage Propargation calculation 
static void calculateEquivalentStrain();
static void calculateJacobian();
static void calculateKappa_nonlocal();
static void calculateDamage();

//Solid-Liquid Couling Algorithm
static void calculateDragForce();
static void calculateBuoyancyForce();
static void calculatePressureGradientForceOnStructure();

//Calculation of the Energy conservation term
static void calculateEnergyConservation();
static void calculateTemperature();

static void initialDisplacement();

// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cVirial=0, cOther=0;



const double L = 20.0e-2;        // Length in meters (20 cm)
const double k = 1.875 / L;      // k value for first vibration mode

// Function to compute f(x1)
double compute_fx1(double x1) {
    double kL = k * L;
    double kx1 = k * x1;

    double term1 = (cos(kL) + cosh(kL)) * (sin(kx1) - sinh(kx1));
    double term2 = (sin(kL) + sinh(kL)) * (cos(kx1) - cosh(kx1));

    return term1 - term2;
}




int main(int argc, char *argv[])
{
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
	    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
   // selectFreeGPU();
    
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( CORE_NUMBER );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeDomain();

//	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
//	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc enter data create(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT],GranularSize[0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	#pragma acc update device(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT])
	#pragma acc update device(LambdaLames[0:ParticleCount],MuLames[0:ParticleCount])
	#pragma acc update device(Strain[0:ParticleCount][0:DIM][0:DIM],Spin[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM],Stress[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc update device(Acceleration[0:ParticleCount][0:DIM],DiffusiveCoefficient[0:ParticleCount],ShearRate[0:ParticleCount],alpha[0:ParticleCount],DragC[0:ParticleCount])
    #pragma acc update device(InitialTemperature[0:TYPE_COUNT],ThermalConductivity[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],SolidusTemperature[0:TYPE_COUNT],LiquidusTemperature[0:TYPE_COUNT],LatentHeat[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
    #pragma acc update device(Temperature[0:ParticleCount],TemperatureOld[0:ParticleCount],SolidFraction[0:ParticleCount],Conductivity[0:ParticleCount],Cp[0:ParticleCount],SolidusTemp[0:ParticleCount],LiquidusTemp[0:ParticleCount],LatentH[0:ParticleCount])
	#pragma acc update device(EquivalentStrain[0:ParticleCount],Damage[0:ParticleCount],MaxEquivalentStrain[0:ParticleCount],DamageOld[0:ParticleCount],MaxEquivalentStrainOld[0:ParticleCount])
	#pragma acc update device(FractureEnergy[0:TYPE_COUNT])
	{
	calculateNeighbor();
	calculateDensityA();
	calculateGravityCenter();
	calculateDensityP();   
	calculateLamesconstant();   
    calculateStrainRateTensor();
    calculateSpinTensor();
	writeVtkFile("output.vtk");
		
	{
		time_t t=time(NULL);
		log_printf("start main roop at %s\n",ctime(&t));
	}
	int iStep=(int)(Time/Dt);
	cStart = clock();
	cFrom = cStart;
	while(Time < EndTime + 1.0e-5*Dt){
			
		if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
		}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

            initialDisplacement();
			
			// wall calculation
			calculateWall();
			
			// periodic boundary calculation
			calculatePeriodicBoundary();
			
			// reset Force to calculate conservative interaction
			resetForce();
			resetAcceleration();
		
		
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			// calculate Neighbor
			//if(neighborCalculation()==1){
			calculateNeighbor();
		//	}
			cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;
			
			#ifdef FLUID
			// calculate density
			calculateDensityA();
			calculateGravityCenter();
			calculateDensityP();
			calculateDivergenceP();
			#endif
			
			// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
			calculatePhysicalCoefficients();

            #ifdef TEMPERATURE
            calculateEnergyConservation();

             #endif
			
		

			#ifdef FLUID
			// calculate pressure 
	        calculatePressureP();
			
			// calculate P(s,rho) s:fixed
		  	calculatePressureA();
			
			// calculate diffuse interface force
			calculateDiffuseInterface();
			
		    // calculate shear viscosity
			calculateViscosityV();
			#endif

            calculateBuoyancyForce();

            calculateDragForce();


			calculateGravity();
			
            // calculate intermidiate Velocity
            calculateAcceleration();    		
				 
           
			 calculateConvection();



   			int substeps = (int)(Dt / Elastic_Dt); 

            #ifdef ELASTOPLASTIC
  			for (int substep = 0; substep < substeps; ++substep) {

			implicitElasticStepVelocityBased();

  			}
			#endif
           

			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;

				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			
			Time += Dt;
			iStep++;
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		}
	}
	cEnd = cTill;
	
	{
		time_t t=time(NULL);
		log_printf("end main roop at %s\n",ctime(&t));
		log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
		log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
		log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
		log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
		log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cVirial+cOther)/CLOCKS_PER_SEC);
		log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
	}
	
	
	#pragma acc exit data delete(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT],GranularSize[0:TYPE_COUNT])
	#pragma acc exit data delete(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc exit data delete(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT])
	#pragma acc exit data delete(LambdaLames[0:ParticleCount],MuLames[0:ParticleCount])
	#pragma acc exit data delete(Strain[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(Stress[0:ParticleCount][0:DIM][0:DIM],Acceleration[0:ParticleCount][0:DIM],alpha[0:ParticleCount],DragC[0:ParticleCount] )
    #pragma acc exit data delete(InitialTemperature[0:TYPE_COUNT],ThermalConductivity[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],SolidusTemperature[0:TYPE_COUNT],LiquidusTemperature[0:TYPE_COUNT],LatentHeat[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
    #pragma acc exit data delete(Temperature[0:ParticleCount],TemperatureOld[0:ParticleCount],SolidFraction[0:ParticleCount],Conductivity[0:ParticleCount],Cp[0:ParticleCount],SolidusTemp[0:ParticleCount],LiquidusTemp[0:ParticleCount],LatentH[0:ParticleCount])
	#pragma acc exit data delete(EquivalentStrain[0:ParticleCount],Damage[0:ParticleCount],MaxEquivalentStrain[0:ParticleCount],DamageOld[0:ParticleCount],MaxEquivalentStrainOld[0:ParticleCount],KappaTmp[0:ParticleCount])
	#pragma acc exit data delete(FractureEnergy[0:TYPE_COUNT])
	return 0;
	
}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

 
    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
       else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
       else if(sscanf(buf," ElasticDt %lf",&Elastic_Dt)==1){mode=reading_global;}
       else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
       else if(sscanf(buf," Density %lf %lf %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3],&Density[4],&Density[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkModulus %lf %lf %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3],&BulkModulus[4],&BulkModulus[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3],&BulkViscosity[4],&BulkViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3],&ShearViscosity[4],&ShearViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[4],&SurfaceTension[5])==4){mode=reading_global;}
       else if(sscanf(buf," Temperature %lf %lf %lf %lf %lf %lf",&InitialTemperature[0],&InitialTemperature[1],&InitialTemperature[2],&InitialTemperature[3],&InitialTemperature[4],&InitialTemperature[5])==6){mode=reading_global;}
       else if(sscanf(buf," ThermalConductivity %lf %lf %lf %lf %lf %lf",&ThermalConductivity[0],&ThermalConductivity[1],&ThermalConductivity[2],&ThermalConductivity[3],&ThermalConductivity[4],&ThermalConductivity[5])==6){mode=reading_global;}
       else if(sscanf(buf," SpecificHeat %lf %lf %lf %lf %lf %lf",&SpecificHeat[0],&SpecificHeat[1],&SpecificHeat[2],&SpecificHeat[3],&SpecificHeat[4],&SpecificHeat[5])==6){mode=reading_global;}
       else if(sscanf(buf," SolidusTemperature %lf %lf %lf %lf %lf %lf",&SolidusTemperature[0],&SolidusTemperature[1],&SolidusTemperature[2],&SolidusTemperature[3],&SolidusTemperature[4],&SolidusTemperature[5])==6){mode=reading_global;}
       else if(sscanf(buf," LiquidusTemperature %lf %lf %lf %lf %lf %lf",&LiquidusTemperature[0],&LiquidusTemperature[1],&LiquidusTemperature[2],&LiquidusTemperature[3],&LiquidusTemperature[4],&LiquidusTemperature[5])==6){mode=reading_global;}
       else if(sscanf(buf," LatentHeat %lf %lf %lf %lf %lf %lf",&LatentHeat[0],&LatentHeat[1],&LatentHeat[2],&LatentHeat[3],&LatentHeat[4],&LatentHeat[5])==6){mode=reading_global;}
       else if(sscanf(buf," CriticalSolidFraction %lf %lf %lf %lf",&CriticalSolidFraction[0],&CriticalSolidFraction[1],&CriticalSolidFraction[2],&CriticalSolidFraction[3])==4){mode=reading_global;}
       else if(sscanf(buf," YoungModulus %lf %lf %lf %lf",&YoungModulus[2],&YoungModulus[3],&YoungModulus[4],&YoungModulus[5])==4){mode=reading_global;}
       else if(sscanf(buf," PoissonRatio %lf %lf %lf %lf ",&PoissonRatio[2],&PoissonRatio[3],&PoissonRatio[4],&PoissonRatio[5])==4){mode=reading_global;}
       else if(sscanf(buf," Cohesion %lf %lf %lf %lf",&Cohesion[2],&Cohesion[3],&Cohesion[4],&Cohesion[5])==4){mode=reading_global;}
       else if(sscanf(buf," FractureEnergy %lf %lf %lf %lf",&FractureEnergy[2],&FractureEnergy[3],&FractureEnergy[4],&FractureEnergy[5])==4){mode=reading_global;}
       else if(sscanf(buf," InternalFrictionAngle %lf %lf %lf %lf ",&InternalFrictionAngle[2],&InternalFrictionAngle[3],&InternalFrictionAngle[4],&InternalFrictionAngle[5])==4){mode=reading_global;}
       else if(sscanf(buf," DilatancyFrictionAngle %lf %lf %lf %lf ",&DilatancyFrictionAngle[2],&DilatancyFrictionAngle[3],&DilatancyFrictionAngle[4],&DilatancyFrictionAngle[5])==4){mode=reading_global;}
       else if(sscanf(buf," ActualDebrisSize %lf %lf ",&GranularSize[2],&GranularSize[3])==2){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3],&InteractionRatio[0][4],&InteractionRatio[0][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3],&InteractionRatio[1][4],&InteractionRatio[1][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3],&InteractionRatio[2][4],&InteractionRatio[2][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3],&InteractionRatio[3][4],&InteractionRatio[3][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type4) %lf %lf %lf %lf %lf %lf",&InteractionRatio[4][0],&InteractionRatio[4][1],&InteractionRatio[4][2],&InteractionRatio[4][3],&InteractionRatio[4][4],&InteractionRatio[4][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type5) %lf %lf %lf %lf %lf %lf",&InteractionRatio[5][0],&InteractionRatio[5][1],&InteractionRatio[5][2],&InteractionRatio[5][3],&InteractionRatio[5][4],&InteractionRatio[5][5])==6){mode=reading_global;}
       else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
       else if(sscanf(buf," Wall2  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[4][0],  &WallCenter[4][1],  &WallCenter[4][2],  &WallVelocity[4][0],  &WallVelocity[4][1],  &WallVelocity[4][2],  &WallOmega[4][0],  &WallOmega[4][1],  &WallOmega[4][2])==9){mode=reading_global;}
       else if(sscanf(buf," Wall3  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[5][0],  &WallCenter[5][1],  &WallCenter[5][2],  &WallVelocity[5][0],  &WallVelocity[5][1],  &WallVelocity[5][2],  &WallOmega[5][0],  &WallOmega[5][1],  &WallOmega[5][2])==9){mode=reading_global;}
       else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
	
	#pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc enter data create(Gravity[0:DIM])
//	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
	#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc enter data create(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT],GranularSize[0:TYPE_COUNT])
	#pragma acc enter data create(InitialTemperature[0:TYPE_COUNT],ThermalConductivity[0:TYPE_COUNT],LatentHeat[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],LiquidusTemperature[0:TYPE_COUNT],SolidusTemperature[0:TYPE_COUNT])
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
	char buf[1024];   
	
	
	try{
		
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%lf",&Time);
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
			&ParticleCount,
			&ParticleSpacing,
			&DomainMin[0], &DomainMax[0],
			&DomainMin[1], &DomainMax[1],
			&DomainMin[2], &DomainMax[2]);
		#ifdef TWO_DIMENSIONAL
		ParticleVolume = ParticleSpacing*ParticleSpacing;
		#else
		ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
		#endif


		
		Property = (int *)malloc(ParticleCount*sizeof(int));
        Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PositionOld = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
	     PositionNewtonBase = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        InitialPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		VelocityOld = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DensityA = (double *)malloc(ParticleCount*sizeof(double));
		GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PressureA = (double *)malloc(ParticleCount*sizeof(double));
		VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
		DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
		PressureP = (double *)malloc(ParticleCount*sizeof(double));
		VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
		VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		Mass = (double (*))malloc(ParticleCount*sizeof(double));
		Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Mu = (double (*))malloc(ParticleCount*sizeof(double));
		Lambda = (double (*))malloc(ParticleCount*sizeof(double));
		Kappa = (double (*))malloc(ParticleCount*sizeof(double));
		
		#pragma acc enter data create(Property[0:ParticleCount])
		#pragma acc enter data create(Position[0:ParticleCount][0:DIM])
		#pragma acc enter data create(PositionOld[0:ParticleCount][0:DIM])
		#pragma acc enter data create(PositionNewtonBase[0:ParticleCount][0:DIM])
		#pragma acc enter data create(InitialPosition[0:ParticleCount][0:DIM])
		#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM])
		#pragma acc enter data create(VelocityOld[0:ParticleCount][0:DIM])
		#pragma acc enter data create(DensityA[0:ParticleCount])
		#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM])
		#pragma acc enter data create(PressureA[0:ParticleCount])
		#pragma acc enter data create(VolStrainP[0:ParticleCount])
		#pragma acc enter data create(DivergenceP[0:ParticleCount])
		#pragma acc enter data create(PressureP[0:ParticleCount])
		#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount])
		#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(Mass[0:ParticleCount])
		#pragma acc enter data create(Force[0:ParticleCount][0:DIM])
		#pragma acc enter data create(Mu[0:ParticleCount])
		#pragma acc enter data create(Lambda[0:ParticleCount])
		#pragma acc enter data create(Kappa[0:ParticleCount])
		#pragma acc enter data attach(Property,Position,PositionOld,PositionNewtonBase,InitialPosition,Velocity,VelocityOld,DensityA,GravityCenter,PressureA)
		#pragma acc enter data attach(VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mass,Force,Mu,Lambda,Kappa)
        
    
      	LambdaLames = (double (*))malloc(ParticleCount*sizeof(double));
       	MuLames = (double (*))malloc(ParticleCount*sizeof(double));
        Strain = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Spin = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        PlasticStrainRate = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Stress = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		StressOld = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        StrainTmp = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Acceleration = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        DiffusiveCoefficient = (double (*))malloc(ParticleCount*sizeof(double));	
        ShearRate = (double (*))malloc(ParticleCount*sizeof(double));	
        EquivalentStrain = (double (*))malloc(ParticleCount*sizeof(double));
		MaxEquivalentStrain = (double (*))malloc(ParticleCount*sizeof(double));
		Damage = (double (*))malloc(ParticleCount*sizeof(double));
		MaxEquivalentStrainOld = (double (*))malloc(ParticleCount*sizeof(double));
		DamageOld = (double (*))malloc(ParticleCount*sizeof(double));
        KappaTmp = (double (*))malloc(ParticleCount*sizeof(double));

        for (int iP = 0; iP < ParticleCount; ++iP) {
            EquivalentStrain[iP] = 0.0;
            MaxEquivalentStrain[iP] = 0.0;
            Damage[iP] = 0.0;
            MaxEquivalentStrainOld[iP] = 0.0;
            DamageOld[iP] = 0.0;
            KappaTmp[iP] = 0.0;
        }

        #pragma acc enter data create(LambdaLames[0:ParticleCount])
		#pragma acc enter data create(MuLames[0:ParticleCount])
		#pragma acc enter data create(Strain[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(Spin[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(Stress[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(StressOld[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(Acceleration[0:ParticleCount][0:DIM])
		#pragma acc enter data create(DiffusiveCoefficient[0:ParticleCount])
		#pragma acc enter data create(ShearRate[0:ParticleCount])
		#pragma acc enter data create(EquivalentStrain[0:ParticleCount])
		#pragma acc enter data create(MaxEquivalentStrain[0:ParticleCount])
		#pragma acc enter data create(Damage[0:ParticleCount])
		#pragma acc enter data create(MaxEquivalentStrainOld[0:ParticleCount])
		#pragma acc enter data create(DamageOld[0:ParticleCount])
        #pragma acc enter data create(KappaTmp[0:ParticleCount])
		#pragma acc enter data attach(EquivalentStrain,MaxEquivalentStrain,Damage,MaxEquivalentStrainOld,DamageOld)
		#pragma acc enter data attach(LambdaLames,MuLames,Strain,Spin,PlasticStrainRate,Stress,StressOld,Acceleration,DiffusiveCoefficient,ShearRate,StrainTmp,KappaTmp)

		alpha= (double (*))malloc(ParticleCount*sizeof(double));
		DragC= (double (*))malloc(ParticleCount*sizeof(double));
		#pragma acc enter data create(alpha[0:ParticleCount])
		#pragma acc enter data create(DragC[0:ParticleCount])
		#pragma acc enter data attach(alpha,DragC)

        // Parameters related to melting and solidification
        Temperature = (double *)malloc(ParticleCount * sizeof(double));
        TemperatureOld = (double *)malloc(ParticleCount * sizeof(double));
        SolidFraction = (double *)malloc(ParticleCount * sizeof(double));
        Conductivity = (double *)malloc(ParticleCount * sizeof(double));
        LatentH = (double *)malloc(ParticleCount * sizeof(double));
        Cp = (double *)malloc(ParticleCount * sizeof(double));
        SolidusTemp = (double *)malloc(ParticleCount * sizeof(double));
        LiquidusTemp = (double *)malloc(ParticleCount * sizeof(double));
        #pragma acc enter data create(Temperature[0:ParticleCount])
    	#pragma acc enter data create(TemperatureOld[0:ParticleCount])
	    #pragma acc enter data create(SolidFraction[0:ParticleCount])
	    #pragma acc enter data create(Conductivity[0:ParticleCount])
	    #pragma acc enter data create(LatentH[0:ParticleCount])
	    #pragma acc enter data create(Cp[0:ParticleCount])
	    #pragma acc enter data create(SolidusTemp[0:ParticleCount])
	    #pragma acc enter data create(LiquidusTemp[0:ParticleCount])
	    #pragma acc enter data attach(Temperature,TemperatureOld,SolidFraction,Conductivity,LatentH,Cp,SolidusTemp,LiquidusTemp)


		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
		Neighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
		NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		
		#pragma acc enter data create(NeighborCount[0:ParticleCount])
		#pragma acc enter data create(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM])
		#pragma acc enter data attach(NeighborCount,Neighbor,NeighborCalculatedPosition)

		VelocityNewtonBase = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DeltaVelocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		ResidualV = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		InternalForceBase = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

		Residual = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DeltaPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		InternalForceBuf = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		InternalForceOldBuf = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		ExternalForceBuf = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		NewtonDiagVec = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		NewtonDiag = (double *)malloc(ParticleCount*sizeof(double));
		#pragma acc enter data create(VelocityNewtonBase[0:ParticleCount][0:DIM])
		#pragma acc enter data create(DeltaVelocity[0:ParticleCount][0:DIM])
		#pragma acc enter data create(ResidualV[0:ParticleCount][0:DIM])
		#pragma acc enter data create(InternalForceBase[0:ParticleCount][0:DIM])
		#pragma acc enter data create(NewtonDiagVec[0:ParticleCount][0:DIM])
		#pragma acc enter data create(Residual[0:ParticleCount][0:DIM])
		#pragma acc enter data create(DeltaPosition[0:ParticleCount][0:DIM])
		#pragma acc enter data create(InternalForceBuf[0:ParticleCount][0:DIM])
		#pragma acc enter data create(InternalForceOldBuf[0:ParticleCount][0:DIM])
		#pragma acc enter data create(ExternalForceBuf[0:ParticleCount][0:DIM])
		#pragma acc enter data create(NewtonDiag[0:ParticleCount])
		#pragma acc enter data attach(VelocityNewtonBase,DeltaVelocity,ResidualV,InternalForceBase,NewtonDiagVec)
		#pragma acc enter data attach(Residual,DeltaPosition,InternalForceBuf,InternalForceOldBuf,ExternalForceBuf,NewtonDiag)

		KrylovR = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		KrylovP = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		KrylovZ = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		KrylovAp = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		TangentStress = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		TangentForce = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		YieldActive = (int *)malloc(ParticleCount*sizeof(int));
		PlasticDf = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		PlasticDg = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		PlasticCepDen = (double *)malloc(ParticleCount*sizeof(double));	

		#pragma acc enter data create(KrylovR[0:ParticleCount][0:DIM])
		#pragma acc enter data create(KrylovP[0:ParticleCount][0:DIM])
		#pragma acc enter data create(KrylovZ[0:ParticleCount][0:DIM])
		#pragma acc enter data create(KrylovAp[0:ParticleCount][0:DIM])
		#pragma acc enter data create(TangentStress[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(TangentForce[0:ParticleCount][0:DIM])
		#pragma acc enter data create(YieldActive[0:ParticleCount])
		#pragma acc enter data create(PlasticDf[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(PlasticDg[0:ParticleCount][0:DIM][0:DIM])
		#pragma acc enter data create(PlasticCepDen[0:ParticleCount])
		#pragma acc enter data attach(KrylovR,KrylovP,KrylovZ,KrylovAp,TangentStress,TangentForce,YieldActive,PlasticDf,PlasticDg,PlasticCepDen)

		BcgR = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgRhat = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgP = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgV = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgS = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgT = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgPhat = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		BcgShat = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

		#pragma acc enter data create(BcgR[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgRhat[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgP[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgV[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgS[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgT[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgPhat[0:ParticleCount][0:DIM])
		#pragma acc enter data create(BcgShat[0:ParticleCount][0:DIM])
		#pragma acc enter data attach(BcgR,BcgRhat,BcgP,BcgV,BcgS,BcgT,BcgPhat,BcgShat)
	//	double (*q)[DIM] = Position;
		double (*v)[DIM] = Velocity;
		
		for(int iP=0;iP<ParticleCount;++iP){
			if(fgets(buf,sizeof(buf),fp)==NULL)break;
			sscanf(buf,"%d  %lf %lf %lf %lf %lf %lf  %lf %lf %lf",
				&Property[iP],
				&Position[iP][0],&Position[iP][1],&Position[iP][2],
                &InitialPosition[iP][0],&InitialPosition[iP][1],&InitialPosition[iP][2],
				&v[iP][0],&v[iP][1],&v[iP][2]
			);
		}
	}catch(...){};
	
	fclose(fp);
	
    FluidParticleBegin = -1;
    FluidParticleEnd = -1;
    StructureParticleBegin = -1;
    StructureParticleEnd = -1;
    WallParticleBegin = -1;
    WallParticleEnd = -1;

    for (int iP = 0; iP < ParticleCount; ++iP) {
        int prop = Property[iP];

        if (FLUID_BEGIN <= prop && prop < FLUID_END) {
            if (FluidParticleBegin == -1) FluidParticleBegin = iP;
            FluidParticleEnd = iP + 1;
        } else if (STRUCTURE_BEGIN <= prop && prop < STRUCTURE_END) {
            if (StructureParticleBegin == -1) StructureParticleBegin = iP;
            StructureParticleEnd = iP + 1;
        } else if (WALL_BEGIN <= prop && prop < WALL_END) {
            if (WallParticleBegin == -1) WallParticleBegin = iP;
            WallParticleEnd = iP + 1;
        }
    }

    if (FluidParticleBegin != -1)
        printf("Fluid Particles: %d\n", FluidParticleEnd - FluidParticleBegin);
    else
        printf("Fluid Particles: 0\n");

    if (StructureParticleBegin != -1)
        printf("Structure Particles: %d\n", StructureParticleEnd - StructureParticleBegin);
    else
        printf("Structure Particles: 0\n");

    if (WallParticleBegin != -1)
        printf("Wall Particles: %d\n", WallParticleEnd - WallParticleBegin);
    else
        printf("Wall Particles: 0\n");

	#pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(InitialPosition[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
//	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)


	

	
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

 //   const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e %e %e %e  %e %e %e\n",
                    Property[iP],
                    Position[iP][0], Position[iP][1], Position[iP][2],
                    InitialPosition[iP][0],InitialPosition[iP][1],InitialPosition[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}


static void writeVtkFile(char *filename)
{
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount],Mass[0:ParticleCount],DiffusiveCoefficient[0:ParticleCount])
	#pragma acc update host(NeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM],Stress[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc update host(Strain[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM],Spin[0:ParticleCount][0:DIM][0:DIM],Damage[0:ParticleCount])
    const double (*v)[DIM] = Velocity;

    FILE *fp=fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)Position[iP][0], (float)Position[iP][1], (float)Position[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");

    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");

    /*
    fprintf(fp, "\n");
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS displacement float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        const double displacement[DIM]={Position[iP][0]-InitialPosition[iP][0],Position[iP][1]-InitialPosition[iP][1],Position[iP][2]-InitialPosition[iP][2]};
        fprintf(fp, "%e %e %e\n", (float)displacement[0], (float)displacement[1], (float)displacement[2]);
    }
    */
   
    for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS stress[%d][%d] float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Stress[iP][iD][jD]);
    }
    }
    }
        for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS strain[%d][%d] float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Strain[iP][iD][jD]);
    }
    }
    }
    fprintf(fp, "SCALARS Damage float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n",(float) Damage[iP]);
    }
    fprintf(fp, "SCALARS EquivalentStrain float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n",(float)EquivalentStrain[iP]);
    }
    fprintf(fp, "SCALARS YieldStress float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n",(float)YieldActive[iP]);
    }
  //  for (int iD=0;iD<DIM;iD++){
 //  for(int jD=0;jD<DIM;jD++){
//fprintf(fp, "\n"); fprintf(fp," SCALARS spin[%d][%d] float \n", iD, jD);
//fprintf(fp, "LOOKUP_TABLE default\n");
//for(int iP=0;iP<ParticleCount;++iP){
//    fprintf(fp, "%e\n", (float)Spin[iP][iD][jD]);
//}
//}
//}

for (int iD=0;iD<DIM;iD++){
    for(int jD=0;jD<DIM;jD++){
 fprintf(fp, "\n"); fprintf(fp," SCALARS plastic[%d][%d] float \n", iD, jD);
 fprintf(fp, "LOOKUP_TABLE default\n");
 for(int iP=0;iP<ParticleCount;++iP){
     fprintf(fp, "%e\n", (float)PlasticStrainRate[iP][iD][jD]);
 }
 }
 }
    fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
    fprintf(fp, "SCALARS Mass float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n",(float) Mass[iP]);
    }
    fprintf(fp,"SCALARS temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int iP=0;iP<ParticleCount;iP++){
        fprintf(fp,"%e\n",(float) Temperature[iP]);
    }
   // fprintf(fp, "SCALARS Diffuse float 1\n");
   // fprintf(fp, "LOOKUP_TABLE default\n");
   // for(int iP=0;iP<ParticleCount;++iP){
   //      fprintf(fp, "%e\n",(float) DiffusiveCoefficient[iP]);
    //}
  //  fprintf(fp, "SCALARS shear float 1\n");
  //  fprintf(fp, "LOOKUP_TABLE default\n");
  //  for(int iP=0;iP<ParticleCount;++iP){
  //       fprintf(fp, "%e\n",(float) ShearRate[iP]);
  //  }

    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");


    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for 
    }
	fprintf(fp, "\n");
	
    fflush(fp);
    fclose(fp);

}

static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else	//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
	
	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	

}


static void initializeFluid()
{
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
    for (int iP = 0; iP < ParticleCount; ++iP) {
        Temperature[iP] = InitialTemperature[Property[iP]];
    }

	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}
    
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])

}



static void initializeWall()
{
	
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
		
	}
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}

static void initializeDomain( void )
{
	CellWidth = ParticleSpacing;
	
	double cellCount[DIM];
	
	cellCount[0] = round((DomainMax[0] - DomainMin[0])/CellWidth);
	cellCount[1] = round((DomainMax[1] - DomainMin[1])/CellWidth);
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = round((DomainMax[2] - DomainMin[2])/CellWidth);
	#endif
	
	CellCount[0] = (int)cellCount[0];
	CellCount[1] = (int)cellCount[1];
	CellCount[2] = (int)cellCount[2];
	CellCounts   = cellCount[0]*cellCount[1]*cellCount[2];
	
	if(cellCount[0]!=(double)CellCount[0] || cellCount[1]!=(double)CellCount[1] ||cellCount[2]!=(double)CellCount[2]){
		fprintf(stderr,"DomainWidth/CellWidth is not integer\n");
		DomainMax[0] = DomainMin[0] + CellWidth*(double)CellCount[0];
		DomainMax[1] = DomainMin[1] + CellWidth*(double)CellCount[1];
		DomainMax[2] = DomainMin[2] + CellWidth*(double)CellCount[2];
		fprintf(stderr,"Changing the Domain Max to (%e,%e,%e)\n", DomainMax[0], DomainMax[1], DomainMax[2]);
	}
	DomainWidth[0] = DomainMax[0] - DomainMin[0];
	DomainWidth[1] = DomainMax[1] - DomainMin[1];
	DomainWidth[2] = DomainMax[2] - DomainMin[2];
	
	CellParticleBegin = (int *)malloc( CellCounts * sizeof(int) );
	CellParticleEnd   = (int *)malloc( CellCounts * sizeof(int) );
	#pragma acc enter data create(CellParticleBegin[0:CellCounts])
	#pragma acc enter data create(CellParticleEnd  [0:CellCounts])
	#pragma acc enter data attach(CellParticleBegin,CellParticleEnd)
	
	
	// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
	ParticleCountPower=0;  
	while((ParticleCount>>ParticleCountPower)!=0){
		++ParticleCountPower;
	}
	PowerParticleCount = (1<<ParticleCountPower);
	fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
	CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	#pragma acc enter data create(CellIndex   [0:PowerParticleCount])
	#pragma acc enter data create(CellParticle[0:PowerParticleCount])
	#pragma acc enter data attach(CellIndex,CellParticle)
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((RadiusG>MaxRadius) ? RadiusG : MaxRadius);
	MaxRadius = ((RadiusP>MaxRadius) ? RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
	
	#pragma acc update device(CellWidth,CellCount[0:DIM],CellCounts)
	#pragma acc update device(DomainMax[0:DIM],DomainMin[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
	#pragma acc update device(MaxRadius)
}


static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction (max:maxShift2)
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Mod(Position[iP][iD] - NeighborCalculatedPosition[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}




static void calculateNeighbor( void )
{
	
	// calculate CellIndex[iP]
	#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<(1<<ParticleCountPower); ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = CellCount[0]*CellCount[1]*CellCount[2];
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	{
		// sort with CellIndex
		// https://edom18.hateblo.jp/entry/2020/09/21/150416
		for(int iMain=0;iMain<ParticleCountPower;++iMain){
			for(int iSub=0;iSub<=iMain;++iSub){
				
				int dist = (1<< (iMain-iSub));
				
				#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
				#pragma acc loop independent
				#pragma omp parallel for
				for(int iP=0;iP<(1<<ParticleCountPower);++iP){
					bool up = ((iP >> iMain) & 2) == 0;
					
					if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
						int tmpCellIndex    = CellIndex[ iP ];
						int tmpCellParticle = CellParticle[ iP ];
						CellIndex[ iP ]     = CellIndex[ iP | dist ];
						CellParticle[ iP ]  = CellParticle[ iP | dist ];
						CellIndex[ iP | dist ]    = tmpCellIndex;
						CellParticle[ iP | dist ] = tmpCellParticle;
					}
				}
			}
		}
	}
	
	// search for CellParticleBegin[iC]
	#pragma acc kernels present(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iC=0;iC<CellCounts;++iC){
			CellParticleBegin[iC]=0;
			CellParticleEnd[iC]=0;
		}
		
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0; iP<ParticleCount; ++iP){
			if( CellIndex[iP]<CellIndex[iP+1] ){
				CellParticleEnd[ CellIndex[iP] ]   =iP+1;
				CellParticleBegin[ CellIndex[iP+1] ]=iP+1;
			}
		}
	}
    
    // calculate neighbor
	#pragma acc kernels present(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborCount[iP]=0;
    	#pragma acc loop seq
    	for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
    		Neighbor[iP][iN]=-1;
    	}
    }
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int range = (int)(ceil((MaxRadius+MARGIN)/CellWidth));
    	const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
    	const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
    	const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];

#ifdef TWO_DIMENSIONAL
        #pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
            #pragma acc loop seq
            for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
                const int jCZ=0;
                const int jC=CellId(jCX,jCY,jCZ);
                #pragma acc loop seq
                for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                    const int jP=CellParticle[jCP];
                    double qij[DIM];
                    #pragma acc loop seq
                    for(int iD=0;iD<DIM;++iD){
                        qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                    }
                    const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                    if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                        if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        	NeighborCount[iP]++;
                        }
                        else if(iP!=jP){
                            Neighbor[iP][NeighborCount[iP]] = jP;
                            NeighborCount[iP]++;
                        }
                    }
                }
            }
        }

    	    	
#else // TWO_DIMENSIONAL
        #pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        	#pragma acc loop seq
        	for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
                #pragma acc loop seq
                for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
                    const int jC=CellId(jCX,jCY,jCZ);
                    #pragma acc loop seq
                    for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                        const int jP=CellParticle[jCP];
                        double qij[DIM];
                        #pragma acc loop seq
                        for(int iD=0;iD<DIM;++iD){
                            qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                        }
                        const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                        if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                            if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        		NeighborCount[iP]++;
                        	}
                        	else if(iP!=jP){
                            	Neighbor[iP][NeighborCount[iP]] = jP;
                            	NeighborCount[iP]++;
                        	}
                        }
                    }
                }
            }
        }
#endif // TWO_DIMENSIONAL
    }
	
	#pragma acc kernels present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
		}
	}
	
}




static void calculateConvection()
{
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){

        Acceleration[iP][0] += Force[iP][0]/Mass[iP];
        Acceleration[iP][1] += Force[iP][1]/Mass[iP];
        Acceleration[iP][2] += Force[iP][2]/Mass[iP];

        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}


static void resetForce()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{	
  #pragma acc kernels present (Property[0:ParticleCount],Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mass[iP]=Density[Property[iP]]*ParticleVolume;
    }
    
    #pragma acc kernels present (Kappa[0:ParticleCount],Property[0:ParticleCount],VolStrainP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Kappa[iP]=BulkModulus[Property[iP]];
        if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
    }
   
    
    #pragma acc kernels present(Lambda[0:ParticleCount],VolStrainP[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
  Lambda[iP]=BulkViscosity[Property[iP]];
        if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
    }
    
    #pragma acc kernels present (Property[0:ParticleCount],Mu[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mu[iP]=ShearViscosity[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],Conductivity[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Conductivity[iP]=ThermalConductivity[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],LatentH[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        LatentH[iP]=LatentHeat[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],Cp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Cp[iP]=SpecificHeat[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],SolidusTemp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        SolidusTemp[iP]=SolidusTemperature[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],LiquidusTemp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        LiquidusTemp[iP]=LiquidusTemperature[Property[iP]];
    }

}


static void initialDisplacement()
{
#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for (int iP = 0; iP < ParticleCount; ++iP) {

        const double x = Position[iP][0];
        const double y = Position[iP][1];
        if (x > 0.224 && x < 0.226 && y> 0.098) {
    //    if (x > 0.46 && x < 0.47 && y< 0.26) {
    //        Velocity[iP][1] -= 5.0e-7/Dt;
        }
    }
}


#ifndef DEBRIS_SOLID_FRACTION
#define DEBRIS_SOLID_FRACTION 0.10
#endif

#ifndef TWO_WAY_DEBRIS_FLUID_COUPLING
#define TWO_WAY_DEBRIS_FLUID_COUPLING 1
#endif


#pragma acc routine seq
static inline double clampAlphaDebris(const double a)
{
    if (a < 1.0e-4) return 1.0e-4;
    if (a > 0.64)   return 0.64;
    return a;
}


#pragma acc routine seq
static inline double calcCdSchillerNaumannDebris(const double Re)
{
    const double eps = 1.0e-30;

    if (Re <= eps) return 0.0;

    if (Re <= 1000.0) {
        return 24.0 / Re * (1.0 + 0.15 * pow(Re, 0.687));
    }

    return 0.44;
}


#pragma acc routine seq
static inline double calcInterphaseBetaDebris(
    const double alpha_s_in,
    const double rho_f,
    const double mu_f,
    const double dp,
    const double ur_abs)
{
    const double eps = 1.0e-12;

    double alpha_s = clampAlphaDebris(alpha_s_in);
    double alpha_f = 1.0 - alpha_s;

    if (alpha_f < 1.0e-4) {
        alpha_f = 1.0e-4;
    }

    const double Re =
        rho_f * ur_abs * dp / (mu_f + eps);

    const double Cd =
        calcCdSchillerNaumannDebris(Re);

    double beta = 0.0;

    if (alpha_f <= 0.80) {
        beta =
            150.0 * alpha_s * alpha_s * mu_f
            / ((alpha_f + eps) * dp * dp + eps)
            +
            1.75 * alpha_s * rho_f * ur_abs
            / (dp + eps);
    } else {
        beta =
            0.75 * Cd * rho_f * alpha_s * alpha_f * ur_abs
            / (dp + eps)
            * pow(alpha_f, -2.65);
    }

    if (beta < 0.0) {
        beta = 0.0;
    }

    return beta;
}


/* ============================================================
   Structure/debris side only.
   This is called inside buildExternalForce().
   It modifies only ExternalForceBuf[iP].
   ============================================================ */

static void addDebrisFluidForceToStructure()
{
    const double eps = 1.0e-12;
    const double alpha_s = clampAlphaDebris(DEBRIS_SOLID_FRACTION);

    #pragma acc parallel loop present( \
        ExternalForceBuf[0:ParticleCount][0:DIM], \
        Position[0:ParticleCount][0:DIM], \
        Velocity[0:ParticleCount][0:DIM], \
        NeighborCount[0:ParticleCount], \
        Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
        DomainWidth[0:DIM], \
        Property[0:ParticleCount], \
        Density[0:TYPE_COUNT], \
        Mu[0:ParticleCount], \
        Mass[0:ParticleCount], \
        Gravity[0:DIM], \
        GranularSize[0:TYPE_COUNT]) \
        firstprivate(alpha_s)
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        if (iP < 0 || iP >= ParticleCount) continue;

        const int type_i = Property[iP];

        if (type_i < 0 || type_i >= TYPE_COUNT) continue;
        if (!(STRUCTURE_BEGIN <= type_i && type_i < STRUCTURE_END)) continue;

        const double dp = GranularSize[type_i];
        const double rho_s = Density[type_i];

        if (dp <= eps) continue;
        if (rho_s <= eps) continue;
        if (Mass[iP] <= eps) continue;

        const double V_control =
            Mass[iP] / (alpha_s * rho_s + eps);

        if (V_control <= eps) continue;

        int nNei = NeighborCount[iP];

        if (nNei > MAX_NEIGHBOR_COUNT) nNei = MAX_NEIGHBOR_COUNT;
        if (nNei < 0) nNei = 0;
        if (nNei <= 0) continue;

        double sum_w = 0.0;
        double rho_f_avg = 0.0;
        double mu_f_avg  = 0.0;
        double uf_avg[DIM];

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            uf_avg[d] = 0.0;
        }

        #pragma acc loop seq
        for (int n = 0; n < nNei; ++n) {

            const int jP = Neighbor[iP][n];

            if (jP < 0 || jP >= ParticleCount) continue;
            if (jP == iP) continue;

            const int type_j = Property[jP];

            if (type_j < 0 || type_j >= TYPE_COUNT) continue;
            if (!(FLUID_BEGIN <= type_j && type_j < FLUID_END)) continue;

            double rij2 = 0.0;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                const double dx =
                    Mod(Position[jP][d] - Position[iP][d]
                        + 0.5 * DomainWidth[d],
                        DomainWidth[d])
                    - 0.5 * DomainWidth[d];

                rij2 += dx * dx;
            }

            if (rij2 <= 1.0e-20) continue;
            if (rij2 > RadiusP * RadiusP) continue;

            const double w = 1.0;

            sum_w += w;

            rho_f_avg += w * Density[type_j];
            mu_f_avg  += w * Mu[jP];

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                uf_avg[d] += w * Velocity[jP][d];
            }
        }

        if (sum_w <= eps) continue;

        const double inv_sum_w = 1.0 / sum_w;

        rho_f_avg *= inv_sum_w;
        mu_f_avg  *= inv_sum_w;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            uf_avg[d] *= inv_sum_w;
        }

        if (rho_f_avg <= eps) continue;
        if (mu_f_avg <= eps) continue;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            ExternalForceBuf[iP][d] +=
                -rho_f_avg * alpha_s * V_control * Gravity[d];
        }

        double ur[DIM];
        double ur2 = 0.0;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            ur[d] = uf_avg[d] - Velocity[iP][d];
            ur2 += ur[d] * ur[d];
        }

        const double ur_abs = sqrt(ur2);

        if (ur_abs <= 1.0e-15) continue;

        const double beta =
            calcInterphaseBetaDebris(
                alpha_s,
                rho_f_avg,
                mu_f_avg,
                dp,
                ur_abs
            );

        if (beta <= eps) continue;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            ExternalForceBuf[iP][d] +=
                beta * V_control * ur[d];
        }
    }
}


/* ============================================================
   Fluid side drag reaction only.
   This function modifies only Force[jP].
   Call this immediately before calculateAcceleration().
   ============================================================ */

static void addDebrisDragReactionForceToFluid()
{
    const double eps = 1.0e-12;
    const double alpha_s = clampAlphaDebris(DEBRIS_SOLID_FRACTION);

    #pragma acc parallel loop present( \
        Force[0:ParticleCount][0:DIM], \
        Position[0:ParticleCount][0:DIM], \
        Velocity[0:ParticleCount][0:DIM], \
        NeighborCount[0:ParticleCount], \
        Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
        DomainWidth[0:DIM], \
        Property[0:ParticleCount], \
        Density[0:TYPE_COUNT], \
        Mu[0:ParticleCount], \
        Mass[0:ParticleCount], \
        GranularSize[0:TYPE_COUNT]) \
        firstprivate(alpha_s)
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        if (iP < 0 || iP >= ParticleCount) continue;

        const int type_i = Property[iP];

        if (type_i < 0 || type_i >= TYPE_COUNT) continue;
        if (!(STRUCTURE_BEGIN <= type_i && type_i < STRUCTURE_END)) continue;

        const double dp = GranularSize[type_i];
        const double rho_s = Density[type_i];

        if (dp <= eps) continue;
        if (rho_s <= eps) continue;
        if (Mass[iP] <= eps) continue;

        const double V_control =
            Mass[iP] / (alpha_s * rho_s + eps);

        if (V_control <= eps) continue;

        int nNei = NeighborCount[iP];

        if (nNei > MAX_NEIGHBOR_COUNT) nNei = MAX_NEIGHBOR_COUNT;
        if (nNei < 0) nNei = 0;
        if (nNei <= 0) continue;

        double sum_w = 0.0;
        double rho_f_avg = 0.0;
        double mu_f_avg  = 0.0;
        double uf_avg[DIM];

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            uf_avg[d] = 0.0;
        }

        #pragma acc loop seq
        for (int n = 0; n < nNei; ++n) {

            const int jP = Neighbor[iP][n];

            if (jP < 0 || jP >= ParticleCount) continue;
            if (jP == iP) continue;

            const int type_j = Property[jP];

            if (type_j < 0 || type_j >= TYPE_COUNT) continue;
            if (!(FLUID_BEGIN <= type_j && type_j < FLUID_END)) continue;

            double rij2 = 0.0;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                const double dx =
                    Mod(Position[jP][d] - Position[iP][d]
                        + 0.5 * DomainWidth[d],
                        DomainWidth[d])
                    - 0.5 * DomainWidth[d];

                rij2 += dx * dx;
            }

            if (rij2 <= 1.0e-20) continue;
            if (rij2 > RadiusP * RadiusP) continue;

            const double w = 1.0;

            sum_w += w;

            rho_f_avg += w * Density[type_j];
            mu_f_avg  += w * Mu[jP];

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                uf_avg[d] += w * Velocity[jP][d];
            }
        }

        if (sum_w <= eps) continue;

        const double inv_sum_w = 1.0 / sum_w;

        rho_f_avg *= inv_sum_w;
        mu_f_avg  *= inv_sum_w;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            uf_avg[d] *= inv_sum_w;
        }

        if (rho_f_avg <= eps) continue;
        if (mu_f_avg <= eps) continue;

        double ur[DIM];
        double ur2 = 0.0;

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            ur[d] = uf_avg[d] - Velocity[iP][d];
            ur2 += ur[d] * ur[d];
        }

        const double ur_abs = sqrt(ur2);

        if (ur_abs <= 1.0e-15) continue;

        const double beta =
            calcInterphaseBetaDebris(
                alpha_s,
                rho_f_avg,
                mu_f_avg,
                dp,
                ur_abs
            );

        if (beta <= eps) continue;

        double F_inter[DIM];

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            F_inter[d] =
                beta * V_control * ur[d];
        }

#if TWO_WAY_DEBRIS_FLUID_COUPLING

        #pragma acc loop seq
        for (int n = 0; n < nNei; ++n) {

            const int jP = Neighbor[iP][n];

            if (jP < 0 || jP >= ParticleCount) continue;
            if (jP == iP) continue;

            const int type_j = Property[jP];

            if (type_j < 0 || type_j >= TYPE_COUNT) continue;
            if (!(FLUID_BEGIN <= type_j && type_j < FLUID_END)) continue;

            double rij2 = 0.0;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                const double dx =
                    Mod(Position[jP][d] - Position[iP][d]
                        + 0.5 * DomainWidth[d],
                        DomainWidth[d])
                    - 0.5 * DomainWidth[d];

                rij2 += dx * dx;
            }

            if (rij2 <= 1.0e-20) continue;
            if (rij2 > RadiusP * RadiusP) continue;

            const double share = inv_sum_w;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                #pragma acc atomic update
                Force[jP][d] -= share * F_inter[d];
            }
        }

#endif
    }
}


/* ============================================================
   Fluid side buoyancy reaction only.
   This function modifies only Force[jP].
   Call this immediately before calculateAcceleration().
   ============================================================ */

static void addDebrisBuoyancyReactionForceToFluid()
{
    const double eps = 1.0e-12;
    const double alpha_s = clampAlphaDebris(DEBRIS_SOLID_FRACTION);

    #pragma acc parallel loop present( \
        Force[0:ParticleCount][0:DIM], \
        Position[0:ParticleCount][0:DIM], \
        NeighborCount[0:ParticleCount], \
        Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
        DomainWidth[0:DIM], \
        Property[0:ParticleCount], \
        Density[0:TYPE_COUNT], \
        Mass[0:ParticleCount], \
        Gravity[0:DIM], \
        GranularSize[0:TYPE_COUNT]) \
        firstprivate(alpha_s)
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        if (iP < 0 || iP >= ParticleCount) continue;

        const int type_i = Property[iP];

        if (type_i < 0 || type_i >= TYPE_COUNT) continue;
        if (!(STRUCTURE_BEGIN <= type_i && type_i < STRUCTURE_END)) continue;

        const double dp = GranularSize[type_i];
        const double rho_s = Density[type_i];

        if (dp <= eps) continue;
        if (rho_s <= eps) continue;
        if (Mass[iP] <= eps) continue;

        const double V_control =
            Mass[iP] / (alpha_s * rho_s + eps);

        if (V_control <= eps) continue;

        int nNei = NeighborCount[iP];

        if (nNei > MAX_NEIGHBOR_COUNT) nNei = MAX_NEIGHBOR_COUNT;
        if (nNei < 0) nNei = 0;
        if (nNei <= 0) continue;

        double sum_w = 0.0;
        double rho_f_avg = 0.0;

        #pragma acc loop seq
        for (int n = 0; n < nNei; ++n) {

            const int jP = Neighbor[iP][n];

            if (jP < 0 || jP >= ParticleCount) continue;
            if (jP == iP) continue;

            const int type_j = Property[jP];

            if (type_j < 0 || type_j >= TYPE_COUNT) continue;
            if (!(FLUID_BEGIN <= type_j && type_j < FLUID_END)) continue;

            double rij2 = 0.0;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                const double dx =
                    Mod(Position[jP][d] - Position[iP][d]
                        + 0.5 * DomainWidth[d],
                        DomainWidth[d])
                    - 0.5 * DomainWidth[d];

                rij2 += dx * dx;
            }

            if (rij2 <= 1.0e-20) continue;
            if (rij2 > RadiusP * RadiusP) continue;

            sum_w += 1.0;
            rho_f_avg += Density[type_j];
        }

        if (sum_w <= eps) continue;

        rho_f_avg /= sum_w;

        double F_buoy_fluid[DIM];

        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            F_buoy_fluid[d] =
                rho_f_avg * alpha_s * V_control * Gravity[d];
        }

        const double inv_sum_w = 1.0 / sum_w;

        #pragma acc loop seq
        for (int n = 0; n < nNei; ++n) {

            const int jP = Neighbor[iP][n];

            if (jP < 0 || jP >= ParticleCount) continue;
            if (jP == iP) continue;

            const int type_j = Property[jP];

            if (type_j < 0 || type_j >= TYPE_COUNT) continue;
            if (!(FLUID_BEGIN <= type_j && type_j < FLUID_END)) continue;

            double rij2 = 0.0;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                const double dx =
                    Mod(Position[jP][d] - Position[iP][d]
                        + 0.5 * DomainWidth[d],
                        DomainWidth[d])
                    - 0.5 * DomainWidth[d];

                rij2 += dx * dx;
            }

            if (rij2 <= 1.0e-20) continue;
            if (rij2 > RadiusP * RadiusP) continue;

            const double share = inv_sum_w;

            #pragma acc loop seq
            for (int d = 0; d < DIM; ++d) {
                #pragma acc atomic update
                Force[jP][d] += share * F_buoy_fluid[d];
            }
        }
    }
}



/* ============================================================
   Keep these names if your main loop already calls them.
   They affect only fluid particles.
   ============================================================ */

static void calculateDragForce(void)
{
    addDebrisDragReactionForceToFluid();
}


static void calculateBuoyancyForce(void)
{
    addDebrisBuoyancyReactionForceToFluid();
}


static void calculateEnergyConservation()
{
    const double eps = 1.0e-30;

#ifdef _OPENACC
    #pragma acc parallel loop present( \
        Temperature[0:ParticleCount], \
        TemperatureOld[0:ParticleCount])
#else
    #pragma omp parallel for
#endif
    for (int iP = 0; iP < ParticleCount; ++iP) {
        TemperatureOld[iP] = Temperature[iP];
    }


#ifdef _OPENACC
    #pragma acc parallel loop present( \
        Property[0:ParticleCount], \
        Position[0:ParticleCount][0:DIM], \
        Temperature[0:ParticleCount], \
        TemperatureOld[0:ParticleCount], \
        SolidusTemp[0:ParticleCount], \
        LiquidusTemp[0:ParticleCount], \
        LatentH[0:ParticleCount], \
        Cp[0:ParticleCount], \
        Conductivity[0:ParticleCount], \
        NeighborCount[0:ParticleCount], \
        Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
        DomainWidth[0:DIM], \
        Density[0:TYPE_COUNT])
#else
    #pragma omp parallel for
#endif
    for (int iP = 0; iP < ParticleCount; ++iP) {

        const double Ti = TemperatureOld[iP];
        const double Ts = SolidusTemp[iP];
        const double Tl = LiquidusTemp[iP];

        double cp_eff = Cp[iP];

        if (Tl > Ts + eps) {
            if (Ti >= Ts && Ti <= Tl) {
                cp_eff = Cp[iP] + LatentH[iP] / (Tl - Ts);
            }
        }

        if (cp_eff < eps) cp_eff = eps;

        double heat_rate = 0.0;
        #pragma acc loop seq
        for (int iN = 0; iN < NeighborCount[iP]; ++iN) {

            const int jP = Neighbor[iP][iN];
            if (iP == jP) continue;

            double xij[DIM];
 #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                xij[iD] =
                    Mod(Position[jP][iD] - Position[iP][iD]
                    + 0.5 * DomainWidth[iD], DomainWidth[iD])
                    - 0.5 * DomainWidth[iD];
            }

            const double radius = RadiusV;

            const double rij2 =
                xij[0] * xij[0]
              + xij[1] * xij[1]
              + xij[2] * xij[2];

            if (rij2 < radius * radius && rij2 > eps) {

                const double rij  = sqrt(rij2);
                const double dwij = -dwvdr(rij, radius);
                const double wwij = dwij / rij;

                const double ki = Conductivity[iP];
                const double kj = Conductivity[jP];

                double kij = 0.0;

                if (ki + kj > eps) {
                    kij = 2.0 * ki * kj / (ki + kj);
                }

                heat_rate +=
                    2.0 * kij
                    / Density[Property[iP]]
                    * (TemperatureOld[jP] - TemperatureOld[iP])
                    * wwij;
            }
        }

        const double dT = heat_rate * Dt / cp_eff;

        Temperature[iP] = TemperatureOld[iP] + dT;
    }
}




static void calculateDensityA()
{
    
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	{	
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
        
			double sum = 0.0;
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
                  	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double radius = RadiusA;
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(radius*radius - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wa(rij,radius);
					sum += weight;
				}
			}
			DensityA[iP]=sum;
		}
	}
}


static void calculateGravityCenter()
{
 

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
			double sum[DIM]={0.0,0.0,0.0};
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
                  	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(RadiusG*RadiusG - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wg(rij,RadiusG);
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						sum[iD] += xij[iD]*weight/R2g*RadiusG;
					}
				}
			}
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				GravityCenter[iP][iD] = sum[iD];
			}
		}
	}
}

static void calculatePressureA()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],PressureA[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureA[0:ParticleCount],DensityA[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
      
    	double force[DIM]={0.0,0.0,0.0};
        #pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateDiffuseInterface()
{
	
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
       
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VolStrainP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void calculateDivergenceP()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],DivergenceP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum -= uij[iD]*eij[iD]*dw;
				}
			}
		}
		DivergenceP[iP]=sum;
	}
}

static void calculatePressureP()
{
	
	#pragma acc kernels present (PressureP[0:ParticleCount],Lambda[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],Kappa[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureP[iP] = -Lambda[iP]*DivergenceP[iP];
		if(VolStrainP[iP]>0.0){
			PressureP[iP]+=Kappa[iP]*VolStrainP[iP];
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureP[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 > 0){
			
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}



static void calculateViscosityV(){

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
  
        #pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
              	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
         if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
            	double fij[DIM] = {0.0,0.0,0.0};
            	#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}




//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//=================Elastoplastic calculation============================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//



static void calculateLamesconstant()
{
#pragma acc kernels present(Property[0:ParticleCount],MuLames[0:ParticleCount],LambdaLames[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
	 for(int iP=0;iP<ParticleCount;++iP){
    const double E = YoungModulus[Property[iP]];
    const double v = PoissonRatio[Property[iP]];

        LambdaLames[iP]=(E*v)/((1+v)*(1-2*v));
        MuLames[iP]=E/(2*(1+v));
    }
}


static void resetAcceleration()
{
	#pragma acc kernels loop present(Acceleration[0:ParticleCount][0:DIM])
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Acceleration[iP][iD]=0.0;
			
        }
	}
}

static void resetStructureForce()
{
#pragma acc parallel loop present(Force[0:ParticleCount][0:DIM])
#pragma omp parallel for
for(int iP=0;iP<ParticleCount;++iP){
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Force[iP][iD] = 0.0;
        }
    }
}

#pragma acc routine seq
static inline int isStructureParticleImplicit(const int p)
{
	return (Property[p] >= STRUCTURE_BEGIN && Property[p] < STRUCTURE_END);
}
/* if your rigid particles are fixed / prescribed boundaries for structure,
   keep RIGID here. If not, remove the rigid branch. */
#pragma acc routine seq
static inline int isDirichletLikeParticleImplicit(const int p)
{
	return (Property[p] >= WALL_BEGIN  && Property[p] < WALL_END);
}

static void calculateStrainRateTensor() {
	#pragma acc parallel loop present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM], \
									  Velocity[0:ParticleCount][0:DIM], \
									  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
									  NeighborCount[0:ParticleCount], \
									  Strain[0:ParticleCount][0:DIM][0:DIM], \
									  ShearRate[0:ParticleCount], \
									  DomainWidth[0:DIM])
 #pragma omp parallel for
 for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
		double eps[3][3] = {{0.0}};
	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			if(iP==jP) continue;
            if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;

            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }

            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if (rij2 <= RadiusP * RadiusP && rij2 > 1.0e-20) {

			const double rij = sqrt(rij2);
			const double dw = dwpdr(rij, RadiusP);
			const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};

		
		double uij[DIM];
        #pragma acc loop seq
			for (int d = 0; d < DIM; ++d) {
				uij[d] = Velocity[jP][d] - Velocity[iP][d];
			}
            #pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
            #pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				eps[iD][jD] += 0.5 * (uij[iD] * eij[jD] + uij[jD] * eij[iD]) * dw;
			}
		}
	}
		}
		double ss = 0.0;
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD)
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				Strain[iP][iD][jD] = eps[iD][jD];
				ss += eps[iD][jD] * eps[iD][jD];
		}
		ShearRate[iP] = sqrt(0.5 * ss);
	}
}



static void calculateSpinTensor() {
	#pragma acc parallel loop present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM], \
									  Velocity[0:ParticleCount][0:DIM], \
									  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
									  NeighborCount[0:ParticleCount], \
									  DomainWidth[0:DIM], \
									  Spin[0:ParticleCount][0:DIM][0:DIM])
		 #pragma omp parallel for
		 for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
		double omega[DIM][DIM] = {{0.0}};
#pragma acc loop seq
		for (int iN = 0; iN < NeighborCount[iP]; ++iN) {
			const int jP=Neighbor[iP][iN];
			if(iP==jP) continue;
            if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;
			double xij[DIM];

			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD) {
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];			
			}

            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if (rij2 <= RadiusP * RadiusP && rij2 > 1.0e-20) {
			const double rij = sqrt(rij2);
			const double dw = dwpdr(rij, RadiusP);
			const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
		
		double uij[DIM];
        #pragma acc loop seq
			for (int d = 0; d < DIM; ++d) {
				uij[d] = Velocity[jP][d] - Velocity[iP][d];
			}

			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD)
				#pragma acc loop seq
				for (int jD = 0; jD < DIM; ++jD)
					omega[iD][jD] += 0.5 * (uij[iD] * eij[jD] - uij[jD] * eij[iD]) * dw;
		}
	}

		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD)
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD)
				Spin[iP][iD][jD] = omega[iD][jD];
	}
}


static void initializeImplicitVelocityGuess()
{
#pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM], \
                                  VelocityOld[0:ParticleCount][0:DIM], \
                                  Position[0:ParticleCount][0:DIM], \
                                  PositionOld[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Velocity[iP][iD] = VelocityOld[iP][iD];
            Position[iP][iD] = PositionOld[iP][iD];// + Elastic_Dt * Velocity[iP][iD];
        }
    }
}

static void buildVelocityResidual()
{
    const double dt = Elastic_Dt;
    const double beta_force = 1.0;   /* Crank–Nicolson */

#pragma acc parallel loop present(ResidualV[0:ParticleCount][0:DIM], \
                                  Velocity[0:ParticleCount][0:DIM], \
                                  VelocityOld[0:ParticleCount][0:DIM], \
                                  Mass[0:ParticleCount], \
                                  InternalForceBuf[0:ParticleCount][0:DIM], \
                                  InternalForceOldBuf[0:ParticleCount][0:DIM], \
                                  ExternalForceBuf[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {

            const double fint =
                (1.0 - beta_force) * InternalForceOldBuf[iP][iD]
              + beta_force         * InternalForceBuf[iP][iD];

            ResidualV[iP][iD] =
                Mass[iP] * (Velocity[iP][iD] - VelocityOld[iP][iD])
              - dt * (fint + ExternalForceBuf[iP][iD]);
        }
    }
}

static double dotVelocityVector(const double (*a)[DIM],
                                const double (*b)[DIM])
{
    double sum = 0.0;

#pragma acc parallel loop reduction(+:sum) present(a[0:ParticleCount][0:DIM], \
                                                   b[0:ParticleCount][0:DIM])
	 #pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
#pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            sum += a[iP][iD] * b[iP][iD];
        }
    }

    return sum;
}


static double normVelocityVector(const double (*a)[DIM])
{
    const double n2 = dotVelocityVector(a, a);
    return sqrt(fmax(n2, 0.0));
}

static void applyVelocityDiagonalPreconditioner(const double (*r)[DIM],
                                                double (*z)[DIM])
{
#pragma acc parallel loop present(r[0:ParticleCount][0:DIM],       \
                                  z[0:ParticleCount][0:DIM],       \
                                  NewtonDiagVec[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
#pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {

            double diag = NewtonDiagVec[iP][iD];

            if (fabs(diag) < 1.0e-30) {
                diag = (diag >= 0.0) ? 1.0e-30 : -1.0e-30;
            }

            z[iP][iD] = r[iP][iD] / diag;
        }
    }
}

static double dotStructureVector(double (*A)[DIM], double (*B)[DIM])
{
    double sum = 0.0;

#pragma acc parallel loop reduction(+:sum) present(A[0:ParticleCount][0:DIM], \
                                                   B[0:ParticleCount][0:DIM])
#pragma omp parallel for reduction(+:sum)
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            sum += A[iP][iD] * B[iP][iD];
        }
    }

    return sum;
}


static double computeVelocityResidualNorm()
{
    double norm2 = 0.0;

	#pragma acc parallel loop reduction(+:norm2) present(ResidualV[0:ParticleCount][0:DIM])
	#pragma omp parallel for reduction(+:norm2)
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            norm2 += ResidualV[iP][iD] * ResidualV[iP][iD];
        }
    }
    return sqrt(norm2);
}

static void saveNewtonBaseVelocity()
{
#pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM], \
                                  VelocityNewtonBase[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            VelocityNewtonBase[iP][iD] = Velocity[iP][iD];
        }
    }
}

static void restoreNewtonBaseVelocity()
{
#pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM], \
                                  VelocityNewtonBase[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Velocity[iP][iD] = VelocityNewtonBase[iP][iD];
        }
    }
}

static void applyVelocityUpdateFromBase(const double alpha)
{
#pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM], \
                                  VelocityNewtonBase[0:ParticleCount][0:DIM], \
                                  DeltaVelocity[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Velocity[iP][iD] = VelocityNewtonBase[iP][iD] + alpha * DeltaVelocity[iP][iD];
        }
    }
}

static void updatePositionFromVelocityImplicit()
{
    const double dt = Elastic_Dt;
    const double theta = 1.0;

#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
                                  PositionOld[0:ParticleCount][0:DIM], \
                                  Velocity[0:ParticleCount][0:DIM], \
                                  VelocityOld[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Position[iP][iD] = PositionOld[iP][iD]
                             + dt * ((1.0 - theta) * VelocityOld[iP][iD]
                             + theta * Velocity[iP][iD]);
        }
    }
}






static void saveElasticOldState()
{
#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
                                  Velocity[0:ParticleCount][0:DIM], \
                                  Stress[0:ParticleCount][0:DIM][0:DIM], \
                                  Damage[0:ParticleCount], \
                                  MaxEquivalentStrain[0:ParticleCount], \
                                  PositionOld[0:ParticleCount][0:DIM], \
                                  VelocityOld[0:ParticleCount][0:DIM], \
                                  StressOld[0:ParticleCount][0:DIM][0:DIM], \
                                  DamageOld[0:ParticleCount], \
                                  MaxEquivalentStrainOld[0:ParticleCount])
#pragma omp parallel for
for(int iP=0;iP<ParticleCount;++iP){
        DamageOld[iP] = Damage[iP];
        MaxEquivalentStrainOld[iP] = MaxEquivalentStrain[iP];

        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            PositionOld[iP][iD] = Position[iP][iD];
            VelocityOld[iP][iD] = Velocity[iP][iD];
            #pragma acc loop seq
            for (int jD = 0; jD < DIM; ++jD) {
                StressOld[iP][iD][jD] = Stress[iP][iD][jD];
            }
        }
    }
}

static void restoreDamageHistoryFromOldState()
{
#pragma acc parallel loop present(Damage[0:ParticleCount], \
                                  MaxEquivalentStrain[0:ParticleCount], \
                                  DamageOld[0:ParticleCount], \
                                  MaxEquivalentStrainOld[0:ParticleCount])
#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {
        Damage[iP] = DamageOld[iP];
        MaxEquivalentStrain[iP] = MaxEquivalentStrainOld[iP];
    }
}

static void buildExternalForce()
{
#pragma acc parallel loop present(ExternalForceBuf[0:ParticleCount][0:DIM], \
                                  Mass[0:ParticleCount], \
                                  Gravity[0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        #pragma acc loop seq
        for (int d = 0; d < DIM; ++d) {
            ExternalForceBuf[iP][d] = Mass[iP] * Gravity[d];
        }
    }

    addDebrisFluidForceToStructure();
}

static void copyStructureForceToInternalBuffer()
{
#pragma acc parallel loop present(Force[0:ParticleCount][0:DIM], \
                                  InternalForceBuf[0:ParticleCount][0:DIM])
#pragma omp parallel for
for(int iP=0;iP<ParticleCount;++iP){
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            InternalForceBuf[iP][iD] = Force[iP][iD];
        }
    }
}

#pragma acc routine seq
static inline void buildElasticTangentIsotropic(
    const double lambda,
    const double mu,
    double Ce[DIM][DIM][DIM][DIM])
{
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            for (int k = 0; k < DIM; ++k) {
                for (int l = 0; l < DIM; ++l) {
                    const double dij = (i == j) ? 1.0 : 0.0;
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    const double dik = (i == k) ? 1.0 : 0.0;
                    const double djl = (j == l) ? 1.0 : 0.0;
                    const double dil = (i == l) ? 1.0 : 0.0;
                    const double djk = (j == k) ? 1.0 : 0.0;

                    Ce[i][j][k][l] =
                        lambda * dij * dkl
                      + mu * (dik * djl + dil * djk);
                }
            }
        }
    }
}

#pragma acc routine seq
static inline double traceTensor2(const double A[DIM][DIM])
{
    double tr = 0.0;
    for (int i = 0; i < DIM; ++i) tr += A[i][i];
    return tr;
}


#pragma acc routine seq
static inline void buildConsistentTangentCurrentModel(
    const int iP,
    double Cep[DIM][DIM][DIM][DIM])
{
    const double lambda = LambdaLames[iP];
    const double mu     = MuLames[iP];

    double Ce[DIM][DIM][DIM][DIM];
    buildElasticTangentIsotropic(lambda, mu, Ce);


    if (!YieldActive[iP]) {
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                for (int k = 0; k < DIM; ++k)
                    for (int l = 0; l < DIM; ++l)
                        Cep[i][j][k][l] = Ce[i][j][k][l];
        return;
    }

    double df[DIM][DIM], dg[DIM][DIM];
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            df[i][j] = PlasticDf[iP][i][j];
            dg[i][j] = PlasticDg[iP][i][j];
        }
    }

    const double tr_df = traceTensor2(df);
    const double tr_dg = traceTensor2(dg);

    double A[DIM][DIM], B[DIM][DIM];

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            const double dij = (i == j) ? 1.0 : 0.0;
            A[i][j] = lambda * tr_dg * dij + 2.0 * mu * dg[i][j];
            B[i][j] = lambda * tr_df * dij + 2.0 * mu * df[i][j];
        }
    }

    double denom = PlasticCepDen[iP];
    if (fabs(denom) < 1.0e-20) denom = (denom >= 0.0) ? 1.0e-20 : -1.0e-20;

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            for (int k = 0; k < DIM; ++k) {
                for (int l = 0; l < DIM; ++l) {
                    Cep[i][j][k][l] =
                        Ce[i][j][k][l]
                      - (A[i][j] * B[k][l]) / denom;
                }
            }
        }
    }
}
	

#pragma acc routine seq
static inline void buildPairKelConsistent(
    const int iP,
    const int jP,
    const double eij[DIM],
    const double dw,
    double Kel[DIM][DIM])
{
    const double dt = Elastic_Dt;

    double CepI[DIM][DIM][DIM][DIM];
    buildConsistentTangentCurrentModel(iP, CepI);

    const int jIsStruct = isStructureParticleImplicit(jP);

    double CepJ[DIM][DIM][DIM][DIM];

    if (jIsStruct) {
        buildConsistentTangentCurrentModel(jP, CepJ);
    }

    const double pairFactor = jIsStruct ? 4.0 : 2.0;
    #pragma acc loop seq
    for (int a = 0; a < DIM; ++a) {
        #pragma acc loop seq
        for (int b = 0; b < DIM; ++b) {

            double kab = 0.0;

            for (int m = 0; m < DIM; ++m) {
                for (int n = 0; n < DIM; ++n) {

                    double CepPair = CepI[a][m][b][n];

                    if (jIsStruct) {
                        CepPair =
                            0.5 * (CepI[a][m][b][n] + CepJ[a][m][b][n]);
                    }

                    kab += CepPair * eij[m] * eij[n];
                }
            }

            Kel[a][b] =
                pairFactor * dt * dt * ParticleVolume * dw * dw * kab;
        }
    }
}


#pragma acc routine seq
static inline void decomposeStressPQ(
    const double sigma[DIM][DIM],
    double *p,
    double s[DIM][DIM],
    double *q)
{
    double I1 = 0.0;
    for (int a = 0; a < DIM; ++a) I1 += sigma[a][a];
    *p = I1 / (double)DIM;

    double J2 = 0.0;
    for (int a = 0; a < DIM; ++a) {
        for (int b = 0; b < DIM; ++b) {
            const double delta = (a == b) ? 1.0 : 0.0;
            s[a][b] = sigma[a][b] - (*p) * delta;
            J2 += 0.5 * s[a][b] * s[a][b];
        }
    }

    *q = sqrt(fmax(3.0 * J2, 1.0e-24));
}





#pragma acc routine seq
static inline int localReturnMappingDPNewton(
    const double sigma_tr[DIM][DIM],
    const double mu,
    const double K,
    const double alpha_phi,
    const double alpha_psi,
    const double k_dp,
    const double dt,
    double sigma_new[DIM][DIM],
    double df_out[DIM][DIM],
    double dg_out[DIM][DIM],
    double dep_out[DIM][DIM],
    double *dgamma_out,
    double *cep_denom_out)
{
    const int    max_iter  = 25;
    const double tol_f     = 1.0e-10;
    const double tol_dg    = 1.0e-12;
    const double q_tol     = 1.0e-12;

    double p_tr, q_tr;
    double s_tr[DIM][DIM];
    decomposeStressPQ(sigma_tr, &p_tr, s_tr, &q_tr);

    const double f_tr = q_tr - alpha_phi * p_tr - k_dp;


    if (f_tr <= 1.0e-8) {
        for (int a = 0; a < DIM; ++a) {
            for (int b = 0; b < DIM; ++b) {
                sigma_new[a][b] = 0.5 * (sigma_tr[a][b] + sigma_tr[b][a]);
                df_out[a][b]    = 0.0;
                dg_out[a][b]    = 0.0;
                dep_out[a][b]   = 0.0;
            }
        }
        *dgamma_out   = 0.0;
        *cep_denom_out = 0.0;
        return 0;
    }


    double denom = 3.0 * mu + K * alpha_phi * alpha_psi;
    if (fabs(denom) < 1.0e-20) denom = 1.0e-20;

    double gamma = 0.0;


    for (int iter = 0; iter < max_iter; ++iter) {

        const double q = fmax(q_tr - 3.0 * mu * gamma, 0.0);
        const double p = p_tr - K * alpha_psi * gamma;

        const double r = q - alpha_phi * p - k_dp;

        if (fabs(r) < tol_f) break;

      
        const double dr = -denom;

        double dgamma = -r / dr;
        gamma += dgamma;

        if (gamma < 0.0) gamma = 0.0;

        if (fabs(dgamma) < tol_dg) break;
    }

    double q_new = fmax(q_tr - 3.0 * mu * gamma, 0.0);
    double p_new = p_tr - K * alpha_psi * gamma;

  
    const double p_apex =
        (alpha_phi > 1.0e-12) ? (-k_dp / alpha_phi) : -1.0e30;


    int apex_return = 0;
    if (q_new <= q_tol || p_new < p_apex) {
        apex_return = 1;
        q_new = 0.0;
        p_new = fmax(p_new, p_apex);
    }

    if (apex_return) {
        for (int a = 0; a < DIM; ++a) {
            for (int b = 0; b < DIM; ++b) {
                const double delta = (a == b) ? 1.0 : 0.0;
                sigma_new[a][b] = p_new * delta;
            }
        }
    } else {
        const double scale = (q_tr > 1.0e-20) ? (q_new / q_tr) : 0.0;
        for (int a = 0; a < DIM; ++a) {
            for (int b = 0; b < DIM; ++b) {
                const double delta = (a == b) ? 1.0 : 0.0;
                sigma_new[a][b] = scale * s_tr[a][b] + p_new * delta;
            }
        }
    }


    for (int a = 0; a < DIM; ++a) {
        for (int b = 0; b < DIM; ++b) {
            sigma_new[a][b] = 0.5 * (sigma_new[a][b] + sigma_new[b][a]);
        }
    }

   {
        double p_ret, q_ret;
        double s_ret[DIM][DIM];
        decomposeStressPQ(sigma_new, &p_ret, s_ret, &q_ret);

        if (q_ret > 1.0e-20) {
            for (int a = 0; a < DIM; ++a) {
                for (int b = 0; b < DIM; ++b) {
                    const double delta = (a == b) ? 1.0 : 0.0;
                    const double n_dev = (3.0 / (2.0 * q_ret)) * s_ret[a][b];

                    df_out[a][b]  = n_dev - (alpha_phi / (double)DIM) * delta;
                    dg_out[a][b]  = n_dev + (alpha_psi / (double)DIM) * delta;
                    dep_out[a][b] = (gamma / dt) * dg_out[a][b];
                }
            }
        } else {
            for (int a = 0; a < DIM; ++a) {
                for (int b = 0; b < DIM; ++b) {
                    const double delta = (a == b) ? 1.0 : 0.0;
                    df_out[a][b]  = -(alpha_phi / (double)DIM) * delta;
                    dg_out[a][b]  = +(alpha_psi / (double)DIM) * delta;
                    dep_out[a][b] = (gamma / dt) * dg_out[a][b];
                }
            }
        }
    }

    *dgamma_out    = gamma;
    *cep_denom_out = denom;

    return 1;
}



static void calculateStressImplicitLocal()
{
#ifdef _OPENACC
#pragma acc parallel loop present( \
    Stress[0:ParticleCount][0:DIM][0:DIM], \
    StressOld[0:ParticleCount][0:DIM][0:DIM], \
    Strain[0:ParticleCount][0:DIM][0:DIM], \
    Spin[0:ParticleCount][0:DIM][0:DIM], \
    PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM], \
    PlasticDf[0:ParticleCount][0:DIM][0:DIM], \
    PlasticDg[0:ParticleCount][0:DIM][0:DIM], \
    PlasticCepDen[0:ParticleCount], \
    YieldActive[0:ParticleCount], \
    MuLames[0:ParticleCount], \
    LambdaLames[0:ParticleCount], \
    InternalFrictionAngle[0:TYPE_COUNT], \
    DilatancyFrictionAngle[0:TYPE_COUNT], \
    Cohesion[0:TYPE_COUNT], \
    Damage[0:ParticleCount], \
    Property[0:ParticleCount])
#else
#pragma omp parallel for
#endif
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        double eps_dot[DIM][DIM]   = {{0.0}};
        double omega[DIM][DIM]     = {{0.0}};
        double sigma_old[DIM][DIM] = {{0.0}};
        double sigma_tr[DIM][DIM]  = {{0.0}};
        double sigma_new[DIM][DIM] = {{0.0}};
        double sigma_dmg[DIM][DIM] = {{0.0}};

        double df_loc[DIM][DIM]  = {{0.0}};
        double dg_loc[DIM][DIM]  = {{0.0}};
        double dep_loc[DIM][DIM] = {{0.0}};

        #pragma acc loop seq    
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {
                eps_dot[a][b]   = Strain[iP][a][b];
                omega[a][b]     = Spin[iP][a][b];
                sigma_old[a][b] = StressOld[iP][a][b];

                PlasticStrainRate[iP][a][b] = 0.0;
                PlasticDf[iP][a][b]         = 0.0;
                PlasticDg[iP][a][b]         = 0.0;
            }
        }

        YieldActive[iP]   = 0;
        PlasticCepDen[iP] = 0.0;

        double tr_eps = 0.0;
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            tr_eps += eps_dot[a][a];
        }

        const double mu     = MuLames[iP];
        const double lambda = LambdaLames[iP];
        const double K      = lambda + 2.0 * mu / 3.0;
        const double dt     = Elastic_Dt;
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {

                const double delta = (a == b) ? 1.0 : 0.0;

                double jaumann = 0.0;

                #pragma acc loop seq
                for (int k = 0; k < DIM; ++k) {
                    jaumann += omega[a][k] * sigma_old[k][b]
                             - sigma_old[a][k] * omega[k][b];
                }

                sigma_tr[a][b] =
                    sigma_old[a][b]
                  + dt * (
                        lambda * tr_eps * delta
                      + 2.0 * mu * eps_dot[a][b]
                      + jaumann
                    );
            }
        }


        #pragma acc loop seq    
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = a + 1; b < DIM; ++b) {
                const double s = 0.5 * (sigma_tr[a][b] + sigma_tr[b][a]);
                sigma_tr[a][b] = s;
                sigma_tr[b][a] = s;
            }
        }

        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {
                sigma_new[a][b] = sigma_tr[a][b];
            }
        }

        /*
          ============================================================
          3. Drucker-Prager return mapping
          ============================================================
        */
        const int mat = Property[iP];

        const double phi = InternalFrictionAngle[mat] * M_PI / 180.0;
        const double psi = DilatancyFrictionAngle[mat] * M_PI / 180.0;
        const double c   = Cohesion[mat];

        const double tan_phi = tan(phi);
        const double tan_psi = tan(psi);

        const double alpha_phi =
            (3.0 * tan_phi) / sqrt(9.0 + 12.0 * tan_phi * tan_phi);

        const double alpha_psi =
            (3.0 * tan_psi) / sqrt(9.0 + 12.0 * tan_psi * tan_psi);

        const double k_dp =
            (3.0 * c) / sqrt(9.0 + 12.0 * tan_phi * tan_phi);

        double dgamma    = 0.0;
        double cep_denom = 0.0;

        const int yielded = localReturnMappingDPNewton(
            sigma_tr,
            mu, K,
            alpha_phi, alpha_psi, k_dp,
            dt,
            sigma_new,
            df_loc,
            dg_loc,
            dep_loc,
            &dgamma,
            &cep_denom
        );

        YieldActive[iP]   = yielded;
        PlasticCepDen[iP] = cep_denom;



        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {
                sigma_dmg[a][b] = sigma_new[a][b];
            }
        }

        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {

                Stress[iP][a][b] = sigma_dmg[a][b];

                PlasticDf[iP][a][b] = df_loc[a][b];
                PlasticDg[iP][a][b] = dg_loc[a][b];
                PlasticStrainRate[iP][a][b] = dep_loc[a][b];
            }
        }
    }
}


   static int    StructNonzeroCountA = 0;
   static double *StructCsrValA = NULL;
   static int    *StructCsrIndA = NULL;
   static int    *StructCsrPtrA = NULL;
   static double *StructVectorB = NULL;
   static double *StructDiagA   = NULL;
   static int    StructSystemSizeN = 0;
   #pragma acc declare create(StructNonzeroCountA,StructCsrValA,StructCsrIndA,StructCsrPtrA,StructVectorB,StructDiagA,StructSystemSizeN)

   
   /* ------------------------- helpers ------------------------- */
   
   
   #pragma acc routine seq
   static inline int structureLocalIndex(const int iP)
   {
	   return iP - StructureParticleBegin;
   }
   
   #pragma acc routine seq
   static inline void getEffectiveLamesImplicit(const int iP,
												double *mu_eff,
												double *lambda_eff)
   {
	   *mu_eff     = MuLames[iP];
	   *lambda_eff = LambdaLames[iP];
   }
   
   static void freeStructureImplicitSystem()
   {
#ifdef _OPENACC
	   const int N = StructSystemSizeN;
	   if (StructCsrValA && StructNonzeroCountA > 0 &&
		   acc_is_present(StructCsrValA, (size_t)StructNonzeroCountA * sizeof(double))) {
		   #pragma acc exit data delete(StructCsrValA[0:StructNonzeroCountA])
	   }
	   if (StructCsrIndA && StructNonzeroCountA > 0 &&
		   acc_is_present(StructCsrIndA, (size_t)StructNonzeroCountA * sizeof(int))) {
		   #pragma acc exit data delete(StructCsrIndA[0:StructNonzeroCountA])
	   }
	   if (StructCsrPtrA && N >= 0 &&
		   acc_is_present(StructCsrPtrA, (size_t)(N + 1) * sizeof(int))) {
		   #pragma acc exit data delete(StructCsrPtrA[0:N+1])
	   }
	   if (StructVectorB && N > 0 &&
		   acc_is_present(StructVectorB, (size_t)N * sizeof(double))) {
		   #pragma acc exit data delete(StructVectorB[0:N])
	   }
	   if (StructDiagA && N > 0 &&
		   acc_is_present(StructDiagA, (size_t)N * sizeof(double))) {
		   #pragma acc exit data delete(StructDiagA[0:N])
	   }
#endif
	   if (StructCsrValA) { free(StructCsrValA); StructCsrValA = NULL; }
	   if (StructCsrIndA) { free(StructCsrIndA); StructCsrIndA = NULL; }
	   if (StructCsrPtrA) { free(StructCsrPtrA); StructCsrPtrA = NULL; }
	   if (StructVectorB) { free(StructVectorB); StructVectorB = NULL; }
	   if (StructDiagA)   { free(StructDiagA);   StructDiagA   = NULL; }
	   StructNonzeroCountA = 0;
	   StructSystemSizeN = 0;
   }
   
   static void gatherStructureVelocityToVector(double *x)
   {
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
#ifdef _OPENACC
	   const int N = DIM * nStruct;
	   if (acc_is_present(x, (size_t)N * sizeof(double))) {
		   #pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM],x[0:N])
		   for (int iS = 0; iS < nStruct; ++iS) {
			   const int iP = StructureParticleBegin + iS;
			   #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   x[DIM * iS + a] = Velocity[iP][a];
			   }
		   }
		   return;
	   }
	   #pragma acc update self(Velocity[0:ParticleCount][0:DIM])
#endif
	   #pragma omp parallel for
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   for (int a = 0; a < DIM; ++a) {
			   x[DIM * iS + a] = Velocity[iP][a];
		   }
	   }
   }
   
   static void scatterStructureVelocityFromVector(const double *x)
   {
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
#ifdef _OPENACC
	   const int N = DIM * nStruct;
	   if (acc_is_present((void *)x, (size_t)N * sizeof(double))) {
		   #pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM],x[0:N])
		   for (int iS = 0; iS < nStruct; ++iS) {
			   const int iP = StructureParticleBegin + iS;
			   #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   Velocity[iP][a] = x[DIM * iS + a];
			   }
		   }
		   return;
	   }
#endif
	   #pragma omp parallel for
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   for (int a = 0; a < DIM; ++a) {
			   Velocity[iP][a] = x[DIM * iS + a];
		   }
	   }
#ifdef _OPENACC
	   #pragma acc update device(Velocity[0:ParticleCount][0:DIM])
#endif
   }
   
   static double vectorDotImplicit(const int n, const double *x, const double *y)
   {
	   double sum = 0.0;
#ifdef _OPENACC
	   if (acc_is_present((void *)x, (size_t)n * sizeof(double)) &&
		   acc_is_present((void *)y, (size_t)n * sizeof(double))) {
		   #pragma acc parallel loop present(x[0:n],y[0:n]) reduction(+:sum)
		   for (int i = 0; i < n; ++i) sum += x[i] * y[i];
		   return sum;
	   }
#endif
	   #pragma omp parallel for reduction(+:sum)
	   for (int i = 0; i < n; ++i) sum += x[i] * y[i];
	   return sum;
   }
   
   static double vectorNormImplicit(const int n, const double *x)
   {
	   return sqrt(fmax(vectorDotImplicit(n, x, x), 0.0));
   }
   
   static void csrMatVecImplicit(const int n,
								 const int nnz,
								 const double *csrVal,
								 const int *csrPtr,
								 const int *csrInd,
								 const double *x,
								 double *y)
   {
	   (void)nnz;
       #pragma acc parallel loop present(csrVal[0:nnz],x[0:n],y[0:n],csrPtr[0:n+1],csrInd[0:nnz])
	   #pragma omp parallel for
	   for (int i = 0; i < n; ++i) {
		   double sum = 0.0;
           #pragma acc loop seq
		   for (int k = csrPtr[i]; k < csrPtr[i + 1]; ++k) {
			   sum += csrVal[k] * x[csrInd[k]];
		   }
		   y[i] = sum;
	   }
   }
   
   static void applyDiagonalPreconditionerImplicit(const int n,
												   const double *diag,
												   const double *r,
												   double *z)
   { 
	#pragma omp parallel for
    #pragma acc parallel loop present(diag[0:n],r[0:n],z[0:n])
	   for (int i = 0; i < n; ++i) {
		   double d = diag[i];
		   if (fabs(d) < 1.0e-30) d = (d >= 0.0) ? 1.0e-30 : -1.0e-30;
		   z[i] = r[i] / d;
	   }
   }

   static void calculateStressForce()
{
#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
                                  Stress[0:ParticleCount][0:DIM][0:DIM], \
                                  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
                                  NeighborCount[0:ParticleCount], \
                                  DomainWidth[0:DIM], \
                                  Force[0:ParticleCount][0:DIM], \
                                  Property[0:ParticleCount])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        double force[DIM] = {0.0, 0.0, 0.0};

        #pragma acc loop seq
        for (int iN = 0; iN < NeighborCount[iP]; ++iN) {

            const int jP = Neighbor[iP][iN];
            if (iP == jP) continue;
            if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;
            double xij[DIM];
            #pragma acc loop seq
            for (int a = 0; a < DIM; ++a) {
                xij[a] = Mod(Position[jP][a] - Position[iP][a] + 0.5 * DomainWidth[a],
                             DomainWidth[a]) - 0.5 * DomainWidth[a];
            }

            const double rij2 = xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2];
            if (rij2 > RadiusP * RadiusP || rij2 <= 1.0e-20) continue;

            const double rij  = sqrt(rij2);
            const double dwij = dwpdr(rij, RadiusP);

            double eij[DIM];
            #pragma acc loop seq
            for (int a = 0; a < DIM; ++a) eij[a] = xij[a] / rij;
#pragma acc loop seq
				for (int a = 0; a < DIM; ++a) {
                    #pragma acc loop seq
					for (int b = 0; b < DIM; ++b) {
						const double sij = Stress[iP][a][b] + Stress[jP][a][b];
						force[a] += 2.0 * sij * eij[b] * dwij * ParticleVolume;
					}
				}
			}
            #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            Force[iP][a] = force[a];
        }
    }
}
   
   /* ------------------ consistent real stress force ------------------ */
   static void calculateStressForceFromGivenTensor(
    const double Sigma[][DIM][DIM],
    double ForceOut[][DIM])
{
	#pragma acc parallel loop present(ForceOut[0:ParticleCount][0:DIM])
	#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            ForceOut[iP][a] = 0.0;
        }
    }
    #pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
        Sigma[0:ParticleCount][0:DIM][0:DIM], \
        Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
        NeighborCount[0:ParticleCount], \
        DomainWidth[0:DIM], \
        ForceOut[0:ParticleCount][0:DIM], \
        Property[0:ParticleCount])
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {

        double force[DIM] = {0.0, 0.0, 0.0};
        #pragma acc loop seq
        for (int iN = 0; iN < NeighborCount[iP]; ++iN) {

            const int jP = Neighbor[iP][iN];
            if (iP == jP) continue;
            if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;
            double xij[DIM];
            #pragma acc loop seq
            for (int a = 0; a < DIM; ++a) {
                xij[a] = Mod(Position[jP][a] - Position[iP][a] + 0.5 * DomainWidth[a],
                             DomainWidth[a]) - 0.5 * DomainWidth[a];
            }

            const double rij2 = xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2];
            if (rij2 > RadiusP * RadiusP || rij2 <= 1.0e-20) continue;

            const double rij = sqrt(rij2);
            const double dwij  = dwpdr(rij, RadiusP);

            double eij[DIM];
            #pragma acc loop seq
            for (int a = 0; a < DIM; ++a) eij[a] = xij[a] / rij;

        #pragma acc loop seq
				for (int a = 0; a < DIM; ++a) {
                    #pragma acc loop seq
					for (int b = 0; b < DIM; ++b) {
						const double sij = Sigma[iP][a][b] + Sigma[jP][a][b];
						force[a] += 2.0 * sij * eij[b] * dwij * ParticleVolume;
					}
				}
			}
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            ForceOut[iP][a] = force[a];
        }
    }
}


   static void assembleStructureNewtonSystem()
   {
	   freeStructureImplicitSystem();
   
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
	   const int N       = DIM * nStruct;
   
	   if (N <= 0) return;
	   StructSystemSizeN = N;
   
	   StructCsrPtrA = (int *)malloc((N + 1) * sizeof(int));
       #pragma acc enter data create(StructCsrPtrA[0:N+1])
	   StructCsrPtrA[0] = 0;
       #pragma acc update device(StructCsrPtrA[0:1])
   
	   /* ---------- CSR pattern count ---------- */
	   #pragma omp parallel for
       #pragma acc parallel loop present(Property[0:ParticleCount], StructCsrPtrA[0:N+1],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount],DomainWidth[0:DIM])
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   int nStructNei = 0;
           #pragma acc loop seq
		   for (int iN = 0; iN < NeighborCount[iP]; ++iN) {
			   const int jP = Neighbor[iP][iN];
			   if (jP == iP) continue;
			   if (!isStructureParticleImplicit(jP)) continue;
               if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;
			   double xij[DIM];
               #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   xij[a] = Mod(Position[jP][a] - Position[iP][a] + 0.5 * DomainWidth[a],
								DomainWidth[a]) - 0.5 * DomainWidth[a];
			   }
   
			   const double rij2 = xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2];
			   if (rij2 > RadiusP * RadiusP || rij2 <= 1.0e-20) continue;
   
			   ++nStructNei;
		   }
   
           #pragma acc loop seq
		   for (int a = 0; a < DIM; ++a) {
			   const int row = DIM * iS + a;
			   StructCsrPtrA[row + 1] = DIM * (1 + nStructNei);
		   }
	   }
   
       #pragma acc serial present(StructCsrPtrA[0:N+1])
       {
		   for (int i = 0; i < N; ++i) {
			   StructCsrPtrA[i + 1] += StructCsrPtrA[i];
		   }
       }
       #pragma acc update self(StructCsrPtrA[N:1])
   
	   StructNonzeroCountA = StructCsrPtrA[N];

	   StructCsrValA = (double *)malloc(StructNonzeroCountA * sizeof(double));
	   StructCsrIndA = (int    *)malloc(StructNonzeroCountA * sizeof(int));
	   StructVectorB = (double *)malloc(N * sizeof(double));
	   StructDiagA   = (double *)malloc(N * sizeof(double));
       #pragma acc enter data create(StructCsrValA[0:StructNonzeroCountA],StructCsrIndA[0:StructNonzeroCountA],StructVectorB[0:N],StructDiagA[0:N])
       

   #pragma acc parallel loop present(StructCsrValA[0:StructNonzeroCountA])
	   for (int i = 0; i < StructNonzeroCountA; ++i) StructCsrValA[i] = 0.0;
   #pragma acc parallel loop present(StructVectorB[0:N],StructDiagA[0:N])
	   for (int i = 0; i < N; ++i) {
		   StructVectorB[i] = 0.0;
		   StructDiagA[i]   = 0.0;
	   }
   
	   /* ---------- CSR index pattern ---------- */
   #pragma acc parallel loop present(Property[0:ParticleCount],StructCsrPtrA[0:N+1],StructCsrIndA[0:StructNonzeroCountA],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount],DomainWidth[0:DIM])
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   int structSlot = 0;
           #pragma acc loop seq
		   for (int a = 0; a < DIM; ++a) {
			   const int row  = DIM * iS + a;
			   const int base = StructCsrPtrA[row];
               #pragma acc loop seq
			   for (int b = 0; b < DIM; ++b) {
				   StructCsrIndA[base + b] = DIM * iS + b;
			   }
		   }
           #pragma acc loop seq
		   for (int iN = 0; iN < NeighborCount[iP]; ++iN) {
			   const int jP = Neighbor[iP][iN];
			   if (jP == iP) continue;
			   if (!isStructureParticleImplicit(jP)) continue;
   
			   double xij[DIM];
               #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   xij[a] = Mod(Position[jP][a] - Position[iP][a] + 0.5 * DomainWidth[a],
								DomainWidth[a]) - 0.5 * DomainWidth[a];
			   }
   
			   const double rij2 = xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2];
			   if (rij2 > RadiusP * RadiusP || rij2 <= 1.0e-20) continue;
   
			   const int jS = structureLocalIndex(jP);
               #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   const int row  = DIM * iS + a;
				   const int base = StructCsrPtrA[row] + DIM * (1 + structSlot);
                   #pragma acc loop seq
				   for (int b = 0; b < DIM; ++b) {
					   StructCsrIndA[base + b] = DIM * jS + b;
				   }
			   }
   
			   ++structSlot;
		   }
	   }
   
	   /* ---------- RHS = - current residual ---------- */
   #pragma acc parallel loop present(Property[0:ParticleCount],StructCsrValA[0:StructNonzeroCountA],StructVectorB[0:N],Mass[0:ParticleCount],ResidualV[0:ParticleCount][0:DIM],StructCsrPtrA[0:N+1])
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
           #pragma acc loop seq
		   for (int a = 0; a < DIM; ++a) {
			   const int row  = DIM * iS + a;
			   const int base = StructCsrPtrA[row];
   
			   /* dR/dv self mass part */
			   StructCsrValA[base + a] += Mass[iP];
   
			   /* Newton correction RHS */
			   StructVectorB[row] = -ResidualV[iP][a];
		   }
	   }
   
/* ---------- tangent contribution ---------- */
#pragma acc parallel loop present(Property[0:ParticleCount],StructCsrValA[0:StructNonzeroCountA],StructCsrPtrA[0:N+1],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount],DomainWidth[0:DIM])
#pragma omp parallel for
for (int iS = 0; iS < nStruct; ++iS) {
    const int iP = StructureParticleBegin + iS;
    int structSlot = 0;

    #pragma acc loop seq
    for (int iN = 0; iN < NeighborCount[iP]; ++iN) {

        const int jP = Neighbor[iP][iN];
        if (jP == iP) continue;

        const int jIsStruct = isStructureParticleImplicit(jP);
        const int jIsWall   = isDirichletLikeParticleImplicit(jP);
            if (Property[jP] >= FLUID_BEGIN && Property[jP] < FLUID_END) continue;
        if (!jIsStruct && !jIsWall) continue;

        double xij[DIM];

        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            xij[a] =
                Mod(Position[jP][a] - Position[iP][a] + 0.5 * DomainWidth[a],
                    DomainWidth[a]) - 0.5 * DomainWidth[a];
        }

        const double rij2 =
            xij[0] * xij[0] +
            xij[1] * xij[1] +
            xij[2] * xij[2];

        if (rij2 > RadiusP * RadiusP || rij2 <= 1.0e-20) continue;

        const double rij = sqrt(rij2);
        const double dw  = dwpdr(rij, RadiusP);

        double eij[DIM];
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            eij[a] = xij[a] / rij;
        }

        double Kel[DIM][DIM];
        buildPairKelConsistent(iP, jP, eij, dw, Kel);

        /*
          Always add self contribution.
          For wall neighbor, this is the only matrix contribution.
        */
        #pragma acc loop seq
        for (int a = 0; a < DIM; ++a) {
            const int row      = DIM * iS + a;
            const int selfBase = StructCsrPtrA[row];

            #pragma acc loop seq
            for (int b = 0; b < DIM; ++b) {
                StructCsrValA[selfBase + b] += Kel[a][b];
            }
        }

        /*
          Structure neighbor:
              A_ij -= Kel

          Wall neighbor:
              no off-diagonal column.
              wall velocity is prescribed and appears in residual.
        */
        if (jIsStruct) {
            #pragma acc loop seq
            for (int a = 0; a < DIM; ++a) {
                const int row     = DIM * iS + a;
                const int offBase = StructCsrPtrA[row] + DIM * (1 + structSlot);

                #pragma acc loop seq
                for (int b = 0; b < DIM; ++b) {
                    StructCsrValA[offBase + b] -= Kel[a][b];
                }
            }

            ++structSlot;
        }
    }
}
   
	   /* ---------- diagonal preconditioner ---------- */
   #pragma acc parallel loop present(StructCsrValA[0:StructNonzeroCountA],StructCsrPtrA[0:N+1],StructDiagA[0:N])
	   #pragma omp parallel for
	   for (int iS = 0; iS < nStruct; ++iS) {
           #pragma acc loop seq
		   for (int a = 0; a < DIM; ++a) {
			   const int row  = DIM * iS + a;
			   const int base = StructCsrPtrA[row];
			   StructDiagA[row] = StructCsrValA[base + a];
   
			   if (fabs(StructDiagA[row]) < 1.0e-30) {
				   StructDiagA[row] = (StructDiagA[row] >= 0.0) ? 1.0e-30 : -1.0e-30;
			   }
		   }
	   }
   }
   
   
   static void solveStructureCorrectionBiCGStab()
   {
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
	   const int N = DIM * nStruct;
   
	   if (N <= 0) return;
   
	   double *x    = (double *)malloc(N * sizeof(double));
	   double *r    = (double *)malloc(N * sizeof(double));
	   double *rhat = (double *)malloc(N * sizeof(double));
	   double *p    = (double *)malloc(N * sizeof(double));
	   double *phat = (double *)malloc(N * sizeof(double));
	   double *v    = (double *)malloc(N * sizeof(double));
	   double *s    = (double *)malloc(N * sizeof(double));
	   double *shat = (double *)malloc(N * sizeof(double));
	   double *t    = (double *)malloc(N * sizeof(double));
       #pragma acc enter data create(x[0:N],r[0:N],rhat[0:N],p[0:N],phat[0:N],v[0:N],s[0:N],shat[0:N],t[0:N])
   
   #pragma acc parallel loop present(x[0:N])
	   for (int i = 0; i < N; ++i) {
		   x[i] = 0.0;  /* solve for correction dv */
	   }
   
	   csrMatVecImplicit(N, StructNonzeroCountA,
						 StructCsrValA, StructCsrPtrA, StructCsrIndA,
						 x, v);
   #pragma acc parallel loop present(StructVectorB[0:N],r[0:N],rhat[0:N],p[0:N],v[0:N])
	#pragma omp parallel for
	   for (int i = 0; i < N; ++i) {
		   r[i]    = StructVectorB[i] - v[i];
		   rhat[i] = r[i];
		   p[i]    = 0.0;
		   v[i]    = 0.0;
	   }
   
	   const int maxIter = 100;
	   const double tol_rel = 1.0e-7;
	   const double eps_break = 1.0e-30;
   
	   const double bnorm = fmax(vectorNormImplicit(N, StructVectorB), 1.0e-30);
	   double rnorm = vectorNormImplicit(N, r);
   
	   if (rnorm / bnorm < tol_rel) {
        #pragma acc parallel loop present(DeltaVelocity[0:ParticleCount][0:DIM])
		   for (int iS = 0; iS < nStruct; ++iS) {
			   const int iP = StructureParticleBegin + iS;
               #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   DeltaVelocity[iP][a] = 0.0;
			   }
		   }
#ifdef _OPENACC
		   #pragma acc exit data delete(x[0:N],r[0:N],rhat[0:N],p[0:N],phat[0:N],v[0:N],s[0:N],shat[0:N],t[0:N])
#endif
		   free(x); free(r); free(rhat); free(p); free(phat); free(v);
		   free(s); free(shat); free(t);
		   return;
	   }
   
	   double rho_old = 1.0;
	   double alpha   = 1.0;
	   double omega   = 1.0;

	   for (int iter = 0; iter < maxIter; ++iter) {
   
		   const double rho_new = vectorDotImplicit(N, rhat, r);
		   if (fabs(rho_new) < eps_break) break;
   
		   const double beta = (rho_new / rho_old) * (alpha / omega);
   
           #pragma acc parallel loop
		   for (int i = 0; i < N; ++i) {
			   p[i] = r[i] + beta * (p[i] - omega * v[i]);
		   }
   
		   applyDiagonalPreconditionerImplicit(N, StructDiagA, p, phat);
		   csrMatVecImplicit(N, StructNonzeroCountA,
							 StructCsrValA, StructCsrPtrA, StructCsrIndA,
							 phat, v);
   
		   const double rhat_v = vectorDotImplicit(N, rhat, v);
		   if (fabs(rhat_v) < eps_break) break;
   
		   alpha = rho_new / rhat_v;
   
		   #pragma acc parallel loop
		   for (int i = 0; i < N; ++i) {
			   s[i] = r[i] - alpha * v[i];
		   }
   
		   const double snorm = vectorNormImplicit(N, s);
		   if (snorm / bnorm < tol_rel) {
            #pragma acc parallel loop
			   for (int i = 0; i < N; ++i) x[i] += alpha * phat[i];
			   break;
		   }
   
		   applyDiagonalPreconditionerImplicit(N, StructDiagA, s, shat);
		   csrMatVecImplicit(N, StructNonzeroCountA,
							 StructCsrValA, StructCsrPtrA, StructCsrIndA,
							 shat, t);
   
		   const double t_s = vectorDotImplicit(N, t, s);
		   const double t_t = vectorDotImplicit(N, t, t);
		   if (fabs(t_t) < eps_break) break;
   
		   omega = t_s / t_t;
		   if (fabs(omega) < eps_break) break;
   
		   #pragma acc parallel loop
		   for (int i = 0; i < N; ++i) {
			   x[i] += alpha * phat[i] + omega * shat[i];
		   }
   
		   #pragma acc parallel loop
		   for (int i = 0; i < N; ++i) {
			   r[i] = s[i] - omega * t[i];
		   }
   
		   rnorm = vectorNormImplicit(N, r);
		   if (rnorm / bnorm < tol_rel) break;
   
		   rho_old = rho_new;
	   }
   
	   #pragma acc parallel loop present(DeltaVelocity[0:ParticleCount][0:DIM],x[0:N])
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
           #pragma acc loop seq
		   for (int a = 0; a < DIM; ++a) {
			   DeltaVelocity[iP][a] = x[DIM * iS + a];
		   }
	   }
   
#ifdef _OPENACC
	   #pragma acc exit data delete(x[0:N],r[0:N],rhat[0:N],p[0:N],phat[0:N],v[0:N],s[0:N],shat[0:N],t[0:N])
#endif
	   free(x); free(r); free(rhat); free(p); free(phat); free(v);
	   free(s); free(shat); free(t);
   }
   

   /* ------------------ outer nonlinear Picard loop ------------------ */
   
   static double computeStructureVelocityChangeNorm(const double *vOldVec)
   {
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
	   const int N = DIM * nStruct;
	   double norm2 = 0.0;

#ifdef _OPENACC
	   if (acc_is_present((void *)vOldVec, (size_t)N * sizeof(double))) {
		   #pragma acc parallel loop reduction(+:norm2) present(Velocity[0:ParticleCount][0:DIM],vOldVec[0:N])
		   for (int iS = 0; iS < nStruct; ++iS) {
			   const int iP = StructureParticleBegin + iS;
			   #pragma acc loop seq
			   for (int a = 0; a < DIM; ++a) {
				   const double dv = Velocity[iP][a] - vOldVec[DIM * iS + a];
				   norm2 += dv * dv;
			   }
		   }
		   return sqrt(norm2);
	   }
	   #pragma acc update self(Velocity[0:ParticleCount][0:DIM])
#endif
	   #pragma omp parallel for reduction(+:norm2)
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   for (int a = 0; a < DIM; ++a) {
			   const double dv = Velocity[iP][a] - vOldVec[DIM * iS + a];
			   norm2 += dv * dv;
		   }
	   }
	   return sqrt(norm2);
   }
   
   static void relaxStructureVelocity(const double omega_relax,
									  const double *vOldVec)
   {
	   const int nStruct = StructureParticleEnd - StructureParticleBegin;
	   #pragma omp parallel for
	   for (int iS = 0; iS < nStruct; ++iS) {
		   const int iP = StructureParticleBegin + iS;
		   for (int a = 0; a < DIM; ++a) {
			   const double vnew = Velocity[iP][a];
			   const double vold = vOldVec[DIM * iS + a];
			   Velocity[iP][a] = (1.0 - omega_relax) * vold + omega_relax * vnew;
		   }
	   }
#ifdef _OPENACC
	   #pragma acc update device(Velocity[0:ParticleCount][0:DIM])
#endif
   }

   static void applyDamageToCurrentStress(void)
   {
   #pragma acc parallel loop present( \
       Stress[0:ParticleCount][0:DIM][0:DIM], \
       Damage[0:ParticleCount], \
       Property[0:ParticleCount])
       for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
   
           double D = Damage[iP];
   
           if (D < 0.0)  D = 0.0;
           if (D > 0.99) D = 0.99;
   
           const double fac = 1.0 - D;
   
           double meanStress = 0.0;
   #pragma acc loop seq
           for (int a = 0; a < DIM; ++a) {
               meanStress += Stress[iP][a][a];
           }
           meanStress /= (double)DIM;
   
           /*
             compression positive, tension negative.
             If mean stress is compressive, do not apply tensile damage.
           */
           if (meanStress > 0.0) {
               continue;
           }
   
   #pragma acc loop seq
           for (int a = 0; a < DIM; ++a) {
   #pragma acc loop seq
               for (int b = 0; b < DIM; ++b) {
                   Stress[iP][a][b] *= fac;
               }
           }
       }
   }

   
   static double rebuildFullElastoplasticStateAndResidual()
   {
	   updatePositionFromVelocityImplicit();
       buildExternalForce();
   
	   resetStructureForce();
   
	   calculateStrainRateTensor();
	   calculateSpinTensor();
	   calculateStressImplicitLocal();
   restoreDamageHistoryFromOldState();
       calculateEquivalentStrain();
       calculateKappa_nonlocal();
       calculateDamage();
       applyDamageToCurrentStress();
	   calculateStressForce();
	   copyStructureForceToInternalBuffer();
	   buildVelocityResidual();
   
	   return computeVelocityResidualNorm();
   }

  
static void implicitElasticStepVelocityBased()
{
	const int nStruct = StructureParticleEnd - StructureParticleBegin;
	if (nStruct <= 0) return;

	const int N = DIM * nStruct;

	const int maxOuter = 12;
	const double tol_res_rel = 1.0e-6;
	const double tol_dv_rel  = 1.0e-6;

	double *vBefore = (double *)malloc((size_t)N * sizeof(double));
	if (vBefore == NULL) {
		printf("implicitElasticStepVelocityBased: malloc failed for vBefore\n");
		exit(1);
	}
#ifdef _OPENACC
	#pragma acc enter data create(vBefore[0:N])
#endif

	saveElasticOldState();
	/* old state の internal force を保存 */
calculateStressForceFromGivenTensor(StressOld, InternalForceOldBuf);
	initializeImplicitVelocityGuess();
	buildExternalForce();

	/* build initial current state */
	double resNow = rebuildFullElastoplasticStateAndResidual();
	double res0   = fmax(resNow, 1.0e-20);

	for (int outer = 0; outer < maxOuter; ++outer) {

		gatherStructureVelocityToVector(vBefore);
		saveNewtonBaseVelocity();

		assembleStructureNewtonSystem();
		solveStructureCorrectionBiCGStab();
		freeStructureImplicitSystem();

		/* backtracking / line search */
		double alpha = 1.0;
		double bestRes = 1.0e300;
		double acceptedAlpha = 0.0;
		int accepted = 0;

		for (int ls = 0; ls < 8; ++ls) {
			restoreNewtonBaseVelocity();
			applyVelocityUpdateFromBase(alpha);

			const double resTry = rebuildFullElastoplasticStateAndResidual();

			if (resTry < bestRes) {
				bestRes = resTry;
				acceptedAlpha = alpha;
			}

			if (resTry < resNow) {
				resNow = resTry;
				accepted = 1;
				break;
			}

			alpha *= 0.5;
		}

		if (!accepted) {
			restoreNewtonBaseVelocity();
			applyVelocityUpdateFromBase(acceptedAlpha);
			resNow = rebuildFullElastoplasticStateAndResidual();
		}

		const double dv   = computeStructureVelocityChangeNorm(vBefore);
		const double vref = fmax(vectorNormImplicit(N, vBefore), 1.0e-30);

		const double resRel = resNow / res0;
		const double dvRel  = dv / vref;

		if (resRel < tol_res_rel && dvRel < tol_dv_rel) {
			break;
		}
	}

#ifdef _OPENACC
	#pragma acc exit data delete(vBefore[0:N])
#endif
	free(vBefore);
}   

/* ============================================================
   Kurumatani-type fracture-energy damage model for concrete.

   Assumptions:
     - Modified von-Mises equivalent strain is used to account for
       the compression/tension strength ratio of concrete.

     - Strain tensor in Strain[][][] is strain-rate tensor.
       Therefore kappa is updated by:
          kappa += eps_eq_dot * Elastic_Dt

     - If your Strain[][][] is total strain, remove * Elastic_Dt.
   ============================================================ */


   #pragma acc routine seq
   static inline void principalValuesSym3x3(const double A[DIM][DIM],double *lam_min,double *lam_mid,double *lam_max)
   {
	const double a00 = A[0][0];
	const double a11 = A[1][1];
	const double a22 = A[2][2];
	const double a01 = 0.5 * (A[0][1] + A[1][0]);
	const double a02 = 0.5 * (A[0][2] + A[2][0]);
	const double a12 = 0.5 * (A[1][2] + A[2][1]);
	
	const double p1 = a01*a01 + a02*a02 + a12*a12;
	
	double l1, l2, l3;
	
	if (p1 < 1.0e-30) {
		l1 = a00;
		l2 = a11;
		l3 = a22;
	} else {
		const double trace = a00 + a11 + a22;
		const double q = trace / 3.0;
	
		const double b00 = a00 - q;
		const double b11 = a11 - q;
		const double b22 = a22 - q;
	
		const double p2 = b00*b00 + b11*b11 + b22*b22 + 2.0*p1;
		const double p = sqrt(fmax(p2 / 6.0, 1.0e-30));
	
		const double c00 = b00 / p;
		const double c11 = b11 / p;
		const double c22 = b22 / p;
		const double c01 = a01 / p;
		const double c02 = a02 / p;
		const double c12 = a12 / p;
	
		const double detC =
			  c00 * (c11*c22 - c12*c12)
			- c01 * (c01*c22 - c12*c02)
			+ c02 * (c01*c12 - c11*c02);
	
		double r = 0.5 * detC;
		if (r < -1.0) r = -1.0;
		if (r >  1.0) r =  1.0;
	
		const double phi = acos(r) / 3.0;
	
		l1 = q + 2.0 * p * cos(phi);
		l3 = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
		l2 = 3.0 * q - l1 - l3;
	}
	
	double mn = l1;
	double md = l2;
	double mx = l3;
	
	if (mn > md) {
		const double t = mn;
		mn = md;
		md = t;
	}
	if (md > mx) {
		const double t = md;
		md = mx;
		mx = t;
	}
	if (mn > md) {
		const double t = mn;
		mn = md;
		md = t;
	}
	
	*lam_min = mn;
	*lam_mid = md;
	*lam_max = mx;
	
	}
	
	

    #pragma acc routine seq
    static inline double tensilePrincipalEquivalentStrainRate3D_checked(
        const double eps[DIM][DIM],
        const double sigma[DIM][DIM],
        const double ft)
    {
        /*
          Stress sign convention:
              compression : positive
              tension     : negative
    
          Damage should grow only when the particle is really under tensile stress.
          Therefore, use the most tensile principal stress sigma_min.
        */
    
        double s_min, s_mid, s_max;
        principalValuesSym3x3(sigma, &s_min, &s_mid, &s_max);
        (void)s_mid;
        (void)s_max;
    
        /*
          No tensile stress state.
          sigma_min >= 0 means all principal stresses are compressive or zero.
        */
        if (s_min >= 0.0) {
            return 0.0;
        }
    
        /*
          Optional but recommended:
          do not start damage for tiny tensile numerical noise.
          Use e.g. 5% of tensile strength.
        */
        const double tensile_tol = 0.05 * fmax(ft, 1.0e-30);
    
        if (-s_min < tensile_tol) {
            return 0.0;
        }
    
        /*
          For strain-rate tensor:
          usually tensile strain-rate is the largest positive principal value.
          If your Strain sign is opposite, change this part.
        */
        double e_min, e_mid, e_max;
        principalValuesSym3x3(eps, &e_min, &e_mid, &e_max);
        (void)e_min;
        (void)e_mid;
    
        const double eps_tensile_dot = e_max;
    
        return (eps_tensile_dot > 0.0) ? eps_tensile_dot : 0.0;
    }
   



static void calculateEquivalentStrain(void)
{
#pragma acc parallel loop present( \
    EquivalentStrain[0:ParticleCount], \
    MaxEquivalentStrain[0:ParticleCount], \
    Strain[0:ParticleCount][0:DIM][0:DIM], \
    Stress[0:ParticleCount][0:DIM][0:DIM], \
    Property[0:ParticleCount], \
    YoungModulus[0:TYPE_COUNT], \
    Cohesion[0:TYPE_COUNT])
#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {

        EquivalentStrain[iP] = 0.0;

        if (iP < StructureParticleBegin || iP >= StructureParticleEnd) {
            continue;
        }

        const int mat = Property[iP];

        const double ft = fmax(Cohesion[mat], 1.0e-30);

        double eps_eq_dot =
            tensilePrincipalEquivalentStrainRate3D_checked(
                Strain[iP],
                Stress[iP],
                ft
            );

        if (eps_eq_dot < 1.0e-14) {
            EquivalentStrain[iP] = 0.0;
            continue;
        }

        EquivalentStrain[iP] = eps_eq_dot;

        double kappa_old = MaxEquivalentStrain[iP];

        if (kappa_old < 0.0) {
            kappa_old = 0.0;
        }

        const double dkappa = eps_eq_dot * Elastic_Dt;

        double kappa_new = kappa_old + dkappa;

        if (kappa_new < kappa_old) {
            kappa_new = kappa_old;
        }

        MaxEquivalentStrain[iP] = kappa_new;
    }
}

   
   static void calculateKappa_nonlocal(void)
   {
	   /*
		 Keep this almost local at first.
   
		 For sharp cracks:
			 eta = 0.0
   
		 For more stable but slightly smeared cracks:
			 eta = 0.05 - 0.15
   
		 Since the current problem is "damage is too diffusive",
		 eta = 0.0 is recommended.
	   */
   
	   const double h = 0.5*ParticleSpacing;
	   const double inv_h2 = 1.0 / (h * h + 1.0e-30);
   
	   const double eta = 0.0;
   
   #pragma acc parallel loop present( \
	   EquivalentStrain[0:ParticleCount], \
	   MaxEquivalentStrain[0:ParticleCount], \
	   KappaTmp[0:ParticleCount], \
	   Property[0:ParticleCount], \
	   Position[0:ParticleCount][0:DIM], \
	   NeighborCount[0:ParticleCount], \
	   Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	   for (int iP = 0; iP < ParticleCount; ++iP) {
   
		   KappaTmp[iP] = MaxEquivalentStrain[iP];
   
		   if (iP < StructureParticleBegin || iP >= StructureParticleEnd) {
			   continue;
		   }
   
		   /*
			 Do not spread kappa from inactive particles.
			 This is important for sharp crack localization.
		   */
		   if (EquivalentStrain[iP] <= 1.0e-14) {
			   continue;
		   }
   
		   const double k_i = MaxEquivalentStrain[iP];
   
		   double sumw = 1.0;
		   double sumk = k_i;
   
   #pragma acc loop seq
		   for (int n = 0; n < NeighborCount[iP]; ++n) {
   
			   const int jP = Neighbor[iP][n];
   
			   if (jP < 0 || jP == iP) {
				   continue;
			   }
   
			   if (jP < StructureParticleBegin || jP >= StructureParticleEnd) {
				   continue;
			   }
   
			   if (Property[jP] != Property[iP]) {
				   continue;
			   }
   
			   /*
				 Only use active or already damaged neighbors.
				 This avoids wide artificial damage diffusion.
			   */
			   if (MaxEquivalentStrain[jP] <= 0.0 &&
				   EquivalentStrain[jP] <= 1.0e-14) {
				   continue;
			   }
   
			   double r2 = 0.0;
   
   #pragma acc loop seq
			   for (int d = 0; d < DIM; ++d) {
				   const double dx = Position[jP][d] - Position[iP][d];
				   r2 += dx * dx;
			   }
   
			   const double w = exp(-r2 * inv_h2);
   
			   sumw += w;
			   sumk += w * MaxEquivalentStrain[jP];
		   }
   
		   const double k_avg = sumk / fmax(sumw, 1.0e-30);
   
		   double k_bar = (1.0 - eta) * k_i + eta * k_avg;
   
		   /*
			 Irreversibility.
		   */
		   if (k_bar < k_i) {
			   k_bar = k_i;
		   }
   
		   KappaTmp[iP] = k_bar;
	   }
   
   #pragma acc parallel loop present( \
	   MaxEquivalentStrain[0:ParticleCount], \
	   KappaTmp[0:ParticleCount])
	   for (int iP = 0; iP < ParticleCount; ++iP) {
		   if (KappaTmp[iP] > MaxEquivalentStrain[iP]) {
			   MaxEquivalentStrain[iP] = KappaTmp[iP];
		   }
	   }
   }
   
   static void calculateDamage(void)
{
    /*
      Fracture-energy-based tensile damage evolution.

      D(kappa) =
          1 - kappa0 / kappa
              * exp[- beta * (kappa - kappa0)]

      kappa0 = ft / E
      beta   = ft * h / Gf
    */

    const double D_MAX = 0.999;

    /*
      Characteristic length.
    */
    const double h = 0.5* ParticleSpacing;

    /*
      Concrete fracture energy.
      100 N/m is a reasonable first value.
    */
    const double Gf_default = 100.0;  /* [N/m] */

#pragma acc parallel loop present( \
    Damage[0:ParticleCount], \
    EquivalentStrain[0:ParticleCount], \
    MaxEquivalentStrain[0:ParticleCount], \
    Property[0:ParticleCount], \
    YoungModulus[0:TYPE_COUNT], \
    Cohesion[0:TYPE_COUNT], \
    FractureEnergy[0:TYPE_COUNT])
#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {

        if (iP < StructureParticleBegin || iP >= StructureParticleEnd) {
            Damage[iP] = 0.0;
            continue;
        }

        const int mat = Property[iP];

        const double E = fmax(YoungModulus[mat], 1.0e-30);
        double D_old = Damage[iP];

        if (D_old < 0.0) {
            D_old = 0.0;
        }

        if (D_old > D_MAX) {
            D_old = D_MAX;
        }

        double ft = Cohesion[mat];

        if (ft <= 0.0) {
            Damage[iP] = D_old;
            continue;
        }

        ft = fmax(ft, 1.0e-30);

        double Gf = FractureEnergy[mat];
        if (Gf <= 0.0) {
            Gf = Gf_default;
        }
        Gf = fmax(Gf, 1.0e-30);

        const double kappa0 = fmax(ft / E, 1.0e-12);
        const double kappa  = MaxEquivalentStrain[iP];

        /*
          Damage may remain, but new damage is allowed only while the
          current equivalent strain-rate is tensile-active.
        */
        if (EquivalentStrain[iP] <= 1.0e-14) {
            Damage[iP] = D_old;
            continue;
        }

        /*
          No new damage before onset.
          Keep previous damage for irreversibility.
        */
        if (kappa <= kappa0) {
            Damage[iP] = D_old;
            continue;
        }

        /*
          beta = E*kappa0*h/Gf = ft*h/Gf
        */
        double beta = ft * h / Gf;

        if (beta < 0.0) {
            beta = 0.0;
        }

        double x = beta * (kappa - kappa0);

        double expo;

        if (x > 700.0) {
            expo = 0.0;
        } else {
            expo = exp(-x);
        }

        double D_cand = 1.0 - (kappa0 / kappa) * expo;

        if (D_cand < 0.0) {
            D_cand = 0.0;
        }

        if (D_cand > D_MAX) {
            D_cand = D_MAX;
        }

        /*
          Irreversibility.
        */
        double D_new = D_old;

        if (D_cand > D_old) {
            D_new = D_cand;
        }

        if (D_new > D_MAX) {
            D_new = D_MAX;
        }

        Damage[iP] = D_new;
    }
}
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//=================Elastoplastic calculation============================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//


static void calculateGravity(){
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
    
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
    Force[iP][0] += Mass[iP]*Gravity[0];
    Force[iP][1] += Mass[iP]*Gravity[1];
    Force[iP][2] += Mass[iP]*Gravity[2];
    }
}

static void calculateAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
}


static void calculateWall()
{
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
	//if(Time<0.40){
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
		const int iProp = Property[iP];
		double r[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
		const double (&R)[DIM][DIM] = WallRotation[iProp];
		const double (&w)[DIM] = WallOmega[iProp];
		r[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
		r[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
		r[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];
		Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
		Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
		Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
		Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
		Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
		Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
		
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
		WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
		WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
	}
	//}
}



static void calculateVirialStressAtParticle()
{
	//const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels present (VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			// pressureP
			if(RadiusP*RadiusP - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// pressureA
			if(RadiusA*RadiusA - rij2 > 0){
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dwij = ratio * dwadr(rij,RadiusA);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}

	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// viscosity term
			if(RadiusV*RadiusV - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// diffuse interface force (1st term)
			if(RadiusG*RadiusG - rij2 > 0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
			
			// diffuse interface force (2nd term)
			if(RadiusG*RadiusG - rij2 > 0.0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dw = ratio * dwgdr(rij,RadiusG);
				const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (                     -GravityCenter[iP][iD])*xij[iD];
				}
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}	
	

	#pragma acc kernels present(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else 
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}



static void calculatePeriodicBoundary( void )
{
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}

