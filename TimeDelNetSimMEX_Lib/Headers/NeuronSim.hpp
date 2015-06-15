#ifndef NEURONSIM_HPP
#define NEURONSIM_HPP
#include "Network.hpp"
#include "MexMem.hpp"
#include "FiltRandomTBB.hpp"
#include <mex.h>
#include <matrix.h>
#include <xutility>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <tbb\atomic.h>
#include <tbb\parallel_for.h>

#include <emmintrin.h>
#include <smmintrin.h>

#define DEFAULT_STORAGE_STEP 500
#define DEFAULT_STATUS_DISPLAY_STEP 400
using namespace std;

struct OutOps{
	enum {
		CURRENT_QINDS_REQ   = (1 << 1 ), 
		FINAL_STATE_REQ     = (1 << 2 ),
		INITIAL_STATE_REQ   = (1 << 3 ), 
		I_EXT_GEN_STATE_REQ = (1 << 4 ),
		I_EXT_REQ           = (1 << 5 ),
		I_IN_REQ            = (1 << 6 ), 
		I_TOT_REQ           = (1 << 7 ), 
		LASTSPIKED_NEU_REQ  = (1 << 8 ), 
		LASTSPIKED_SYN_REQ  = (1 << 9 ), 
		SPIKE_LIST_REQ      = (1 << 10), 
		SPIKE_QUEUE_REQ     = (1 << 11), 
		TIME_REQ            = (1 << 12), 
		U_REQ               = (1 << 13), 
		V_REQ               = (1 << 14),
		WEIGHT_DERIV_REQ    = (1 << 15), 
		WEIGHT_REQ          = (1 << 16), 
	};
};

typedef vector<tbb::atomic<long long>, tbb::zero_allocator<long long> > atomicLongVect;
typedef vector<tbb::atomic<int>, tbb::zero_allocator<int> > atomicIntVect;

// Incomplete declarations
struct InputArgs;
struct StateVarsOutStruct;
struct SingleStateStruct;
struct FinalStateStruct;
struct InitialStateStruct;
struct OutputVarsStruct;
struct InternalVars;

struct CurrentUpdate
{
	InternalVars &IntVars;

	CurrentUpdate(InternalVars &IntVars_) :
		IntVars(IntVars_){};
	void operator() (const tbb::blocked_range<int*> &BlockedRange) const;
};
struct NeuronSimulate{
	InternalVars &IntVars;

	NeuronSimulate(
		InternalVars &IntVars_
		) :
		IntVars(IntVars_)
	{};
	void operator() (tbb::blocked_range<int> &Range) const;
};

struct CurrentAttenuate{
	InternalVars &IntVars;

	CurrentAttenuate(
		InternalVars &IntVars_) :
		IntVars(IntVars_){}

	void operator() (tbb::blocked_range<int> &Range) const; 
};


struct InputArgs{
	static void IExtFunc(InternalVars &);
	MexVector<Synapse> Network;
	MexVector<Neuron> Neurons;
	MexVector<int> InterestingSyns;
	MexVector<float> V;
	MexVector<float> U;
	MexVector<float> Iin;
	MexVector<float> Iext;
	MexVector<float> WeightDeriv;
	
	MexVector<uint32_t> IExtGenState;
	MexVector<MexVector<int> > SpikeQueue;
	MexVector<int> LSTNeuron;
	MexVector<int> LSTSyn;
	int onemsbyTstep;
	int NoOfms;
	int DelayRange;
	int Time;
	int CurrentQIndex;
	int OutputControl;
	int StorageStepSize;
	int StatusDisplayInterval;

	InputArgs() :
		Network(),
		Neurons(),
		InterestingSyns(),
		V(),
		U(),
		Iin(),
		Iext(),
		IExtGenState(),
		WeightDeriv(),
		SpikeQueue(),
		LSTNeuron(),
		LSTSyn() {}
};

struct InternalVars{
	size_t N;
	size_t NExc;
	size_t M;
	size_t MExc;
	size_t i;		//This is the most important loop index that is definitely a state variable
				// and plays a crucial role in deciding the index into which the output must be performed
	size_t Time;	// must be initialized befor beta
	size_t beta;	// This is another parameter that plays a rucial role when storing sparsely.
				// It is the first value of i for which the sparse storage must be done.
				// goes from 1 to StorageStepSize * onemsbyTstep
	size_t onemsbyTstep;
	size_t NoOfms;
	size_t DelayRange;
	size_t CurrentQIndex;
	const float I0;
	const float CurrentDecayFactor;
	const float IExtScaleFactor;
	const float IExtDecayFactor;
	const float STDPDecayFactor;
	const int STDPMaxWinLen;
	const float MaxSynWeight;
	const float W0;
	const float alpha;
	const float StdDev;

	size_t OutputControl;
	size_t StorageStepSize;
	const size_t StatusDisplayInterval;

	// Parameters that control C=Spike Storage Buffering
	size_t CacheBuffering;

	MexVector<Synapse> &Network;
	MexVector<Neuron> &Neurons;
	MexVector<int> &InterestingSyns;
	MexVector<float> &V;
	MexVector<float> &U;
	atomicLongVect Iin;
	MexVector<float> &WeightDeriv;
	MexVector<float> &Iext; // made reference cuz Iext is now a state variable

	XorShiftPlus IExtGen;
	size_t CurrentGenNeuron;
	MexVector<MexVector<int> > &SpikeQueue;
	MexVector<int> &LSTNeuron;
	MexVector<int> &LSTSyn;

	MexVector<size_t> AuxArray;						    // Auxillary Array that is an indirection between Network
													    // and an array sorted lexicographically by (NEnd, NStart)
	MexVector<size_t> PreSynNeuronSectionBeg;	        // PreSynNeuronSectionBeg[j] Maintains the list of the 
														// index of the first synapse in Network with NStart = j+1
	MexVector<size_t> PreSynNeuronSectionEnd;	        // PostSynNeuronSectionEnd[j] Maintains the list of the 
														// indices one greater than index of the last synapse in 
														// Network with NStart = j+1

	MexVector<size_t> PostSynNeuronSectionBeg;	        // PostSynNeuronSectionBeg[j] Maintains the list of the 
														// index of the first synapse in AuxArray with NEnd = j+1
	MexVector<size_t> PostSynNeuronSectionEnd;	        // PostSynNeuronSectionEnd[j] Maintains the list of the 
														// indices one greater than index of the last synapse in 
														// AuxArray with NEnd = j+1


	MexVector<float> ExpVect;

	// These vectors are instrumental in the Cache aligned 
	// implementation of Spike Storage
	MexVector<__m128i> BinningBuffer;	//each element is 16 bytes
	MexVector<int> BufferInsertIndex;
	MexVector<int> AddressOffset;

	InternalVars(InputArgs &IArgs) :
		N                     (IArgs.Neurons.size()),
		NExc                  (0),
		M                     (IArgs.Network.size()),
		MExc                  (0),
		i                     (0),
		Time                  (IArgs.Time),
		// beta defined conditionally below
		CurrentQIndex         (IArgs.CurrentQIndex),
		OutputControl         (IArgs.OutputControl),
		StorageStepSize       (IArgs.StorageStepSize),
		StatusDisplayInterval (IArgs.StatusDisplayInterval),
		Network               (IArgs.Network),
		Neurons               (IArgs.Neurons),
		InterestingSyns       (IArgs.InterestingSyns),
		V                     (IArgs.V),
		U                     (IArgs.U),
		Iin                   (N), 
		// Iin is defined separately as an atomic vect.
		WeightDeriv           (IArgs.WeightDeriv),
		Iext                  (IArgs.Iext),
		IExtGen               (),
		CurrentGenNeuron      (0),
		SpikeQueue            (IArgs.SpikeQueue),
		LSTNeuron             (IArgs.LSTNeuron),
		LSTSyn                (IArgs.LSTSyn),

		AuxArray                (M),
		PreSynNeuronSectionBeg  (N, -1),
		PreSynNeuronSectionEnd  (N, -1),
		PostSynNeuronSectionBeg (N, -1),
		PostSynNeuronSectionEnd (N, -1),
		ExpVect                 (STDPMaxWinLen),

		BinningBuffer     (CacheBuffering*onemsbyTstep*DelayRange / 4),
		BufferInsertIndex (onemsbyTstep*DelayRange, 0),
		AddressOffset     (onemsbyTstep*DelayRange, 0),

		onemsbyTstep       (IArgs.onemsbyTstep),
		NoOfms             (IArgs.NoOfms),
		DelayRange         (IArgs.DelayRange),
		CacheBuffering     (128),
		I0                 (1.0f),
		STDPMaxWinLen      (int(onemsbyTstep*(log(0.0001) / log(pow((double)STDPDecayFactor, (double)onemsbyTstep))))),
		CurrentDecayFactor (powf(1.0f / 3.5f, 1.0f / onemsbyTstep)),
		IExtDecayFactor    (2.0f / 3),
		IExtScaleFactor    (20),
		STDPDecayFactor    (powf(0.95f, 1.0f / onemsbyTstep)),
		W0                 (0.1f),
		MaxSynWeight       (10.0),
		alpha              (0.5), 
		StdDev             (3.5)
	{

		// Setting value of beta
		if (StorageStepSize)
			beta = (onemsbyTstep * StorageStepSize) - Time % (onemsbyTstep * StorageStepSize);
		else
			beta = 0;

		// Setting Initial Conditions of V and U
		if (U.istrulyempty()){
			U.resize(N);
			for (int j = 0; j<N; ++j)
				U[j] = Neurons[j].b*(Neurons[j].b - 5.0f - sqrt((5.0f - Neurons[j].b)*(5.0f - Neurons[j].b) - 22.4f)) / 0.08f;
		}
		else if (U.size() != N){
			// GIVE ERROR MESSAGE HERE
			return;
		}

		if (V.istrulyempty()){
			V.resize(N);
			for (int j = 0; j<N; ++j){
				V[j] = (Neurons[j].b - 5.0f - sqrt((5.0f - Neurons[j].b)*(5.0f - Neurons[j].b) - 22.4f)) / 0.08f;
			}
		}
		else if (V.size() != N){
			// GIVE ERROR MESSAGE HEREx
			return;
		}

		// Setting Initial Conditions for INTERNAL CURRENT
		if (IArgs.Iin.size() == N){
			for (int j = 0; j < N; ++j){
				Iin[j] = (long long int)(IArgs.Iin[j] * (1i64 << 32));
			}
		}
		else if (IArgs.Iin.size()){
			// GIVE ERROR MESSAGE HERE
			return;
		}
		//else{
		//	Iin is already initialized to zero by tbb::zero_allocator<long long>
		//}

		// Setting Initial Conditions for External Current (Iext)
		if (Iext.istrulyempty()){
			Iext.resize(N, 0.0f);
		}
		else if (Iext.size() != N){
			// Show error
			return;
		}

		// Setting Initial Conditions for IExt RNo Generator
		if (IArgs.IExtGenState.size() == 1){ // single number seeding
			IExtGen = XorShiftPlus(IArgs.IExtGenState[0]);
		}
		else if (IArgs.IExtGenState.size() == 4){ // previous state initializing
			XorShiftPlus::StateStruct tempState;
			tempState.ConvertVecttoState(IArgs.IExtGenState);
			IExtGen = XorShiftPlus(tempState);
		}
		else if (!IArgs.IExtGenState.istrulyempty()){ // error
			// Throw Exception here
			return;
		}
		//else{
		//	// This has already been acheived by the default constructor
		//}

		// Setting Initial Conditions for Weight Derivative
		if (IArgs.WeightDeriv.istrulyempty()){
			WeightDeriv.resize(M, 0);
		}
		else if (IArgs.WeightDeriv.size() != M){
			// Return Exception
			return;
		}

		// Setting Initial Conditions of SpikeQueue
		if (SpikeQueue.istrulyempty()){
			SpikeQueue = MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>());
		}
		else if (SpikeQueue.size() != onemsbyTstep * DelayRange){
			// GIVE ERROR MESSAGE HERE
			return;
		}

		// Setting Initial Conditions for LSTs
		if (LSTNeuron.istrulyempty()){
			LSTNeuron = MexVector<int>(N, -1);
		}
		else if (LSTNeuron.size() != N){
			//GIVE ERROR MESSAGE HERE
			return;
		}
		if (LSTSyn.istrulyempty()){
			LSTSyn = MexVector<int>(M, -1);
		}
		else if (LSTSyn.size() != M){
			//GIVE ERROR MESSAGE HERE
			return;
		}
	}
	void DoOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){
		DoFullOutput(StateOut, OutVars);
		if (StorageStepSize && !(Time % (StorageStepSize*onemsbyTstep))){
			DoSparseOutput(StateOut, OutVars);
		}
	}
	void DoSingleStateOutput(SingleStateStruct &FinalStateOut);
private:
	void DoSparseOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars);
	void DoFullOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars);
};

struct OutputVarsStruct{
	MexMatrix<float> WeightOut;
	MexMatrix<float> Itot;
	MexVector<int> IExtNeuron;

	struct SpikeListStruct{
		MexVector<int> SpikeSynInds;
		MexVector<int> TimeRchdStartInds;
		SpikeListStruct() : SpikeSynInds(), TimeRchdStartInds(){}
	} SpikeList;

	OutputVarsStruct() :
		WeightOut(),
		Itot(),
		SpikeList(),
		IExtNeuron(){}

	void initialize(const InternalVars &);
};

struct StateVarsOutStruct{
	MexMatrix<float> WeightOut;
	MexMatrix<float> VOut;
	MexMatrix<float> UOut;
	MexMatrix<float> IinOut;
	MexMatrix<float> IextOut;
	MexMatrix<float> WeightDerivOut;
	MexMatrix<uint32_t> IExtGenStateOut;
	MexVector<int> TimeOut;

	MexVector<MexVector<MexVector<int> > > SpikeQueueOut;
	MexVector<int> CurrentQIndexOut;
	MexMatrix<int> LSTNeuronOut;
	MexMatrix<int> LSTSynOut;

	StateVarsOutStruct() :
		WeightOut(),
		VOut(),
		UOut(),
		IinOut(),
		IextOut(),
		WeightDerivOut(),
		TimeOut(),
		IExtGenStateOut(),
		SpikeQueueOut(),
		CurrentQIndexOut(),
		LSTNeuronOut(),
		LSTSynOut() {}

	void initialize(const InternalVars &);
};
struct SingleStateStruct{
	MexVector<float> Weight;
	MexVector<float> V;
	MexVector<float> U;
	MexVector<float> Iin;
	MexVector<float> Iext;
	MexVector<float> WeightDeriv;

	MexVector<uint32_t> IExtGenState;
	MexVector<MexVector<int > > SpikeQueue;
	MexVector<int> LSTNeuron;
	MexVector<int> LSTSyn;
	int Time;
	int CurrentQIndex;

	SingleStateStruct() :
		Weight(),
		V(),
		U(),
		Iin(),
		Iext(),
		WeightDeriv(),
		IExtGenState(),
		SpikeQueue(),
		LSTNeuron(),
		LSTSyn() {}

	virtual void initialize(const InternalVars &) {}
};
struct FinalStateStruct : public SingleStateStruct{
	void initialize(const InternalVars &);
};

struct InitialStateStruct : public SingleStateStruct{
	void initialize(const InternalVars &);
};

void CountingSort(int N, MexVector<Synapse> &Network, MexVector<int> &indirection);

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVarsStruct &PureOutputs,
	StateVarsOutStruct &StateVarsOutput,
	FinalStateStruct &FinalStateOutput,
	InitialStateStruct &InitalStateOutput);

#endif