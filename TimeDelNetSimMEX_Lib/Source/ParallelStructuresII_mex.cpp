#include <vector>
#include <iostream>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <tbb\atomic.h>
#include <fstream>
#include <chrono>
#include <cmath>
#include "..\Headers\MexMem.hpp"
#include "..\Headers\Network.hpp"
#include "..\Headers\NeuronSim.hpp"
#include "..\Headers\FiltRandomTBB.hpp"

#include <emmintrin.h>
#include <smmintrin.h>

using namespace std;

#define VECTOR_IMPLEMENTATION

void CountingSort(int N, MexVector<Synapse> &Network, MexVector<size_t> &indirection)
{
	MexVector<int> CumulativeCountStart(N,0);
	MexVector<int> IOutsertIndex(N);
	size_t M = Network.size();
	if (indirection.size() != M) indirection.resize(M);

	for (int i = 0; i < M; ++i){
		CumulativeCountStart[Network[i].NEnd - 1]++;
	}
	IOutsertIndex[0] = 0;
	for (int i = 1; i < N; ++i){
		CumulativeCountStart[i] += CumulativeCountStart[i - 1];
		IOutsertIndex[i] = CumulativeCountStart[i - 1];
	}
	int indirIter = 0;
	for (int i = 0; i < M; ++i){
		int NeuronEnd = Network[i].NEnd;
		indirection[IOutsertIndex[NeuronEnd-1]] = i;
		IOutsertIndex[NeuronEnd - 1]++;
	}
}

void CurrentUpdate::operator () (const tbb::blocked_range<int*> &BlockedRange) const{
	const float &I0 = IntVars.I0;
	auto &Network              = IntVars.Network;
	auto &Iin                  = IntVars.Iin;
	auto &LastSpikedTimeSyn    = IntVars.LSTSyn;
	auto &LastSpikedTimeNeuron = IntVars.LSTNeuron;
	auto &WeightDeriv          = IntVars.WeightDeriv;
	auto &time                 = IntVars.Time;
	auto &ExpVect              = IntVars.ExpVect;

	auto &STDPMaxWinLen        = IntVars.STDPMaxWinLen;

	int *begin = BlockedRange.begin();
	int *end = BlockedRange.end();
	for (int * iter = begin; iter < end; ++iter){
		Synapse CurrentSynapse = Network[*iter];
		int CurrentSynapseInd = *iter;
		__m128 AddedCurrent;
		AddedCurrent.m128_u64[0] = 0; AddedCurrent.m128_u64[1] = 0;
		AddedCurrent.m128_f32[0] = CurrentSynapse.Weight;
		AddedCurrent.m128_u32[0] += (32 << 23);

		Iin[CurrentSynapse.NEnd - 1].fetch_and_add((long long)AddedCurrent.m128_f32[0]);

		LastSpikedTimeSyn[CurrentSynapseInd] = time;
		size_t SpikeTimeDiffCurr = time - LastSpikedTimeNeuron[CurrentSynapse.NEnd];
		WeightDeriv[CurrentSynapseInd] -= ((SpikeTimeDiffCurr < STDPMaxWinLen) ? 1.2*ExpVect[SpikeTimeDiffCurr] : 0);
		
	}
}

void NeuronSimulate::operator() (tbb::blocked_range<int> &Range) const{

	auto &Vnow = IntVars.V;
	auto &Unow = IntVars.U;

	auto &Neurons = IntVars.Neurons;
	auto &Network = IntVars.Network;

	auto &Iin = IntVars.Iin;
	auto &Iext = IntVars.Iext;
	auto &RandMat = IntVars.RandMat;
	auto &WeightDeriv = IntVars.WeightDeriv;

	auto &PreSynNeuronSectionBeg = IntVars.PreSynNeuronSectionBeg;
	auto &PreSynNeuronSectionEnd = IntVars.PreSynNeuronSectionEnd;
	auto &PostSynNeuronSectionBeg = IntVars.PostSynNeuronSectionBeg;
	auto &PostSynNeuronSectionEnd = IntVars.PostSynNeuronSectionEnd;

	auto &ExpVect = IntVars.ExpVect;

	auto &AuxArray = IntVars.AuxArray;

	auto &LastSpikedTimeNeuron = IntVars.LSTNeuron;
	auto &LastSpikedTimeSynapse = IntVars.LSTSyn;

	auto &StdDev = IntVars.StdDev;
	auto &onemsbyTstep = IntVars.onemsbyTstep;
	auto &time = IntVars.Time;
	auto &STDPMaxWinLen = IntVars.STDPMaxWinLen;
	auto k = (IntVars.i - 1) % 8192;

	int QueueSize = onemsbyTstep*IntVars.DelayRange;
	int RangeBeg = Range.begin();
	int RangeEnd = Range.end();
	for (int j = RangeBeg; j < RangeEnd; ++j){
		if (Vnow[j] == 30.0f){
			//Implementing Izhikevich resetting
			Vnow[j] = Neurons[j].c;
			Unow[j] += Neurons[j].d;
		}
		else{
			//Implementing Izhikevich differential equation
			float Vnew, Unew;
			Vnew = Vnow[j] + (Vnow[j] * (0.04f*Vnow[j] + 5.0f) + 140.0f - Unow[j] + (float)(Iin[j]) / (1i64 << 32) + Iext[j] + StdDev*RandMat(k,j)) / onemsbyTstep;
			Unew = Unow[j] + (Neurons[j].a*(Neurons[j].b*Vnow[j] - Unow[j])) / onemsbyTstep;
			Vnow[j] = (Vnew > -100)? Vnew: -100;
			Unow[j] = Unew;

			//Implementing Network Computation in case a Neuron has spiked in the current interval
			if (Vnow[j] >= 30.0f){
				Vnow[j] = 30.0f;
				//NSpikesGenminProc += ((PreSynNeuronSectionBeg[j] >= 0) ? PreSynNeuronSectionEnd[j] - PreSynNeuronSectionBeg[j] : 0);
				LastSpikedTimeNeuron[j] = time;
				//Space to implement any causal Learning Rule
				MexVector<size_t>::iterator kbegin = AuxArray.begin() + PostSynNeuronSectionBeg[j];
				MexVector<size_t>::iterator kend   = AuxArray.begin() + PostSynNeuronSectionEnd[j];
				if (kbegin != kend)
				for (MexVector<size_t>::iterator k = kbegin; k < kend; ++k){
					size_t CurrSynIndex = *k;
					MexVector<Synapse>::iterator CurrSynPtr = Network.begin() + CurrSynIndex;
					size_t SpikeTimeDiffCurr = time - LastSpikedTimeSynapse[CurrSynIndex];
					
					WeightDeriv[CurrSynIndex] += ((SpikeTimeDiffCurr < STDPMaxWinLen) ? ExpVect[SpikeTimeDiffCurr] : 0);
				}
			}
		}
	}
}
void CurrentAttenuate::operator() (tbb::blocked_range<int> &Range) const {

	auto &Iin = IntVars.Iin;
	auto &attenFactor = IntVars.CurrentDecayFactor;

	tbb::atomic<long long> *Begin1 = &Iin[Range.begin()];
	tbb::atomic<long long> *End1 = &Iin[Range.end()-1] + 1;\

	for (tbb::atomic<long long> *i = Begin1; i < End1; ++i){
		(*i) = (long long)(float(i->load()) * attenFactor);
	}
}
void InputArgs::IExtFunc(float time, MexVector<float> &Iext)
{
	//((int)(time / 0.1))
	int N = Iext.size();
	if (time - 0.1 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 9;
	}
	else if (time - 0.8 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 9;
	}
	else{
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 9;
	}
}

void StateVarsOutStruct::initialize(const InternalVars &IntVars) {

	int onemsbyTstep = IntVars.onemsbyTstep;
	int NoOfms = IntVars.NoOfms;
	int StorageStepSize = IntVars.StorageStepSize;
	int Tbeg = IntVars.Time;
	int nSteps = onemsbyTstep * NoOfms;
	int OutputControl = IntVars.OutputControl;
	int beta = IntVars.beta;
	int N = IntVars.N;
	int M = IntVars.M;
	int DelayRange = IntVars.DelayRange;

	int TimeDimLen;  // beta is the time offset from Tbeg which 
	// corresponds to the first valid storage location
	if (StorageStepSize){
		TimeDimLen = (nSteps - beta) / (StorageStepSize*onemsbyTstep) + 1;	//No. of times (StorageStepSize * onemsbyTstep)|time happens
	}
	else{
		TimeDimLen = nSteps;
	}
	if (OutputControl & OutOps::WEIGHT_REQ)
		if (!(IntVars.InterestingSyns.size()))
			this->WeightOut = MexMatrix<float>(TimeDimLen, M);

	if (OutputControl & OutOps::V_REQ)
		this->VOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::U_REQ)
		this->UOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::I_IN_REQ)
		this->IinOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::WEIGHT_DERIV_REQ)
		this->WeightDerivOut = MexMatrix<float>(TimeDimLen, M);

	if (OutputControl & OutOps::I_RAND_REQ)
		this->IrandOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::GEN_STATE_REQ)
		this->GenStateOut = MexMatrix<uint32_t>(TimeDimLen, 4);

	this->TimeOut = MexVector<int>(TimeDimLen);

	if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
		this->LSTNeuronOut = MexMatrix<int>(TimeDimLen, N);

	if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
		this->LSTSynOut = MexMatrix<int>(TimeDimLen, M);

	if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
		this->SpikeQueueOut = MexVector<MexVector<MexVector<int> > >(TimeDimLen,
			MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>()));

	if (OutputControl & OutOps::CURRENT_QINDS_REQ)
		this->CurrentQIndexOut = MexVector<int>(TimeDimLen);
}
void OutputVarsStruct::initialize(const InternalVars &IntVars){
	int TimeDimLen;
	int N = IntVars.N;
	int onemsbyTstep = IntVars.onemsbyTstep;
	int NoOfms = IntVars.NoOfms;
	int StorageStepSize = IntVars.StorageStepSize;
	int Tbeg = IntVars.Time;
	int nSteps = onemsbyTstep * NoOfms;
	int OutputControl = IntVars.OutputControl;
	int beta = IntVars.beta;

	if (IntVars.StorageStepSize){
		TimeDimLen = (nSteps - beta) / (StorageStepSize*onemsbyTstep) + 1;	//No. of times (StorageStepSize * onemsbyTstep)|time happens
	}
	else{
		TimeDimLen = nSteps;
	}

	if (OutputControl & OutOps::WEIGHT_REQ)
		if (IntVars.InterestingSyns.size())
			this->WeightOut = MexMatrix<float>(TimeDimLen, IntVars.InterestingSyns.size());
	if (OutputControl & OutOps::I_TOT_REQ)
		this->Itot = MexMatrix<float>(TimeDimLen, N);
	if ((OutputControl & OutOps::SPIKE_LIST_REQ) && !StorageStepSize)
		// the size used below is to store all the spikes generated but not yet arrived
		// also, spiek storage is always only done in non-sparse mode.
		this->SpikeList = MexVector<MexVector<int >>(nSteps + IntVars.DelayRange*onemsbyTstep - 1, MexVector<int>());
}
void FinalStateStruct::initialize(const InternalVars &IntVars){
	int OutputControl	= IntVars.OutputControl;
	int DelayRange		= IntVars.DelayRange;
	int onemsbyTstep	= IntVars.onemsbyTstep;
	int N				= IntVars.N;
	int M				= IntVars.M;

	if (OutputControl & OutOps::FINAL_STATE_REQ){
		this->V = MexVector<float>(N);
		this->U = MexVector<float>(N);
		this->Iin = MexVector<float>(N);
		this->WeightDeriv = MexVector<float>(M);
		this->Irand = MexVector<float>(N);
		this->GenState = MexVector<uint32_t>(4);
		this->Weight = MexVector<float>(M);
		this->LSTNeuron = MexVector<int>(N);
		this->LSTSyn = MexVector<int>(M);
		this->SpikeQueue = MexVector<MexVector<int> >(DelayRange*onemsbyTstep, MexVector<int>());
	}
	this->CurrentQIndex = -1;
	this->Time = -1;
}
void InitialStateStruct::initialize(const InternalVars &IntVars){
	int OutputControl = IntVars.OutputControl;
	int DelayRange = IntVars.DelayRange;
	int onemsbyTstep = IntVars.onemsbyTstep;
	int N = IntVars.N;
	int M = IntVars.M;

	if (OutputControl & OutOps::INITIAL_STATE_REQ){
		this->V = MexVector<float>(N);
		this->U = MexVector<float>(N);
		this->Iin = MexVector<float>(N);
		this->WeightDeriv = MexVector<float>(M);
		this->GenState = MexVector<uint32_t>(4);
		this->Weight = MexVector<float>(M);
		this->Weight = MexVector<float>(M);
		this->LSTNeuron = MexVector<int>(N);
		this->LSTSyn = MexVector<int>(M);
		this->SpikeQueue = MexVector<MexVector<int> >(DelayRange*onemsbyTstep, MexVector<int>());
	}
	this->CurrentQIndex = -1;
	this->Time = -1;
}
void InternalVars::DoSparseOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){

	int CurrentInsertPos = (i - beta) / (onemsbyTstep * StorageStepSize);
	int iint = i % 8192;
	int QueueSize = onemsbyTstep * DelayRange;
	// Storing U,V,Iin and Time
	if (OutputControl & OutOps::V_REQ)
		StateOut.VOut[CurrentInsertPos] = V;
	if (OutputControl & OutOps::U_REQ)
		StateOut.UOut[CurrentInsertPos] = U;
	if (OutputControl & OutOps::I_IN_REQ)
		for (int j = 0; j < N; ++j)
			StateOut.IinOut(CurrentInsertPos, j) = (float)Iin[j] / (1i64 << 32);

	//Storing Weight Derivative
	if (OutputControl & OutOps::WEIGHT_DERIV_REQ){
		StateOut.WeightDerivOut[CurrentInsertPos] = WeightDeriv;
	}

	// Storing Random Current related state vars
	if (OutputControl & OutOps::I_RAND_REQ)
		StateOut.IrandOut[CurrentInsertPos] = RandMat[iint];
	if (OutputControl & OutOps::GEN_STATE_REQ){
		StateOut.GenStateOut[CurrentInsertPos] = GenMat[iint];
	}
	StateOut.TimeOut[CurrentInsertPos] = Time;

	// Storing Weights
	if (OutputControl & OutOps::WEIGHT_REQ && InterestingSyns.size()){
		size_t tempSize = InterestingSyns.size();
		for (int j = 0; j < tempSize; ++j)
			OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
	}
	else if (OutputControl & OutOps::WEIGHT_REQ){
		for (int j = 0; j < M; ++j)
			StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
	}

	// Storing Spike Queue related state informations
	if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
		for (int j = 0; j < QueueSize; ++j)
			StateOut.SpikeQueueOut[CurrentInsertPos][j] = SpikeQueue[j];
	if (OutputControl & OutOps::CURRENT_QINDS_REQ)
		StateOut.CurrentQIndexOut[CurrentInsertPos] = CurrentQIndex;

	// Storing last Spiked timings
	if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
		StateOut.LSTNeuronOut[CurrentInsertPos] = LSTNeuron;
	if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
		StateOut.LSTSynOut[CurrentInsertPos] = LSTSyn;

	// Storing Itot
	if (OutputControl & OutOps::I_TOT_REQ){
		for (int j = 0; j < N; ++j)
			OutVars.Itot(CurrentInsertPos, j) = Iext[j] + StdDev*RandMat(iint,j) + (float)(Iin[j]) / (1i64 << 32);
	}

}
void InternalVars::DoFullOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){
	if (!StorageStepSize){
		int nSteps = onemsbyTstep*NoOfms;
		int CurrentInsertPos = i - 1;
		int iint = i % 8192;
		int QueueSize = onemsbyTstep * DelayRange;
		// Storing U,V,Iout and Time
		if (OutputControl & OutOps::V_REQ)
			StateOut.VOut[CurrentInsertPos] = V;
		if (OutputControl & OutOps::U_REQ)
			StateOut.UOut[CurrentInsertPos] = U;
		if (OutputControl & OutOps::I_IN_REQ)
			for (int j = 0; j < N; ++j)
				StateOut.IinOut(CurrentInsertPos, j) = (float)Iin[j] / (1i64 << 32);

		//Storing Weight Derivative
		if (OutputControl & OutOps::WEIGHT_DERIV_REQ){
			StateOut.WeightDerivOut[CurrentInsertPos] = WeightDeriv;
		}

		// Storing Random Curent Related State vars
		if (OutputControl & OutOps::I_RAND_REQ)
			StateOut.IrandOut[CurrentInsertPos] = RandMat[iint];
		if (OutputControl & OutOps::GEN_STATE_REQ){
			StateOut.GenStateOut[CurrentInsertPos] = GenMat[iint];
		}
		StateOut.TimeOut[CurrentInsertPos] = Time;

		// Storing Weights
		if (OutputControl & OutOps::WEIGHT_REQ && InterestingSyns.size()){
			size_t tempSize = InterestingSyns.size();
			for (int j = 0; j < tempSize; ++j)
				OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
		}
		else if (OutputControl & OutOps::WEIGHT_REQ){
			for (int j = 0; j < M; ++j)
				StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
		}

		// Storing Spike Queue related state informations
		if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
			for (int j = 0; j < QueueSize; ++j)
				StateOut.SpikeQueueOut[CurrentInsertPos][j] = SpikeQueue[j];
		if (OutputControl & OutOps::CURRENT_QINDS_REQ)
			StateOut.CurrentQIndexOut[CurrentInsertPos] = CurrentQIndex;

		// Storing last Spiked timings
		if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
			StateOut.LSTNeuronOut[CurrentInsertPos] = LSTNeuron;
		if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
			StateOut.LSTSynOut[CurrentInsertPos] = LSTSyn;

		// Storing Itot
		if (OutputControl & OutOps::I_TOT_REQ){
			for (int j = 0; j < N; ++j)
				OutVars.Itot(CurrentInsertPos, j) = Iext[j] + StdDev*RandMat(iint, j) + (float)(Iin[j]) / (1i64 << 32);
		}

		// Storing Spike List
		if (OutputControl & OutOps::SPIKE_LIST_REQ){
			OutVars.SpikeList[CurrentInsertPos] = SpikeQueue[CurrentQIndex];
			if (i == nSteps){
				// Storing spikes which are generated but not gonna arrive next turn
				for (int j = 1; j < DelayRange*onemsbyTstep; ++j){
					OutVars.SpikeList[CurrentInsertPos + j] = SpikeQueue[(CurrentQIndex + j) % (DelayRange*onemsbyTstep)];
				}
			}
		}
	}
}
void InternalVars::DoSingleStateOutput(SingleStateStruct &SingleStateOut){
	int QueueSize = onemsbyTstep * DelayRange;
	for (int j = 0; j < N; ++j){
		SingleStateOut.Iin[j] = (float)Iin[j] / (1i64 << 32);
	}
	// storing Random curret related state vars
	SingleStateOut.Irand = RandMat[i % 8192];
	SingleStateOut.GenState = GenMat[i % 8192];

	SingleStateOut.V = V;
	SingleStateOut.U = U;
	for (int j = 0; j < M; ++j){
		SingleStateOut.Weight[j] = Network[j].Weight;
	}
	SingleStateOut.WeightDeriv = WeightDeriv;
	for (int j = 0; j < QueueSize; ++j){
		SingleStateOut.SpikeQueue[j] = SpikeQueue[j];
	}
	SingleStateOut.CurrentQIndex = CurrentQIndex;
	SingleStateOut.LSTNeuron = LSTNeuron;
	SingleStateOut.LSTSyn = LSTSyn;
	SingleStateOut.Time = Time;
}

void CachedSpikeStorage(InternalVars &IntVars){

	auto &N = IntVars.N;
	auto &Network = IntVars.Network;

	auto &SpikeQueue = IntVars.SpikeQueue;
	auto &LastSpikedTimeNeuron = IntVars.LSTNeuron;

	auto &preSynNeuronSectionBeg = IntVars.PreSynNeuronSectionBeg;
	auto &preSynNeuronSectionEnd = IntVars.PreSynNeuronSectionEnd;

	auto &CurrentQIndex = IntVars.CurrentQIndex;
	auto &time = IntVars.Time;

	const int nBins = IntVars.onemsbyTstep * IntVars.DelayRange;
	const int CacheBuffering = 128;	// Each time a cache of size 64 will be pulled in 
	static MexVector<__m128i> BinningBuffer(CacheBuffering*nBins/4);	//each element is 16 bytes
	static MexVector<int> BufferInsertIndex(nBins, 0);
	static MexVector<int> AddressOffset(nBins, 0);

	
	for (int j = 0; j < nBins; ++j){
		AddressOffset[j] = (reinterpret_cast<size_t>(SpikeQueue[j].end()) & 0x0F) >> 2;
	}
	for (int j = 0; j < N; ++j){
		if (LastSpikedTimeNeuron[j] == time){
			int k = preSynNeuronSectionBeg[j];
			int kend = preSynNeuronSectionEnd[j];

			if (k != kend){
				int NoofCurrNeuronSpikes = kend - k;
				MexVector<Synapse>::iterator iSyn = Network.begin() + k;
				MexVector<Synapse>::iterator iSynEnd = Network.begin() + kend;

				//TotalSpikesTemp += iSynEnd - iSyn;
				for (; iSyn < iSynEnd; ++iSyn, ++k){
					int CurrIndex = (CurrentQIndex + iSyn->DelayinTsteps) % nBins;
					int CurrAddressOffset = AddressOffset[CurrIndex];
					int BufferIndex = BufferInsertIndex[CurrIndex];
					int *BufferIndexPtr = &BufferInsertIndex[CurrIndex];
					int *CurrAddressOffsetPtr = &AddressOffset[CurrIndex];

					if (BufferIndex == CacheBuffering){
						if (CurrAddressOffset){
							for (int k = 0; k < 4-CurrAddressOffset; ++k){
								SpikeQueue[CurrIndex].push_back(*(reinterpret_cast<int*>(BinningBuffer.begin() + (CurrIndex + 1)*CacheBuffering / 4)
									                                      - (4-CurrAddressOffset) + k));
								// This is bcuz, the buffer at CurrIndex ends at
								// BinningBuffer.begin() + (CurrIndex + 1)*CacheBuffering / 4 [__m128*]
								// therefore we move 4-CurrAddressOffset (which is the no. of elems to be inserted)
								// behind and push back.
								}
							BufferIndex -= 4 - CurrAddressOffset;
							*BufferIndexPtr -= 4 - CurrAddressOffset;
							*CurrAddressOffsetPtr = 0;
						}
						else{
							SpikeQueue[CurrIndex].push_size(CacheBuffering);
							//TotalSpikesTemp += CacheBuffering;
							__m128i* kbeg = reinterpret_cast<__m128i*>(SpikeQueue[CurrIndex].end() - CacheBuffering);
							__m128i* kend = reinterpret_cast<__m128i*>(SpikeQueue[CurrIndex].end());
							__m128i* lbeg = reinterpret_cast<__m128i*>(BinningBuffer.begin() + CurrIndex * CacheBuffering / 4);

							for (__m128i* k = kbeg, *l = lbeg; k < kend; k++, l++){
								_mm_stream_si128(k, _mm_load_si128(l));
							}
							BufferIndex = 0;
							*BufferIndexPtr = 0;
						}
						
					}

					reinterpret_cast<int *>(BinningBuffer.begin())[CurrIndex*CacheBuffering + BufferIndex]
						= k;
					++*BufferIndexPtr;
				}
			}
		}
	}

	for (int i = 0; i < nBins; ++i){
		size_t CurrNElems = BufferInsertIndex[i];
		SpikeQueue[i].push_size(CurrNElems);
		int* kbeg = SpikeQueue[i].end() - CurrNElems;
		int* kend = SpikeQueue[i].end();
		int* lbeg = reinterpret_cast<int*>(BinningBuffer.begin() + i * CacheBuffering/4);
		for (int* k = kbeg, *l = lbeg; k < kend; ++k, ++l){
			*k = *l;
		}
		BufferInsertIndex[i] = 0;
	}
}

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVarsStruct &PureOutputs,
	StateVarsOutStruct &StateVarsOutput,
	FinalStateStruct &FinalStateOutput,
	InitialStateStruct &InitialStateOutput
)
{
	// Aliasing Input Arguments Into Appropriate
	// "In Function" state and input variables

	// Initialization and aliasing of All the input / State / parameter variables.
	InternalVars IntVars(InputArguments);

	// Aliasing of Data members in IntVar
	MexVector<Synapse>			&Network              = IntVars.Network;
	MexVector<Neuron>			&Neurons              = IntVars.Neurons;
	MexVector<float>			&Vnow                 = IntVars.V;
	MexVector<float>			&Unow                 = IntVars.U;
	MexVector<int>				&InterestingSyns      = IntVars.InterestingSyns;
	atomicLongVect              &Iin                  = IntVars.Iin;
	MexVector<float>            &WeightDeriv          = IntVars.WeightDeriv;
	BandLimGaussVect            &Irand                = IntVars.Irand;
	MexMatrix<float>			&RandMat              = IntVars.RandMat;
	MexMatrix<uint32_t>			&GenMat               = IntVars.GenMat;
	MexVector<float>			&Iext                 = IntVars.Iext;
	MexVector<MexVector<int> >	&SpikeQueue           = IntVars.SpikeQueue;
	MexVector<int>				&LastSpikedTimeNeuron = IntVars.LSTNeuron;
	MexVector<int>				&LastSpikedTimeSyn    = IntVars.LSTSyn;
	

	size_t &NoOfms              = IntVars.NoOfms;
	const size_t &onemsbyTstep  = IntVars.onemsbyTstep;
	size_t Tbeg                 = IntVars.Time;				//Tbeg is just an initial constant, 
	size_t &time                = IntVars.Time;				//time is the actual changing state variable
	size_t &DelayRange          = IntVars.DelayRange;
	size_t &CurrentQueueIndex   = IntVars.CurrentQIndex;
	size_t &StorageStepSize     = IntVars.StorageStepSize;
	size_t &OutputControl       = IntVars.OutputControl;
	size_t &i                   = IntVars.i;
	size_t &NExc                = IntVars.NExc;
	size_t &MExc                = IntVars.MExc;

	const float &I0			= IntVars.I0;	// Value of the current factor to be multd with weights (constant)
	// calculate value of alpha for filtering
	// alpha = 0 => no filtering
	// alpha = 1 => complete filtering
	const float &alpha		= IntVars.alpha;
	const float &StdDev		= IntVars.StdDev;
	const float &CurrentDecayFactor	= IntVars.CurrentDecayFactor;	//Current Decay Factor in the current model (possibly input in future)

	const size_t &StatusDisplayInterval = IntVars.StatusDisplayInterval;

	// other data members. probably derived from inputs or something
	// I think should be a constant. (note that it is possible that 
	// I club some of these with the inputs in future revisions like
	// CurrentDecayFactor

	size_t QueueSize = SpikeQueue.size();
	int nSteps = NoOfms*onemsbyTstep;
	size_t N = InputArguments.Neurons.size(), M = InputArguments.Network.size();			

	
	// VARIOuS ARRAYS USED apart from those in the argument list and Output List.
	// Id like to call them intermediate arrays, required for simulation but are
	// not state, input or output vectors.
	// they are typically of the form of some processed version of an input vector
	// thus they dont change with time and are prima facie not used to generate output

	MexVector<size_t> &AuxArray                = IntVars.AuxArray;

	MexVector<size_t> &PreSynNeuronSectionBeg  = IntVars.PreSynNeuronSectionBeg;	
	
	MexVector<size_t> &PreSynNeuronSectionEnd  = IntVars.PreSynNeuronSectionEnd;

	MexVector<size_t> &PostSynNeuronSectionBeg = IntVars.PostSynNeuronSectionBeg;
	
	MexVector<size_t> &PostSynNeuronSectionEnd = IntVars.PostSynNeuronSectionEnd;
	
	MexVector<float>  &ExpVect                 = IntVars.ExpVect;
	//----------------------------------------------------------------------------------------------//
	//--------------------------------- Initializing output Arrays ---------------------------------//
	//----------------------------------------------------------------------------------------------//

	

	StateVarsOutput.initialize(IntVars);
	PureOutputs.initialize(IntVars);
	FinalStateOutput.initialize(IntVars);
	InitialStateOutput.initialize(IntVars);
	
	//---------------------------- Initializing the Intermediate Arrays ----------------------------//
	CountingSort(N, Network, AuxArray);	// Perform counting sort by (NEnd, NStart)
	                                    // to get AuxArray

	// Sectioning the Network and AuxArray Arrays as according to 
	// definition of respective variables above
	PreSynNeuronSectionBeg[Network[0].NStart - 1] = 0;
	PreSynNeuronSectionEnd[Network[0].NStart - 1] = 0;
	PostSynNeuronSectionBeg[Network[AuxArray[0]].NEnd - 1] = 0;
	PostSynNeuronSectionEnd[Network[AuxArray[0]].NEnd - 1] = 0;

	PreSynNeuronSectionEnd[Network[M - 1].NStart - 1] = M;
	PostSynNeuronSectionEnd[Network[AuxArray[M - 1]].NEnd - 1] = M;

	for (int j = 1; j<M; ++j){
		if (Network[j - 1].NStart != Network[j].NStart){
			PreSynNeuronSectionBeg[Network[j].NStart - 1] = j;
			PreSynNeuronSectionEnd[Network[j - 1].NStart - 1] = j;
		}
		if (Network[AuxArray[j - 1]].NEnd != Network[AuxArray[j]].NEnd){
			PostSynNeuronSectionBeg[Network[AuxArray[j]].NEnd - 1] = j;
			PostSynNeuronSectionEnd[Network[AuxArray[j - 1]].NEnd - 1] = j;
		}
	}
	for (int j = 1; j < N; ++j){
		if (PreSynNeuronSectionBeg[j] == -1){
			PreSynNeuronSectionBeg[j] = PreSynNeuronSectionEnd[j]
				= PreSynNeuronSectionEnd[j - 1];
		}
		if (PostSynNeuronSectionBeg[j] == -1){
			PostSynNeuronSectionBeg[j] = PostSynNeuronSectionEnd[j]
				= PostSynNeuronSectionEnd[j - 1];
		}
	}

	float Val = IntVars.W0;
	for (int j = 0 ; j < IntVars.STDPMaxWinLen; ++j){
		ExpVect[j] = Val;
		Val *= IntVars.STDPDecayFactor;
	}

	NExc = 0;
	for (int j = 0; Neurons[j].a == Neurons[0].a; ++j, ++NExc);
	MExc = PreSynNeuronSectionEnd.operator[](NExc - 1);

	// Here I assume that the first neuron is always excitatory
	// and all excitatory neurons are stored contiguously starting 
	// from the first neuron, and that excitatory and inhibitory
	// neurons differe in their parameter a (which is tru in izhikevich case)

	// The Structure of iteration below is given below
	/*
		->Iin[t-1] ----------
		|                    \
		|                     \
		|                      >Itemp ---------- * CurrentDecayFactor ---- Iin[t]
		|                     /             \                                |
		|                    /               \                               |
		->SpikeQueue[t-1] ---                 > V,U[t] ----- SpikeQueue[t]   |
		|       |                            /    |                |         |
		|       |               V,U[t-1] ----     |                |         |
		|       |                  |              |                |         |
		|===<===|====<=========<===|==Loop Iter<==|=======<========|===<=====|

		The vector corresponding to the spikes processed in the current 
		iteration is cleared after the calculation of Itemp
	*/
	// Giving Initial State if Asked For
	
	if (OutputControl & OutOps::INITIAL_STATE_REQ){
		IntVars.DoSingleStateOutput(InitialStateOutput);
	}
	size_t maxSpikeno = 0;
	tbb::affinity_partitioner apCurrentUpdate;
	tbb::affinity_partitioner apNeuronSim;
	int TotalStorageStepSize = (StorageStepSize*onemsbyTstep); // used everywhere
	int epilepsyctr = 0;

	// ------------------------------------------------------------------------------ //
	// ------------------------------ Simulation Loop ------------------------------- //
	// ------------------------------------------------------------------------------ //
	// Profiling Times.
	size_t IExtGenTime = 0,
		   IRandGenTime = 0,
		   IUpdateTime = 0,
		   SpikeStoreTime = 0,
		   NeuronCalcTime = 0,
		   OutputTime = 0;
	std::chrono::system_clock::time_point
		IExtGenTimeBeg, IExtGenTimeEnd,
		IRandGenTimeBeg, IRandGenTimeEnd,
		IUpdateTimeBeg, IUpdateTimeEnd,
		SpikeStoreTimeBeg, SpikeStoreTimeEnd,
		NeuronCalcTimeBeg, NeuronCalcTimeEnd,
		OutputTimeBeg, OutputTimeEnd;

	for (i = 1; i<=nSteps; ++i){
		
		time = time + 1;

		IExtGenTimeBeg = std::chrono::system_clock::now();
		InputArgs::IExtFunc(time*0.001f / onemsbyTstep, Iext);
		IExtGenTimeEnd = std::chrono::system_clock::now();
		IExtGenTime += std::chrono::duration_cast<std::chrono::microseconds>(IExtGenTimeEnd - IExtGenTimeBeg).count();

		// This iteration applies time update equation for internal current
		// in this case, it is just an exponential attenuation
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 3000),
			CurrentAttenuate(IntVars));

		size_t QueueSubEnd = SpikeQueue[CurrentQueueIndex].size();
		maxSpikeno += QueueSubEnd;
		// Epilepsy Check
		if (QueueSubEnd > (2*M) / (5)){
			epilepsyctr++;
			if (epilepsyctr > 100){
			#ifdef MEX_LIB
				mexErrMsgTxt("Epileptic shit");
			#elif defined MEX_EXE
				printf("Epilepsy Nyuh!!\n");
			#endif
				return;
			}
		}
		IUpdateTimeBeg = std::chrono::system_clock::now();
		// This iter calculates Itemp as in above diagram
		if (SpikeQueue[CurrentQueueIndex].size() != 0)
			tbb::parallel_for(tbb::blocked_range<int*>((int*)&SpikeQueue[CurrentQueueIndex][0],
				(int*)&SpikeQueue[CurrentQueueIndex][QueueSubEnd - 1] + 1, 10000), 
				CurrentUpdate(IntVars), apCurrentUpdate);
		SpikeQueue[CurrentQueueIndex].clear();
		IUpdateTimeEnd = std::chrono::system_clock::now();
		IUpdateTime += std::chrono::duration_cast<std::chrono::microseconds>(IUpdateTimeEnd - IUpdateTimeBeg).count();

		// Calculation of V,U[t] from V,U[t-1], Iin = Itemp
		NeuronCalcTimeBeg = std::chrono::system_clock::now();
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 100), NeuronSimulate(IntVars), apNeuronSim);
		NeuronCalcTimeEnd = std::chrono::system_clock::now();
		NeuronCalcTime += std::chrono::duration_cast<std::chrono::microseconds>(NeuronCalcTimeEnd - NeuronCalcTimeBeg).count();

		// Generating RandMat
		int LoopLimit = (8192 + i <= nSteps) ? 8192 : nSteps - i + 1;
		IRandGenTimeBeg = std::chrono::system_clock::now();
		if (i % 8192 == 0){
			for (int j = 0; j < LoopLimit; ++j){
				Irand.generate();
				RandMat[j] = Irand;
				Irand.generator1().getstate().ConvertStatetoVect(GenMat[j]);
			}
		}
		IRandGenTimeEnd = std::chrono::system_clock::now();
		IRandGenTime += std::chrono::duration_cast<std::chrono::microseconds>(IRandGenTimeEnd - IRandGenTimeBeg).count();

		// This is code for storing spikes
		SpikeStoreTimeBeg = std::chrono::system_clock::now();
		CachedSpikeStorage(IntVars);
		SpikeStoreTimeEnd = std::chrono::system_clock::now();
		SpikeStoreTime += std::chrono::duration_cast<std::chrono::microseconds>(SpikeStoreTimeEnd - SpikeStoreTimeBeg).count();

		CurrentQueueIndex = (CurrentQueueIndex + 1) % QueueSize;

		if (!(time % (1000 * onemsbyTstep))){
			for (int j = 0; j < MExc; ++j){
				Network[j].Weight += WeightDeriv[j] + 0.01;
				Network[j].Weight = (Network[j].Weight > 0) ? Network[j].Weight : 0;
				Network[j].Weight = (Network[j].Weight < IntVars.MaxSynWeight) ? Network[j].Weight : IntVars.MaxSynWeight;
				WeightDeriv[j] *= 0.9;
			}
		}

		OutputTimeBeg = std::chrono::system_clock::now();
		IntVars.DoOutput(StateVarsOutput, PureOutputs);
		OutputTimeEnd = std::chrono::system_clock::now();
		OutputTime += std::chrono::duration_cast<std::chrono::microseconds>(OutputTimeEnd - OutputTimeBeg).count();

		// Status Display Section
		if (!(i % StatusDisplayInterval)){
		#ifdef MEX_LIB
			mexPrintf("Completed  %d steps with Total no. of Spikes = %d\n", i, maxSpikeno);
			mexEvalString("drawnow");
		#elif defined MEX_EXE
			printf("Completed  %d steps with Total no. of Spikes = %d\n", i, maxSpikeno);
		#endif
			maxSpikeno = 0;
		}
	}
	// This needs to be done as the state corresponds to i = nSteps not i = nSteps+1
	i = i - 1;
	if ((OutputControl & OutOps::FINAL_STATE_REQ)){
		IntVars.DoSingleStateOutput(FinalStateOutput);
	}

	cout << "CurrentExt Generation Time = " << IExtGenTime / 1000 << "millisecs" << endl;
	cout << "CurrentRand Generation Time = " << IRandGenTime / 1000 << "millisecs" << endl;
	cout << "Current Update Time = " << IUpdateTime / 1000 << "millisecs" << endl;
	cout << "Spike Storage Time = " << SpikeStoreTime / 1000 << "millisecs" << endl;
	cout << "Nuron Calculation Time = " << NeuronCalcTime / 1000 << "millisecs" << endl;
	cout << "Output Time = " << OutputTime / 1000 << "millisecs" << endl;
}

