#include <matrix.h>
#include <crtdefs.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "PGrpFind_Header.hpp"

using namespace PGrpFind;

PGrpFind::OutputVariables::OutputVariables():
	PNGSpikeNeuronsVect(),
	PNGSpikeTimingsVect(),
	PNGSpikeSynapsesVect(),
	PNGIndexVectorVect(),
	PNGMaxLengthVect(),
	PNGCombinationKeyVect(){}

SimulationVars::SimulationVars(mxArray *MatlabInputStruct): SimulationVars(){
	initialize(MatlabInputStruct);
}

void SimulationVars::initialize(mxArray *MatlabInputStruct){

	this->N = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "a"));
	this->M = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "NStart"));

	this->onemsbyTstep = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "onemsbyTstep")));
	this->DelayRange = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "DelayRange")));

	float*      genFloatPtr[4];     // Generic float Pointers used around the place to access data
	int*        genIntPtr[2];       // Generic int Pointers used around the place to access data
	uint32_t*	genUIntPtr[1];		// Generic unsigned int Pointers used around the place to access data (generator bits)
	short *     genCharPtr;         // Generic short Pointer used around the place to access data (delays specifically)
	mxArray *   genmxArrayPtr;      // Generic mxArray Pointer used around the place to access data

	// Initializing neuron specification structure array Neurons
	genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "a")));	// a[N]
	genFloatPtr[1] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "b")));	// b[N]
	genFloatPtr[2] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "c")));	// c[N]
	genFloatPtr[3] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "d")));	// d[N]

	this->Neurons = MexVector<Neuron>(N);

	for (int i = 0; i < N; ++i){
		this->Neurons[i].a = genFloatPtr[0][i];
		this->Neurons[i].b = genFloatPtr[1][i];
		this->Neurons[i].c = genFloatPtr[2][i];
		this->Neurons[i].d = genFloatPtr[3][i];
	}

	// Initializing network (Synapse) specification structure array Network
	genIntPtr[0] = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NStart")));	  // NStart[M]
	genIntPtr[1] = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NEnd")));      // NEnd[M]
	genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Weight")));    // Weight[M]
	genFloatPtr[1] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Delay")));     // Delay[M]

	this->Network = MexVector<Synapse>(M);

	for (int i = 0; i < M; ++i){
		this->Network[i].NStart = genIntPtr[0][i];
		this->Network[i].NEnd = genIntPtr[1][i];
		this->Network[i].Weight = genFloatPtr[0][i];
		this->Network[i].DelayinTsteps = (int(genFloatPtr[1][i] * this->onemsbyTstep + 0.5) > 0) ?
			int(genFloatPtr[1][i] * this->onemsbyTstep + 0.5) : 1;
	}
	
	this->FlippedExcNetwork.resize(0);
	this->StrippedNetworkMapping.resize(0);

	int SpikeQueueSize = this->DelayRange*this->onemsbyTstep;
	this->SpikeQueue = MexVector<MexVector<int> >(SpikeQueueSize, MexVector<int>());
	this->MaxLengthofSpike  = MexVector<MexVector<int> >(SpikeQueueSize, MexVector<int>());

	this->SpikeState = MexVector<int>(N, 0);

	this->HasSpikedNow.resize(0);
	this->HasSpikedPreviously.resize(0);

	this->PreSynNeuronSectionBeg.resize(0);
	this->PreSynNeuronSectionEnd.resize(0);
	this->PostSynNeuronSectionBeg.resize(0);
	this->PostSynNeuronSectionEnd.resize(0);

	this->CurrentContribSyn.resize(N, MexVector<int>());
	this->PrevContribSyn.resize(N, MexVector<int>());

	this->CurrentNonZeroIinNeurons.clear();
	this->PreviousNonZeroIinNeurons.clear();

	this->CurrentPreSynNeurons.clear();
	//this->MaxLenUptilNow = MexVector<uint32_T>(N, uint32_T(0));
	this->MaxLenInCurrIter = MexVector<uint32_T>(N, uint32_T(0));

	this->PolychronousGroupMap.clear();
	this->ProhibitedCombinationSet.clear();
}

void SimulationVars::initNExcMExc(){
	// Requirements
	//   Requires the Pre-Syn Section Arrays to be properly initialized
	
	// This assumes that all excitatory neurons are located  sequentially starting from 
	// Neuron 1 and all Exc Neurons are of type RS whereas all Inhibitory neurons are of type FS
	// The condition Neurons[NExc].a < 0.08 is basically meant to find all neurons whose 'a' is 0.02
	// and not 0.1. This is just in case of floating point roundoff errors
	// Also assuming the prescence of at least one Exc Neuron.
	NExc = 0;
	for (NExc = 0; NExc < N && Neurons[NExc].a < 0.08; ++NExc);
	MExc = PreSynNeuronSectionEnd[NExc - 1];
}

void SimulationVars::initFlippedExcNetwork(){
	// Requirements.
	//   NExc and MExc must be set correctly.
	//   Network must be final (Stripped) Network.

	this->FlippedExcNetwork.resize(MExc);
	FlippedExcNetwork.copyArray(0, this->Network.begin(), MExc);
	sort(this->FlippedExcNetwork.begin(), this->FlippedExcNetwork.end(), SimulationVars::SynapseComp_NEnd_NStart);
}

void SimulationVars::initPreSynSectionArrays(){
	// Requirements
	//   N, M should be correctly set
	//   Network should be the final (Stripped) Network

	PreSynNeuronSectionBeg = MexVector<int>(N, -1);
	PreSynNeuronSectionEnd = MexVector<int>(N, -1);

	PreSynNeuronSectionBeg[Network[0].NStart - 1] = 0;
	PreSynNeuronSectionEnd[Network.last().NStart - 1] = M;

	PreSynNeuronSectionBeg[0] = 0;
	PreSynNeuronSectionEnd[0] = 0;

	for (int i = 1; i < M; ++i){
		if (Network[i].NStart != Network[i - 1].NStart){
			PreSynNeuronSectionBeg[Network[i].NStart - 1] = i;
			PreSynNeuronSectionEnd[Network[i - 1].NStart - 1] = i;
		}
	}
	for (int i = 1; i < N; ++i){
		if (PreSynNeuronSectionBeg[i] == -1){
			PreSynNeuronSectionBeg[i] = PreSynNeuronSectionEnd[i]
				= PreSynNeuronSectionEnd[i - 1];
		}
	}
}

void SimulationVars::initPostSynSectionArrays(){
	// Requirements
	//   NExc, MExc must be correctly set
	//   FlippedExcNetwork must be correctly initialized
	
	PostSynNeuronSectionBeg = MexVector<int>(N, -1);
	PostSynNeuronSectionEnd = MexVector<int>(N, -1);
	
	PostSynNeuronSectionBeg[FlippedExcNetwork[0].NEnd - 1] = 0;
	PostSynNeuronSectionEnd[FlippedExcNetwork.last().NEnd - 1] = MExc;

	PostSynNeuronSectionBeg[0] = 0;
	PostSynNeuronSectionEnd[0] = 0;

	for (int i = 1; i < MExc; ++i){
		if (FlippedExcNetwork[i].NEnd != FlippedExcNetwork[i - 1].NEnd){
			PostSynNeuronSectionBeg[FlippedExcNetwork[i].NEnd - 1] = i;
			PostSynNeuronSectionEnd[FlippedExcNetwork[i - 1].NEnd - 1] = i;
		}
	}
	for (int i = 1; i < N; ++i){
		if (PostSynNeuronSectionBeg[i] == -1){
			PostSynNeuronSectionBeg[i] = PostSynNeuronSectionEnd[i]
				= PostSynNeuronSectionEnd[i - 1];
		}
	}
}

void PGrpFind::StoreSpikes(SimulationVars &SimVars, bool isInitialCase){
	
	// Aliasing SimVars Variables
	#pragma region Aliasing SimVars Variables
	auto &HasSpikedNow = SimVars.HasSpikedNow;
	auto &HasSpikedPreviously = SimVars.HasSpikedPreviously;
	auto &SpikeQueue = SimVars.SpikeQueue;
	auto &Network = SimVars.Network;
	auto &SpikeState = SimVars.SpikeState;

	auto &PreSynNeuronSectionBeg = SimVars.PreSynNeuronSectionBeg;
	auto &PreSynNeuronSectionEnd = SimVars.PreSynNeuronSectionEnd;

	auto &MaxLengthofSpike = SimVars.MaxLengthofSpike;
	auto &MaxLenInCurrIter = SimVars.MaxLenInCurrIter;

	auto &onemsbyTstep = SimVars.onemsbyTstep;
	auto &DelayRange = SimVars.DelayRange;
	auto &CurrentQIndex = SimVars.CurrentQIndex;
	#pragma endregion

	int QueueSize = onemsbyTstep*DelayRange;
	int nCurrSpikedNeu = HasSpikedNow.size();

	for (int i = 0; i < nCurrSpikedNeu; ++i){
		int CurrNeuron = HasSpikedNow[i];
		// Perform SpikeState Setting
		SpikeState[CurrNeuron - 1] = 1;
		HasSpikedPreviously.push_back(CurrNeuron);

		int k = PreSynNeuronSectionBeg[CurrNeuron - 1];
		int kend = PreSynNeuronSectionEnd[CurrNeuron - 1];
		uint32_t CurrMaxLen = MaxLenInCurrIter[CurrNeuron - 1];
		if (isInitialCase){
			for (; k < kend; ++k){
				Synapse CurrSyn = Network[k];
				int CurrInsertIndex = (CurrSyn.DelayinTsteps + CurrentQIndex) % QueueSize;
				MaxLengthofSpike[CurrInsertIndex].push_back(CurrMaxLen);
				SpikeQueue[CurrInsertIndex].push_back(~k);
			}
		}
		else{
			for (; k < kend; ++k){
				Synapse CurrSyn = Network[k];
				int CurrInsertIndex = (CurrSyn.DelayinTsteps + CurrentQIndex) % QueueSize;
				MaxLengthofSpike[CurrInsertIndex].push_back(CurrMaxLen);
				SpikeQueue[CurrInsertIndex].push_back(k);
			}
		}
	}
	HasSpikedNow.clear();
}

void PGrpFind::ProcessArrivingSpikes(SimulationVars &SimVars){

	// Aliasing SimVars Variables
	#pragma region Aliasing SimVars Variables
	auto &Network = SimVars.Network;
	auto &SpikeQueue = SimVars.SpikeQueue;
	auto &MaxLengthofSpike = SimVars.MaxLengthofSpike;

	auto &CurrentContribSyn = SimVars.CurrentContribSyn;
	auto &PrevContribSyn    = SimVars.PrevContribSyn;
	auto &CurrentNZIinNeurons = SimVars.CurrentNonZeroIinNeurons;
	auto &MaxLenInCurrIter = SimVars.MaxLenInCurrIter;

	auto &CurrentQIndex = SimVars.CurrentQIndex;
	auto &ZeroCurrentThresh = SimVars.ZeroCurrentThresh;
	auto &InitialWeight = SimVars.InitialWeight;
	#pragma endregion

	auto SpikeIterBeg = SpikeQueue[CurrentQIndex].begin();
	auto SpikeIterEnd = SpikeQueue[CurrentQIndex].end();
	auto SpikeIter = SpikeIterBeg;

	auto MaxLenIterBeg = MaxLengthofSpike[CurrentQIndex].begin();
	auto MaxLenIterEnd = MaxLengthofSpike[CurrentQIndex].end();
	auto MaxLenIter = MaxLenIterBeg;

	for (; SpikeIter < SpikeIterEnd; ++SpikeIter, ++MaxLenIter){
		int SynapseInd = *SpikeIter;
		int ActualSynapseInd = (SynapseInd < 0) ? ~SynapseInd : SynapseInd;

		Synapse CurrSynapse = Network[ActualSynapseInd];
		int CurrNEnd = CurrSynapse.NEnd;

		if (CurrentContribSyn[CurrNEnd - 1].isempty()){ // First contributing synapse to current NEnd
			CurrentNZIinNeurons.push_back(CurrNEnd);    // Pushing Neuron into the list of those cur-
			                                            // rently receiving currents
			if (PrevContribSyn[CurrNEnd - 1].isempty()){
				MaxLenInCurrIter[CurrNEnd - 1] = *MaxLenIter + 1;
			}
		}
		else{
			MaxLenInCurrIter[CurrNEnd - 1] = (*MaxLenIter >= MaxLenInCurrIter[CurrNEnd - 1]) ? 
				                              *MaxLenIter + 1 : MaxLenInCurrIter[CurrNEnd - 1];
			// This is code to Update the MaxLen for CurrNEnd given the current Spike
			// This is because the first spike has already come and this must be gre-
			// ater than it in order to replace it
			//  (*MaxLenIter >= MaxLenInCurrIter[CurrNEnd - 1]) 
			//                     |||
			// (*MaxLenIter + 1 > MaxLenInCurrIter[CurrNEnd - 1])
		}
		CurrentContribSyn[CurrNEnd - 1].push_back(ActualSynapseInd); // Pushing current synapse into list of
		                                                             // Contributing Synapses

		//// Assigning Current
		//if (SynapseInd < 0){ 
		//	// For case of Initial spike release
		//	Iin[CurrNEnd - 1] += InitialWeight;
		//}
		//else{
		//	// For all other cases
		//	Iin[CurrNEnd - 1] += CurrSynapse.Weight;
		//}
	}

	// Clearing Current Vector of SpikeQueue and MaxLengthofSpike Queues
	SpikeQueue[CurrentQIndex].clear();
	MaxLengthofSpike[CurrentQIndex].clear();
}

void PGrpFind::PublishCurrentSpikes(SimulationVars &SimVars, PolyChrNeuronGroup &PNGCurrent){

	// Aliasing SimVars Variables
	#pragma region Aliasing SimVars variables
	auto &Network                = SimVars.Network;
	auto &StrippedNetworkMapping = SimVars.StrippedNetworkMapping;

	auto &CurrentContribSyn      = SimVars.CurrentContribSyn;
	auto &PrevContribSyn         = SimVars.PrevContribSyn;
	auto &CurrentNZIinNeurons    = SimVars.CurrentNonZeroIinNeurons;
	auto &SpikeState             = SimVars.SpikeState;
	auto &HasSpikedNow           = SimVars.HasSpikedNow;
	auto &CurrentPreSynNeurons   = SimVars.CurrentPreSynNeurons;

	auto &MaxLenInCurrIter       = SimVars.MaxLenInCurrIter;
	auto &PolychronousGroupMap   = SimVars.PolychronousGroupMap;
	auto &ProhibCombiSet         = SimVars.ProhibitedCombinationSet;

	auto &NExc = SimVars.NExc;
	auto &time = SimVars.time;
	#pragma endregion

	auto NeuronListIterBeg = CurrentNZIinNeurons.begin();
	auto NeuronListIterEnd = CurrentNZIinNeurons.end();

	// This loop Publishes all Spikes into PNGCurrent
	// and into HasSpikedNow
	for (auto NeuronIter = NeuronListIterBeg; NeuronIter != NeuronListIterEnd; ++NeuronIter){
		int CurrNeuron = *NeuronIter;
		if (CurrentContribSyn[CurrNeuron - 1].size() + 
		    PrevContribSyn[CurrNeuron - 1].size() >= SimVars.RequiredConcurrency
		    && SpikeState[CurrNeuron - 1] == 0){
			// IN CASE A NEURON SPIKES (and hasnt spiked in previous time instant)

			// Update the MaxLength of the current PNG according to the new Maxlengths
			// accorded to the spiking neurons
			int CurrNeuronMaxLen = MaxLenInCurrIter[CurrNeuron - 1];
			PNGCurrent.MaxLength = (PNGCurrent.MaxLength > CurrNeuronMaxLen) ? PNGCurrent.MaxLength : CurrNeuronMaxLen;

			// Signal the issue of a spike if the Neuron is Excitatory
			if (CurrNeuron <= NExc)
				HasSpikedNow.push_back(CurrNeuron);

			// Code to Store into the PNG
			PNGCurrent.SpikeNeurons.push_back(CurrNeuron);
			PNGCurrent.SpikeTimings.push_back(time);
			PNGCurrent.IndexVector.push_back(PNGCurrent.SpikeSynapses.size());
			for (auto Elem : CurrentContribSyn[CurrNeuron - 1]){
				PNGCurrent.SpikeSynapses.push_back(StrippedNetworkMapping[Elem]);
			}
			for (auto Elem : PrevContribSyn[CurrNeuron - 1]){
				PNGCurrent.SpikeSynapses.push_back(StrippedNetworkMapping[Elem]);
			}
			
		}
	}

}

void PGrpFind::PerformOutput(SimulationVars &SimVars, OutputVariables &OutVars){

	// Aliasing Simvars variables
	#pragma region Aliasing SimVars Varibles
	auto &PolychronousGroupMap = SimVars.PolychronousGroupMap;
	#pragma endregion

	auto MapIterBeg = PolychronousGroupMap.begin();
	auto MapIterEnd = PolychronousGroupMap.end();

	for (auto Iter = MapIterBeg; Iter != MapIterEnd; ++Iter){
		
		// Outputting SpikeNeurons -> PNGSpikeNeuronsVect
		OutVars.PNGSpikeNeuronsVect.push_back(MexVector<int>());
		OutVars.PNGSpikeNeuronsVect.last().swap(Iter->second.SpikeNeurons);

		// Outputting SpikeTimings -> PNGSpikeTimingsVect
		OutVars.PNGSpikeTimingsVect.push_back(MexVector<int>());
		OutVars.PNGSpikeTimingsVect.last().swap(Iter->second.SpikeTimings);

		// Outputting SpikeSynapses -> PNGSpikeSynapsesVect
		OutVars.PNGSpikeSynapsesVect.push_back(MexVector<int>());
		OutVars.PNGSpikeSynapsesVect.last().swap(Iter->second.SpikeSynapses);
		
		// Outputting IndexVector -> PNGIndexVectorVect
		OutVars.PNGIndexVectorVect.push_back(MexVector<int>());
		OutVars.PNGIndexVectorVect.last().swap(Iter->second.IndexVector);

		// Outputting MaxLen -> PNGMaxLengthVect
		OutVars.PNGMaxLengthVect.push_back(Iter->second.MaxLength);

		// Outputting CombinationKey -> PNGCombinationKeyVect
		OutVars.PNGCombinationKeyVect.push_back(Iter->first);
	}
}

void PGrpFind::ResetIntermediateVars(SimulationVars &SimVars){
	// Resets the following lists and arrays produced by ProcessArrivingSpikes
	// in the mentioned manner
	// PreviousNonZeroIinNeurons - This is parsed in  order to clear PrevContribSyn.  Then cleared.
	// PrevContribSyn            - This is cleared by parsing through PrevContribSyn.
	// CurrentNonZeroIinNeurons  - This list is swapped with the cleared PreviousNonZeroIinNeurons.
	// 
	// CurrentContribSyn         - This vectvect is cleared for all neurons in HasSpiked. Then, it
	//                           - is swapped with PreviousContribSyn.
	// SpikeState                - Set to zero by parsing through HasSpikedPreviously
	// HasSpikedPreviously       - This is cleared after being used to reset SpikeState.

	// Aliasing Simvars Variables
	#pragma region Aliasing Simvars Variables
	auto &CurrentNonZeroIinNeurons  = SimVars.CurrentNonZeroIinNeurons;
	auto &PreviousNonZeroIinNeurons = SimVars.PreviousNonZeroIinNeurons;
	auto &HasSpikedNow              = SimVars.HasSpikedNow;
	auto &HasSpikedPreviously       = SimVars.HasSpikedPreviously;
	auto &CurrentContribSyn         = SimVars.CurrentContribSyn;
	auto &PrevContribSyn            = SimVars.PrevContribSyn;
	auto &SpikeState                = SimVars.SpikeState;
	#pragma endregion

	auto PrevNZINeuronIterBeg = PreviousNonZeroIinNeurons.begin();
	auto PrevNZINeuronIterEnd = PreviousNonZeroIinNeurons.end();

	for (auto PrevIter = PrevNZINeuronIterBeg; PrevIter != PrevNZINeuronIterEnd; ++PrevIter){
		int Neuron = *PrevIter;
		PrevContribSyn[Neuron - 1].clear();
	}

	PreviousNonZeroIinNeurons.clear();
	PreviousNonZeroIinNeurons.swap(CurrentNonZeroIinNeurons);

	auto HasSpikedNowIterBeg = HasSpikedNow.begin();
	auto HasSpikedNowIterEnd = HasSpikedNow.end();

	for (auto CurrentIter = HasSpikedNowIterBeg; CurrentIter != HasSpikedNowIterEnd; ++CurrentIter){
		int CurrNeuron = *CurrentIter;
		CurrentContribSyn[CurrNeuron - 1].clear();
	}
	PrevContribSyn.swap(CurrentContribSyn);

	auto HasSpikedPrevIterBeg = HasSpikedPreviously.begin();
	auto HasSpikedPrevIterEnd = HasSpikedPreviously.end();

	for (auto Iter = HasSpikedPrevIterBeg; Iter != HasSpikedPrevIterEnd; ++Iter){
		int CurrNeuron = *Iter;
		SpikeState[CurrNeuron - 1] = 0;
	}
	HasSpikedPreviously.clear();
}

void PGrpFind::GetPolychronousGroups(SimulationVars &SimVars, OutputVariables &OutVars){

	// Aliasing SimVars variables
	#pragma region Aliasing SimVars
	auto &Neurons = SimVars.Neurons;
	auto &Network = SimVars.Network;
	auto &FlippedExcNetwork = SimVars.FlippedExcNetwork;
	auto &StrippedNetworkMapping = SimVars.StrippedNetworkMapping;

	auto &N = SimVars.N;
	auto &M = SimVars.M;
	auto &NExc = SimVars.NExc;
	auto &MExc = SimVars.MExc;
	auto &onemsbyTstep = SimVars.onemsbyTstep;
	auto &DelayRange = SimVars.DelayRange;
	auto &isCurrentPNGRecurrent = SimVars.isCurrentPNGRecurrent;

	auto &time = SimVars.time;
	auto &CurrentQIndex = SimVars.CurrentQIndex;

	auto &PreSynNeuronSectionBeg  = SimVars.PreSynNeuronSectionBeg;
	auto &PreSynNeuronSectionEnd  = SimVars.PreSynNeuronSectionEnd;
	auto &PostSynNeuronSectionBeg = SimVars.PostSynNeuronSectionBeg;
	auto &PostSynNeuronSectionEnd = SimVars.PostSynNeuronSectionEnd;
	
	auto &HasSpikedNow = SimVars.HasSpikedNow;
	auto &HasSpikedPreviously = SimVars.HasSpikedPreviously;
	auto &SpikeState = SimVars.SpikeState;
	auto &SpikeQueue = SimVars.SpikeQueue;

	auto &PrevContribSyn            = SimVars.PrevContribSyn;
	auto &PreviousNonZeroIinNeurons = SimVars.PreviousNonZeroIinNeurons;

	auto &MaxLengthofSpike = SimVars.MaxLengthofSpike;
	auto &MaxLenInCurrIter = SimVars.MaxLenInCurrIter;

	auto &PolychronousGroupMap = SimVars.PolychronousGroupMap;
	auto &ProhibitedCombinationSet = SimVars.ProhibitedCombinationSet;
	#pragma endregion

	int QueueSize = onemsbyTstep*DelayRange;
	// Discard all edges with weights less than 8
	#pragma region Stripping Network
	MexVector<Synapse> NetworkTemp;

	for (int i = 0; i < M; ++i){
		if (Network[i].Weight > SimVars.MinWeightSyn){
			NetworkTemp.push_back(Network[i]);
			StrippedNetworkMapping.push_back(i);
		}
	}
	
	Network.swap(NetworkTemp);
	M = Network.size();
	#pragma endregion
	
	// Initializing Pre-Syn Section Begin/End Arrays
	SimVars.initPreSynSectionArrays();

	// Calculating NExc and MExc.
	SimVars.initNExcMExc();
	
	// Initializing FlippedExcNetwork
	SimVars.initFlippedExcNetwork();

	// Initializing Post-Syn Section Begin/End Arrays
	SimVars.initPostSynSectionArrays();
	
	// Temporary Variables Used in the loop below
	PolyChrNeuronGroup CurrentGrp;     // This stores the current Polychronous Group
	MexVector<Synapse> SynapseSet(2);  // This stores the triplet of synapses corresponding to the current triplet.
	MexVector<int>     DelaySet(2, 0); // This stores the triplet of delays of the synapses held by SynapseSet

	for (int NeuTarget = 1; NeuTarget <= NExc; ++NeuTarget){
		int nPreSynNeurons = PostSynNeuronSectionEnd[NeuTarget - 1] - PostSynNeuronSectionBeg[NeuTarget - 1];
		MexVector<Synapse>::iterator IncomingSynBeg = FlippedExcNetwork.begin() + PostSynNeuronSectionBeg[NeuTarget - 1];

		// Loop that iterates over all combinations of Excitatory Presynaptic Neurons of NeuTarget
		for (int NeuIndex1 = 0            ; NeuIndex1 < nPreSynNeurons - 1; ++NeuIndex1){
		for (int NeuIndex2 = NeuIndex1 + 1; NeuIndex2 < nPreSynNeurons    ; ++NeuIndex2){

			int Neuron1 = (IncomingSynBeg + NeuIndex1)->NStart;
			int Neuron2 = (IncomingSynBeg + NeuIndex2)->NStart;

			// Calculate Unique Key pertaining to this Combination
			uint64_t CombinationKey = (uint64_t)(NeuTarget-1)*NExc*NExc +
										(uint64_t)(Neuron1-1)*NExc +
					                    (uint64_t)(Neuron2-1);

			// Checking if the current neuron combination is prohibited.
			auto ProhibCombSetEnd = ProhibitedCombinationSet.end();
			auto CurrentKeyElem = ProhibitedCombinationSet.find(CombinationKey);

			// Analyse the PNG of the current Neuron combination
			#pragma region Analyze PNG of current Neuron Combination
			// Initializing Time Related States
			time = 0;
			CurrentQIndex = 0;
			int NeuronCursor = 0; // This is a cursor  used to iterate through the
				                    // different elements in SynapseSet and DelaySet
				                    // when issuing and storing the initial spikes

			// Initializing SynapseSet and DelaySet and Sorting
			SynapseSet[0] = *(IncomingSynBeg + NeuIndex1);
			SynapseSet[1] = *(IncomingSynBeg + NeuIndex2);

			sort(SynapseSet.begin(), SynapseSet.end(), SimulationVars::SynapseComp_Delays);

			DelaySet[0] = SynapseSet[0].DelayinTsteps;
			DelaySet[1] = SynapseSet[1].DelayinTsteps;
			
			// Initializing Group Related States
			CurrentGrp.reset();

			// Initializing SpikeState  using SimVars.HasSpikedPreviously we
			// make zero those elements that belong in NonZeroIinNeurons and 
			// then clear NonZeroIinNeurons
			auto HasSpikedPrevBeg = HasSpikedPreviously.begin();
			auto HasSpikedPrevEnd = HasSpikedPreviously.end();
			for (auto Iter = HasSpikedPrevBeg; Iter != HasSpikedPrevEnd; ++Iter){
				SpikeState[*Iter - 1] = 0;
			}
			HasSpikedPreviously.clear();

			// Initializing SpikeQueue and MaxLengthofSpike
			for (int i = 0; i < QueueSize; ++i){
				SpikeQueue[i].clear();
				MaxLengthofSpike[i].clear();
			}
				
			// Initializing MaxLenInCurrIter as 0 for the neurons of current triplet
			// This is done just before the  emmission of spikes as otherwise it can
			// get erased by the  arrival of another spike  into the neuron prior to 
			// its firing
			
			// Initializing PreviousNonZeroIinNeurons and PrevContribSyn
			// This are the uncleared memories of the simulation of the previous Co-
			// mbination
			auto PrevNZINeuronBeg = PreviousNonZeroIinNeurons.begin();
			auto PrevNZINeuronEnd = PreviousNonZeroIinNeurons.end();

			for (auto Iter = PrevNZINeuronBeg; Iter < PrevNZINeuronEnd; ++Iter){
				PrevContribSyn[*Iter - 1].clear();
			}
			PreviousNonZeroIinNeurons.clear();

			// Initializing Constants used for conditional evaluation
			bool isSpikeQueueEmpty = false;
			isCurrentPNGRecurrent = 0;

			// initial iteration which releases the spike of the PreSynaptic neuron
			// connected to the synapse with the highest delay
			{
				int CurrInitNeuron = SynapseSet[1].NStart; 
				// Publishing this spike into CurrentGrp as it is not done during
				// the publish spikes procedure
				CurrentGrp.SpikeNeurons.push_back(CurrInitNeuron);
				CurrentGrp.SpikeTimings.push_back(time);
				CurrentGrp.IndexVector.push_back(CurrentGrp.SpikeSynapses.size());

				HasSpikedNow.push_back(CurrInitNeuron);
				// Initializind MaxLenInCurrIter for this neuron
				MaxLenInCurrIter[CurrInitNeuron - 1] = 0;
				StoreSpikes(SimVars, false);
				time++;
				CurrentQIndex = (CurrentQIndex + 1) % QueueSize;
				NeuronCursor++;
			}
			// initializing HasSpiked, CurrentNonZeroIinNeurons, CurrentContribSyn
			// These are  expected to  be aready in a  clear state so  no need for 
			// initialization

			// This loop simulates the network upto termination (OR a detection of
			// recurrence) and determines the structure and apiking squence of the
			// PNG created by the current combination of Neurons. The initial ite-
			// ration has beed done outside the loop itself
			while (!isSpikeQueueEmpty && isCurrentPNGRecurrent != 2 && CurrentGrp.MaxLength < 15){
					
				// Calling the functions to update current, process spikes, and analyse
				// the generated  spikes for repetitions in  groups and to ward against
				// recurrent groups, and to finally store the spikes in the spike queue
				// for parsing at the time of arrival

				ProcessArrivingSpikes(SimVars);
				PublishCurrentSpikes(SimVars, CurrentGrp);
				ResetIntermediateVars(SimVars);
				StoreSpikes(SimVars, false);
				if (NeuronCursor < 2 && time == (DelaySet[1] - DelaySet[1 - NeuronCursor])){

					int CurrInitNeuron = SynapseSet[1 - NeuronCursor].NStart;
					// Publishing this spike into CurrentGrp as it is not done during
					// the publish spikes procedure
					CurrentGrp.SpikeNeurons.push_back(CurrInitNeuron);
					CurrentGrp.SpikeTimings.push_back(time);
					CurrentGrp.IndexVector.push_back(CurrentGrp.SpikeSynapses.size());

					// Initializind MaxLenInCurrIter for this neuron
					MaxLenInCurrIter[CurrInitNeuron - 1] = 0;
					HasSpikedNow.push_back(CurrInitNeuron);
					StoreSpikes(SimVars, false);
					NeuronCursor++;
				}

				// Temporal Variable Update
				time++;
				CurrentQIndex = (CurrentQIndex + 1) % QueueSize;

				// Calculating isSpikeQueueEmpty
				int j;
				for (j = 0; j < QueueSize && SpikeQueue[j].isempty(); ++j);
				isSpikeQueueEmpty = (j == QueueSize); // means SpikeQueue[j].isempty() is true for all j in 1
					                                    // to QueueSize
			}
			// Inserting the currently calculated PNG into the Map only if its 
			// length exceeds a certain minimum threshold (in this case 1)
			if (CurrentGrp.MaxLength > 1){
				CurrentGrp.IndexVector.push_back(CurrentGrp.SpikeSynapses.size());
				PolychronousGroupMap.emplace(CombinationKey, CurrentGrp);
			}
			#pragma endregion
			
		}
		}
		
	}
	
	// Performing Output Conversion from unordered_map<uint64_T, PolyChrNeuronGroup>
	// to OutputVars
	PerformOutput(SimVars, OutVars);

	// All shit done and completed.
	// At this point the ProhibitedCombinationSet and PolychronousGroupMap
	// are left as they are. They will be used by the parent function as 
	// seen fit. 
	// 
	// The state of the other variables is as
	// HasSpikedNow                 - Cleared
	// SpikeQueue and MaxLenofSpike - Possibly uncleared if last iter was recursive PNG
	// Everything Else              - Uncleared
	// The clearance for this will basically constitute a reinitialization
	// of SimVars (using initialize) in the parent function.
	
}