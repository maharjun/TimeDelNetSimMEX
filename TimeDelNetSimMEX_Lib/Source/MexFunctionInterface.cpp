#include <mex.h>
#include <matrix.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include <type_traits>
#include "..\Headers\NeuronSim.hpp"
#include "..\Headers\MexMem.hpp"


using namespace std;

int getOutputControl(char* OutputControlSequence){
	char * SequenceWord;
	char * NextNonDelim = NULL;
	char * Delims = " -,";
	int OutputControl = 0x00000000;
	SequenceWord = strtok_s(OutputControlSequence, Delims, &NextNonDelim);
	bool AddorRemove; // TRUE for ADD
	while (SequenceWord != NULL) {
		AddorRemove = true;
		if (SequenceWord[0] == '/') {
			AddorRemove = false;
			SequenceWord++;
		}
		if (!_strcmpi(SequenceWord, "Initial"))
			OutputControl |= OutOps::INITIAL_STATE_REQ;
		if (AddorRemove && !_strcmpi(SequenceWord, "VCF"))
			OutputControl |= OutOps::V_REQ | OutOps::U_REQ | OutOps::I_TOT_REQ
		                   | OutOps::FINAL_STATE_REQ;
		if (AddorRemove && !_strcmpi(SequenceWord, "VCWF"))
			OutputControl |= OutOps::V_REQ | OutOps::U_REQ | OutOps::I_TOT_REQ 
			               | OutOps::WEIGHT_REQ
			               | OutOps::FINAL_STATE_REQ;
		if (AddorRemove && !_strcmpi(SequenceWord, "FSF"))
			OutputControl |= OutOps::V_REQ | OutOps::U_REQ 
						   | OutOps::I_IN_REQ
						   | OutOps::I_EXT_REQ | OutOps::I_EXT_GEN_STATE_REQ
						   | OutOps::WEIGHT_DERIV_REQ
			               | OutOps::WEIGHT_REQ
						   | OutOps::CURRENT_QINDS_REQ
						   | OutOps::SPIKE_QUEUE_REQ
						   | OutOps::LASTSPIKED_NEU_REQ
						   | OutOps::LASTSPIKED_SYN_REQ
						   | OutOps::FINAL_STATE_REQ;
		if (!_strcmpi(SequenceWord, "V"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::V_REQ : 
					 OutputControl & ~(OutOps::V_REQ);
		if (!_strcmpi(SequenceWord, "U"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::U_REQ : 
					 OutputControl & ~(OutOps::U_REQ);
		if (!_strcmpi(SequenceWord, "Iin"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::I_IN_REQ : 
					 OutputControl & ~(OutOps::I_IN_REQ);
		if (!_strcmpi(SequenceWord, "Iext"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::I_EXT_REQ: 
					 OutputControl & ~(OutOps::I_EXT_REQ);
		if (!_strcmpi(SequenceWord, "IExtGenState"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::I_EXT_GEN_STATE_REQ: 
					 OutputControl & ~(OutOps::I_EXT_GEN_STATE_REQ);
		if (!_strcmpi(SequenceWord, "WeightDeriv"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::WEIGHT_DERIV_REQ : 
					 OutputControl & ~(OutOps::WEIGHT_DERIV_REQ);
		if (!_strcmpi(SequenceWord, "Weight"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::WEIGHT_REQ : 
					 OutputControl & ~(OutOps::WEIGHT_REQ);
		if (!_strcmpi(SequenceWord, "CurrentQInds"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::CURRENT_QINDS_REQ : 
					 OutputControl & ~(OutOps::CURRENT_QINDS_REQ);
		if (!_strcmpi(SequenceWord, "SpikeQueue"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::SPIKE_QUEUE_REQ : 
					 OutputControl & ~(OutOps::SPIKE_QUEUE_REQ);
		if (!_strcmpi(SequenceWord, "LSTNeuron"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::LASTSPIKED_NEU_REQ : 
					 OutputControl & ~(OutOps::LASTSPIKED_NEU_REQ);
		if (!_strcmpi(SequenceWord, "LSTSyn"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::LASTSPIKED_SYN_REQ : 
					 OutputControl & ~(OutOps::LASTSPIKED_SYN_REQ);
		if (!_strcmpi(SequenceWord, "Itot"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::I_TOT_REQ : 
					 OutputControl & ~(OutOps::I_TOT_REQ);
		if (!_strcmpi(SequenceWord, "SpikeList"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::SPIKE_LIST_REQ : 
					 OutputControl & ~(OutOps::SPIKE_LIST_REQ);
		if (!_strcmpi(SequenceWord, "Final"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::FINAL_STATE_REQ : 
					 OutputControl & ~(OutOps::FINAL_STATE_REQ);
		SequenceWord = strtok_s(NULL, Delims, &NextNonDelim);
	}
	return OutputControl;
}

void takeInputFromMatlabStruct(mxArray* MatlabInputStruct, InputArgs &InputArgList){

	size_t N = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "a"));
	size_t M = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "NStart"));

	InputArgList.onemsbyTstep = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "onemsbyTstep")));
	InputArgList.NoOfms = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NoOfms")));
	InputArgList.DelayRange = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "DelayRange")));
	InputArgList.CurrentQIndex = 0;
	InputArgList.Time = 0;
	InputArgList.StorageStepSize = DEFAULT_STORAGE_STEP;
	InputArgList.OutputControl = 0;
	InputArgList.StatusDisplayInterval = DEFAULT_STATUS_DISPLAY_STEP;

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

	InputArgList.Neurons = MexVector<Neuron>(N);

	for (int i = 0; i < N; ++i){
		InputArgList.Neurons[i].a = genFloatPtr[0][i];
		InputArgList.Neurons[i].b = genFloatPtr[1][i];
		InputArgList.Neurons[i].c = genFloatPtr[2][i];
		InputArgList.Neurons[i].d = genFloatPtr[3][i];
	}

	// Initializing network (Synapse) specification structure array Network
	genIntPtr[0]   = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NStart")));	  // NStart[M]
	genIntPtr[1]   = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NEnd")));      // NEnd[M]
	genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Weight")));    // Weight[M]
	genFloatPtr[1] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Delay")));     // Delay[M]

	InputArgList.Network = MexVector<Synapse>(M);

	for (int i = 0; i < M; ++i){
		InputArgList.Network[i].NStart = genIntPtr[0][i];
		InputArgList.Network[i].NEnd = genIntPtr[1][i];
		InputArgList.Network[i].Weight = genFloatPtr[0][i];
		InputArgList.Network[i].DelayinTsteps = (int(genFloatPtr[1][i] * InputArgList.onemsbyTstep + 0.5) > 0) ?
			int(genFloatPtr[1][i] * InputArgList.onemsbyTstep + 0.5) : 1;
	}

	// Initializing Time
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "Time");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.Time = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing StorageStepSize
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "StorageStepSize");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.StorageStepSize = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing StatusDisplayInterval
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "StatusDisplayInterval");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.StatusDisplayInterval = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing InterestingSyns
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "InterestingSyns");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		size_t NumElems = mxGetNumberOfElements(genmxArrayPtr);
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.InterestingSyns = MexVector<int>(NumElems);
		InputArgList.InterestingSyns.copyArray(0, genIntPtr[0], NumElems);
	}

	// Initializing V, U and Iin, Iext
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "V");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.V = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.V.copyArray(0, genFloatPtr[0], N);
	}
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "U");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.U = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.U.copyArray(0, genFloatPtr[0], N);
	}
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "Iin");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.Iin = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.Iin.copyArray(0, genFloatPtr[0], N);
	}
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "Iext");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.Iext = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.Iext.copyArray(0, genFloatPtr[0], N);
	}

	// Initializing IExtGenState
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "IExtGenState");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		size_t NElems = mxGetNumberOfElements(genmxArrayPtr);
		InputArgList.IExtGenState = MexVector<uint32_t>(NElems);
		genUIntPtr[0] = reinterpret_cast<uint32_t *>(mxGetData(genmxArrayPtr));
		InputArgList.IExtGenState.copyArray(0, genUIntPtr[0], NElems);
	}

	// Initializing WeightDeriv
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "WeightDeriv");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.WeightDeriv = MexVector<float>(M);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.WeightDeriv.copyArray(0, genFloatPtr[0], M);
	}

	// Initializing CurrentQueueIndex
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "CurrentQIndex");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.CurrentQIndex = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing InitSpikeQueue
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "SpikeQueue");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		mxArray **SpikeQueueArr = reinterpret_cast<mxArray **>(mxGetData(genmxArrayPtr));
		int SpikeQueueSize = InputArgList.onemsbyTstep * InputArgList.DelayRange;
		InputArgList.SpikeQueue = MexVector<MexVector<int> >(SpikeQueueSize);
		for (int i = 0; i < SpikeQueueSize; ++i){
			size_t NumOfSpikes = mxGetNumberOfElements(SpikeQueueArr[i]);
			InputArgList.SpikeQueue[i] = MexVector<int>(NumOfSpikes);
			int * CurrQueueArr = reinterpret_cast<int *>(mxGetData(SpikeQueueArr[i]));
			InputArgList.SpikeQueue[i].copyArray(0, CurrQueueArr, NumOfSpikes);
		}
	}

	// Initializing InitLastSpikedTimeNeuron
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "LSTNeuron");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.LSTNeuron = MexVector<int>(N);
		InputArgList.LSTNeuron.copyArray(0, genIntPtr[0], N);
	}

	// Initializing InitLastSpikedTimeSyn
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "LSTSyn");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.LSTSyn = MexVector<int>(M);
		InputArgList.LSTSyn.copyArray(0, genIntPtr[0], M);
	}

	// Initializing OutputControl
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "OutputControl");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		char * OutputControlSequence = mxArrayToString(genmxArrayPtr);
		InputArgList.OutputControl = getOutputControl(OutputControlSequence);
		mxFree(OutputControlSequence);
	}
}

template<typename T> mxArray * assignmxArray(T &ScalarOut, mxClassID ClassID){

	mxArray * ReturnPointer;
	if (is_arithmetic<T>::value){
		ReturnPointer = mxCreateNumericMatrix_730(1, 1, ClassID, mxREAL);
		*reinterpret_cast<T *>(mxGetData(ReturnPointer)) = ScalarOut;
	}
	else{
		ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	}

	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexMatrix<T> &VectorOut, mxClassID ClassID){

	mxArray * ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	if (VectorOut.ncols() && VectorOut.nrows()){
		mxSetM(ReturnPointer, VectorOut.ncols());
		mxSetN(ReturnPointer, VectorOut.nrows());
		mxSetData(ReturnPointer, VectorOut.releaseArray());
	}

	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexVector<T> &VectorOut, mxClassID ClassID){

	mxArray * ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	if (VectorOut.size()){
		mxSetM(ReturnPointer, VectorOut.size());
		mxSetN(ReturnPointer, 1);
		mxSetData(ReturnPointer, VectorOut.releaseArray());
	}
	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexVector<MexVector<T> > &VectorOut, mxClassID ClassID){
	
	mxArray * ReturnPointer;
	if (VectorOut.size()){
		ReturnPointer = mxCreateCellMatrix(VectorOut.size(), 1);
		
		size_t VectVectSize = VectorOut.size();
		for (int i = 0; i < VectVectSize; ++i){
			mxSetCell(ReturnPointer, i, assignmxArray(VectorOut[i], ClassID));
		}
	}
	else{
		ReturnPointer = mxCreateCellMatrix_730(0, 0);
	}
	return ReturnPointer;
}

mxArray * putOutputToMatlabStruct(OutputVarsStruct &Output){
	const char *FieldNames[] = { 
		"WeightOut",
		"Itot",
		"IExtNeuron",
		"SpikeList",
		nullptr
	};

	int NFields = 0;
	for (; FieldNames[NFields] != nullptr; ++NFields);
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);
	
	// Assigning Weightout
	mxSetField(ReturnPointer, 0, "WeightOut", assignmxArray(Output.WeightOut, mxSINGLE_CLASS));
	// Assigning Itot
	mxSetField(ReturnPointer, 0, "Itot", assignmxArray(Output.Itot, mxSINGLE_CLASS));
	// Assigning IExtNeuron
	mxSetField(ReturnPointer, 0, "IExtNeuron", assignmxArray(Output.IExtNeuron, mxINT32_CLASS));

	// Assigning SpikeList
	mxArray * SpikeListStructPtr;
		const char *SpikeListFieldNames[] = {
			"SpikeSynInds",
			"TimeRchdStartInds"
		};
		SpikeListStructPtr = mxCreateStructArray(1, StructArraySize, 2, SpikeListFieldNames);
		Output.SpikeList.SpikeSynInds.trim();
		Output.SpikeList.TimeRchdStartInds.trim();
		mxSetField(SpikeListStructPtr, 0, "SpikeSynInds"     , assignmxArray(Output.SpikeList.SpikeSynInds, mxINT32_CLASS));
		mxSetField(SpikeListStructPtr, 0, "TimeRchdStartInds", assignmxArray(Output.SpikeList.TimeRchdStartInds, mxINT32_CLASS));
	mxSetField(ReturnPointer, 0, "SpikeList", SpikeListStructPtr);

	return ReturnPointer;
}

mxArray * putStateToMatlabStruct(StateVarsOutStruct &Output){
	const char *FieldNames[] = {
		"V",
		"Iin",
		"Iext",
		"IExtGenState",
		"WeightDeriv",
		"Time",
		"U",
		"Weight",
		"CurrentQIndex",
		"SpikeQueue",
		"LSTNeuron",
		"LSTSyn",
		nullptr
	};
	int NFields = 0;
	for (; FieldNames[NFields] != nullptr; ++NFields);
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);

	// Assigning V, U, I, Time
	mxSetField(ReturnPointer, 0, "V"             , assignmxArray(Output.VOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "U"             , assignmxArray(Output.UOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "Iin"           , assignmxArray(Output.IinOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "Iext"          , assignmxArray(Output.IextOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "IExtGenState"  , assignmxArray(Output.IExtGenStateOut, mxUINT32_CLASS));
	mxSetField(ReturnPointer, 0, "WeightDeriv"   , assignmxArray(Output.WeightDerivOut, mxSINGLE_CLASS));
	
	mxSetField(ReturnPointer, 0, "Time"          , assignmxArray(Output.TimeOut, mxINT32_CLASS));

	// Assigning Weight
	mxSetField(ReturnPointer, 0, "Weight"        , assignmxArray(Output.WeightOut, mxSINGLE_CLASS));

	// Assigning Spike Queue Related Shiz
	mxSetField(ReturnPointer, 0, "CurrentQIndex" , assignmxArray(Output.CurrentQIndexOut, mxINT32_CLASS));
	// Assigning SpikeQueue

	mxSetField(ReturnPointer, 0, "SpikeQueue"    , assignmxArray(Output.SpikeQueueOut, mxINT32_CLASS));

	// Assigning Last Spiked Time related information
	mxSetField(ReturnPointer, 0, "LSTNeuron"     , assignmxArray(Output.LSTNeuronOut, mxINT32_CLASS));
	mxSetField(ReturnPointer, 0, "LSTSyn"        , assignmxArray(Output.LSTSynOut, mxINT32_CLASS));

	return ReturnPointer;
}

mxArray * putSingleStatetoMatlabStruct(SingleStateStruct &SingleStateList){
	const char *FieldNames[] = {
		"V",
		"Iin",
		"Iext",
		"IExtGenState",
		"WeightDeriv",
		"Time",
		"U",
		"Weight",
		"CurrentQIndex",
		"SpikeQueue",
		"LSTNeuron",
		"LSTSyn",
		nullptr
	};
	int NFields = 0;
	for (; FieldNames[NFields] != nullptr; ++NFields);
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);

	// Assigning vout, Uout, Iout, TimeOut
	mxSetField(ReturnPointer, 0, "V"                 , assignmxArray(SingleStateList.V, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "U"                 , assignmxArray(SingleStateList.U, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "Iin"               , assignmxArray(SingleStateList.Iin, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "Iext"              , assignmxArray(SingleStateList.Iext, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "IExtGenState"      , assignmxArray(SingleStateList.IExtGenState, mxUINT32_CLASS));
	mxSetField(ReturnPointer, 0, "WeightDeriv"       , assignmxArray(SingleStateList.WeightDeriv, mxSINGLE_CLASS));
	
	if (SingleStateList.Time >= 0)
		mxSetField(ReturnPointer, 0, "Time"          , assignmxArray(SingleStateList.Time, mxINT32_CLASS));
	else
		mxSetField(ReturnPointer, 0, "Time"          , mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxREAL));

	// Assigning WeightOut
	mxSetField(ReturnPointer, 0, "Weight"            , assignmxArray(SingleStateList.Weight, mxSINGLE_CLASS));

	// Assigning Spike Queue Related Shiz
	if (SingleStateList.CurrentQIndex >= 0)
		mxSetField(ReturnPointer, 0, "CurrentQIndex" , assignmxArray(SingleStateList.CurrentQIndex, mxINT32_CLASS));
	else
		mxSetField(ReturnPointer, 0, "CurrentQIndex" , mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxREAL));
	mxSetField(ReturnPointer, 0, "SpikeQueue"        , assignmxArray(SingleStateList.SpikeQueue, mxINT32_CLASS));

	// Assigning Last Spiked Time related information
	mxSetField(ReturnPointer, 0, "LSTNeuron"         , assignmxArray(SingleStateList.LSTNeuron, mxINT32_CLASS));
	mxSetField(ReturnPointer, 0, "LSTSyn"            , assignmxArray(SingleStateList.LSTSyn, mxINT32_CLASS));

	return ReturnPointer;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]){
	// NOTE THAT THERE IS NO DATA VALIDATION AS THIS IS EXPECTED TO HAVE 
	// BEEN DONE IN THE MATLAB SIDE OF THE INTERFACE TO THIS MEX FUNCTION

	InputArgs InputArgList;
	takeInputFromMatlabStruct(prhs[0], InputArgList);

	// Declaring Output Vectors
	OutputVarsStruct PureOutput;
	StateVarsOutStruct StateVarsOutput;
	FinalStateStruct FinalStateOutput;
	InitialStateStruct InitialStateOutput;

	// Running Simulation Function.
	chrono::system_clock::time_point TStart = chrono::system_clock::now();
	try{
		
		SimulateParallel(
			move(InputArgList),
			PureOutput,
			StateVarsOutput,
			FinalStateOutput,
			InitialStateOutput);
	}
	catch(ExOps::ExCodes A){
		if (A == ExOps::EXCEPTION_MEM_FULL){
		#ifdef MEX_LIB
			char OutputString[256];
			sprintf_s(OutputString, 256, "Mem Limit of %lld MB Exceeded\n", (MemCounter::MemUsageLimit) >> 20);
			mexErrMsgIdAndTxt("CppSimException:MemOverFlow", OutputString);
		#elif defined MEX_EXE
			throw A;
		#endif
		}
	}

	chrono::system_clock::time_point TEnd = chrono::system_clock::now();
#ifdef MEX_LIB
	mexPrintf("The Time taken = %d milliseconds\n", chrono::duration_cast<chrono::milliseconds>(TEnd - TStart).count());
	mexEvalString("drawnow");
#elif defined MEX_EXE
	printf("The Time taken = %d milliseconds\n", chrono::duration_cast<chrono::milliseconds>(TEnd - TStart).count());
#endif
	mwSize StructArraySize[2] = { 1, 1 };
	
	plhs[0] = putOutputToMatlabStruct(PureOutput);
	plhs[1] = putStateToMatlabStruct(StateVarsOutput);
	plhs[2] = putSingleStatetoMatlabStruct(FinalStateOutput);

	if (nlhs == 4){
		plhs[3] = putSingleStatetoMatlabStruct(InitialStateOutput);
	}
}