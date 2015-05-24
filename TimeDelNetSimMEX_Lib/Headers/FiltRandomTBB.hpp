#ifndef FILT_RANDOM_TBB_HPP
#define FILT_RANDOM_TBB_HPP

#include <random>
#include <chrono>
#include <stdint.h>
#include <tbb/task.h>
#include <tbb/task_group.h>

// For providing random seed
#include "MexMem.hpp"

using namespace std;

// Random Number Generator
class XorShiftPlus{

public:
	struct StateStruct{
		uint32_t w, x, y, z;
		void ConvertVecttoState(const MexVector<uint32_t> &SeedState) {
			this->w = (SeedState.size() > 0) ? 
				SeedState[0] : (uint32_t)chrono::system_clock::now().time_since_epoch().count();
			this->x = (SeedState.size() > 1) ? 
				SeedState[1] : (uint32_t)chrono::system_clock::now().time_since_epoch().count();
			this->y = (SeedState.size() > 2) ?
				SeedState[2] : (uint32_t)chrono::system_clock::now().time_since_epoch().count();
			this->z = (SeedState.size() > 3) ?
				SeedState[3] : (uint32_t)chrono::system_clock::now().time_since_epoch().count();
		}
		void ConvertStatetoVect(MexVector<uint32_t> &SeedState) const{
			if (SeedState.size() != 4)
				SeedState.resize(4);
			SeedState[0] = w;
			SeedState[1] = x;
			SeedState[2] = y;
			SeedState[3] = z;
		}
		void ConvertStatetoVect(const MexVector<uint32_t> &SeedState) const{
			if (SeedState.size() != 4)
				throw ExOps::EXCEPTION_CONST_MOD;
			SeedState[0] = w;
			SeedState[1] = x;
			SeedState[2] = y;
			SeedState[3] = z;
		}
	};
private:
	StateStruct State;

public:
	XorShiftPlus(){
		State.w = 0;
		State.x = (uint32_t)chrono::system_clock::now().time_since_epoch().count();
		State.y = 0;
		State.z = 0;
	}
	XorShiftPlus(uint32_t Seed){
		State.w = 0;
		State.x = Seed;
		State.y = 0;
		State.z = 0;
	}
	XorShiftPlus(const StateStruct &SeedState){
		State = SeedState;
	}

	inline uint32_t operator()(void) {
		uint32_t &w = State.w;
		uint32_t &x = State.x;
		uint32_t &y = State.y;
		uint32_t &z = State.z;
		uint32_t t = x ^ (x << 11);
		x = y; y = z; z = w;
		return w = w ^ (w >> 19) ^ t ^ (t >> 8);
	}
	inline const StateStruct& getstate(void) const{
		return State;
	}
	void setstate(const StateStruct &SeedState) {
		State = SeedState;
	}
	

};


typedef float resTyp;

class BandLimGaussVect : public MexVector<resTyp>{

	XorShiftPlus Generator1;
	XorShiftPlus Generator2;
	resTyp alpha;
public:
	struct StateStruct{
		XorShiftPlus Generator1;
		XorShiftPlus Generator2;
		MexVector<resTyp> Values;
		resTyp alpha;
		StateStruct() : Generator1(), Generator2(), alpha(resTyp(0)), Values(){};
	};
	BandLimGaussVect() : alpha(resTyp(0)),  MexVector<resTyp>(), Generator1(), Generator2(){};
	BandLimGaussVect(resTyp alpha_) : alpha(alpha_), MexVector<resTyp>(), Generator1(), Generator2(){};

	BandLimGaussVect(const XorShiftPlus &Generator1_, const XorShiftPlus &Generator2_, resTyp alpha_ = resTyp(0)) :
		alpha(alpha_), MexVector<resTyp>(), Generator1(Generator1_), Generator2(Generator2_){};

	BandLimGaussVect(int n, resTyp alpha_ = resTyp(0)) : alpha(alpha_), MexVector<resTyp>(n, resTyp(0)), Generator1(), Generator2(){};

	BandLimGaussVect(int n, const XorShiftPlus &Generator1_, const XorShiftPlus &Generator2_, resTyp alpha_ = resTyp(0)) :
		MexVector<resTyp>(n, resTyp(0)),
		Generator1(Generator1_),
		Generator2(Generator2_),
		alpha(alpha_){};

	BandLimGaussVect(const MexVector<resTyp> &V, 
		const XorShiftPlus &Generator1_, 
		const XorShiftPlus &Generator2_, 
		resTyp alpha_ = resTyp(0)) :
		MexVector<resTyp>(V),
		Generator1(Generator1_),
		Generator2(Generator2_),
		alpha(alpha_){};

	void reset(){
		Generator1 = XorShiftPlus();
		Generator2 = XorShiftPlus();
	}
	void reset(resTyp alpha_) {
		Generator1 = XorShiftPlus();
		Generator2 = XorShiftPlus();
		alpha = alpha_;
	}

	void configure(resTyp alpha_){
		alpha = alpha_;
	};
	void configure(const XorShiftPlus &Generator1_, const XorShiftPlus &Generator2_, resTyp alpha_ = resTyp(-1)){
		Generator1 = Generator1_;
		Generator2 = Generator2_;
		alpha = (alpha_ > 0)?alpha_ : alpha;
	}

	void setstate(const StateStruct &State){
		Generator1 = State.Generator1;
		Generator2 = State.Generator2;
		this->assign(State.Values);
		alpha = State.alpha;
	}
	void getstate(StateStruct &State) const{
		State.Generator1 = Generator1;
		State.Generator2 = Generator2;
		State.Values = *this;
		State.alpha = alpha;
	}
	void readstate(StateStruct &State) const{
		State.Generator1 = Generator1;
		State.Generator2 = Generator2;
		this->sharewith(State.Values);
		State.alpha = alpha;
	}
	void resize(int NewSize){
		MexVector<resTyp>::resize(NewSize, resTyp(0));
	}
	const XorShiftPlus& generator1(){
		return Generator1;
	}
	const XorShiftPlus& generator2(){
		return Generator2;
	}

	__inline pair<resTyp, resTyp> gaussRandVal(XorShiftPlus &Gen){
		pair<resTyp, resTyp> retVal;
		resTyp &U = retVal.first;
		resTyp &V = retVal.second;
		resTyp S = 0;
		bool value_invalid = true;
		while (S < 1){
			U = (resTyp(Gen()) / resTyp(1ui32 << 31)) - 1;
			V = (resTyp(Gen()) / resTyp(1ui32 << 31)) - 1;
			S = 1 / (U*U + V*V);
		}
		if (is_same<resTyp, float>::value){
			U *= sqrtf(2 * logf(S) * S);
			V *= sqrtf(2 * logf(S) * S);
		}
		else{
			U *= sqrt(2 * log(S));
			V *= sqrt(2 * log(S));
		}
		return retVal;
	}

	void generate(){
		iterator Array_Beg = this->begin();
		iterator Array_Mid = this->begin() + this->size() / 2;
		iterator Array_Last = this->end();
		iterator i, j;
		
		float &alpha = this->alpha;
		pair<resTyp, resTyp> gaussRandValues;
		for (i = Array_Beg + 1; i < Array_Last; i += 2){
			gaussRandValues = gaussRandVal(Generator1);
			*(i - 1) = *(i-1) * alpha + (1 - alpha) * gaussRandValues.first;
			*i = *i * alpha + (1 - alpha) * gaussRandValues.second;
		}
		if (i == Array_Last){
			gaussRandValues = gaussRandVal(Generator1);
			*(i - 1) = *(i - 1) * alpha + (1 - alpha) * gaussRandValues.first;
		}
	}
};

class mt19937Extended : public mt19937{
public:
	void getState(MexVector<unsigned int> &V){
		V.resize(624);
		for (int i = 0; i < 624; ++i){
			V[i] = this->_At(i);
		}
	}
	void setState(MexVector<unsigned int> &V){
		for (int i = 0; i < 624; ++i)
			this->_Ax[i] = V[i] & _WMSK;
		this->_Idx = 624;
	}
	void setState(unsigned int Seed){
		this->seed(Seed);
	}
};
// // Random Code mainly to open libraries
// // Completely useless otherwise
// void Coolshit(){
	// FiltRandomTBB YEAHHNN;
	// normal_distribution<> Yo;
	// Yo(YEAHHNN);
	// YEAHHNN();
// }
#endif
