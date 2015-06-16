#ifndef MEXMEM_HPP
#define MEXMEM_HPP
#include <xutility>
#include <matrix.h>
#include <type_traits>

template<typename T> class MexVector;
template<typename T> class MexMatrix;

struct ExOps{
	enum ExCodes{
		EXCEPTION_MEM_FULL = 0xFF,
		EXCEPTION_EXTMEM_MOD = 0x7F,
		EXCEPTION_CONST_MOD = 0x3F
	};
};

class MemCounter{
	static size_t MemUsageCount;
public:
	const static size_t MemUsageLimit;

	template<typename T>
	friend class MexVector;

	template<typename T>
	friend class MexMatrix;
};

template<typename T>
class MexVector{
	bool isCurrentMemExternal;
	T* Array_Beg;
	T* Array_Last;	// Note Not Array_End as it is not representative
					// of the capacity
	T* Array_End;	// Note Array End is true end of allocated array
public:
	typedef T* iterator;

	inline MexVector() : Array_End(NULL), isCurrentMemExternal(false), Array_Beg(NULL), Array_Last(NULL){};
	inline explicit MexVector(size_t Size){
		if (Size > 0){
			size_t NumExtraBytes = Size*sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(Size, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
			}
			if (Array_Beg == NULL)     // Full Memory exception
				throw ExOps::EXCEPTION_MEM_FULL;
			for (size_t i = 0; i < Size; ++i)
				new (Array_Beg + i) T;	// Constructing Default Objects
		}
		else{
			Array_Beg = NULL;
		}
		Array_Last = Array_Beg + Size;
		Array_End = Array_Beg + Size;
		isCurrentMemExternal = false;
	}
	inline MexVector(const MexVector<T> &M){
		size_t Size = M.size();
		if (Size > 0){
			size_t NumExtraBytes = Size*sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(Size, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
			}
			if (Array_Beg != NULL)
				for (size_t i = 0; i < Size; ++i)
					new (Array_Beg + i) T(M.Array_Beg[i]);
			else{	// Checking for memory full shit
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array_Beg = NULL;
		}
		Array_Last = Array_Beg + Size;
		Array_End = Array_Beg + Size;
		isCurrentMemExternal = false;
	}
	inline MexVector(MexVector<T> &&M){
		isCurrentMemExternal = M.isCurrentMemExternal;
		Array_Beg = M.Array_Beg;
		Array_Last = M.Array_Last;
		Array_End = M.Array_End;
		if (!(M.Array_Beg == NULL)){
			M.isCurrentMemExternal = true;
		}
	}
	inline explicit MexVector(size_t Size, const T &Elem){
		if (Size > 0){
			size_t NumExtraBytes = Size*sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(Size, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
			}
			if (Array_Beg == NULL){
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array_Beg = NULL;
		}
		Array_Last = Array_Beg + Size;
		Array_End = Array_Beg + Size;
		isCurrentMemExternal = false;
		for (T* i = Array_Beg; i < Array_Last; ++i)
			new (i) T(Elem);
	}
	inline explicit MexVector(size_t Size, T* Array_, bool SelfManage = 1) :
		Array_Beg(Size ? Array_ : NULL), 
		Array_Last(Array_ + Size), 
		Array_End(Array_ + Size), 
		isCurrentMemExternal(Size ? !SelfManage : false){}

	inline ~MexVector(){
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0);
			trim();    // Ensure destruction of all elements
		}
	}
	inline void operator = (const MexVector<T> &M){
		assign(M);
	}
	inline void operator = (const MexVector<T> &&M){
		assign(move(M));
	}
	inline void operator = (const MexVector<T> &M) const{
		this->assign(M);
	}
	inline T& operator[] (size_t Index) const{
		return Array_Beg[Index];
	}

	// If Ever this operation is called, no funcs except will work (Vector will point to empty shit) unless 
	// the assign function is explicitly called to self manage another array.
	inline T* releaseArray(){
		if (isCurrentMemExternal)
			return NULL;
		else{
			isCurrentMemExternal = false;
			T* temp = Array_Beg;
			Array_Beg = NULL;
			Array_Last = NULL;
			Array_End = NULL;
			return temp;
		}
	}
	inline void assign(const MexVector<T> &M){
		size_t ExtSize = M.size();
		size_t currCapacity = this->capacity();
		if (ExtSize > currCapacity && !isCurrentMemExternal){
			if (Array_Beg != NULL){
				resize(0);		// this ensures destruction of all elements
				trim();
			}
			size_t NumExtraBytes = ExtSize*sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(ExtSize, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
			}
			if (Array_Beg == NULL)
				throw ExOps::EXCEPTION_MEM_FULL;
			for (size_t i = 0; i < ExtSize; ++i)
				new (Array_Beg+i) T(M.Array_Beg[i]);	// needs to be copy constructed
			Array_Last = Array_Beg + ExtSize;
			Array_End = Array_Beg + ExtSize;
		}
		else if (ExtSize <= currCapacity && !isCurrentMemExternal){
			for (size_t i = 0; i < ExtSize; ++i)
				Array_Beg[i] = M.Array_Beg[i];			// operator= needs to be defined
														// else standard shallow copy
			Array_Last = Array_Beg + ExtSize;
		}
		else if (ExtSize == this->size()){
			for (size_t i = 0; i < ExtSize; ++i)
				Array_Beg[i] = M.Array_Beg[i];
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;	// Attempted resizing or reallocation of Array_Beg holding External Memory
		}
	}
	inline void assign(MexVector<T> &&M){
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0);
			trim();    // Ensure destruction of all elements
		}
		isCurrentMemExternal = M.isCurrentMemExternal;
		Array_Beg = M.Array_Beg;
		Array_Last = M.Array_Last;
		Array_End = M.Array_End;
		if (Array_Beg != NULL){
			M.isCurrentMemExternal = true;
		}
	}
	inline void assign(const MexVector<T> &M) const{
		size_t ExtSize = M.size();
		if (ExtSize == this->size()){
			for (size_t i = 0; i < ExtSize; ++i)
				Array_Beg[i] = M.Array_Beg[i];
		}
		else{
			throw ExOps::EXCEPTION_CONST_MOD;	// Attempted resizing or reallocation or reassignment of const Array_Beg
		}
	}
	inline void assign(size_t Size, T* Array_, bool SelfManage = 1){
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0);
			trim();
		}
		if (Size > 0){
			isCurrentMemExternal = !SelfManage;
			Array_Beg = Array_;
		}
		else{
			isCurrentMemExternal = false;
			Array_Beg = NULL;
		}
		Array_Last = Array_Beg + Size;
		Array_End = Array_Beg + Size;
	}
	inline void push_back(const T &Val){
		if (Array_Last != Array_End){
			*Array_Last = Val;
			++Array_Last;
		}
		else {
			size_t Capacity = this->capacity();
			int Exp = 0;
			Capacity = Capacity ? Capacity + (Capacity >> 1) : 4;
			reserve(Capacity);
			*Array_Last = Val;
			++Array_Last;
		}
	}
	inline void push_size(size_t Increment){
		if (Array_Last + Increment> Array_End){
			size_t CurrCapacity = this->capacity();
			size_t CurrSize = this->size();
			CurrCapacity = CurrCapacity ? CurrCapacity : 4;
			while (CurrCapacity <= CurrSize + Increment){
				CurrCapacity += ((CurrCapacity >> 1) + 1);
			}
			reserve(CurrCapacity);
		}
		Array_Last += Increment;
	}
	inline void copyArray(size_t Position, T* ArrBegin, size_t NumElems) const{
		if (Position + NumElems > this->size()){
			throw ExOps::EXCEPTION_CONST_MOD;
		}
		else{
			for (size_t i = 0; i<NumElems; ++i)
				Array_Beg[i + Position] = ArrBegin[i];
		}
	}
	inline void reserve(size_t Cap){
		size_t currCapacity = this->capacity();
		if (!isCurrentMemExternal && Cap > currCapacity){
			T* Temp;
			size_t prevSize = this->size();

			if (Array_Beg != NULL){
				// This is special bcuz reallocation requires (currCapacity + Cap)
				// Locations to be free but increases memory by only (Cap - currCapacity)
				size_t NumExtraBytes = (Cap)*sizeof(T);
				if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
					MemCounter::MemUsageCount += (Cap - currCapacity)*sizeof(T);
					Temp = reinterpret_cast<T*>(mxRealloc(Array_Beg, Cap*sizeof(T)));
				}
				else{
					throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
				}
			}
			else{
				size_t NumExtraBytes = Cap*sizeof(T);
				if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
					MemCounter::MemUsageCount += NumExtraBytes;
					Temp = reinterpret_cast<T*>(mxCalloc(Cap, sizeof(T)));
				}
				else{
					throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
				}
			}
			if (Temp != NULL){
				Array_Beg = Temp;
				for (size_t i = currCapacity; i < Cap; ++i)
					new (Array_Beg + i) T;
				Array_Last = Array_Beg + prevSize;
				Array_End = Array_Beg + Cap;
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL;
		}
		else if (isCurrentMemExternal)
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted reallocation of external memory
	}
	inline void resize(size_t NewSize) {
		if (NewSize > this->capacity() && !isCurrentMemExternal){
			reserve(NewSize);
		}
		else if (isCurrentMemExternal){
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted resizing of External memory
		}
		Array_Last = Array_Beg + NewSize;
	}
	inline void resize(size_t NewSize, const T &Val){
		size_t prevSize = this->size();
		resize(NewSize);
		T* End = Array_Beg + NewSize;
		if (NewSize != 0)
			for (T* j = Array_Beg + prevSize; j < End; ++j)
				*j = Val;
	}
	inline void resize(size_t NewSize, T &&Val){
		size_t prevSize = this->size();
		resize(NewSize);
		T* End = Array_Beg + NewSize;
		if (NewSize != 0)
			for (T* j = Array_Beg + prevSize; j < End; ++j)
				*j = Val;
	}
	inline void sharewith(MexVector<T> &M) const{
		if (!M.isCurrentMemExternal && M.Array_Beg != NULL){
			M.resize(0);
			M.trim();
		}
		if (Array_End){
			M.Array_Beg = Array_Beg;
			M.Array_Last = Array_Last;
			M.Array_End = Array_End;
			M.isCurrentMemExternal = true;
		}
		else{
			M.Array_Beg = NULL;
			M.Array_Last = NULL;
			M.Array_End = NULL;
			M.isCurrentMemExternal = false;
		}
	}
	inline void swap(MexVector<T> &M) {
		T* Temp_Beg;
		int Temp_Size;
		int Temp_Capacity;
		bool Temp_isCurrentMemExternal;

		Temp_Size = M.size();
		Temp_Capacity = M.capacity();
		Temp_isCurrentMemExternal = M.ismemext();
		Temp_Beg = M.releaseArray();

		M.Array_Beg = Array_Beg;
		M.Array_Last = Array_Last;
		M.Array_End = Array_End;
		M.isCurrentMemExternal = isCurrentMemExternal;

		Array_Beg = Temp_Beg;
		Array_Last = Temp_Beg + Temp_Size;
		Array_End = Temp_Beg + Temp_Capacity;
		isCurrentMemExternal = Temp_isCurrentMemExternal;
	}
	inline void trim(){
		if (!isCurrentMemExternal){
			size_t currSize = this->size();
			// Run Destructors
			if (!std::is_trivially_destructible<T>::value)
				for (T* j = Array_Last; j < Array_End; ++j) {
					j->~T();
				}
			
			T* Temp;
			size_t NumExtraBytes = (currSize)*sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount -= (this->capacity() - currSize)*sizeof(T);
				Temp = reinterpret_cast<T*>(mxRealloc(Array_Beg, currSize*sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL; // Memory Quota Exceeded
			}

			if (currSize > 0){
				if (Temp != NULL)
					Array_Beg = Temp;
				else
					throw ExOps::EXCEPTION_MEM_FULL;
			}
			else{
				Array_Beg = NULL;
			}
			Array_Last = Array_Beg + currSize;
			Array_End = Array_Beg +  currSize;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;
		}
	}
	inline void clear(){
		if (!isCurrentMemExternal)
			Array_Last = Array_Beg;
		else
			throw ExOps::EXCEPTION_EXTMEM_MOD; //Attempt to resize External memory
	}
	inline iterator begin() const{
		return Array_Beg;
	}
	inline iterator end() const{
		return Array_Last;
	}
	inline T &last() const{
		return *(Array_Last - 1);
	}
	inline size_t size() const{
		return Array_Last - Array_Beg;
	}
	inline size_t capacity() const{
		return Array_End - Array_Beg;
	}
	inline bool ismemext() const{
		return isCurrentMemExternal;
	}
	inline bool isempty() const{
		return Array_Beg == Array_Last;
	}
	inline bool istrulyempty() const{
		return Array_End == Array_Beg;
	}
};


template<class T>
class MexMatrix{
	size_t NRows, NCols;
	size_t Capacity;
	MexVector<T> RowReturnVector;
	T* Array_Beg;
	bool isCurrentMemExternal;

public:
	typedef T* iterator;

	inline MexMatrix() : NRows(0), NCols(0), Capacity(0), isCurrentMemExternal(false), Array_Beg(NULL), RowReturnVector(){};
	inline explicit MexMatrix(size_t NRows_, size_t NCols_) : RowReturnVector() {
		if (NRows_*NCols_ > 0){
			size_t NumExtraBytes = NRows_ * NCols_ * sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(NRows_ * NCols_, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL;
			}
			if (Array_Beg == NULL)     // Full Memory exception
				throw ExOps::EXCEPTION_MEM_FULL;
			for (size_t i = 0; i < NRows_*NCols_; ++i)
				new (Array_Beg + i) T;
		}
		else{
			Array_Beg = NULL;
		}
		NRows = NRows_;
		NCols = NCols_;
		Capacity = NRows_*NCols_;
		isCurrentMemExternal = false;
	}
	inline MexMatrix(const MexMatrix<T> &M) : RowReturnVector(){
		size_t MNumElems = M.NRows * M.NCols;
		if (MNumElems > 0){
			int NumExtraBytes = MNumElems * sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(MNumElems, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL;
			}
			if (Array_Beg != NULL)
				for (size_t i = 0; i < MNumElems; ++i)
					new (Array_Beg + i) T(M.Array_Beg[i]);
			else{	// Checking for memory full shit
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array_Beg = NULL;
		}
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = MNumElems;
		isCurrentMemExternal = false;
	}
	inline MexMatrix(MexMatrix<T> &&M) : RowReturnVector(){
		isCurrentMemExternal = M.isCurrentMemExternal;
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = M.Capacity;
		Array_Beg = M.Array_Beg;
		if (!(M.Array_Beg == NULL)){
			M.isCurrentMemExternal = true;
		}
	}
	inline explicit MexMatrix(size_t NRows_, size_t NCols_, const T &Elem) : RowReturnVector(){
		size_t NumElems = NRows_*NCols_;
		if (NumElems > 0){
			int NumExtraBytes = NumElems * sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(NumElems, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL;
			}
			if (Array_Beg == NULL){
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array_Beg = NULL;
		}

		NRows = NRows_;
		NCols = NCols_;
		Capacity = NumElems;
		isCurrentMemExternal = false;
		for (size_t i = 0; i < NumElems; ++i){
			Array_Beg[i] = Elem;
		}
	}
	inline MexMatrix(size_t NRows_, size_t NCols_, T* Array_, bool SelfManage = 1) :
		RowReturnVector(),
		Array_Beg((NRows_*NCols_) ? Array_ : NULL),
		NRows(NRows_), NCols(NCols_),
		Capacity(NRows_*NCols_),
		isCurrentMemExternal((NRows_*NCols_) ? ~SelfManage : false){}

	inline ~MexMatrix(){
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0, 0);		// Ensure destruction of elements
			trim();
		}
	}
	inline void operator = (const MexMatrix<T> &M){
		assign(M);
	}
	inline void operator = (const MexMatrix<T> &&M){
		assign(move(M));
	}
	inline void operator = (const MexMatrix<T> &M) const{
		assign(M);
	}
	inline const MexVector<T>& operator[] (size_t Index) {
		RowReturnVector.assign(NCols, Array_Beg + Index*NCols, false);
		return  RowReturnVector;
	}
	inline T& operator()(size_t RowIndex, size_t ColIndex){
		return *(Array_Beg + RowIndex*NCols + ColIndex);
	}
	// If Ever this operation is called, no funcs except will work (Vector will point to NULL) unless 
	// the assign function is explicitly called to self manage another array.
	inline T* releaseArray(){
		if (isCurrentMemExternal)
			return NULL;
		else{
			isCurrentMemExternal = false;
			T* temp = Array_Beg;
			Array_Beg = NULL;
			NRows = 0;
			NCols = 0;
			Capacity = 0;
			return temp;
		}
	}
	inline void assign(const MexMatrix<T> &M){

		size_t MNumElems = M.NRows * M.NCols;

		if (MNumElems > this->Capacity && !isCurrentMemExternal){
			if (Array_Beg != NULL){
				resize(0, 0);		// Ensure destruction of elements
				trim();
			}
			size_t NumExtraBytes = MNumElems * sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				Array_Beg = reinterpret_cast<T*>(mxCalloc(MNumElems, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL;
			}
			if (Array_Beg == NULL)
				throw ExOps::EXCEPTION_MEM_FULL;
			for (size_t i = 0; i < MNumElems; ++i)
				new (Array_Beg+i) T(M.Array_Beg[i]);	// needs to be copy constructed
			NRows = M.NRows;
			NCols = M.NCols;
			Capacity = MNumElems;
		}
		else if (MNumElems <= this->Capacity && !isCurrentMemExternal){
			for (size_t i = 0; i < MNumElems; ++i)
				Array_Beg[i] = M.Array_Beg[i];
			NRows = M.NRows;
			NCols = M.NCols;
		}
		else if (MNumElems == this->NRows * this->NCols){
			for (size_t i = 0; i < MNumElems; ++i)
				Array_Beg[i] = M.Array_Beg[i];
			NRows = M.NRows;
			NCols = M.NCols;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;	// Attempted resizing or reallocation of Array_Beg holding External Memory
		}
	}
	inline void assign(MexMatrix<T> &&M){
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0, 0);		// Ensure destruction of elements
			trim();

		}
		isCurrentMemExternal = M.isCurrentMemExternal;
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = M.Capacity;
		Array_Beg = M.Array_Beg;
		if (Array_Beg != NULL){
			M.isCurrentMemExternal = true;
		}
	}
	inline void assign(const MexMatrix<T> &M) const{
		size_t MNumElems = M.NRows * M.NCols;
		if (M.NRows == NRows && M.NCols == NCols){
			for (size_t i = 0; i < MNumElems; ++i)
				Array_Beg[i] = M.Array_Beg[i];
		}
		else{
			throw ExOps::EXCEPTION_CONST_MOD;
		}
	}
	inline void assign(size_t NRows_, size_t NCols_, T* Array_, bool SelfManage = 1){
		NRows = NRows_;
		NCols = NCols_;
		Capacity = NRows_*NCols_;
		if (!isCurrentMemExternal && Array_Beg != NULL){
			resize(0, 0);		// Ensure destruction of elements
			trim();
		}
		if (Capacity > 0){
			isCurrentMemExternal = !SelfManage;
			Array_Beg = Array_;
		}
		else{
			isCurrentMemExternal = false;
			Array_Beg = NULL;
		}
	}
	inline void copyArray(size_t RowPos, size_t ColPos, T* ArrBegin, size_t NumElems) const{
		size_t Position = RowPos*NCols + ColPos;
		if (Position + NumElems > NRows*NCols){
			throw ExOps::EXCEPTION_CONST_MOD;
		}
		else{
			for (size_t i = 0; i<NumElems; ++i)
				Array_Beg[i + Position] = ArrBegin[i];
		}
	}
	inline void reserve(size_t Cap){
		// this is a bit shitty as it basically deletes all previous values
		if (!isCurrentMemExternal && Cap > Capacity){
			T* temp;
			if (Array_Beg != NULL){
				resize(0, 0);
				trim();
			}
			size_t NumExtraBytes = Cap * sizeof(T);
			if (MemCounter::MemUsageCount + NumExtraBytes <= MemCounter::MemUsageLimit){
				MemCounter::MemUsageCount += NumExtraBytes;
				temp = reinterpret_cast<T*>(mxCalloc(Cap, sizeof(T)));
			}
			else{
				throw ExOps::EXCEPTION_MEM_FULL;
			}
			
			if (temp != NULL){
				Array_Beg = temp;
				Capacity = Cap;
				for (int i = 0; i < Cap; ++i){
					new (Array_Beg + i) T;	// Defult constructing memory locations.
				}
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL; // Full memory
		}
		else if (isCurrentMemExternal)
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted reallocation of external memory
	}
	inline void resize(size_t NewNRows, size_t NewNCols){
		size_t NewSize = NewNRows * NewNCols;
		if (NewSize > Capacity && !isCurrentMemExternal){
			reserve(NewSize);
		}
		else if (isCurrentMemExternal){
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted resizing of External memory
		}
		NRows = NewNRows;
		NCols = NewNCols;
	}
	inline void resize(size_t NewNRows, size_t NewNCols, const T &Val){
		size_t PrevSize = NRows * NCols;
		size_t NewSize = NewNRows * NewNCols;
		resize(NewNRows, NewNCols);
		for (T *j = Array_Beg + PrevSize; j < Array_Beg + NewSize; ++j){
			*j = Val;
		}
	}
	inline void resize(size_t NewNRows, size_t NewNCols, T &&Val){
		size_t PrevSize = NRows * NCols;
		size_t NewSize = NewNRows * NewNCols;
		resize(NewNRows, NewNCols);
		for (T *j = Array_Beg + PrevSize; j < Array_Beg + NewSize; ++j){
			*j = std::move(Val);
		}
	}
	inline void trim(){
		if (!isCurrentMemExternal){
			if (NRows*NCols > 0){
				T* Temp = reinterpret_cast<T*>(mxRealloc(Array_Beg, NRows*NCols*sizeof(T)));
				if (Temp != NULL)
					Array_Beg = Temp;
				else
					throw ExOps::EXCEPTION_MEM_FULL;
			}
			else{
				if (Array_Beg != NULL){
					MemCounter::MemUsageCount -= (this->Capacity)*sizeof(T);
					mxFree(Array_Beg);
				}
				Array_Beg = NULL;
			}
			Capacity = NRows*NCols;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD; // trying to reallocate external memory
		}
	}
	inline void sharewith(MexMatrix<T> &M) const{
		if (!M.isCurrentMemExternal && M.Array_Beg != NULL){
			resize(0, 0);		// Ensure destruction of elements
			trim();
		}
		if (Capacity > 0){
			M.NRows = NRows;
			M.NCols = NCols;
			M.Capacity = Capacity;
			M.Array_Beg = Array_Beg;
			M.isCurrentMemExternal = true;
		}
		else{
			M.NRows = 0;
			M.NCols = 0;
			M.Capacity = 0;
			M.Array_Beg = NULL;
			M.isCurrentMemExternal = false;
		}
	}
	inline void swap(MexMatrix<T> &M){
		size_t Temp_nRows, Temp_nCols;
		size_t Temp_Capacity;
		T* Temp_Beg;

		Temp_nRows = M.NRows;
		Temp_nCols = M.NCols;
		Temp_Capacity = M.Capacity;
		Temp_Beg = M.releaseArray();

		M.Array_Beg = Array_Beg;
		M.NRows = NRows;
		M.NCols = NCols;
		M.Capacity = Capacity;

		Array_Beg = Temp_Beg;
		NRows = Temp_nRows;
		NCols = Temp_nCols;
		Capacity = Temp_Capacity;
	}
	inline void clear(){
		if (!isCurrentMemExternal)
			NRows = 0;
		else
			throw ExOps::EXCEPTION_EXTMEM_MOD; //Attempt to resize External memory
	}
	inline iterator begin() const{
		return Array_Beg;
	}
	inline iterator end() const{
		return Array_Beg + (NRows * NCols);
	}
	inline size_t nrows() const{
		return NRows;
	}
	inline size_t ncols() const{
		return NCols;
	}
	inline size_t capacity() const{
		return Capacity;
	}
	inline bool ismemext() const{
		return isCurrentMemExternal;
	}
	inline bool isempty() const{
		return (NCols*NRows == 0);
	}
	inline bool istrulyempty() const{
		return Capacity == 0;
	}
};
#endif