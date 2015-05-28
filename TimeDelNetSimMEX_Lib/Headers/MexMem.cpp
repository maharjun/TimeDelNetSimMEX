#include "MexMem.hpp"

const size_t MemCounter::MemUsageLimit = (size_t(3) << 29);
size_t MemCounter::MemUsageCount = 0;