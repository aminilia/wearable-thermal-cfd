# make cpu          -> bin/gputherm_cpu   (g++, OpenMP)
# make cuda         -> bin/gputherm_cuda  (nvcc; ARCH=sm_75 for a Colab/Kaggle T4,
#                                           sm_80 A100, sm_86 RTX 30xx, sm_89 L4/RTX 40xx)
# make all          -> both
CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= sm_75
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
NVFLAGS  ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
INC       = -Iinclude
HDR       = include/gputherm/problem.hpp

.PHONY: all cpu cuda clean
all: cpu cuda
cpu: bin/gputherm_cpu
cuda: bin/gputherm_cuda

bin/gputherm_cpu: src/cpu_main.cpp $(HDR)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

bin/gputherm_cuda: src/cuda_main.cu $(HDR)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $(INC) -o $@ $<

clean:
	rm -rf bin
