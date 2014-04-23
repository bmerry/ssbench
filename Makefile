NVCC = nvcc
NVCCFLAGS = -arch sm_20
CXX = g++
CXXFLAGS = -fopenmp -Wall -Wno-unused-local-typedefs -g -O3 -std=c++11 -I$(HOME)/src/compute/include -I$(HOME)/devel/vexcl
LDFLAGS = -lboost_system -lclogs -lOpenCL -Xcompiler -fopenmp

scanbench: scanbench.o scanbench_cuda.o Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ scanbench.o scanbench_cuda.o $(LDFLAGS)
	# $(CXX) -o $@ scanbench.o $(LDFLAGS)

scanbench.o: scanbench.cpp
	$(CXX) -c $< $(CXXFLAGS)

scanbench_cuda.o: scanbench_cuda.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f scanbench
