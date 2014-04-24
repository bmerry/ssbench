NVCC = nvcc
NVCCFLAGS = -arch sm_20
CXX = g++
CXXFLAGS = -g -D__CL_ENABLE_EXCEPTIONS -fopenmp -Wall -Wno-unused-local-typedefs -O3 -std=c++11 -I$(HOME)/src/compute/include -I$(HOME)/devel/vexcl
LDFLAGS = -g -lboost_system -lclogs -lOpenCL -Xcompiler -fopenmp
CXX_SOURCES = scanbench.cpp scanbench_vex.cpp scanbench_compute.cpp scanbench_clogs.cpp
CU_SOURCES = scanbench_cuda.cu
OBJECTS = $(patsubst %.cpp, %.o, $(CXX_SOURCES)) $(patsubst %.cu, %.o, $(CU_SOURCES))

scanbench: $(OBJECTS) Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)
	# $(CXX) -o $@ scanbench.o $(LDFLAGS)

%.o: %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS)

scanbench.o: $(wildcard *.h)

%.o: %.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f scanbench $(OBJECTS)
