USE_CUDA ?= 1
USE_CLOGS ?= 1
USE_VEX ?= 1
USE_COMPUTE ?= 1

VEX_HOME ?= $(HOME)/devel/vexcl
COMPUTE_HOME ?= $(HOME)/src/compute

CXX = g++
NVCC = nvcc
NVCCFLAGS = -arch sm_20
CXXFLAGS = -g -D__CL_ENABLE_EXCEPTIONS -fopenmp -Wall -Wno-unused-local-typedefs -O3 -std=c++11
LDFLAGS = -g $(LDFLAG_PREFIX) -fopenmp

CXX_SOURCES = scanbench.cpp

ifeq ($(USE_CUDA),1)
    NVCC = nvcc
    NVCCFLAGS = -arch sm_20
    LDFLAG_PREFIX = -Xcompiler
    LINK = $(NVCC) $(NVCCFLAGS)
else
    LDFLAG_PREFIX =
    LINK = $(CXX)
endif

ifeq ($(USE_CLOGS),1)
    LDFLAGS += -lclogs
    CXX_SOURCES += scanbench_clogs.cpp
endif

ifeq ($(USE_VEX),1)
    CXXFLAGS += -I$(VEX_HOME)
    LDFLAGS += -lboost_system -lOpenCL
    CXX_SOURCES += scanbench_vex.cpp
endif

ifeq ($(USE_COMPUTE),1)
    CXXFLAGS += -I$(COMPUTE_HOME)/include
    LDFLAGS += -lOpenCL
    CXX_SOURCES += scanbench_compute.cpp
endif

ifeq ($(USE_CUDA),1)
    CU_SOURCES += scanbench_cuda.cu
endif

OBJECTS = $(patsubst %.cpp, %.o, $(CXX_SOURCES)) $(patsubst %.cu, %.o, $(CU_SOURCES))

scanbench: $(OBJECTS) Makefile
	$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS)

scanbench.o: $(wildcard *.h)

%.o: %.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f scanbench $(OBJECTS)
