USE_CUDA ?= 1
USE_CLOGS ?= 1
USE_VEX ?= 1
USE_COMPUTE ?= 1
USE_CPU ?= 1

VEX_HOME ?= $(HOME)/devel/vexcl
COMPUTE_HOME ?= $(HOME)/src/compute

CXX = g++
NVCC = nvcc
NVCCFLAGS = -arch sm_20
CXXFLAGS = -g -D__CL_ENABLE_EXCEPTIONS -Wall -Wno-unused-local-typedefs -O3 -std=c++11
LDFLAGS = -g

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
    CXXFLAGS += -DUSE_CLOGS=1
endif

ifeq ($(USE_VEX),1)
    CXXFLAGS += -I$(VEX_HOME) -DUSE_VEX=1
    LDFLAGS += -lboost_system -lOpenCL
    CXX_SOURCES += scanbench_vex.cpp
endif

ifeq ($(USE_COMPUTE),1)
    CXXFLAGS += -I$(COMPUTE_HOME)/include -DUSE_COMPUTE=1
    LDFLAGS += -lOpenCL
    CXX_SOURCES += scanbench_compute.cpp
endif

ifeq ($(USE_CUDA),1)
    CXXFLAGS += -DUSE_CUDA=1
    CU_SOURCES += scanbench_cuda.cu
endif

ifeq ($(USE_CPU),1)
    CXX_SOURCES += scanbench_cpu.cpp
    CXXFLAGS += -fopenmp -DUSE_CPU=1
    LDFLAGS += $(LDFLAG_PREFIX) -fopenmp
endif

OBJECTS = $(patsubst %.cpp, %.o, $(CXX_SOURCES)) $(patsubst %.cu, %.o, $(CU_SOURCES))

scanbench: $(OBJECTS) Makefile
	$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS)

scanbench.o: $(wildcard *.h) Makefile

%.o: %.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f scanbench $(OBJECTS)
