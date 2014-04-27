USE_THRUST ?= 1
USE_CLOGS ?= 1
USE_VEX ?= 1
USE_COMPUTE ?= 1
USE_CPU ?= 1

VEX_HOME ?= $(HOME)/devel/vexcl
COMPUTE_HOME ?= $(HOME)/src/compute

CXX = g++
NVCC = nvcc
NVCCFLAGS = -arch sm_20
CXXFLAGS = -g -O3 -D__CL_ENABLE_EXCEPTIONS -Wall -Wno-unused-local-typedefs -std=c++11
LDFLAGS = -g -lboost_program_options

CXX_SOURCES = scanbench.cpp

ifeq ($(USE_THRUST),1)
    NVCC = nvcc
    LDFLAG_PREFIX = -Xcompiler
    LINK = $(NVCC) $(NVCCFLAGS)
else
    LDFLAG_PREFIX =
    LINK = $(CXX)
endif

ifeq ($(USE_CLOGS),1)
    LDFLAGS += -lclogs -lOpenCL
    CXX_SOURCES += scanbench_clogs.cpp
endif

ifeq ($(USE_VEX),1)
    CXXFLAGS += -I$(VEX_HOME)
    LDFLAGS += -lclogs -lboost_system -lOpenCL
    CXX_SOURCES += scanbench_vex.cpp
endif

ifeq ($(USE_COMPUTE),1)
    CXXFLAGS += -I$(COMPUTE_HOME)/include
    LDFLAGS += -lOpenCL
    CXX_SOURCES += scanbench_compute.cpp
endif

ifeq ($(USE_THRUST),1)
    CXX_SOURCES += scanbench_thrust_register.cpp
    CU_SOURCES += scanbench_thrust.cu
endif

ifeq ($(USE_CPU),1)
    CXX_SOURCES += scanbench_cpu.cpp
    CXXFLAGS += -fopenmp
    LDFLAGS += $(LDFLAG_PREFIX) -fopenmp
endif

OBJECTS = $(patsubst %.cpp, %.o, $(CXX_SOURCES)) $(patsubst %.cu, %.o, $(CU_SOURCES))

scanbench: $(OBJECTS) Makefile
	$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp $(wildcard *.h)
	$(CXX) -c $< $(CXXFLAGS)

%.o: %.cu $(wildcard *.h)
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f scanbench $(OBJECTS)
