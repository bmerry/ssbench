CXX = g++
CXXFLAGS = -fopenmp -Wall -Wno-unused-local-typedefs -g -O0 -std=c++11 -I$(HOME)/src/compute/include -I$(HOME)/devel/vexcl
LDFLAGS = -fopenmp -lOpenCL -lboost_system -lclogs

scanbench: scanbench.cpp Makefile
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f scanbench
