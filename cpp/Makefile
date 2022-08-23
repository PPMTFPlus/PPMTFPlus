CXXFLAGS	= -march=native -O3
CXXFLAGS	+= -std=c++11
CXXFLAGS	+= -isystem ./include
CXXFLAGS	+= -MMD
CC	= g++

TARGET	=  PPMTF SynData_PPMTF SynData_SGD

all: $(TARGET)

t: EvalPPITFError
	./$<

t0: SynData_SGD.0
	./$< PF TK 10 5

PPMTF: PPMTF.o PPXTF.o common.o

SynData_PPMTF: SynData_PPMTF.o common.o

SynData_SGD: SynData_SGD.o common.o

clean:
	rm -f *~ $(TARGET) *.d *.o

-include *.d
