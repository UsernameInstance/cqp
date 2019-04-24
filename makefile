#Source directory
SDIR := ./

#Source file extension
SEXT := cxx

#Object directory
ODIR := object

$(shell mkdir -p $(SDIR); mkdir -p $(ODIR);) 

#Source files
SFILES = $(wildcard $(SDIR)/*.$(SEXT))

#Object files
OFILES = $(patsubst $(SDIR)/%.$(SEXT),$(ODIR)/%.o,$(SFILES))

#Executable file name 
EXE = cqp_tests 

#C++ compiler
CXX = g++

#C++ COMPILER_FLAGS 
CXXFLAGS = 

#Include Paths
IPATHS = -I/cygdrive/e/cygwin64/usr/include/eigen3

#Library Paths
LPATHS =

#Linker Flags
LFLAGS = -lgmpxx -lgmp

.PHONY: all
all: $(OFILES) 
	$(CXX) -g $(OFILES) $(LPATHS) $(LFLAGS) -o $(EXE)

$(ODIR)/%.o: $(SDIR)/%.$(SEXT) test_problem.hxx cqp.hxx misc_tests.hxx test_problem_general_form.hxx
	$(CXX) -c $< $(CXXFLAGS) $(IPATHS) -o $@

.PHONY: debug release
debug: CXXFLAGS += -Og -g -Wall
release: CXXFLAGS += -O3
debug release: all

.PHONY: clean
clean: 
	rm $(ODIR)/*.o
