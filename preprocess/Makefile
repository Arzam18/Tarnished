CXX ?= clang++
CXXFLAGS := -O3 -std=c++20 -pthread -DNDEBUG

ARCH_LEVEL ?= native
ifeq ($(ARCH_LEVEL),native)
	ARCH := -march=native
else ifeq ($(ARCH_LEVEL),v3)
	ARCH := -march=x86-64-v3 -static
else ifeq ($(ARCH_LEVEL),v4)
	ARCH := -march=x86-64-v4 -static
else ifeq ($(ARCH_LEVEL),arm64)
	ARCH := -march=armv8-a+dotprod+crypto
else ifeq ($(ARCH_LEVEL),neon)
	ARCH := -march=armv8-a
endif

EXE_SUFFIX =
ifeq ($(OS),Windows_NT)
	EXE_SUFFIX := .exe
endif

all: permute$(EXE_SUFFIX)

permute$(EXE_SUFFIX): permute.cpp
	$(CXX) $(CXXFLAGS) $(ARCH) permute.cpp -o permute$(EXE_SUFFIX)

clean:
	rm -f permute permute.exe
