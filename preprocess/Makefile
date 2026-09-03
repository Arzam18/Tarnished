HOST_CXX ?= g++
CXXFLAGS := -O3 -std=c++20 -pthread -DNDEBUG

ARCH_LEVEL ?= native
ifeq ($(ARCH_LEVEL),v3)
	PERMUTE_FLAGS := -DUSE_AVX2
else ifeq ($(ARCH_LEVEL),v4)
	PERMUTE_FLAGS := -DUSE_AVX512
else ifeq ($(ARCH_LEVEL),arm64)
	PERMUTE_FLAGS := -DUSE_NEON
else ifeq ($(ARCH_LEVEL),neon)
	PERMUTE_FLAGS := -DUSE_NEON
else
	PERMUTE_FLAGS := -march=native
endif

EXE_SUFFIX =
ifeq ($(OS),Windows_NT)
	EXE_SUFFIX := .exe
endif

all: permute$(EXE_SUFFIX)

permute$(EXE_SUFFIX): permute.cpp
	$(HOST_CXX) $(CXXFLAGS) $(PERMUTE_FLAGS) permute.cpp -o permute$(EXE_SUFFIX)

clean:
	rm -f permute permute.exe
