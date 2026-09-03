# sp yoink
EXE_SUFFIX =
LDFLAGS = -fuse-ld=lld
SOURCES := $(wildcard src/*.cpp)
SOURCES += src/external/format.cpp
_ROOT := $(_THIS)
CXX := clang++

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
else
	$(error Invalid ARCH_LEVEL: $(ARCH_LEVEL). Use native, v3, v4, arm64, or neon)
endif

ifeq ($(OS),Windows_NT)
	STACK_FLAGS := -Wl,/STACK:8388608
	RM = del /Q /F
	EXE_SUFFIX := .exe
	fixpath = $(subst /,\,$1)
else
	STACK_FLAGS :=
	RM = rm -f
	EXE_SUFFIX = 
	fixpath = $1
endif

cat := $(if $(filter $(OS),Windows_NT),type,cat)
DEFAULT_NET := $(shell $(cat) network.txt)
EVALFILE_PROCESSED = processed.bin

ifndef EVALFILE
	EVALFILE = $(DEFAULT_NET).bin
	NO_EVALFILE_SET = true
endif

ifeq ($(NO_EVALFILE_SET),)
$(EVALFILE):
	@true
endif

CXXFLAGS := -O3 $(ARCH) -fno-finite-math-only -funroll-loops -flto -fuse-ld=lld -std=c++20 -DNDEBUG -pthread -DEVALFILE=\"$(EVALFILE_PROCESSED)\"

ifdef NO_EVALFILE_SET
$(EVALFILE):
	$(info Downloading default network $(DEFAULT_NET).bin)
	curl -sOL https://github.com/Bobingstern/tarnished-nets/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).bin
download-net: $(EVALFILE)
endif

.DEFAULT_GOAL := native

ifndef EXE
	EXE = tarnished$(EXE_SUFFIX)
endif

$(EVALFILE_PROCESSED):
	$(MAKE) -C preprocess ARCH_LEVEL=$(ARCH_LEVEL)
	./preprocess/permute$(EXE_SUFFIX) $(EVALFILE) $(EVALFILE_PROCESSED)
	-$(RM) $(call fixpath,preprocess/permute$(EXE_SUFFIX))

.NOTPARALLEL: $(EVALFILE_PROCESSED)

$(EXE): $(EVALFILE) $(EVALFILE_PROCESSED) $(SOURCES) 
	$(CXX) $(CXXFLAGS) $(STACK_FLAGS) $(LDFLAGS) $(SOURCES) -o $@
	-$(RM) $(call fixpath,$(EVALFILE_PROCESSED))

native: $(EXE)

avx2:
	$(MAKE) ARCH_LEVEL=v3

avx512:
	$(MAKE) ARCH_LEVEL=v4

arm64:
	$(MAKE) ARCH_LEVEL=arm64

neon:
	$(MAKE) ARCH_LEVEL=neon
