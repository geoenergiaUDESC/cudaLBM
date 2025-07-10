# common.mk - Shared CUDA/C++ flags
NVCXX = nvcc
CXX = g++

# Auto-detect GPU architecture
GPU_ARCH_CODE := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//' | sort -n | tail -1)
ifeq ($(GPU_ARCH_CODE),)
    GPU_ARCH_CODE := 86  # Fallback to default
endif

# Common build directories (relative to project root)
BUILD_DIR = ../build
BIN_DIR = $(BUILD_DIR)/bin

# CUDA Compiler Flags
NVCXX_STANDARD = -std c++20
NVCXX_OPTFLAGS = -O3 --restrict
NVCXX_MFLAGS = --m64 -arch compute_$(GPU_ARCH_CODE)
NVCXX_WFLAGS = --Wreorder --Wdefault-stream-launch --Wmissing-launch-bounds --Wext-lambda-captures-this -Xptxas -v
NVCXX_COMPILER_WFLAGS = -Xcompiler "-O3 -funroll-loops -march=native -mtune=native -Wall -Wextra -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wshadow -Wsign-conversion -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings"
NVCXX_DFLAGS = -DSCALAR_PRECISION_32 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19
NVCXX_FLAGS = $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_COMPILER_WFLAGS) $(NVCXX_DFLAGS)

# C++ Compiler Flags
CXX_STANDARD = -std=c++20
CXX_OPTFLAGS = -O3 -flto -funroll-loops 
CXX_MFLAGS = -march=native -mtune=native
CXX_WFLAGS = -Wall -Wextra -Wpedantic -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings
CXX_DFLAGS = NVCXX_DFLAGS
CXX_FLAGS = $(CXX_STANDARD) $(CXX_OPTFLAGS) $(CXX_MFLAGS) $(CXX_WFLAGS) $(CXX_DFLAGS)