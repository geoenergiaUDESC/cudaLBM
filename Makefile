NVCXX = nvcc
NVCXX_STANDARD = -std c++20
NVCXX_OPTFLAGS = -O3 --restrict
# NVCXX_OPTFLAGS = -O3
# NVCXX_WFLAGS = --Wreorder --Wdefault-stream-launch --Wmissing-launch-bounds --Wext-lambda-captures-this -Werror all-warnings
NVCXX_WFLAGS = --Wreorder --Wdefault-stream-launch --Wmissing-launch-bounds --Wext-lambda-captures-this
NVCXX_COMPILER_WFLAGS = -Xcompiler "-O3 -funroll-loops -march=native -mtune=native -Wall -Wextra -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wshadow -Wsign-conversion -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings"
# NVCXX_COMPILER_WFLAGS = -Xcompiler "-Wall -Wextra -Wpedantic -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings"

# Define compute capability
GPU_ARCH_CODE := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//' | sort -n | tail -1)
ifeq ($(GPU_ARCH_CODE),)
    GPU_ARCH_CODE := 86  # Default value if it fails
endif

NVCXX_MFLAGS = --m64 -arch compute_$(GPU_ARCH_CODE)

NVCXX_DFLAGS = -DSCALAR_PRECISION_32 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19

CXX = g++
CXX_STANDARD = -std=c++20
CXX_OPTFLAGS = -O3 -flto -funroll-loops 
CXX_MFLAGS = -march=native -mtune=native
CXX_WFLAGS = -Wall -Wextra -Wpedantic -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings
CXX_DFLAGS = -DSCALAR_PRECISION_32 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19

# MPI_INCLUDE_DIR = $(MPI_DIR)/include
# MPI_LIB_DIR = $(MPI_DIR)/lib

# CUDA_INCLUDE_DIR = $(CUDA_DIR)/include
# CUDA_LIB_DIR = $(CUDA_DIR)/lib

NVCXX_FLAGS = $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_COMPILER_WFLAGS) $(NVCXX_DFLAGS)
CXX_FLAGS = $(CXX_STANDARD) $(CXX_OPTFLAGS) $(CXX_MFLAGS) $(CXX_WFLAGS) $(CXX_DFLAGS)

default:
	make clean
	$(NVCXX) $(NVCXX_FLAGS) LBM.cu -o momentBasedD3Q19

install:
	make clean
	make uninstall
	$(NVCXX) $(NVCXX_FLAGS) LBM.cu -o momentBasedD3Q19
	cp -rf momentBasedD3Q19 build/bin/momentBasedD3Q19
	rm -rf momentBasedD3Q19

clean:
	rm -rf momentBasedD3Q19

uninstall:
	rm -rf build/bin/momentBasedD3Q19
