NVCXX = nvcc
NVCXX_STANDARD = -std c++20
NVCXX_OPTFLAGS = -O3 --lto
NVCXX_WFLAGS = --restrict --Wreorder --Wdefault-stream-launch --Wmissing-launch-bounds --Wext-lambda-captures-this -Werror all-warnings
NVCXX_MFLAGS = --m64 -arch compute_89
NVCXX_DFLAGS = -DSCALAR_PRECISION_64 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19

CXX = g++
CXX_STANDARD = -std=c++20
CXX_OPTFLAGS = -O3 -flto -funroll-loops
CXX_MFLAGS = -march=native -mtune=native
CXX_WFLAGS = -Wall -Wextra -Wpedantic -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings
CXX_DFLAGS = -DSCALAR_PRECISION_64 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19

MPI_INCLUDE_DIR = $(HOME)/opt/openMPI/include
MPI_LIB_DIR = $(HOME)/opt/openMPI/lib

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib

NVCXX_FLAGS = $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_DFLAGS)
CXX_FLAGS = $(CXX_STANDARD) $(CXX_OPTFLAGS) $(CXX_MFLAGS) $(CXX_WFLAGS) $(CXX_DFLAGS)

default:
	make clean
	$(NVCXX) $(NVCXX_FLAGS) mbLBM.cu -o mbLBM -L$(CUDA_LIB_DIR) -L$(MPI_LIB_DIR) -lmpi -lm

install:
	make clean
	$(NVCXX) $(NVCXX_FLAGS) mbLBM.cu -o mbLBM -L$(CUDA_LIB_DIR) -L$(MPI_LIB_DIR) -lmpi -lm
	rm -rf bin/mbLBM
	cp -rf mbLBM bin/mbLBM
	rm -rf mbLBM

clean:
	rm -rf mbLBM

uninstall:
	rm -rf bin/mbLBM