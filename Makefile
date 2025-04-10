NVCXX = nvcc
NVCXXSTANDARD = -std c++20
OPTFLAGS = -O3 --lto
WFLAGS = --restrict --Wreorder --Wdefault-stream-launch --Wmissing-launch-bounds --Wext-lambda-captures-this -Werror all-warnings
MFLAGS = --m64 -arch compute_89
DFLAGS = -DSCALAR_PRECISION_64 -DLABEL_SIZE_64 -DSTENCIL_TYPE_D3Q19

NVCXXFLAGS = $(NVCXXSTANDARD) $(OPTFLAGS) $(MFLAGS) $(WFLAGS) $(DFLAGS)

default:
	make clean
	$(NVCXX) $(NVCXXFLAGS) cudaLBM.cu -o cudaLBM

clean:
	rm -rf cudaLBM