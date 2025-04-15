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

NVCXXFLAGS = $(NVCXXSTANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_DFLAGS)
CXXFLAGS = $(CXXSTANDARD) $(CXX_OPTFLAGS) $(CXX_MFLAGS) $(CXX_WFLAGS) $(CXX_DFLAGS)

default:
	make clean
	$(NVCXX) $(NVCXXFLAGS) mbLBM.cu -o mbLBM

install:
	make clean
	$(NVCXX) $(NVCXXFLAGS) mbLBM.cu -o mbLBM
	rm -rf bin/mbLBM
	cp -rf mbLBM bin/mbLBM
	rm -rf mbLBM

clean:
	rm -rf mbLBM

uninstall:
	rm -rf bin/mbLBM