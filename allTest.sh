cd meshTools
cp ../lidDrivenCavity/caseInfo caseInfo
make
./meshTools
cp mesh ../lidDrivenCavity/mesh
rm -rf caseInfo
rm -rf mesh

cd ../
make install

cd lidDrivenCavity

# compute-sanitizer --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory mbLBM -GPU 0
compute-sanitizer --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory mbLBM -GPU 0,1
# compute-sanitizer --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory mbLBM -GPU 0,1,2

cd ../