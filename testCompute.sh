make install

cd lidDrivenCavity

rm -rf *.dat

compute-sanitizer --print-limit 1000 --tool=memcheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 0
#compute-sanitizer --print-limit 1000 --tool=racecheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 0
compute-sanitizer --print-limit 1000 --tool=synccheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 0
compute-sanitizer --print-limit 1000 --tool=initcheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 0

compute-sanitizer --print-limit 1000 --tool=memcheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 1
#compute-sanitizer --print-limit 1000 --tool=racecheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 1
compute-sanitizer --print-limit 1000 --tool=synccheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 1
compute-sanitizer --print-limit 1000 --tool=initcheck --check-bulk-copy=yes --check-device-heap=yes --check-exit-code=yes --racecheck-detect-level=error --racecheck-memcpy-async=yes --racecheck-num-workers=0 --racecheck-report=analysis --check-optix-leaks --check-warpgroup-mma=yes --check-api-memory-access=yes --check-optix --track-unused-memory momentBasedD3Q19 -GPU 1

cd ../