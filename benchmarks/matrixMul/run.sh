#!/bin/bash
source ../../../common.all

if [ ! -f matrixMul ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

for S in {1..10}; do
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./matrixMul --sizemult=$S;
	mkdir -p results/$S;
	mv prof.*.txt results/$S/;
done
cp compute_13_o* results/
../../../compress.sh $(pwd);
