#!/bin/bash
source ../../../common.all

if [ ! -f reduction ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi
cp compute_13_o* results/
for size in 512 511 480 11; do
	for number in 1023 511 480; do
		for kernel in {0..6}; do
			echo "$size $kernel $number";
			LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./reduction --threads=$size --kernel=$kernel --n=$number --type=float > /dev/null;
		done;
		mkdir -p results/$size/$number;
		mv prof.*.txt results/$size/$number;
	done;
done
../../../compress.sh $(pwd)
