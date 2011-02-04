#!/bin/bash
source ../../../common.all

if [ ! -f srad ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

for i in 16 32 48 64 240 256 480 512 ; do
	for j in 16 32 48 64 240 256 480 512; do
		for k in 10 100; do
			echo $i $j $k;
			LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./srad $i $j 0 127 0 127 0.5 $k;
			mkdir -p results/$i/$j/$k;
			mv prof.*.txt results/$i/$j/$k;
		done
	done
done

cp compute_13_o* results/
../../../compress.sh $(pwd);
