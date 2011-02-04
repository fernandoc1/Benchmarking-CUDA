#!/bin/bash
source ../../../common.all

if [ ! -f needle ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

cp compute_13_o* results/

for i in 16 32 48 64 240 480 512 1024; do
	for j in 7 16 32 48 64 240 479 480 511 512 1024 65536; do
		LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./needle $i $j;
		echo $i $j
		mkdir -p results/$i/$j;
		mv prof.*.txt results/$i/$j/;
	done;
done;

../../../compress.sh $(pwd);
