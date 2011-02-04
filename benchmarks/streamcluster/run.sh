#!/bin/bash
source ../../../common.all

if [ ! -f sc ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi
LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./sc 10 20 255 512 512 100 none output.txt 1
rm -rf results;
mkdir results;
mv prof.*.txt results/;
cp compute_13_o* results/;
