#!/bin/bash
source ../../../common.all
if [ ! -f histogram ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./histogram;
mv prof.*.txt results/;
cp compute_13_o* results/
