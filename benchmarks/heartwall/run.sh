#!/bin/bash
source ../../../common.all

if [ ! -f heartwall ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

cp compute_13_o* results/
LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./heartwall test.avi 104
mv prof.*.txt results;

