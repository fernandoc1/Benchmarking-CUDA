#!/bin/bash
source ../../../common.all

if [ ! -f fastWalshTransform ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;
set -x
LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./fastWalshTransform
cp compute_13_o* results/
sleep 1
cp prof.*.txt results/
set +x
../../../compress.sh $(pwd);
