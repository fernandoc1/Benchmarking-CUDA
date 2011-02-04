#!/bin/bash
source ../../../common.all

if [ -z "$@" ]; then
	echo "No input";
	exit 2;
fi

if [ ! -f "$@" ]; then
	make;
fi

if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

for i in compute_13_ins*; do
	code="$(echo $i | sed 's/^.*ins_//')";
	cp $i $@.devcode/$code/compute_13;
done

LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./$@;
mkdir -p results/ori;
mv prof.*.txt results/ori/;

for i in compute_13_opt*; do
	code="$(echo $i | sed 's/^.*opt_//')";
	cp $i $@.devcode/$code/compute_13;
done

LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./$@;
mkdir -p results/opt;
mv prof.*.txt results/opt/;
