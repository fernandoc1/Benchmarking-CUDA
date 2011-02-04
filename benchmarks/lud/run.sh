#!/bin/bash
source ../../../common.all

if [ ! -f lud ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi
cp compute_13_o* results/
for i in 1 2 3 4 7 15 16 31 32 53 61 64 127 128 191 192 200 255 256 321 389 480 512 611 765 768; do
	file="$i.dat";
	echo "$i" >> $file;
	head -$i input.dat | cut -d ' ' -f 1-$i >> $file;
	mkdir results/$i
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./lud -i $file
	mv prof.*.txt results/$i;
	rm -f $file
done
../../../compress.sh $(pwd);
