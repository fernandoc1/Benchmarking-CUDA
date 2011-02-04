#!/bin/bash
source ../../../common.all

if [ ! -f hotspot ]; then
	make;
fi
if [ -d results ]; then
	rm -rf results/*;
else
	mkdir results;
fi;

for i in compute_13_o*; do
	sed 's/cvt.s32.s8/cvt.s32.s16/g' $i > results/$i;
done;

for i in temp_*; do
	size="$(echo $i | cut -d _ -f 2)";

	for rc in 1 2 3 15 256 480 512; do
		if [ $ph -gt $size ]; then
			continue;
		fi
		for ph in 1 7 128 256 480 512; do

			for sim in 1 10 100; do
				LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PROF_BRANCHES=${PROF_BRANCHES} ./hotspot $rc $ph $sim temp_$size power_$size /dev/null
				mkdir -p results/$size/$rc/$ph/$sim;
				a=prof.*.txt;
				if [ "$a" != "" ]; then
					mv prof.*.txt results/$size/$rc/$ph/$sim/;
				fi
			done
		done
	done
done

../../../compress.sh $(pwd);
