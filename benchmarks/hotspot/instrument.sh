#!/bin/bash

#Path to the iptx binary
IPTX="iptx"

####
# search $HOME if iptx file not found
###

if [ ! -f $IPTX ]; then
	echo "File $IPTX not found, searching another one within $HOME..."
	for i in $(find ~/ -name iptx);
	do
		if [ "$(file $i | grep ELF)" != "" ];
		then
			echo "Using file $i";
			IPTX=$i;
			break;
		fi;
	done;
fi;
if [ ! -f $IPTX ]; then
	echo "No iptx found";
	exit 1;
fi
 
set -x
# -e indicates that you are getting number of execution cycles.
if [ -z $INSTRUMENTATION ]; then
	INSTRUMENTATION="-p";
fi

# Name of the executable and folder program:
#EXEC_NAME=$@

# extract ptx

set -x
rm -f sm_* compute_*
for codeDir in *.devcode/*
do
  suffix=${codeDir#*\.devcode/}
  file="$(ls ${codeDir}/compute* | awk -F'/' '{print $NF}')"
  arch="$(echo ${file} | cut -d '_' -f 2)"
  sed 's/cvt.s32.s8/cvt.s32.s16/g' ${codeDir}/${file} > ${file}_ori_${suffix}
  $IPTX $INSTRUMENTATION -i ${file}_ori_${suffix} -o ${file}_ins_${suffix}
  if [[ $? && -f "${file}_ins_${suffix}" ]]; then
    ptxas -O4 -arch sm_$arch ${file}_ins_${suffix} -o sm_${arch}_${suffix}
    cp ${file}_ins_${suffix} ${codeDir}/compute_${arch}
    cp sm_${arch}_${suffix} ${codeDir}/sm_${arch}
  fi
done
set +x
# remove temporary files
# rm -f compute_13_*
