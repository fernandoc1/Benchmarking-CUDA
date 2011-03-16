#! /usr/bin/env python

import sys
import os

print "--- Automated benchmark execution tool ---\n"
if(len(sys.argv)==2):
	numberOfTimes=int(sys.argv[1])
	os.system("./benchmark.py clean")
	os.system("rm -f RESULTS.txt")
	index=0
	while index<numberOfTimes:
		print "-> "+str(index)
		os.system("./benchmark.py all optimized")
		#os.system("./benchmark.py appendResults")
		os.system("./profReader.py "+index)
		os.system("./benchmark.py clean")
		index+=1
else:
	print "Parameters are incorrect."
	print "Please, consult documentation at: https://github.com/fernandoc1/Benchmarking-CUDA/wiki/Benchmark-Tool"
