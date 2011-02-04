#! /usr/bin/env python

import dircache

benchmarkFolder=dircache.listdir('../benchmarks/')
for benchmark in benchmarkFolder:
	print benchmark
	

