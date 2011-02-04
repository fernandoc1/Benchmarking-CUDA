#!/usr/bin/env python

import sys

# input parameter
inputFileName = sys.argv[1]

endOfFile = False
inputFile = file(inputFileName)


# serarch for another kernel or the end of file
endOfFile = True
print "#include \"prof.h\""
print "#include \"prof.cu\""
for line in inputFile:
	line = line.rstrip()
	idx = line.find(">>>")
	if (idx > 0):
		nameAndConfiguration = line[:idx]

		idx = nameAndConfiguration.find("<<<")
		name = nameAndConfiguration[:idx]
		configuration = nameAndConfiguration[idx+3:]

		name = name.strip()

		print "	GpuProfiling::prepareProfiling(", configuration, ");"
		print line
		print "	GpuProfiling::addResults(\"%s\");" % name
	else:
		print line
#	idMain = line.find("main")
#	if (idMain > 0):
#		print line
#		print "GpuProfiling::initProf();"
		

inputFile.close()

