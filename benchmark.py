#! /usr/bin/env python

import dircache
import os
import sys

#Workaround due to Python not accept static methods.
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class Environment:
	_nvccPath="/usr/local/cuda/bin/nvcc"
	_flags="-arch=compute_13 -int=none -ext=all -DUNIX "
	_includePath="../dependencies/includes/"
	_librariesPath="../dependencies/libraries/"
	_ldLibraryPath="/usr/local/cuda/lib64"
	_pathToBenchmarks="../benchmarks/"
	_profBranches="PROF_BRANCHES=0 PROF_KERNEL_EXEC_TIME_CPU=1"
	_ptxOptimizerPath="../ocelot/PTXOptimizer"
	_resultsFolderName="results"
	
	def getNvccPath():
		return Environment._nvccPath
	
	def getFlags():
		return Environment._flags
	
	def getIncludePath():
		return Environment._includePath
		
	def getLibrariesPath():
		return Environment._librariesPath
	
	def getLdLibraryPathFlag():
		return "LD_LIBRARY_PATH="+Environment._ldLibraryPath+" "
		
	def getBenchmarksPath():
		return Environment._pathToBenchmarks
		
	def getProfBranchesFlag():
		return Environment._profBranches
	
	def getPTXOptimizerPath():
		return Environment._ptxOptmizerPath
		
	def getResultsFolderName():
		return Environment._resultsFolderName
		
	getNvccPath=Callable(getNvccPath)
	getFlags=Callable(getFlags)
	getIncludePath=Callable(getIncludePath)
	getLibrariesPath=Callable(getLibrariesPath)
	getLdLibraryPathFlag=Callable(getLdLibraryPathFlag)
	getBenchmarksPath=Callable(getBenchmarksPath)
	getProfBranchesFlag=Callable(getProfBranchesFlag)
	getPTXOptimizerPath=Callable(getPTXOptimizerPath)
	getResultsFolderName=Callable(getResultsFolderName)

class Analyse:
	def _getProfFiles():
		testsFolderFiles=dircache.listdir(".")
		profFiles=[]
		for entry in testsFolderFiles:
			tmp=entry.split(".")
			if(tmp[0]=="prof"):
				profFiles.append(entry)
		return profFiles

	def _getKernelExecutionTime(profFile):
		dataRead=open(profFile, "r").read()
		#This remove the empty spaces from the string.
		values=dataRead.split()
		return values[0]
	
	def _getKernelName(profFile):
		tokens=profFile.split(".")
		return tokens[1]
	
	def appendResultsToFile():
		resultFile=open('../RESULTS.txt', 'a')
		profFiles=Analyse._getProfFiles()
		resultFile.write("++++\n")
		for entry in profFiles:
			resultFile.write(Analyse._getKernelName(entry)+" "+Analyse._getKernelExecutionTime(entry)+"\n")

	def cleanResultsFolder():
		os.system("rm -rvf ./"+Environment.getResultsFolderName())

	def listKernelsData():
		profFiles=Analyse._getProfFiles()
		index=0
		for entry in profFiles:
			index+=1
			print str(index)+"-> "+Analyse._getKernelName(entry)+": "+Analyse._getKernelExecutionTime(entry)+" microsseconds"
	
	_getKernelExecutionTime=Callable(_getKernelExecutionTime)
	_getKernelName=Callable(_getKernelName)
	_getProfFiles=Callable(_getProfFiles)
	cleanResultsFolder=Callable(cleanResultsFolder)
	listKernelsData=Callable(listKernelsData)
	appendResultsToFile=Callable(appendResultsToFile)

class Benchmark:
	_pathToBenchmark=""
	_name=""
	_file=[]
	_files=""
	_libs=""
	_runString=""
	
	def _getFiles(self):
		fileOfFiles=open(self._pathToBenchmark+"/FILES", "r")
		self._files=fileOfFiles.read()
		self._file=self._files.split()
		self._files=""
		for entry in self._file:
			self._files+=self._pathToBenchmark+"/"+entry+" "
		
	def _getLibs(self):
		fileOfLibs=open(self._pathToBenchmark+"/LIBS", "r")
		self._libs=fileOfLibs.read()
		lib=self._libs.split()
		self._libs=""
		for entry in lib:
			self._libs+=entry+" "
		
	def _setName(self):
		path=self._pathToBenchmark
		folders=path.split("/")
		self._name=folders.pop()
		
	def _setRunString(self):
		fileRUN=open(self._pathToBenchmark+"/RUN", "r")
		self._runString=fileRUN.read()
		self._runString=Environment.getLdLibraryPathFlag()+" "+Environment.getProfBranchesFlag()+" "+self._runString
		
	def _setupData(self):
		pathToDataFolder=self._pathToBenchmark+"/data"
		if(os.path.isdir(pathToDataFolder)):
			os.system("cp -rf "+pathToDataFolder+" .")
			print "Data copied successfully for \""+self._name+"\"."
		else:
			print "\""+self._name+"\" has no data."
	
	def __init__(self, absolutePath):
		try:
			self._pathToBenchmark=absolutePath
			self._getFiles()
			self._getLibs()
			self._setName()
			self._setRunString()
			self._setupData()
		except:
			raise
		
	def _getDevcodeDir(self):
		return self._name+".devcode"

	def _getCompileString(self):
		compileString=Environment.getNvccPath()
		compileString+=" "+Environment.getFlags()
		compileString+=" -dir="+self._getDevcodeDir()+" "
		compileString+=" -I"+Environment.getIncludePath()+" "
		compileString+=" -I"+self._pathToBenchmark+" "
		compileString+=" -L"+Environment.getLibrariesPath()+" "
		compileString+=" "+self._libs+" "
		compileString+=" "+self._files+" "
		compileString+=" -o "+self._name
		return compileString
	
	def build(self):
		os.system(self._getCompileString())
		
	def run(self):
		os.system(self._runString)
	
	def clean(self):
		os.system("rm -f "+self._name)
		os.system("rm -rf "+self._getDevcodeDir())
		os.system("rm -rf data")
	
	def _getListOfComputeFolders(self):
		computeFolders=[]
		devcodeDir=self._getDevcodeDir()
		computeFolderEntries=dircache.listdir(devcodeDir)
		for entry in computeFolderEntries:
			computeFolders.append(devcodeDir+"/"+entry)
		return computeFolders
	
	def optimize(self, optimizeParameters):
		computeFolders=self._getListOfComputeFolders()
		for entry in computeFolders:
			compute13File=entry+"/compute_13"
			ptxOptimizerString="../ocelot/PTXOptimizer "+optimizeParameters+" -i "+compute13File+" -o "+compute13File+".optimized"
			print "Optimizing code: "+ptxOptimizerString
			os.system(ptxOptimizerString)
			os.system("rm -v "+compute13File)
			os.system("mv -v "+compute13File+".optimized "+compute13File)
		
	def printProperties(self):
		print "Properties for \""+self._name+"\""
		print "Path -> "+self._pathToBenchmark
		print "Files to compile -> "+self._files
		print "Libraries -> "+self._libs
		print "Compile string -> "+self._getCompileString()
		print "Run string -> "+self._runString


def printListOfBenchmarks():
	benchmarksEntries=dircache.listdir(Environment.getBenchmarksPath())
	index=0
	for benchmarkEntry in benchmarksEntries:
		print str(index)+"-> "+benchmarkEntry
		index+=1

def runBenchmark(benchmarkName):
	benchmark=Benchmark("../benchmarks/"+benchmarkName)
	print ">>> Running \""+benchmarkName+"\" ..."
	benchmark.build()
	benchmark.run()
	benchmark.clean()
	print ">>> \""+benchmarkName+"\" execution done."

def runBenchmarkOptimized(benchmarkName, optimizationString):
	benchmark=Benchmark("../benchmarks/"+benchmarkName)
	print ">>> Running \""+benchmarkName+"\" ..."
	benchmark.build()
	benchmark.optimize(optimizationString)
	benchmark.run()
	benchmark.clean()
	print ">>> \""+benchmarkName+"\" execution done."

def runAllBenchmarks():
	benchmarksEntries=dircache.listdir(Environment.getBenchmarksPath())
	index=0
	for benchmarkEntry in benchmarksEntries:
		try:
			runBenchmark(benchmarkEntry)
		except:
			print "An error has ocurred while trying to run \""+benchmarkEntry+"\"."
			print "Skipping..."

def runAllBenchmarksOptimized(optimizationString):
	if(optimizationString==""):
		print "Using standard PTXOptimizer settings"
	else:
		print "Using \""+optimizationString+"\" as PTXOptimizer settings"
	benchmarksEntries=dircache.listdir(Environment.getBenchmarksPath())
	index=0
	for benchmarkEntry in benchmarksEntries:
		try:
			runBenchmarkOptimized(benchmarkEntry, optimizationString)
		except:
			print "An error has ocurred while trying to run \""+benchmarkEntry+"\"."
			print "Skipping..."
		
def showProperties(benchmarkName):
	benchmark=Benchmark("../benchmarks/"+benchmarkName)
	benchmark.printProperties()

def executeArgs():
	if(len(sys.argv)==2):
		if(sys.argv[1]=="list"):
			printListOfBenchmarks()
		elif(sys.argv[1]=="all"):
			runAllBenchmarks()
		elif(sys.argv[1]=="kernelsData"):
			Analyse.listKernelsData()
		elif(sys.argv[1]=="appendResults"):
			Analyse.appendResultsToFile()
		elif(sys.argv[1]=="clean"):
			Analyse.cleanResultsFolder()
	elif(len(sys.argv)==3):
		if(sys.argv[1]=="run"):
			runBenchmark(sys.argv[2])
		if(sys.argv[1]=="properties"):
			showProperties(sys.argv[2])
		if(sys.argv[1]=="all" and sys.argv[2]=="optimized"):
			runAllBenchmarksOptimized("")
	elif(len(sys.argv)==4):
		if(sys.argv[1]=="all" and sys.argv[2]=="optimized"):
			runAllBenchmarksOptimized(sys.argv[3])
	else:
		print "Please see usage instructions at: https://github.com/fernandoc1/Benchmarking-CUDA/wiki/Benchmark-Tool"
	
def initializeEnvironment():
	resultsFolder=Environment.getResultsFolderName()
	os.system("mkdir -p "+resultsFolder)
	os.chdir("./"+resultsFolder)

initializeEnvironment()
executeArgs()



