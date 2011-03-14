#! /usr/bin/env python

from database import Database
import dircache

class profReader:
	
	_database=None
	_profFolderPath=None
	_regularExpression=""
	_executionCounter=0

	def __init__(self, profFolderPath, regularExpression):
		self._database=Database('results.db')
		self._profFolderPath=profFolderPath
		self._regularExpression=regularExpression

	def _getProfFilesNames(self):
		files=dircache.listdir(self._profFolderPath)
		profFiles=[]
		for entry in files:
			token=entry.split('.')
			if(token[0]=="prof"):
				profFiles.append(entry)
		return profFiles
	
	def incrementExecutionCounter(self):
		self._executionCounter+=1

	def _getKernelName(self, profFileName):
		token=profFileName.split('.')
		return token[1]

	def _insertDataInDatabase(self, data, kernelName):
		label=self._regularExpression.split()
		tokens=data.split()
		index=0
		for entry in tokens:
			print str(index)+" -- "+entry+"->"+label[index]+" kernel="+kernelName
			self._database.insertData(str(self._executionCounter), kernelName, label[index], entry)
			index=(index+1)%len(label)

	def readProfs(self):
		profs=self._getProfFilesNames()
		for entry in profs:
			data=open(self._profFolderPath+"/"+entry).read()
			self._insertDataInDatabase(data, self._getKernelName(entry))

profr=profReader('profs', "a b c")
profr.readProfs()



