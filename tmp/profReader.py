#! /usr/bin/env python

from database import Database
import dircache

class profReader:
	
	_database=None
	_profFolderPath=None
	_regularExpression=""
	_executionCounteir=0

	def __init__(self, profFolderPath, regularExpression):
		self._database=Database('results.db')
		self._profFolderPath=profFolderPath
		self._regularExpression=regularExpression

	def _getProfFilesPath(self):
		files=dircache.listdir(self._profFolderPath)
		profFiles=[]
		for entry in files:
			token=entry.split('.')
			if(token[0]=="prof"):
				profFiles.append(entry)
		return profFiles
	
	def incrementExecutionCounter(self):
		self._executionCounter+=1

	def _getKernelName(profFileName):
		token=profFileName.split('.')
		return token[1]

	def _insertDataInDatabase(self):
		tokens=self._regularExpression.split(' ')
		for entry in tokens:
			#TODO
			print entry

	#regex examples: "numA numB numC"

	def readProfs(self):
		profs=self._getProfFilesPath()
		for entry in profs:
			self._insertDataInDatabase()

profr=profReader('profs', "a b c")
profr.readProfs()



