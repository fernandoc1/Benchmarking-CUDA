#! /usr/bin/env python

from database import Database
import dircache

class profReader:
	
	_database=None
	_profFolderPath=None
	_

	def __init__(self, profFolderPath):
		self._database=Database('results.db')
		self._profFolderPath=profFolderPath

	def _getProfFilesPath(self):
		files=dircache.listdir(self._profFolderPath)
		profFiles=[]
		for entry in files:
			token=entry.split('.')
			if(token[0]=="prof"):
				profFiles.append(entry)
		return profFiles
	
	def readProfs(self):
		


profreader=profReader('profs')
x=profreader._getProfFilesPath()
print x



