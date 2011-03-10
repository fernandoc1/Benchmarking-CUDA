#! /usr/bin/env python

from database import Database
import dircache

class profReader:
	
	_database=None
	_profFolderPath=None

	def __init__(self, profFolderPath):
		self._database=Database('results.db')
		self._profFolderPath=profFolderPath

	def _getProfFilesPath(self):
		files=dircache.listdir(self._profFolderPath)
		for entry in files:
			print entry


