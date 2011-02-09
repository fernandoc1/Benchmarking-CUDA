#! /usr/bin/env python

import sqlite3

def getKernelName(profFilePath):
	token=profFilePath.split('.')
	return token[1]

class ProfReader:
	#These are the tags from the conf file.
	_label=[]
	_connection=None
	

	def __init__(self, confFilePath):
		confData=open(confFilePath, 'r').read()
		self._label=confData.split()
		self._connection=sqlite3.connect('results.db')
		self._initTables()

	def _initTables(self):
		cursor=self._connection.cursor()
		#Create kernel name tables
		try:
			cursor.execute("create table kernelNames (kernelName varchar(30) primary key, benchmarkName varchar(30))")
		except:
			print "kernelNames already created"
		for label in self._label:
			try:
				sqlCommand="create table "+label+" (id integer primary key autoincrement, kernelName varchar(30) references kernelNames(kernelName), value float)"
				#print sqlCommand
				cursor.execute(sqlCommand)
			except:
				print label+" already created"

	def insertProfData(self, profFilePath):
		cursor=self._connection.cursor()
		profFileData=open(profFilePath).read().split()
		kernelName=getKernelName(profFilePath)

		index=0
		for label in self._label:
			sqlCommand="insert into "+label+" (kernelName, value) values (\""+kernelName+"\", "+profFileData[index]+")"
			print sqlCommand
			cursor.execute(sqlCommand)
			self._connection.commit()
			index+=1

pr=ProfReader("conf.txt")
pr.insertProfData("prof.teste.txt")


