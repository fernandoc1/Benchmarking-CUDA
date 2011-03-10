import sqlite3

class Database:
	
	_connection=None
	_cursor=None

	def __init__(self, databaseName):
		self._connection=sqlite3.connect(databaseName)
		self._cursor=self._connection.cursor()
	
	def _createTable(self, tableName):
		try:
			createCommand="create table "+tableName+"(id integer primary key autoincrement, e integer, kernelName varchar(20), data varchar(20))"
			print "++> "+createCommand
			self._cursor.execute(createCommand)
			self._connection.commit()
		except:
			print "!!! "+tableName+" already exists."
	
	def insertData(self, e, kernelName, dataLabel, data):
		self._createTable(dataLabel)
		insertCommand="insert into "+dataLabel+"(e, kernelName, data) values ("+e+", '"+kernelName+"', '"+data+"')"
		print "++> "+insertCommand
		self._cursor.execute(insertCommand)
		self._connection.commit()

	
