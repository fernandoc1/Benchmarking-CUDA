#include <stdio.h>
#include <stdlib.h>

int getFileLength(FILE* file)
{
	rewind(file);
	int count=0;
	getc(file);
	while(!feof(file))
	{
		count++;
		getc(file);
	}
	rewind(file);
	return count;
}

char* fileToCharArray(FILE* file)
{
	int length=getFileLength(file);
	char* buffer=(char*)malloc(sizeof(char)*length);
	int index=0;
	rewind(file);
	while(!feof(file))
	{
		buffer[index]=getc(file);
		index++;
	}
	rewind(file);
	return buffer;
}

int nextCommandIsKernelCall(char* buffer, int lastCommand)
{
	int i=0;
	
}

int main(int argc, char** argv)
{
	if(argc<=2)
	{
		printf("You must give a source code.\nABORTED\n");
		exit(1);
	}
	FILE* source=fopen(argv[1]);
	
	
}
