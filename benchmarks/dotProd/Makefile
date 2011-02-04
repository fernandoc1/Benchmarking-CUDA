NVCC=/usr/local/cuda/bin/nvcc
SOURCE_FILES=dotProd.cu 
INCLUDE_PATH=../../../dependencies/includes
LIBRARIES_PATH=../../../dependencies/libraries
EXECUTABLE_FILE=exec.bin
LIBRARIES=-lcutil
SO_FILES=/usr/local/cuda/lib64
INCLUDES_AND_LIBRARIES=-I$(INCLUDE_PATH) -L$(LIBRARIES_PATH) $(LIBRARIES)
PTX_CODE_DIR=exec.bin.devcode

all:
	$(NVCC) $(SOURCE_FILES) -I$(INCLUDE_PATH) -L$(LIBRARIES_PATH) $(LIBRARIES) -o $(EXECUTABLE_FILE)

run:
	PROF_BRANCHES=1 LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./$(EXECUTABLE_FILE) 64

clean:
	rm -vf $(EXECUTABLE_FILE)

ptx_code:
	rm -rvf $(PTX_CODE_DIR)
	$(NVCC) -arch compute_13 -int=none -ext=all -dir=$(PTX_CODE_DIR) $(INCLUDES_AND_LIBRARIES) ${SOURCE_FILES} -o $(EXECUTABLE_FILE)
	echo Bla ${find . -name compute*}
	echo ../../../ocelot/iptx -e -i $(find . -name compute*) -o instrumented_ptx_code.asm

