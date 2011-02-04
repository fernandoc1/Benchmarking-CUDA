/*!
	\file CFG.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests CFG analysis by forming a CFG from a sequence of instructions known to produce
		a certain CFG and then comparing output
*/

#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <unistd.h>

#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/parser/interface/PTXParser.h>
#include <ocelot/ir/interface/CodeWalker.h>
#include <ocelot/ir/interface/PrinterWorker.h>
#include <ocelot/ir/interface/InstrumentKernelExecTime.h>
#include <ocelot/ir/interface/SimpleBranchProfiler.h>
#include <ocelot/ir/interface/PrecBranchProfiler.h>


/////////////////////////////////////////////////////////////////////////////////////////////////

void analyze(std::string inputFileName, std::string outputFileName, ir::Worker* instrumenter) {
	ir::Module module(inputFileName);

	// instrument code
	if (instrumenter != NULL) {
		ir::CodeWalker w(instrumenter);
		w.visit(&module);
	}

	// print instrumented code
	std::ofstream output_file(outputFileName.c_str(), std::ofstream::out | std::ofstream::trunc);
	
	std::cout << "writing output file: " << outputFileName << std::endl;
	ir::CodeWalker printer(new ir::PrinterWorker(output_file));
	printer.visit(&module);
	
	output_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static void print_usage() {
	std::cout << "CFG <input path>:\n\n";
	std::cout << " - parses the input file, performs control flow analysis on each kernel\n";
	std::cout << "   and emits .dot files for the CFG, dominator tree, and post dominator tree\n";

	std::cout << "\n  To construct graphs of them, use the following command:\n\n";
	std::cout << "    for f in *.dot; do dot -Tpdf -o $f.pdf $f; done\n\n";
}

int main(int argc, char **argv) {
	std::string inputFileName  = "";
	std::string outputFileName = "";
	ir::Worker* instrumenter   = NULL; 
	char c = 0;

	while ((-1) != (c = getopt(argc, argv, "bei:o:p"))) {
		switch (c) {
			case 'b':
				instrumenter = new ir::SimpleBranchProfiler();
				std::cout << "Inserting Simple Branch instrumentation" << std::endl;
				break;
			case 'e':
				instrumenter = new ir::InstrumentKernelExecTime();
				std::cout << "Inserting Kernel Execution Time instrumentation" << std::endl;
				break;
			case 'i':
				inputFileName = optarg;
				break;
			case 'o':
				outputFileName = optarg;
				break;
			case 'p':
				instrumenter = new ir::PrecBranchProfiler();
				std::cout << "Inserting Precise Branch instrumentation" << std::endl;
				break;
			default:
				print_usage();
				exit(-1);			
				break;
		}
	} 
	if (inputFileName == "") {
		print_usage();
		exit(-1);
	}
	if (outputFileName == "") {
		outputFileName = inputFileName + ".out";
	}

	analyze(inputFileName, outputFileName, instrumenter);

	delete instrumenter;
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

