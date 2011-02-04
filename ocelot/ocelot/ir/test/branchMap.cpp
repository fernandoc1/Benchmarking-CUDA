/*!
	\file CFG.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests CFG analysis by forming a CFG from a sequence of instructions known to produce
		a certain CFG and then comparing output
*/


/// \todo
// Fix the block ordering: you are assuming that exit is 1, but that is not
// correct, exit should be the last of the blocks.
// - See if this can fix the last double edge going to exit.
// - See if this adds the edge from entry to first block.


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cxxabi.h>

#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/DominatorTree.h>
#include <ocelot/ir/interface/PostdominatorTree.h>
#include <ocelot/parser/interface/PTXParser.h>


typedef struct {
	unsigned int       branchId;
	unsigned long long timesVisited;
	unsigned long long divergences;
} BranchInformation;

typedef std::vector<BranchInformation> InstrumentationResults;



class BranchMapDotFormatter : public ir::ControlFlowGraph::BasicBlock::DotFormatter {
	public:
	
	BranchMapDotFormatter() {}
	virtual ~BranchMapDotFormatter() {}
	
	virtual std::string toString(const ir::ControlFlowGraph::BasicBlock *block, BranchInformation* branch = NULL) {
		std::stringstream out;
	
		out << "[shape=record,style=filled,color=yellow,";
		out << "label=";
		out << "\"{" << hydrazine::toGraphVizParsableLabel(block->label);
	
		ir::ControlFlowGraph::BasicBlock::InstructionList::const_iterator instrs = block->instructions.begin();	
		for (; instrs != block->instructions.end(); ++instrs) {
			out << " | " << hydrazine::toGraphVizParsableLabel((*instrs)->toString());
		}
		if (branch != NULL) {
			out << " | visited: " << branch->timesVisited << " div: " << branch->divergences;
		}
		out << "}\"]";
	
		return out.str();	
	}
	
};


bool hasDivergences(const unsigned int blockId, BranchInformation* branch) {
	if (branch != NULL) {		
		if (blockId == branch->branchId) {
			if (branch->divergences > 0) {
				return true;
			}
		}
	}
	
	return false;
}

/*!	write a graphviz-compatible file for visualizing instrumentation results */
std::ostream& writeVizualization(std::ostream &out, InstrumentationResults* results, ir::ControlFlowGraph* cfg, ir::ControlFlowGraph::BasicBlock::DotFormatter* blockFormatter) {
	ir::ControlFlowGraph::BlockMap blockIndices;
	ir::ControlFlowGraph::iterator entryBlock = cfg->get_entry_block();
	ir::ControlFlowGraph::iterator exitBlock  = cfg->get_exit_block();
	
	out << "digraph {\n";

	// emit nodes
	out << "  // basic blocks\n\n";

	blockIndices[entryBlock] = 0;
	out << "  bb_0 [shape=Mdiamond,label=\"" << entryBlock->label << "\"];\n";

	int n = 1;
	InstrumentationResults::iterator r_it      = results->begin();
	ir::ControlFlowGraph::const_iterator block = cfg->begin();
	for (; block != cfg->end(); ++block) {
		BranchInformation* branch = NULL;
		
		if (block == entryBlock || block == exitBlock) continue;

		blockIndices[block] = n;

		if (r_it != results->end()) {
			branch = &*r_it;
		}
		
		const ir::ControlFlowGraph::BasicBlock *blockPtr = &*block;
		if (hasDivergences(n, branch)) {
			BranchMapDotFormatter* bmapFormatter = (BranchMapDotFormatter*) blockFormatter;
			out << "  bb_" << n << " " << bmapFormatter->toString(blockPtr, branch) << ";\n";
		} else {
			out << "  bb_" << n << " " << blockFormatter->toString(blockPtr) << ";\n";
		}
		
		n++;
		if (block->endsWithConditionalBranch()) {
			r_it++;	
		}
	}
	
	blockIndices[exitBlock]  = n;
	out << "  bb_" << blockIndices[exitBlock] << " [shape=Msquare,label=\""  << exitBlock->label << "\"];\n";

	// emit edges
	out << "\n\n  // edges\n\n";
	
	for (ir::ControlFlowGraph::const_edge_iterator edge = cfg->edges_begin();
		edge != cfg->edges_end(); ++edge) {
		const ir::ControlFlowGraph::Edge *edgePtr = &*edge;

		if (edgePtr->type != ir::ControlFlowGraph::Edge::Dummy) {
			out << "  " << "bb_" << blockIndices[edge->head] << " -> "
				<< "bb_" << blockIndices[edge->tail];
			out << " " << blockFormatter->toString(edgePtr);
		
			out << ";\n";
		}
	}

	out << "}\n";

	return out;
}

InstrumentationResults readResults(std::string& profFileName) {
	std::ifstream profFile(profFileName.c_str());
	BranchInformation info = {0,0,0};
	InstrumentationResults results(0);
	
	if (profFile.is_open()) {
		profFile >> info.branchId;
		profFile >> info.timesVisited;
		profFile >> info.divergences;
		while ((!profFile.eof()) && (info.branchId != 0)) {
			results.push_back(info);
			
			profFile >> info.branchId;
			profFile >> info.timesVisited;
			profFile >> info.divergences;
		}
		
		profFile.close();
	}
	
	return results;
}

void generateVisualization(std::string& profFile, std::string& outputFile, ir::PTXKernel* kernel) {
	InstrumentationResults results = readResults(profFile);
	BranchMapDotFormatter blockFormatter;
		
	std::ofstream cfg_file(outputFile.c_str());
	cfg_file << "// Kernel: " << kernel->name << "\n";
	cfg_file << "// Control flow graph\n";
	writeVizualization(cfg_file, &results, kernel->cfg(), &blockFormatter);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void analyze(std::string& moduleFileName, std::string& selectedKernel, std::string& prof, std::string& outputFile) {
	ir::Module* module = new ir::Module(moduleFileName);

	ir::Module::KernelMap::const_iterator k_it = module->kernels().begin();
	if (selectedKernel == "") {

		// no specific kernel selected, print the first kernel found
		if (k_it != module->kernels().end()) {
			ir::PTXKernel* kernel = static_cast< ir::PTXKernel* >(k_it->second);
			generateVisualization(prof, outputFile, kernel);
		}

	} else {

		// find the specified kernel and generate a visualization for it
		for ( ;	k_it != module->kernels().end(); ++k_it) {
			ir::PTXKernel* kernel = static_cast< ir::PTXKernel* >(k_it->second);
			int status = -1;

			std::string fullDemangledName = abi::__cxa_demangle(kernel->name.c_str(), 0, 0, &status);
			if (status < 0) {
				std::cerr << "Error: could not demangle kernel name " << kernel->name << std::endl;
				exit(EXIT_FAILURE);
			}
			std::string demangledKernelName = fullDemangledName.substr(0, fullDemangledName.find("("));

			if (demangledKernelName == selectedKernel) {
				generateVisualization(prof, outputFile, kernel);
			}
		}

	}
	
	delete module;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static void print_usage() {
	using namespace std;

	cout << "branchMap [-k <selected kernel>] -m <module file> -p <instrumentation results> -o <output file>:\n\n";
	cout << " - parses the module file, performs control flow analysis on the selected\n";
	cout << "   kernel and emits .dot files for the CFG with instrumentation results.\n";

	cout << "\n  To construct graphs of them, use the following command:\n\n";
	cout << "    for f in *.dot; do dot -Tpdf -o $f.pdf $f; done\n\n";
}

int main(int argc, char **argv) {
	std::string moduleFile;
	std::string kernelName;
	std::string profFile;
	std::string outputFile;
	char c = 0;

	while ((-1) != (c = getopt(argc, argv, "k:m:o:p:"))) {
		switch (c) {
			case 'k': 
				kernelName = optarg;
				break;
			case 'm': 
				moduleFile = optarg;
				break;
			case 'o': 
				outputFile = optarg;
				break;
			case 'p': 
				profFile = optarg;
				break;
			default:
				print_usage();
				exit(-1);			
				break;
		}
	} 
	if (moduleFile == "") {
		print_usage();
		exit(EXIT_FAILURE);
	}
	if (profFile == "") {
		print_usage();
		exit(EXIT_FAILURE);
	}
	if (outputFile == "") {
		if (kernelName != "") {
			outputFile = kernelName + ".dot";
		} else {
			outputFile = moduleFile + ".dot";
		}
	}

	analyze(moduleFile, kernelName, profFile, outputFile);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

