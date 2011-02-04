/*!
	\file CFG.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests CFG analysis by forming a CFG from a sequence of instructions known to produce
		a certain CFG and then comparing output
*/

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


typedef enum {instNone, instSimple, instPrecise} instrumentationType;


bool hasDivergences(const unsigned int blockId, BranchInformation& branch) {		
	if (blockId == branch.branchId) {
		if (branch.divergences > 0) {
			return true;
		}
	}
	
	return false;
}

InstrumentationResults readResults(std::string& profFileName) {
	std::ifstream profFile(profFileName.c_str());
	BranchInformation info = {0,0,0};
	InstrumentationResults results(0);
	
	if (profFile.is_open()) {
		profFile >> info.branchId;
		profFile >> info.timesVisited;
		profFile >> info.divergences;
		while ( !profFile.eof() ) {
			results.push_back(info);
			
			profFile >> info.branchId;
			profFile >> info.timesVisited;
			profFile >> info.divergences;
		}
		
		profFile.close();
	}
	
	return results;
}

void generateKernelStatistics(std::string& demangledKernelName, std::ostream& statFile, ir::PTXKernel* kernel, instrumentationType instType) {
	unsigned int basicBlocks = 0;
	unsigned int conditionalBranches = 0;
	unsigned int divergentBranches = 0;
	unsigned int mostDivergences = 0;
	unsigned int mostDivergentVisits = 0;
	unsigned int mostVisits = 0;
	unsigned int totalDivergencies = 0;
	unsigned int totalInstructions = 0;
	
	std::string profFileName              = "prof." + demangledKernelName + ".txt";
	InstrumentationResults results        = readResults(profFileName);
	InstrumentationResults::iterator r_it = results.begin();

	ir::ControlFlowGraph* cfg  = kernel->cfg();
	ir::ControlFlowGraph::iterator entryBlock = cfg->get_entry_block();
	ir::ControlFlowGraph::iterator exitBlock  = cfg->get_exit_block();
	ir::ControlFlowGraph::BlockPointerVector blocks = cfg->executable_sequence();
	
	ir::ControlFlowGraph::BlockPointerVector::iterator it = blocks.begin();
	for (; it != blocks.end(); ++it) {
		ir::ControlFlowGraph::BlockList::iterator blockPointer = *it;

		if (blockPointer == entryBlock || blockPointer == exitBlock) continue;
		
		ir::ControlFlowGraph::BasicBlock& block = *blockPointer;
		if (instType == instNone) {
			if (block.endsWithConditionalBranch()) {
				BranchInformation& branch = *r_it;

				if (block.id != branch.branchId) {
					std::cerr << "Error: profiling results of file \"" << profFileName << "\" don't seem to be from kernel \"" << demangledKernelName << "\"\n";
					exit(EXIT_FAILURE);
				}

				// test if it's the most visited branch
				if (branch.timesVisited > mostVisits) {
					mostVisits = branch.timesVisited;
				}

				if (hasDivergences(basicBlocks, branch)) {
					divergentBranches++;
					totalDivergencies += branch.divergences;

					// veyfy if it's the most divergent basic block
					if (branch.divergences > mostDivergences) {
						mostDivergences     = branch.divergences;
						mostDivergentVisits = branch.timesVisited;
					}
				}

				conditionalBranches++;
				r_it++;
			}

			basicBlocks++;
		}

		ir::ControlFlowGraph::BasicBlock::InstructionList::const_iterator instr = block.instructions.begin();
		for (; instr != block.instructions.end(); ++instr) {
			totalInstructions++;
		}
	}	
	
	// print results
	if (instType == instNone) {
		statFile << "^ " << demangledKernelName << " |  " << basicBlocks;
		statFile << "|  " << conditionalBranches << "|  " << divergentBranches;
		statFile << "| " << mostDivergences << "/" << mostDivergentVisits;
		statFile << " |  " << mostVisits;
	}

	/*std::cerr << demangledKernelName << " instructions: " << totalInstructions;
	if (instType == instNone) {
		std::cerr << " divergencies: " << totalDivergencies;
	}
	std::cerr << std::endl;*/
}


bool isGoodBranch(ir::ControlFlowGraph* cfg, ir::ControlFlowGraph::BasicBlock& block) {
	ir::ControlFlowGraph::iterator exitBlock = cfg->get_exit_block();
	ir::ControlFlowGraph::BlockPointerVector& sucessors = block.successors;
	ir::PostdominatorTree pdomTree(cfg);
	bool isGood = true;

	// for any successor fallthroughSucc
	ir::ControlFlowGraph::BlockPointerVector::iterator fallthroughSucc = sucessors.begin();
	for (; fallthroughSucc != sucessors.end(); fallthroughSucc++) {

		// for any successor branchSucc such as branchSucc != fallthroughSucc
		ir::ControlFlowGraph::BlockPointerVector::iterator branchSucc = sucessors.begin();
		for (; branchSucc != sucessors.end(); branchSucc++) {
			if (*branchSucc == exitBlock) {
				isGood = false;
				break;
			}

			// branchSucc must not be a postdominator of fallthroughSucc
			if (fallthroughSucc != branchSucc) {
				ir::ControlFlowGraph::iterator fSuccPdom = pdomTree.getPostDominator(*fallthroughSucc);
				while (fSuccPdom != exitBlock) {
					if (fSuccPdom == *branchSucc) {
						isGood = false;
					}

					fSuccPdom = pdomTree.getPostDominator(fSuccPdom);
				}
			}

		} // for any successor fallthroughSucc
	} // for any successor branchSucc

	return isGood;
}

bool isSuperGood(ir::ControlFlowGraph* cfg, ir::ControlFlowGraph::BasicBlock& block) {
	ir::ControlFlowGraph::BlockPointerVector& sucessors = block.successors;
	ir::PostdominatorTree pdomTree(cfg);
	bool superGood = false;

	// for any successor succ1
	ir::ControlFlowGraph::BlockPointerVector::iterator succ1 = sucessors.begin();
	for (; succ1 != sucessors.end(); succ1++) {

		// for any successor succ2
		ir::ControlFlowGraph::BlockPointerVector::iterator succ2 = succ1+1;
		for (; succ2 != sucessors.end(); succ2++) {

			ir::ControlFlowGraph::iterator pdom1 = pdomTree.getPostDominator(*succ1);
			ir::ControlFlowGraph::iterator pdom2 = pdomTree.getPostDominator(*succ2);
			if (pdom1 == pdom2) {
				// super good branch
				superGood = true;
			}
		} // for any successor succ1
	} // for any successor succ2

	return superGood;
}

void findGoodBranches(std::ostream& statFile, ir::PTXKernel* kernel) {
	ir::ControlFlowGraph* cfg  = kernel->cfg();
	ir::ControlFlowGraph::iterator entryBlock = cfg->get_entry_block();
	ir::ControlFlowGraph::iterator exitBlock  = cfg->get_exit_block();
	ir::ControlFlowGraph::BlockPointerVector blocks = cfg->executable_sequence();
	ir::PostdominatorTree pdomTree(cfg);
	unsigned int goodBranches = 0;
	unsigned int superGoods = 0;

	ir::ControlFlowGraph::BlockPointerVector::iterator it = blocks.begin();
	for (; it != blocks.end(); ++it) {
		ir::ControlFlowGraph::BlockList::iterator blockPointer = *it;
		ir::ControlFlowGraph::BasicBlock& block = *blockPointer;

		if (blockPointer == entryBlock || blockPointer == exitBlock) continue;

		if (block.endsWithConditionalBranch()) {
			if (isGoodBranch(cfg, block)) {
				goodBranches++;

				if (isSuperGood(cfg, block)) {
					superGoods++;
				}
			}
		}
	}

	statFile << "|  " << goodBranches << "|  " << superGoods << "|\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void analyze(std::string& moduleFileName, std::string& outputFile, instrumentationType instType) {
	ir::Module* module = new ir::Module(moduleFileName);
	std::ostream* statFile = NULL;

	if (outputFile == "") {
		statFile = &(std::cout);
	} else {
		statFile = (std::ostream*) new std::ofstream(outputFile.c_str());
	}

	if (instType == instNone) {
		*statFile << "|        ^ Basic Blocks ^ Conditional Branches ^ Divergent Branches ^ max num div (num visits) ^ Max Visited ^ Good Branches ^ Super Good ^\n";
	}


	ir::Module::KernelMap::const_iterator k_it = module->kernels().begin();
	for ( ;	k_it != module->kernels().end(); ++k_it) {
		ir::PTXKernel* kernel = static_cast< ir::PTXKernel* >(k_it->second);
		int status = -1;

		std::string fullDemangledName = abi::__cxa_demangle(kernel->name.c_str(), 0, 0, &status);
		if (status < 0) {
			std::cerr << "Error: could not demangle kernel name " << kernel->name << std::endl; 
			exit(EXIT_FAILURE);
		}
		std::string demangledKernelName = fullDemangledName.substr(0, fullDemangledName.find("("));

		generateKernelStatistics(demangledKernelName, *statFile, kernel, instType);
		if (instType == instNone) {
			findGoodBranches(*statFile, kernel);
		}
	}

	if (statFile != &(std::cout)) {
		delete statFile;
	}
	delete module;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static void print_usage() {
	using namespace std;

	cout << "kernelStatistics -m <module file> -o <output file> [-b or -p]\n\n";
}

int main(int argc, char **argv) {
	std::string moduleFile;
	std::string outputFile;
	char c = 0;
	instrumentationType instType = instNone;

	while ((-1) != (c = getopt(argc, argv, "bm:o:p"))) {
		switch (c) {
			case 'm': 
				moduleFile = optarg;
				break;
			case 'o': 
				outputFile = optarg;
				break;
			case 'b':
				if (instType != instNone) {
					print_usage();
					exit(EXIT_FAILURE);
				}

				instType = instSimple;
				break;
			case 'p':
				if (instType != instNone) {
					print_usage();
					exit(EXIT_FAILURE);
				}

				instType = instPrecise;
				break;
			default:
				print_usage();
				exit(EXIT_FAILURE);
				break;
		}
	} 
	if (moduleFile == "") {
		print_usage();
		exit(EXIT_FAILURE);
	}

	analyze(moduleFile, outputFile, instType);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

