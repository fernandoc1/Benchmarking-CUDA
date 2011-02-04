/*
 * BranchInfo.cpp
 *
 *  Created on: Aug 23, 2010
 *      Author: undead
 */

#if 0
#define DEBUG 1
#include <iostream>
#endif

#include <ocelot/analysis/interface/BranchInfo.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/graphs/interface/DivergenceGraph.h>

namespace analysis {

BranchInfo::BranchInfo(const Block *block, const Block *postDominator, const node_type &predicate,
    const DataflowGraph::Instruction &dfgInstruction, const DataflowGraph &dfg,
    graph_utils::DivergenceGraph &divergGraph) :
	_block(block), _postDominator(postDominator), _predicate(predicate), _dfgInstruction(dfgInstruction), _divergGraph(
	    divergGraph)
{
	_fallThrough = &(*_block->fallthrough());
	const DataflowGraph::BlockPointerSet targets = _block->targets();
	DataflowGraph::BlockPointerSet::const_iterator targetBlock = targets.begin();
	DataflowGraph::BlockPointerSet::const_iterator endTargetBlock = targets.end();
	for (; targetBlock != endTargetBlock; targetBlock++) {
		DataflowGraph::BlockVector::const_iterator targetBlockI = *targetBlock;
		if (block->fallthrough() != targetBlockI) {
			_branch = &(*targetBlockI);
		}
#ifdef DEBUG
		std::cout << "Created new BranchInfo:" << std::endl << "\tblock:" << _block->id() << std::endl << "\tpost-dominator:"
		<< _postDominator->id() << std::endl << "\tpredicate:" << predicate << std::endl << "\tfallthrough:"
		<< _fallThrough->id() << std::endl << "\tbranch:" << _branch->id() << std::endl;
#endif
	}
}

bool BranchInfo::isTainted(const graph_utils::DivergenceGraph::node_type &node) const
{
	return ((_branchVariables.find(node) != _branchVariables.end()) || (_fallThroughVariables.find(node)
	    != _fallThroughVariables.end()));
}

void BranchInfo::populate()
{
#ifdef DEBUG
	std::cout << "populating branch of block:" << _block->id() << std::endl << "FALLTHROUGH:" << std::endl;
#endif
	if(_fallThrough != _postDominator){
		_taintBlocks(_fallThrough, _fallThroughBlocks, _fallThroughVariables);
	}
#ifdef DEBUG
	std::cout << "BRANCH" << std::endl;
#endif
	if(_branch != _postDominator){
		_taintBlocks(_branch, _branchBlocks, _branchVariables);
	}
}
;

void BranchInfo::_insertVariable(node_type &variable, node_set &variables)
{
	if ((variables.find(variable) != variables.end()) || (_divergGraph.isDivNode(variable))) {
		return;
	}

#ifdef DEBUG
	std::cout << "Variable " << variable << " taints: " << std::endl;
#endif

	node_set newVariables;
	newVariables.insert(variable);

	while (newVariables.size() > 0) {
		node_type newVar = *newVariables.begin();

		variables.insert(newVar);
		newVariables.erase(newVar);
#ifdef DEBUG
		std::cout << newVar << ' ';
#endif

		node_set dependences = _divergGraph.getOutNodesSet(newVar);
		node_set::const_iterator depend = dependences.begin();
		node_set::const_iterator endDepend = dependences.end();

		for (; depend != endDepend; depend++) {
			if ((variables.find(*depend) == variables.end()) && (!_divergGraph.isDivNode(*depend))) {
				newVariables.insert(*depend);
			}
		}
	}
#ifdef DEBUG
	std::cout << std::endl;
#endif

}

void BranchInfo::_taintVariables(const Block *block, node_set &variables)
{
	const DataflowGraph::InstructionVector instructions = block->instructions();
	DataflowGraph::InstructionVector::const_iterator ins = instructions.begin();
	DataflowGraph::InstructionVector::const_iterator endIns = instructions.end();

	for (; ins != endIns; ins++) {
		DataflowGraph::RegisterPointerVector::const_iterator destiny = ins->d.begin();
		DataflowGraph::RegisterPointerVector::const_iterator endDestiny = ins->d.end();
		for (; destiny != endDestiny; destiny++) {
#ifdef DEBUG
			std::cout << "tainting variable " << *destiny->pointer << std::endl;
#endif
			_insertVariable(*destiny->pointer, variables);
		}
	}
}

void BranchInfo::_taintBlocks(const Block *start, touched_blocks &blocks, node_set &variables)
{

	touched_blocks toCompute;
	toCompute.insert(start);

	while (toCompute.size() > 0) {
		const Block *block = *toCompute.begin();
		if (blocks.find(block) == blocks.end()) {
			blocks.insert(block);

			DataflowGraph::BlockPointerSet::const_iterator newBlock = block->targets().begin();
			DataflowGraph::BlockPointerSet::const_iterator endNewBlock = block->targets().end();

			for (; newBlock != endNewBlock; newBlock++) {
				toCompute.insert(&(*(*newBlock)));
			}

		}

		toCompute.erase(block);
#ifdef DEBUG
		std::cout << "Tainting block " << block->id() << std::endl;
#endif

		_taintVariables(block, variables);
	}
}
}
