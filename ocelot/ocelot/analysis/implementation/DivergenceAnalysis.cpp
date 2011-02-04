#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/ir/interface/PostdominatorTree.h>
#include <assert.h>
#include <sys/time.h>

#if 0
#define DEBUG 1
#include <iostream>
#endif

namespace analysis {

/*! \brief  Analyze the data flow, detecting divergent variables and blocks based on divergent sources, such as t.id, laneId
 *
 * 1) Analyze the data flow adding divergence sources
 * 2) Computes the divergence propagation
 */
void DivergenceAnalysis::_analyzeDataFlow()
{
	DataflowGraph &nonConstGraph = *_kernel->dfg();
	DataflowGraph::const_iterator block = nonConstGraph.begin();
	DataflowGraph::const_iterator endBlock = nonConstGraph.end();

	/* 1) Analyze the data flow adding divergence sources */
	for (; block != endBlock; ++block) {
		DataflowGraph::PhiInstructionVector::const_iterator phiInstruction = block->phis().begin();
		DataflowGraph::PhiInstructionVector::const_iterator endPhiInstruction = block->phis().end();
		for (; phiInstruction != endPhiInstruction; phiInstruction++) {
			for (DataflowGraph::RegisterVector::const_iterator si = phiInstruction->s.begin(); si != phiInstruction->s.end(); ++si) {
				_divergGraph.insertEdge(si->id, phiInstruction->d.id);
				si->type;
			}
		}

		DataflowGraph::InstructionVector::const_iterator ii = block->instructions().begin();
		DataflowGraph::InstructionVector::const_iterator iiEnd = block->instructions().end();
		for (; ii != iiEnd; ++ii) {

			ir::PTXInstruction *ptxInstruction = NULL;
			bool atom = false;

			set<ir::PTXOperand::SpecialRegister> dirtSources;

			if (typeid(ir::PTXInstruction) == typeid(*(ii->i))) {
				ptxInstruction = static_cast<ir::PTXInstruction*> (ii->i);

				if(ptxInstruction->opcode == ir::PTXInstruction::Opcode::Atom){
					atom = true;
				}

				if (ptxInstruction->a.addressMode == ir::PTXOperand::AddressMode::Special) {
					dirtSources.insert(ptxInstruction->a.special);
				}

				if (ptxInstruction->b.addressMode == ir::PTXOperand::AddressMode::Special) {
					dirtSources.insert(ptxInstruction->b.special);
				}

				if (ptxInstruction->c.addressMode == ir::PTXOperand::AddressMode::Special) {
					dirtSources.insert(ptxInstruction->c.special);
				}
			}

			dirtSources.erase(ir::PTXOperand::SpecialRegister_invalid);

			DataflowGraph::RegisterPointerVector::const_iterator destinyReg = ii->d.begin();
			DataflowGraph::RegisterPointerVector::const_iterator destinyEndReg = ii->d.end();

			for (; destinyReg != destinyEndReg; destinyReg++) {

				if (dirtSources.size() != 0) {
					set<ir::PTXOperand::SpecialRegister>::const_iterator dirtSource = dirtSources.begin();
					set<ir::PTXOperand::SpecialRegister>::const_iterator endDirtSource = dirtSources.end();

					for (; dirtSource != endDirtSource; dirtSource++) {
						_divergGraph.insertEdge(*dirtSource, *destinyReg->pointer);
					}
				}

				DataflowGraph::RegisterPointerVector::const_iterator sourceReg = ii->s.begin();
				DataflowGraph::RegisterPointerVector::const_iterator sourceEndReg = ii->s.end();

				for (; sourceReg != sourceEndReg; sourceReg++) {
					_divergGraph.insertEdge(*sourceReg->pointer, *destinyReg->pointer);
				}

				if(atom){
					_divergGraph.setAsDiv(*destinyReg->pointer);
				}
			}
		}
	}
	/* 2) Computes the divergence propagation */
	_divergGraph.computeDivergence();
}

/*! \brief  Makes control flow analysis that detects new divergent variables based on the dependency of variables of variables created on divergent paths
 * Uses the divergence analyzes from _dataFlowAnalyse and discover if divergent blocks affects the divergence of variables considered not divergent
 * 1) Obtain information of all possible divergent branch instructions on the kernel
 * 2) Obtain all branch instructions that depend on a divergent predicate
 * 3) For each divergent branch
 * 3.1) Compute the controlflow dependency
 * 3.2) Search the postdominator block for new divergent variables
 * 3.3) If new divergent variables were found
 * 3.3.1) Re-compute the divergence spread by the new divergence/dataflow graph
 * 3.3.2) Search for new divergent branch instructions
 */
void DivergenceAnalysis::_analyzeControlFlow()
{
	/* Set of possible diverging branches */
	std::set<BranchInfo> branches;

	{ /* 1) Obtain information of all possible divergent branch instructions on the kernel
	 * Create a list of branches that can be divergent, that is, is not a bra.uni and has a predicate */
		DataflowGraph::const_iterator block = _kernel->dfg()->begin();
		DataflowGraph::const_iterator endBlock = _kernel->dfg()->end();

		/* Post-dominator tree */
		ir::PostdominatorTree dtree(_kernel->cfg());

		for (; block != endBlock; ++block) {
			ir::PTXInstruction *ptxInstruction = NULL;

			if (block->instructions().size() > 0) {
				/* Branch instructions can only be the last instruction of a basic block */
				DataflowGraph::Instruction lastInstruction = *(--block->instructions().end());

				if (typeid(ir::PTXInstruction) == typeid(*(lastInstruction.i))) {
					ptxInstruction = static_cast<ir::PTXInstruction*> (lastInstruction.i);

					if ((ptxInstruction->opcode == ir::PTXInstruction::Opcode::Bra) && (ptxInstruction->uni == false)
					    && (lastInstruction.s.size() != 0)) {
						ir::ControlFlowGraph::iterator CFGBlock = _kernel->cfg()->begin();
						ir::ControlFlowGraph::iterator CFGEndBlock = _kernel->cfg()->end();

						for (; CFGBlock != CFGEndBlock; CFGBlock++) {
							if (CFGBlock->label == block->label()) {
								break;
							}
						}
#ifdef DEBUG
						std::cout << "block label: " << CFGBlock->label << std::endl;
#endif
						assert(CFGBlock != CFGEndBlock);
						assert(lastInstruction.s.size() == 1);
						unsigned int id = dtree.getPostDominator(CFGBlock)->id;
						DataflowGraph::const_iterator postDomBlock = _kernel->dfg()->begin();
						DataflowGraph::const_iterator endPostDomBlock = _kernel->dfg()->end();
						for (; postDomBlock != endPostDomBlock; ++postDomBlock) {
							if (postDomBlock->id() == id) {
								break;
							}
						}
#ifdef DEBUG
						std::cout << "CFG block:" << CFGBlock->id << " |CFG-Postdominator:" << id << "|DFG-Postdominator:" << postDomBlock->id() << std::endl;
#endif
						if (postDomBlock != endPostDomBlock) {
							BranchInfo newBranch(&(*block), &(*postDomBlock), *lastInstruction.s.begin()->pointer, lastInstruction,
							    *_kernel->dfg(), _divergGraph);
							branches.insert(newBranch);
						}
					}
				}
			}
		}
	}
	_branches = branches;
	/* 2) Obtain all branch instructions that depend on a divergent predicate
	 * List of branches that are divergent, so their controlflow influence must be tested */
	std::set<BranchInfo> divergent;

	/* Populate the divergent branches set */
	std::set<BranchInfo>::iterator branch = branches.begin();
	std::set<BranchInfo>::iterator endBranch = branches.end();

	while (branch != endBranch) {
		if (isDivBlock(branch->block())) {
			divergent.insert(*branch);
			_divergentBranches.insert(*branch);
			branches.erase(branch);
			branch = branches.begin();
			endBranch = branches.end();
		} else {
			_notDivergentBranches.insert(*branch);
			branch++;
		}
	}

	/*  3) For each divergent branch
	 * Test for divergence on the post-dominator block of every divergent branch instruction */
	while (divergent.size() > 0) {
		BranchInfo branchInfo = *divergent.begin();
		/* 3.1) Compute the controlflow dependency */
		branchInfo.populate();
		/* 3.2) Search the postdominator block for new divergent variables */
		DataflowGraph::PhiInstructionVector phis = branchInfo.postDominator()->phis();
		DataflowGraph::PhiInstructionVector::const_iterator phi = phis.begin();
		DataflowGraph::PhiInstructionVector::const_iterator endphi = phis.end();

		bool newDivegecies = false;
		for (; phi != endphi; phi++) {
			DataflowGraph::RegisterVector::const_iterator source = phi->s.begin();
			DataflowGraph::RegisterVector::const_iterator endSource = phi->s.end();

			for (; source != endSource; source++) {
				if (branchInfo.isTainted(source->id)) {
					_addPredicate(*phi, branchInfo.predicate());
					newDivegecies = true;
				}
			}
		}
		divergent.erase(branchInfo);
		/* 3.3) If new divergent variables were found*/
		if (newDivegecies) {
			/* 3.3.1) Re-compute the divergence spread by the new divergence/dataflow graph */
			_divergGraph.computeDivergence();
			branch = branches.begin();
			/* 3.3.2) Search for new divergent branch instructions */
			while (branch != endBranch) {
				if (isDivBlock(branch->block())) {
					divergent.insert(*branch);
					_divergentBranches.insert(*branch);
					branches.erase(branch);
					branch = branches.begin();
					endBranch = branches.end();
				} else {
					_notDivergentBranches.insert(*branch);
					branch++;
				}
			}
		}
	}
}

/*! \brief Add a predicate as a predecessor of a variable */
void DivergenceAnalysis::_addPredicate(const DataflowGraph::PhiInstruction &phi,
    const graph_utils::DivergenceGraph::node_type &predicate)
{
	_divergGraph.insertEdge(predicate, phi.d.id);
}

/*! \brief Constructor, already making the analysis of a input kernel */
DivergenceAnalysis::DivergenceAnalysis()
{
	_doCFGanalysis = true;
	_kernel = NULL;
}

/*! \brief Analyze the control and data flows searching for divergent variables and blocks
 *
 * 1) Makes data flow analysis that detects divergent variables and blocks based on divergent sources, such as t.id, laneId
 * 2) Makes control flow analysis that detects new divergent variables based on the dependency of variables of variables created on divergent paths
 */
void DivergenceAnalysis::runOnKernel(ir::Kernel &k)
{

	if (typeid(ir::PTXKernel) == typeid(k)) {
		_kernel = (ir::PTXKernel*) &k;
	}

	if (_kernel == NULL) {
		return;
	}

	if(!_kernel->dfg()->ssa())
		_kernel->dfg()->toSsa();

	run();
}


/*\brief re-run the analysis on a already passed kernel */
void DivergenceAnalysis::run()
{
	if (_kernel == NULL) {
		return;
	}

	_divergGraph.clear();
	_analysisTime.tv_sec = 0;
	_analysisTime.tv_usec = 0;
	_branches.clear();
	_divergentBranches.clear();
	_notDivergentBranches.clear();

	struct timeval begin, end;
	graph_utils::DivergenceGraph::node_set predicates;
	gettimeofday(&begin, 0);
	/* 1) Makes data flow analysis that detects divergent variables and blocks based on divergent sources, such as t.id, laneId */
	_analyzeDataFlow();
	/* 2) Makes control flow analysis that detects new divergent variables based on the dependency of variables of variables created on divergent paths */
	if(_doCFGanalysis){
		_analyzeControlFlow();
	}
	gettimeofday(&end, 0);
	_analysisTime.tv_sec = end.tv_sec - begin.tv_sec;
	_analysisTime.tv_usec = end.tv_usec - begin.tv_usec;
}

/*! \brief Tests if a block ends with a divergent branch instruction (isDivBranchInstr) */
bool DivergenceAnalysis::isDivBlock(DataflowGraph::const_iterator &block) const
{
	if (block->instructions().size() == 0) {
		return false;
	}
	return isDivBranch(--block->instructions().end());
}

/*! \brief Tests if a block ends with a divergent branch instruction (isDivBranchInstr) */
bool DivergenceAnalysis::isDivBlock(DataflowGraph::iterator &block) const
{
	if (block->instructions().size() == 0) {
		return false;
	}
	return isDivBranch(--block->instructions().end());
}


/*! \brief Tests if a block ends with a divergent branch instruction (isDivBranchInstr) */
bool DivergenceAnalysis::isDivBlock(const DataflowGraph::Block *block) const
{
	if ((block == NULL) || (block->instructions().size() == 0)) {
		return false;
	}
	return isDivBranch(--block->instructions().end());
}

/*! \brief Tests if a block ends with a branch instruction */
bool DivergenceAnalysis::isPossibleDivBlock(DataflowGraph::const_iterator &block) const
{
	if (block->instructions().size() == 0) {
		return false;
	}
	const DataflowGraph::InstructionVector::const_iterator &instruction = --block->instructions().end();
	if(typeid(ir::PTXInstruction) == typeid(*(instruction->i))) {
		const ir::PTXInstruction &ptxI = *(static_cast<ir::PTXInstruction *> (instruction->i));
		return ((ptxI.opcode == ir::PTXInstruction::Bra) && !ptxI.uni);
	}
	return false;
}

/*! \brief Tests if a block ends with a branch instruction */
bool DivergenceAnalysis::isPossibleDivBlock(const DataflowGraph::Block *block) const
{
	if ((block == NULL) || (block->instructions().size() == 0)) {
		return false;
	}
	const DataflowGraph::InstructionVector::const_iterator &instruction = --block->instructions().end();
	if(typeid(ir::PTXInstruction) == typeid(*(instruction->i))) {
		const ir::PTXInstruction &ptxI = *(static_cast<ir::PTXInstruction *> (instruction->i));
		return ((ptxI.opcode == ir::PTXInstruction::Bra) && !ptxI.uni);
	}
	return false;
}

/*! \brief Tests if the a instruction is a branch instruction instruction and is possible a divergent instruction (isDivInstruction) */
bool DivergenceAnalysis::isDivBranch(const DataflowGraph::InstructionVector::const_iterator &instruction) const
{
	return (isDivInstruction(*instruction) && isPossibleDivBranch(instruction));
}

/*!\brief Tests if a instruction is a branch instruction with possibility of divergence */
bool DivergenceAnalysis::isPossibleDivBranch(const DataflowGraph::InstructionVector::const_iterator &instruction) const
{
	if(typeid(ir::PTXInstruction) == typeid(*(instruction->i))) {
		const ir::PTXInstruction &ptxI = *(static_cast<ir::PTXInstruction *> (instruction->i));
		return ((ptxI.opcode == ir::PTXInstruction::Bra) && (!ptxI.uni));
	}
	return false;
}

/*! \brief Tests if any of the registers of a instruction is possible dirty */
bool DivergenceAnalysis::isDivInstruction(const DataflowGraph::Instruction &instruction) const
{
	bool isDirty = false;
	DataflowGraph::RegisterPointerVector::const_iterator reg = instruction.d.begin();
	DataflowGraph::RegisterPointerVector::const_iterator endReg = instruction.d.end();

	for (; (!isDirty) && (reg != endReg); reg++) {
		isDirty |= _divergGraph.isDivNode(*reg->pointer);
	}

	if (isDirty) {
		return true;
	}

	reg = instruction.s.begin();
	endReg = instruction.s.end();

	for (; (!isDirty) && (reg != endReg); reg++) {
		isDirty |= _divergGraph.isDivNode(*reg->pointer);
	}

	return isDirty;
}

/*! \brief Gives the first instruction block */
DataflowGraph::const_iterator DivergenceAnalysis::beginBlock() const
{
	assert(_kernel != NULL);
	return _kernel->dfg()->begin();
}

/*! \brief Gives the instruction blocks limit */
DataflowGraph::const_iterator DivergenceAnalysis::endBlock() const
{
	assert(_kernel != NULL);
	return _kernel->dfg()->end();
}

}

namespace std {
bool operator<(const analysis::BranchInfo x, const analysis::BranchInfo y)
{
	return x.block()->id() < y.block()->id();
}

bool operator<=(const analysis::BranchInfo x, const analysis::BranchInfo y)
{
	return x.block()->id() <= y.block()->id();
}

bool operator>(const analysis::BranchInfo x, const analysis::BranchInfo y)
{
	return x.block()->id() > y.block()->id();
}

bool operator>=(const analysis::BranchInfo x, const analysis::BranchInfo y)
{
	return x.block()->id() >= y.block()->id();
}
}
