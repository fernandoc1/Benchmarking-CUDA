/*
 * BlockUnificationPass.h
 *
 *  Created on: Aug 9, 2010
 *      Author: coutinho
 */

#ifndef BLOCKUNIFICATIONPASS_H_
#define BLOCKUNIFICATIONPASS_H_

#include <ocelot/analysis/interface/Pass.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/BlockMatcher.h>
#include <ocelot/analysis/interface/BlockExtractor.h>
#include <set>


namespace analysis {

/*! \brief A class for a pass that unifies basic blocks from divergent branches
    of a PTX kernel.

	This implementation identifies divergent branches and unify the basic
	blocks that are target of these branches to reduce divergent execution.
*/
class BlockUnificationPass : public KernelPass
{
private:
	/// return true if the branch points to basic blocks that could be unified,
	/// i.e. a edge don't point to a block that is postdominator of the block
	/// pointed by another edge
	bool thereIsPathFromB1toB2(ir::ControlFlowGraph::const_iterator B1,
			ir::ControlFlowGraph::const_iterator B2,
			ir::ControlFlowGraph::const_iterator c,
			std::set<ir::ControlFlowGraph::BasicBlock*>* visited) const;

	/// Unify basic blocks target1 and target2 in the dataflow graph
	void weaveBlocks(DataflowGraph::iterator branchBlock, DataflowGraph::iterator target1, DataflowGraph::iterator target2, BlockMatcher::MatrixPath& extractionPath, DataflowGraph* dfg);

	/// This method converts every variable in the original program to the new
	/// name that branch fusion has found for this variable.
	void replaceRegisters(DataflowGraph* dfg, BlockExtractor& extractor);

public:
	BlockUnificationPass();
	virtual ~BlockUnificationPass();

	/*! \brief Initialize the pass using a specific module */
	virtual void initialize( const ir::Module& m );
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnKernel( ir::Kernel& k );
	/*! \brief Finalize the pass */
	virtual void finalize();
};

}

#endif /* BLOCKUNIFICATIONPASS_H_ */
