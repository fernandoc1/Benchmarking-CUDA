/*
 * BranchInfo.h
 *
 *  Created on: Aug 23, 2010
 *      Author: undead
 */

#ifndef BRANCHINFO_H_
#define BRANCHINFO_H_

#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/graphs/interface/DivergenceGraph.h>
namespace analysis{
/*!\brief Holds information required to analyze a branch */
class BranchInfo{
	public:
		typedef DataflowGraph::Block Block;
		typedef graph_utils::DirectionalGraph::node_type node_type;
		typedef graph_utils::DirectionalGraph::node_set node_set;
		typedef std::set<const Block*> touched_blocks;

		BranchInfo(const Block *block, const Block *postDominator, const node_type &predicate, const DataflowGraph::Instruction &dfgInstruction, const DataflowGraph &dfg, graph_utils::DivergenceGraph &divergGraph);

		/*!\brief Test if a variable ID, from kernel->DFG in SSA form, is marks as divergent, must call populate() before */
		bool isTainted(const node_type &node) const;

		/*!\brief Compute the influence of the branch in variables and blocks */
		void populate();

		/*!\brief Returns the pointer to the block that holds the branch instruction */
		const Block *block() const {return _block;};
		/*!\brief Returns the pointer to the target block of the branch */
		const Block *branch() const {return _branch;};
		/*!\brief Returns the pointer to the fallthtrough block */
		const Block *fallThrough() const {return _fallThrough;};
		/*!\brief Returns the pointer to the block that postdominates block */
		const Block *postDominator() const {return _postDominator;};
		/*!\brief Returns the the predicate ID, in the kernel->DFG, in SSA form */
		const graph_utils::DirectionalGraph::node_type &predicate() const {return _predicate;};
		/*!\brief Returns the reference to the branch instruction */
		const DataflowGraph::Instruction &instruction() const {return _dfgInstruction;};

	private:
		/* Pointer to the block that holds the branch instruction, the target block of the branch, the fallthrough block, the block that postdominates block */
		const Block *_block, *_branch, *_fallThrough, *_postDominator;
		/*!\brief The predicate ID, in the kernel->DFG, in SSA form */
		const node_type &_predicate;
		/*!\brief Reference to the branch instruction */
		const DataflowGraph::Instruction &_dfgInstruction;
		/*!\brief Reference to divergence graph associated to the kernel */
		graph_utils::DivergenceGraph &_divergGraph;

		/*!\brief Blocks dependency from the branch and fallthrough sides */
		touched_blocks _branchBlocks, _fallThroughBlocks;
		/*!\brief Variables dependency from the branch and fallthrough sides */
		node_set _branchVariables, _fallThroughVariables;

		/*!\brief Insert a variable and all it's successors to the dependency list */
		void _insertVariable(node_type &variable, node_set &variables);
		/*!\brief Walks through the blocks instruction, inserting the destiny variables to the dependency list */
		void _taintVariables(const Block *block, node_set &variables);
		/*!\brief Walks through the blocks creating a list of dependency until reach of the postdominator block */
		void _taintBlocks(const Block *start, touched_blocks &blocks, node_set &variables);

};

}
#endif /* BRANCHINFO_H_ */
