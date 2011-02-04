/*
 * BlockExtractor.h
 *
 *  Created on: Sep 15, 2010
 *      Author: coutinho
 */

#ifndef BLOCKEXTRACTOR_H_
#define BLOCKEXTRACTOR_H_

#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/BlockMatcher.h>
#include <ocelot/analysis/interface/InstructionConverter.h>


namespace analysis {

class BlockExtractor {
private:
	DataflowGraph* dfg;

	const DataflowGraph::const_iterator source1;
	const DataflowGraph::const_iterator source2;
	const BlockMatcher::MatrixPath& extractionPath;

	DataflowGraph::InstructionVector::const_iterator source1Pos;
	DataflowGraph::InstructionVector::const_iterator source2Pos;
	BlockMatcher::MatrixPath::const_iterator extractionPathPos;

	InstructionConverter instConverter;
	VariableAliases operandAliases;
	VariableUnifications operandUnifications;
	VariableAliases operandSubstitutions;

	const ir::PTXOperand& branchCondition;

	/// Add a new instruction to a unified basic block
	void addUnifiedInstruction(DataflowGraph::iterator unifiedBlock);

	/// Add the implementation of a substitution: one instruction of each
	/// source basic block, using guards to control their execution.
	void addSubstitution(DataflowGraph::iterator unifiedBlock);

	/// Insert (making conversions in operandSubstitutions) an instruction in a basic block
	void checkAndInsertInstruction(DataflowGraph::iterator block, ir::Instruction* source);

	/// Check all operands of a unified instruction,
	/// replacing operands by unified ones.
	void checkOperandUnifications(ir::PTXInstruction& unifiedInst, ir::PTXInstruction& source1, ir::PTXInstruction& source2, DataflowGraph::iterator block);

	/// Transforms unifiedOperand into a unified operand capable of replacing
	/// both source1 and source2. Returns true if the unified operand was new
	/// and false otherwise.
	bool getUnifiedOperand(ir::PTXOperand& unifiedOperand, ir::PTXOperand& source1, ir::PTXOperand& source2, DataflowGraph::iterator block);

	/// Get the replacement of a operand.
	ir::PTXOperand getOperandReplacement(ir::PTXOperand& op);

	/// Ocelot renames variables at each basic block, so we need this function
	/// to get a variable real name
	DataflowGraph::RegisterId getVariableRealName(DataflowGraph::RegisterId op)
	{
		VariableAliases::iterator opRealName = operandAliases.find(op);
		if (opRealName == operandAliases.end()) {
			return op;
		} else {
			return opRealName->second;
		}
	}

	/// Add in the basic block a selection instruction selecting between
	/// source1 or source2 to be put unifiedOp based on branch predicate.
	/// source1 is the input that was used in fallthrough block (vertical in
	/// the gain matrix) and source2 is the input that was used in the branch
	/// block (horizontal in the gain matrix).
	void addSelectionInstruction(ir::PTXOperand& unifiedOp, ir::PTXOperand& source1, ir::PTXOperand& source2, DataflowGraph::iterator block);

	/// Add a entry in operand replacement table.
	/// 	\param newOperand New operand name.
	///		\param source Replaced operand.
	void addOperandReplacement(ir::PTXOperand& newOperand, ir::PTXOperand& source);

public:
	BlockExtractor(DataflowGraph* graph, const DataflowGraph::const_iterator block1, const DataflowGraph::const_iterator block2, const BlockMatcher::MatrixPath& path, const ir::PTXOperand& branchPredicate);

	virtual ~BlockExtractor() {}

	bool hasNext()
	{
		return extractionPathPos != extractionPath.end();
	}

	BlockMatcher::StepDirection nextStep()
	{
		return *extractionPathPos;
	}

	/// Returns a map that points, for each variable in the original program,
	/// the new name of this variable in the optimized program.
	inline const VariableAliases& getNewRegisterNames() const {
		return this->operandSubstitutions;
	}

	/// Check if some operand was replaced and change reference by the new one
	void checkOperandReplacements(ir::PTXInstruction& inst);

	/// Create one unified basic block following diagonals extractionPath
	void extractUnifiedBlock(DataflowGraph::iterator unifiedBlock);

	/// Create one or two divergent basic blocks following horizontal and
	/// vertical steps in extractionPath
	void extractDivergentBlocks(DataflowGraph::iterator verticalBlock, DataflowGraph::iterator horizontalBlock);
};

}

#endif /* BLOCKEXTRACTOR_H_ */
