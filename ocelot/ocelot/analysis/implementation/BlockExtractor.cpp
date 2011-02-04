/*
 * BlockExtractor.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: coutinho
 */

#include <ocelot/analysis/interface/BlockExtractor.h>

using namespace analysis;

BlockExtractor::BlockExtractor(DataflowGraph* graph, const DataflowGraph::const_iterator block1, const DataflowGraph::const_iterator block2, const BlockMatcher::MatrixPath& path, const ir::PTXOperand& branchPredicate)
: dfg(graph)
, source1(block1)
, source2(block2)
, extractionPath(path)
, source1Pos(block1->instructions().begin())
, source2Pos(block2->instructions().begin())
, extractionPathPos(extractionPath.begin())
, branchCondition(branchPredicate)
{
	// fill operandAliases map
	DataflowGraph::PhiInstructionVector::const_iterator phi = block1->phis().begin();
	for (; phi != block1->phis().end(); ++phi) {
		assert(phi->s.size() == 1);
		operandAliases[phi->d.id] = phi->s[0].id;
		operandSubstitutions[phi->d.id] = phi->s[0].id;
	}
	phi = block2->phis().begin();
	for (; phi != block2->phis().end(); ++phi) {
		assert(phi->s.size() == 1);
		operandAliases[phi->d.id] = phi->s[0].id;
		operandSubstitutions[phi->d.id] = phi->s[0].id;
	}
}

void BlockExtractor::extractUnifiedBlock(DataflowGraph::iterator unifiedBlock)
{
	while ((extractionPathPos != extractionPath.end()) &&
			(*extractionPathPos == BlockMatcher::Match ||
			*extractionPathPos == BlockMatcher::Substitution)) {
		if (*extractionPathPos == BlockMatcher::Match) {
			addUnifiedInstruction(unifiedBlock);
		} else {
			addSubstitution(unifiedBlock);
		}
		source1Pos++;
		source2Pos++;
		extractionPathPos++;
	}
}

void BlockExtractor::extractDivergentBlocks(DataflowGraph::iterator verticalBlock, DataflowGraph::iterator horizontalBlock)
{
	while ( (extractionPathPos != extractionPath.end()) &&
			(*extractionPathPos == BlockMatcher::Horizontal ||
			*extractionPathPos == BlockMatcher::Vertical) ) {
		if (*extractionPathPos == BlockMatcher::Vertical) {
			checkAndInsertInstruction(verticalBlock, source1Pos->i);
			source1Pos++;
			extractionPathPos++;
		} else {
			checkAndInsertInstruction(horizontalBlock, source2Pos->i);
			source2Pos++;
			extractionPathPos++;
		}
	}
}

void BlockExtractor::addUnifiedInstruction(DataflowGraph::iterator unifiedBlock)
{
	// convert instructions
	DataflowGraph::Instruction converted1;
	DataflowGraph::Instruction converted2;
	bool success = instConverter.normalize(converted1, converted2, *source1Pos, *source2Pos);
	if (!success) {
		std::cerr << "addUnifiedInstruction(): can't unify a instruction that unification path said that we could" << std::endl;
		exit(EXIT_FAILURE);
	}

	ir::PTXInstruction* conv1Ptx = static_cast<ir::PTXInstruction*>(converted1.i);
	ir::PTXInstruction* conv2Ptx = static_cast<ir::PTXInstruction*>(converted2.i);
	ir::PTXInstruction unifiedInstPtx = *conv1Ptx;
	ir::Instruction& unifiedInst = unifiedInstPtx;

	checkOperandUnifications(unifiedInstPtx, *conv1Ptx, *conv2Ptx, unifiedBlock);

	std::cerr << "addUnifiedInstruction(): issuing " << unifiedInstPtx.toString() << std::endl;

	dfg->insert(unifiedBlock, unifiedInst);
}

void BlockExtractor::addSubstitution(DataflowGraph::iterator unifiedBlock)
{
	ir::PTXInstruction* fallthoughInstPtx = static_cast<ir::PTXInstruction*>(source1Pos->i);
	ir::PTXInstruction newFallthoughPtx = *fallthoughInstPtx;
	// fallthroughBlock would be visited if branchCondition was false, so
	// fallthoughInst execution condition is the inverse of branchCondition
	newFallthoughPtx.pg = branchCondition;
	newFallthoughPtx.pg.invertPredicateCondition();
	// remove .uni property in branches
	if (newFallthoughPtx.opcode == ir::PTXInstruction::Bra) {
		newFallthoughPtx.uni = false;
	}
	checkAndInsertInstruction(unifiedBlock, &newFallthoughPtx);

	ir::PTXInstruction* branchInstPtx = static_cast<ir::PTXInstruction*>(source2Pos->i);
	ir::PTXInstruction newBranchPtx = *branchInstPtx;
	// branchBlock would be visited if branchCondition was true, so branchInst
	// execution condition is the same as branchCondition
	newBranchPtx.pg = branchCondition;
	// remove .uni property in branches
	if (newBranchPtx.opcode == ir::PTXInstruction::Bra) {
		newBranchPtx.uni = false;
	}
	checkAndInsertInstruction(unifiedBlock, &newBranchPtx);
}

void BlockExtractor::checkAndInsertInstruction(DataflowGraph::iterator block, ir::Instruction* source)
{
	ir::PTXInstruction* sourcePtx = static_cast<ir::PTXInstruction*>(source);
	ir::PTXInstruction newInstPtx = *sourcePtx;
	ir::Instruction& newInst = newInstPtx;

	// Even if a operand appears in some alias table we will use the original,
	// except if it appears in newOperandAliases, meaning that the original
	// dosen't exists anymore.
	checkOperandReplacements(newInstPtx);

	std::cerr << "copyDivergentInstruction(): issuing " << newInstPtx.toString() << std::endl;

	dfg->insert(block, newInst);
}

void BlockExtractor::checkOperandUnifications(ir::PTXInstruction& unifiedInst, ir::PTXInstruction& source1, ir::PTXInstruction& source2, DataflowGraph::iterator block)
{
	// check source operands
	if (unifiedInst.isUsingOperand(ir::PTXInstruction::PredicateGuard)) {
		bool newOperand = getUnifiedOperand(unifiedInst.pg, source1.pg, source2.pg, block);
		if (newOperand) {
			BlockExtractor::addSelectionInstruction(unifiedInst.pg, source1.pg, source2.pg, block);
		}
	}
	if (unifiedInst.isUsingOperand(ir::PTXInstruction::OperandA)) {
		bool newOperand = getUnifiedOperand(unifiedInst.a, source1.a, source2.a, block);
		if (newOperand) {
			BlockExtractor::addSelectionInstruction(unifiedInst.a, source1.a, source2.a, block);
		}
	}
	if (unifiedInst.isUsingOperand(ir::PTXInstruction::OperandB)) {
		bool newOperand = getUnifiedOperand(unifiedInst.b, source1.b, source2.b, block);
		if (newOperand) {
			BlockExtractor::addSelectionInstruction(unifiedInst.b, source1.b, source2.b, block);
		}
	}
	if (unifiedInst.isUsingOperand(ir::PTXInstruction::OperandC)) {
		bool newOperand = getUnifiedOperand(unifiedInst.c, source1.c, source2.c, block);
		if (newOperand) {
			BlockExtractor::addSelectionInstruction(unifiedInst.c, source1.c, source2.c, block);
		}
	}

	// check destination operands
	if (unifiedInst.isUsingOperand(ir::PTXInstruction::OperandD)) {
		bool newOperand = getUnifiedOperand(unifiedInst.d, source1.d, source2.d, block);
		if (newOperand) {
			if ((unifiedInst.opcode == ir::PTXInstruction::Bra)
					|| (unifiedInst.opcode == ir::PTXInstruction::St)) {
				// in these instructions, D is a source operand
				BlockExtractor::addSelectionInstruction(unifiedInst.d, source1.d, source2.d, block);
			} else {
				BlockExtractor::addOperandReplacement(unifiedInst.d, source1.d);
				BlockExtractor::addOperandReplacement(unifiedInst.d, source2.d);
			}
		}
	}

	if (unifiedInst.isUsingOperand(ir::PTXInstruction::OperandQ)) {
		bool newOperand = getUnifiedOperand(unifiedInst.pq, source1.pq, source2.pq, block);
		if (newOperand) {
			BlockExtractor::addOperandReplacement(unifiedInst.pq, source1.pq);
			BlockExtractor::addOperandReplacement(unifiedInst.pq, source2.pq);
		}
	}
}

void BlockExtractor::checkOperandReplacements(ir::PTXInstruction& inst)
{
	// check source operands
	if (inst.isUsingOperand(ir::PTXInstruction::PredicateGuard))
		inst.pg = getOperandReplacement(inst.pg);
	if (inst.isUsingOperand(ir::PTXInstruction::OperandA))
		inst.a  = getOperandReplacement(inst.a);
	if (inst.isUsingOperand(ir::PTXInstruction::OperandB))
		inst.b  = getOperandReplacement(inst.b);
	if (inst.isUsingOperand(ir::PTXInstruction::OperandC))
		inst.c  = getOperandReplacement(inst.c);

	// check destination operands
	if ((inst.opcode == ir::PTXInstruction::Bra)
			|| (inst.opcode == ir::PTXInstruction::St)) {
		// St doesn't write in operand D, handle it like a source operand
		inst.d  = getOperandReplacement(inst.d);
	}
}

bool BlockExtractor::getUnifiedOperand(ir::PTXOperand& unifiedOperand, ir::PTXOperand& source1, ir::PTXOperand& source2, DataflowGraph::iterator block)
{

	// We accept the following unifications:
	//
	// source operands:
	// address   + address   -> indirect (must have the same offset)
	// address   + indirect  -> indirect (must have the same offset)
	// indirect  + indirect  -> indirect (must have the same offset)
	// register  + register  -> register
	// register  + immediate -> register
	// register  + special   -> register
	// special   + immediate -> register
	// special   + special   -> register
	// immediate + immediate -> register
	// label     + label     -> label (only if they are the same)
	//
	// destination operands:
	// indirect  + indirect  -> indirect (must have the same offset)
	// register  + register  -> register

	if ((source1.addressMode == ir::PTXOperand::Immediate)
			|| (source1.addressMode == ir::PTXOperand::Address)
			|| (source1.addressMode == ir::PTXOperand::Label)
			|| (source1.addressMode == ir::PTXOperand::Special)
			|| (source2.addressMode == ir::PTXOperand::Immediate)
			|| (source2.addressMode == ir::PTXOperand::Address)
			|| (source2.addressMode == ir::PTXOperand::Label)
			|| (source2.addressMode == ir::PTXOperand::Special)) {

		// could not be identified by register id only
		if (source1.equal(source2)) {
			unifiedOperand = source1;
			return false;
		}

		// TODO: leverage previous selections like we do with operands that
		// could be identified by register id

		// Need to use selection. Assemble a PTXOperand that will hold the
		// unified value:

		// identifier
		unifiedOperand.reg = dfg->newRegister();
		std::stringstream stream;
		if (unifiedOperand.type == ir::PTXOperand::pred) {
			stream << "%p" << unifiedOperand.reg;
		} else {
			stream << "%r" << unifiedOperand.reg;
		}
		unifiedOperand.identifier = stream.str();

		// address mode and offset
		if ((source1.addressMode == ir::PTXOperand::Address)
				|| (source1.addressMode == ir::PTXOperand::Indirect)) {
			unifiedOperand.addressMode = ir::PTXOperand::Indirect;
			unifiedOperand.offset = source1.offset;
		} else {
			unifiedOperand.addressMode = ir::PTXOperand::Register;
		}

		// data type
		unifiedOperand.type = source1.type;

		// extra predicate data
		if (unifiedOperand.type == ir::PTXOperand::pred) {
			unifiedOperand.condition = ir::PTXOperand::Pred;
		}

		// currently we only accept unification of non-vector operands
		unifiedOperand.vec = ir::PTXOperand::v1;

		return true;
	}

	// op1 and op2 are register or indirect. They could be identified by
	// register id. Get their real names:
	VariableMatch operandMatch;
	operandMatch.first = getVariableRealName(source1.reg);
	operandMatch.second = getVariableRealName(source2.reg);

	// check if variables are the same
	if (operandMatch.first == operandMatch.second) {
		// already unified
		unifiedOperand = source1;
		unifiedOperand.reg = getVariableRealName(source1.reg);
		return false;
	}

	// check if operands are already unified
	bool isNewOperand = false;
	VariableUnifications::iterator alias = operandUnifications.find(operandMatch);
	if (alias == operandUnifications.end()) {
		// (o1,o2) -> oU isn't in alias table, add it
		DataflowGraph::RegisterId newReg = dfg->newRegister();
		operandUnifications[operandMatch] = newReg;
		alias = operandUnifications.find(operandMatch);

		isNewOperand = true;
	}

	// create a PTXOperand representation of the unified operand
	unifiedOperand = source1;
	unifiedOperand.reg = alias->second;
	std::stringstream stream;
	if (unifiedOperand.type == ir::PTXOperand::pred) {
		stream << "%p" << unifiedOperand.reg;
	} else {
		stream << "%r" << unifiedOperand.reg;
	}
	unifiedOperand.identifier = stream.str();

	return isNewOperand;
}

ir::PTXOperand BlockExtractor::getOperandReplacement(ir::PTXOperand& op)
{
	if ((op.addressMode == ir::PTXOperand::Register)
			|| (op.addressMode == ir::PTXOperand::Indirect)) {
		ir::PTXOperand replacement(op);
		DataflowGraph::RegisterId opRealName = getVariableRealName(op.reg);

		VariableAliases::const_iterator it = operandSubstitutions.find(opRealName);
		if (it != operandSubstitutions.end()) {
			replacement.reg = it->second;

			std::stringstream stream;
			if (replacement.type == ir::PTXOperand::pred) {
				stream << "%p" << replacement.reg;
			} else {
				stream << "%r" << replacement.reg;
			}
			replacement.identifier = stream.str();
		} else {
			replacement.reg = opRealName;
		}

		return replacement;
	} else {
		return op;
	}
}

void BlockExtractor::addSelectionInstruction(ir::PTXOperand& unifiedOp, ir::PTXOperand& source1, ir::PTXOperand& source2, DataflowGraph::iterator block)
{
	// This function accepts the following unifications of source operands:
	//
	// address   + address   -> indirect (converted to register)
	// address   + indirect (converted to register) -> indirect (converted to register)
	// indirect (converted to register) + indirect (converted to register) -> indirect (converted to register)
	// register  + register  -> register
	// register  + immediate -> register
	// register  + special   -> register
	// special   + immediate -> register
	// special   + special   -> register
	// immediate + immediate -> register
	// label     + label     -> label (only if both labels are the same)

	// add to block: selp oU, o2, o1, p
	// Variable o2 is of the branch block (that would be taken if predicate is
	// true) and o1 is fallthrough variable (that would be taken if predicate
	// is false.
	ir::PTXInstruction selectionPtx(ir::PTXInstruction::SelP);

	// Set operands, converting indirect operands to register type
	// because all Selp operands are registers.
	selectionPtx.setDestination(unifiedOp);
	selectionPtx.d.reg = getVariableRealName(selectionPtx.d.reg);
	if (selectionPtx.d.addressMode != ir::PTXOperand::Register) {
		selectionPtx.d.addressMode = ir::PTXOperand::Register;
	}

	ir::PTXOperand* first = NULL;
	ir::PTXOperand* second = NULL;
	if ((branchCondition.condition == ir::PTXOperand::nPT)
			|| (branchCondition.condition == ir::PTXOperand::InvPred)) {
		first = &source1;
		second = &source2;
	} else {
		first = &source2;
		second = &source1;
	}
	selectionPtx.setOperandA(*first);
	selectionPtx.a.reg = getVariableRealName(selectionPtx.a.reg);
	if (selectionPtx.a.addressMode == ir::PTXOperand::Indirect) {
		selectionPtx.a.addressMode = ir::PTXOperand::Register;
	}
	selectionPtx.setOperandB(*second);
	selectionPtx.b.reg = getVariableRealName(selectionPtx.b.reg);
	if (selectionPtx.b.addressMode == ir::PTXOperand::Indirect) {
		selectionPtx.b.addressMode = ir::PTXOperand::Register;
	}
	selectionPtx.setOperandC(branchCondition);

	std::cerr << "addSelectionInstruction(): issuing " << selectionPtx.toString() << std::endl;

	ir::Instruction& selection = selectionPtx;
	dfg->insert(block, selection);
}

void BlockExtractor::addOperandReplacement(ir::PTXOperand& newOperand, ir::PTXOperand& source)
{
	DataflowGraph::RegisterId sourceId = getVariableRealName(source.reg);
	DataflowGraph::RegisterId newId = newOperand.reg;

	operandSubstitutions[sourceId] = newId;
}
