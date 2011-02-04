/*
 * BlockExtractor.cpp
 *
 *  Created on: 14/07/2010
 *      Author: coutinho
 */

#include <limits>
#include <ocelot/analysis/interface/BlockMatcher.h>


using namespace analysis;


BlockMatcher::GainMatrix::GainMatrix(analysis::DataflowGraph* dfg, analysis::DataflowGraph::Block& bb1, analysis::DataflowGraph::Block& bb2, InstructionConverter& ic, ir::PTXInstruction::ComputeCapability cap)
	: deviceCapability(cap)
	, verticalBlock(bb1)
	, horizontelBlock(bb2)
	, height()
	, instMatcher(ic)
	, maxRegister(dfg->maxRegister())
{
	ir::ControlFlowGraph::BasicBlock& ptxBlock1 = *(bb1.block());
	ir::ControlFlowGraph::BasicBlock& ptxBlock2 = *(bb2.block());

	// fill instruction vectors
	ir::ControlFlowGraph::BasicBlock::InstructionList::const_iterator it = ptxBlock1.instructions.begin();
	for (; it != ptxBlock1.instructions.end(); it++) {
		instructions1.push_back( (ir::PTXInstruction*)*it );
	}
	it = ptxBlock2.instructions.begin();
	for (; it != ptxBlock2.instructions.end(); it++) {
		instructions2.push_back( (ir::PTXInstruction*)*it );
	}

	height = instructions1.size()+1;
	width  = instructions2.size()+1;

	cells = (GainMatrix::MatrixCell*) calloc(sizeof(GainMatrix::MatrixCell), height*width); //new GainMatrix::MatrixCell[height*width];
	matrix = (GainMatrix::MatrixCell**) malloc(height*sizeof(GainMatrix::MatrixCell*));
	for (unsigned int i=0; i<height; i++) {
		matrix[i] = &cells[i*width];
	}

	// to left position
	matrix[0][0].gain = getInstructionCost(ir::PTXInstruction::Bra, deviceCapability);
	matrix[0][0].lastStep = Match;
	matrix[0][0].paidBranchCost = false;
	matrix[0][0].paidGotoCost = false;
	VariableUnifications emptyAliasMap;
	matrix[0][0].aliasMap = emptyAliasMap;

	// first line
	for (unsigned int i=1; i<width; i++) {
		matrix[0][i].gain = 0;
		matrix[0][i].lastStep = Horizontal;
		matrix[0][i].paidBranchCost = true;
		matrix[0][i].paidGotoCost = false;
		matrix[0][i].aliasMap = emptyAliasMap;
	}

	// first column
	for (unsigned int i=1; i<height; i++) {
		matrix[i][0].gain = 0;
		matrix[i][0].lastStep = Vertical;
		matrix[i][0].paidBranchCost = true;
		matrix[i][0].paidGotoCost = false;
		matrix[i][0].aliasMap = emptyAliasMap;
	}
}

BlockMatcher::GainMatrix::~GainMatrix()
{
	free(cells);
	free(matrix);
}

float BlockMatcher::GainMatrix::calculateMatch()
{
	// check phi instructions
	DataflowGraph::PhiInstructionVector::const_iterator phi = verticalBlock.phis().begin();
	for (; phi != verticalBlock.phis().end(); ++phi) {
		if (phi->s.size() > 1) {
			// Phi instruction with two arguments: we will not handle this,
			// so give up and make sure that we will never come back here.
			return -(std::numeric_limits<float>::infinity());
		}
		alternativeVarNames[phi->d.id] = phi->s[0].id;
	}
	phi = horizontelBlock.phis().begin();
	for (; phi != horizontelBlock.phis().end(); ++phi) {
		if (phi->s.size() > 1) {
			// Phi instruction with two arguments: we will not handle this,
			// so give up and make that sure we will never come back here.
			return -(std::numeric_limits<float>::infinity());
		}
		alternativeVarNames[phi->d.id] = phi->s[0].id;
	}

	// calculate gain
	for (unsigned int row=1; row<height; row++) {
		for (unsigned int col=1; col<width; col++) {
			calculateCellGain(row, col);
		}
	}

	return matrix[height-1][width-1].gain;
}

BlockMatcher::MatrixPath BlockMatcher::GainMatrix::getPath() const
{
	BlockMatcher::MatrixPath path;

	int row = height - 1;
	int col = width - 1;
	while ((row != 0) || (col != 0)) {
		StepDirection d = matrix[row][col].lastStep;
		path.push_front(d);

		switch (d) {
		case Vertical:
			row--;
			break;
		case Horizontal:
			col--;
			break;
		case Match:
		case Substitution:
			row--;
			col--;
			break;
		}
	}

	return path;
}

void BlockMatcher::GainMatrix::calculateCellGain(unsigned int row, unsigned int col)
{
	// get best right direction
	float bestRightGain = 0.0;
	bool paidGotoCost = false;
	StepDirection bestRightMove = getBestSquareMove(row, col, bestRightGain, paidGotoCost);

	// get diagonal gain
	float bestDiagGain = 0.0;
	VariableUnifications diagAliases = matrix[row-1][col-1].aliasMap;
	StepDirection bestDiagonalMove = getBestDiagonalMove(row, col, bestDiagGain, diagAliases);

	// Choose the best. If gains are equal,
	// choose a right direction because it already paid branch penalty.
	MatrixCell& cell = matrix[row][col];
	if (bestDiagGain > bestRightGain) {
		cell.gain           = bestDiagGain;
		cell.lastStep       = bestDiagonalMove;
		cell.paidBranchCost = false;
		cell.paidGotoCost   = false;
		cell.aliasMap       = diagAliases;
	} else {
		cell.gain           = bestRightGain;
		cell.lastStep       = bestRightMove;
		cell.paidBranchCost = true;
		cell.paidGotoCost   = paidGotoCost;
		if (bestRightMove == Vertical) {
			cell.aliasMap = matrix[row-1][col].aliasMap;
		} else {
			cell.aliasMap = matrix[row][col-1].aliasMap;
		}
	}
}

BlockMatcher::StepDirection BlockMatcher::GainMatrix::getBestSquareMove(unsigned int row, unsigned int col, float& gain, bool& paidGotoCost) const
{
	bool horizMovePaidGotoCost = false;
	bool vertMovePaidGotoCost  = false;

	// Vertical Gain
	float vertGain = calculateSquareMoveGain(row, col, vertMovePaidGotoCost, BlockMatcher::Vertical);

	// Horizontal gain
	float horizGain = calculateSquareMoveGain(row, col, horizMovePaidGotoCost, BlockMatcher::Horizontal);

	// select best right direction (defaults to vertical)
	StepDirection bestRightDirection = Vertical;
	float bestRightGain = vertGain;
	bool willPayGotoCost = vertMovePaidGotoCost;
	if (horizGain > vertGain) {
		bestRightDirection = Horizontal;
		bestRightGain      = horizGain;
		willPayGotoCost    = horizMovePaidGotoCost;
	} else if (horizGain < vertGain) {
		bestRightDirection = Vertical;
		bestRightGain      = vertGain;
		willPayGotoCost    = vertMovePaidGotoCost;
	} else {
		// equal gain, choose who already paid more penalties
		if (horizMovePaidGotoCost && !vertMovePaidGotoCost) {
			bestRightDirection = Horizontal;
			bestRightGain      = horizGain;
			willPayGotoCost    = horizMovePaidGotoCost;
		}
	}

	gain = bestRightGain;
	paidGotoCost = willPayGotoCost;
	return bestRightDirection;
}

BlockMatcher::StepDirection BlockMatcher::GainMatrix::getBestDiagonalMove(unsigned int row, unsigned int col, float& gain, VariableUnifications& aliases)
{
	// match
	VariableUnifications matchAliases = aliases;
	float matchGain = calculateMatchGain(row, col, matchAliases);

	// substitution
	VariableUnifications substAliases = aliases;
	float substGain = calculateSubstGain(row, col, substAliases);

	// if both gains are equal, use a match because further
	// instructions could benefit from operand unifications made here
	StepDirection bestDiagonalMove = Match;
	float bestDiagonalGain = matchGain;
	if (substGain > matchGain) {
		bestDiagonalMove = Substitution;
		bestDiagonalGain = substGain;
		aliases = substAliases;
	} else {
		bestDiagonalMove = Match;
		bestDiagonalGain = matchGain;
		aliases = matchAliases;
	}

	gain = bestDiagonalGain;
	return bestDiagonalMove;
}

float BlockMatcher::GainMatrix::calculateSquareMoveGain(unsigned int row, unsigned int col, bool& paidGotoCost, const StepDirection move) const
{
	if ((move != BlockMatcher::Horizontal) && (move != BlockMatcher::Vertical)) {
		assertM(false, "ERROR: calculateSquareMoveGain() asked to calculate a move that isn't horizontal or vertical\n");
		return -(std::numeric_limits<float>::infinity());
	}

	MatrixCell* sourceCell = NULL;
	if (move == BlockMatcher::Horizontal) {
		sourceCell = &(matrix[row][col-1]);
	} else {
		sourceCell = &(matrix[row-1][col]);
	}
	float gain = sourceCell->gain;

	if (sourceCell->paidBranchCost) {
		// verify if it's needed to pay goto cost
		if (sourceCell->lastStep != move) {
			// goto cost divided by half because only one path need it
			gain -= 0.5 * getInstructionCost(ir::PTXInstruction::Bra, deviceCapability);
			paidGotoCost = true;
		} else {
			paidGotoCost = sourceCell->paidGotoCost;
		}
	} else {
		// pay branch cost
		gain -= getInstructionCost(ir::PTXInstruction::Bra, deviceCapability);
		paidGotoCost = false;
	}

	return gain;
}

float BlockMatcher::GainMatrix::calculateMatchGain(unsigned int row, unsigned int col, VariableUnifications& aliases)
{
	MatrixCell& sourceCell = matrix[row-1][col-1];
	float gain = sourceCell.gain;

	// unify instructions
	ir::PTXInstruction matchedInst1;
	ir::PTXInstruction matchedInst2;
	bool matched = instMatcher.normalize(matchedInst1, matchedInst2, *(instructions1[row-1]), *(instructions2[col-1]));

	if (matched) {
		// subtract source operand selections
		gain -= getOperandsCost(matchedInst1, matchedInst2, aliases, deviceCapability);

		// add match gain
		gain += getInstructionCost(matchedInst1, deviceCapability);
	} else {
		// make sure this move will never be made
		gain = -(std::numeric_limits<float>::infinity());
	}

	return gain;
}

float BlockMatcher::GainMatrix::calculateSubstGain(unsigned int row, unsigned int col, VariableUnifications& aliases) const
{
	MatrixCell& sourceCell = matrix[row-1][col-1];
	float gain = 0.0;

	ir::PTXInstruction* ptxInst1 = instructions1[row-1];
	ir::PTXInstruction* ptxInst2 = instructions2[col-1];

	if ((ptxInst1->isPredicated()) || (ptxInst2->isPredicated())) {
		// do not use substitution on instructions already predicated
		gain = -(std::numeric_limits<float>::infinity());
	} else {
		// penalty for increasing register pressure and polluting code
		gain = sourceCell.gain - 2.0;
	}

	return gain;
}

float BlockMatcher::GainMatrix::getOperandsCost(const ir::PTXInstruction& inst1, const ir::PTXInstruction& inst2, VariableUnifications& aliases, const ir::PTXInstruction::ComputeCapability cap)
{
	float cost = 0.0;

	// just test is instruction 1 is using a operand,
	// because we know that instruction 2 does the same that instruction 1
	if (inst1.isUsingOperand(ir::PTXInstruction::PredicateGuard)) {
		cost += getOperandUnificationCost(inst1.pg, inst2.pg, aliases);
	}
	if (inst1.isUsingOperand(ir::PTXInstruction::OperandA)) {
		cost += getOperandUnificationCost(inst1.a, inst2.a, aliases);
	}
	if (inst1.isUsingOperand(ir::PTXInstruction::OperandB)) {
		cost += getOperandUnificationCost(inst1.b, inst2.b, aliases);
	}
	if (inst1.isUsingOperand(ir::PTXInstruction::OperandC)) {
		cost += getOperandUnificationCost(inst1.c, inst2.c, aliases);
	}

	// add destination operands to aliases table
	if (inst1.isUsingOperand(ir::PTXInstruction::OperandD)) {
		if ((inst1.opcode == ir::PTXInstruction::Bra)
					|| (inst1.opcode == ir::PTXInstruction::St)) {
			// in these instructions, D is a source operand
			cost += getOperandUnificationCost(inst1.d, inst2.d, aliases);
		} else {
			BlockMatcher::GainMatrix::addOperandUnification(inst1.d, inst2.d, aliases);
		}
	}

	if (inst1.isUsingOperand(ir::PTXInstruction::OperandQ)) {
		BlockMatcher::GainMatrix::addOperandUnification(inst1.pq, inst2.pq, aliases);
	}

	return cost;
}

float BlockMatcher::GainMatrix::getOperandUnificationCost(const ir::PTXOperand& op1, const ir::PTXOperand& op2, VariableUnifications& aliases)
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
	// destination operands (only this subset of source operands should appear):
	// indirect  + indirect  -> indirect (must have the same offset)
	// register  + register  -> register

	if ((op1.addressMode == ir::PTXOperand::Immediate)
			|| (op1.addressMode == ir::PTXOperand::Address)
			|| (op1.addressMode == ir::PTXOperand::Label)
			|| (op1.addressMode == ir::PTXOperand::Special)
			|| (op2.addressMode == ir::PTXOperand::Immediate)
			|| (op2.addressMode == ir::PTXOperand::Address)
			|| (op2.addressMode == ir::PTXOperand::Label)
			|| (op2.addressMode == ir::PTXOperand::Special)) {

		// could not be identified by register id only
		if (op1.equal(op2)) {
			return 0.0;
		}

		// do not use selp for addresses
		if ((op1.addressMode == ir::PTXOperand::Address)
				|| (op2.addressMode == ir::PTXOperand::Address)) {
			return std::numeric_limits<float>::infinity();
		}

		// TODO: leverage previous selections like we do with operands that
		// could be identified by register id

		// need to use selection
		return getInstructionCost(ir::PTXInstruction::SelP, deviceCapability);
	}

	// op1 and op2 are register or indirect. They could be identified by
	// register id. Get their real names:
	VariableMatch operandMatch;
	VariableAliases::iterator op1RealName = alternativeVarNames.find(op1.reg);
	VariableAliases::iterator op2RealName = alternativeVarNames.find(op2.reg);
	if (op1RealName == alternativeVarNames.end()) {
		operandMatch.first  = op1.reg;
	} else {
		operandMatch.first  = op1RealName->second;
	}
	if (op2RealName == alternativeVarNames.end()) {
		operandMatch.second  = op2.reg;
	} else {
		operandMatch.second  = op2RealName->second;
	}

	// check if variables are the same
	if (operandMatch.first == operandMatch.second) {
		return 0.0;
	}

	// check if variables are already unified
	VariableUnifications::iterator alias = aliases.find(operandMatch);
	if (alias == aliases.end()) {
		// we can't use selp to select predicates
		if (op1.type == ir::PTXOperand::pred) {
			return std::numeric_limits<float>::infinity();
		}

		// variable isn't unified, must use a selection instruction
		addOperandUnification(op1, op2, aliases);

		return getInstructionCost(ir::PTXInstruction::SelP, deviceCapability);
	}

	return 0.0;
}

void BlockMatcher::GainMatrix::addOperandUnification(const ir::PTXOperand& op1, const ir::PTXOperand& op2, VariableUnifications& aliases)
{
	VariableMatch operandMatch;

	// get variables real names
	VariableAliases::iterator op1RealName = alternativeVarNames.find(op1.reg);
	VariableAliases::iterator op2RealName = alternativeVarNames.find(op2.reg);
	if (op1RealName == alternativeVarNames.end()) {
		operandMatch.first  = op1.reg;
	} else {
		operandMatch.first  = op1RealName->second;
	}
	if (op2RealName == alternativeVarNames.end()) {
		operandMatch.second  = op2.reg;
	} else {
		operandMatch.second  = op2RealName->second;
	}

	// check if variables are the same
	if (operandMatch.first == operandMatch.second) {
		return;
	}

	// check if variables are unified
	VariableUnifications::iterator alias = aliases.find(operandMatch);
	if (alias == aliases.end()) {
		// (o1,o2) -> oU isn't in alias table, add it
		DataflowGraph::RegisterId newReg = getNewRegister();
		std::pair<VariableMatch, analysis::DataflowGraph::RegisterId> p(operandMatch, newReg);
		aliases.insert(p);
	} else {
		assertM(false, "addOperandUnification(): trying to rewrite variable unification\n");
	}
}

float BlockMatcher::GainMatrix::getInstructionCost(ir::PTXInstruction::Opcode op, ir::PTXOperand::DataType type, ir::PTXInstruction::ComputeCapability cap) {
	float cost = 0.0;

	switch (op) {
	// native integer operations in all capabilities
	case ir::PTXInstruction::AddC:
	case ir::PTXInstruction::And:
	case ir::PTXInstruction::Bfe:
	case ir::PTXInstruction::Bfi:
	case ir::PTXInstruction::Bfind:
	case ir::PTXInstruction::Bra:
	case ir::PTXInstruction::Brev:
	case ir::PTXInstruction::Brkpt:
	case ir::PTXInstruction::Call:
	case ir::PTXInstruction::Clz:
	case ir::PTXInstruction::CNot:
	case ir::PTXInstruction::Cvta:
	case ir::PTXInstruction::Exit:
	case ir::PTXInstruction::Isspacep:
	case ir::PTXInstruction::Not:
	case ir::PTXInstruction::Or:
	case ir::PTXInstruction::Pmevent:
	case ir::PTXInstruction::Popc:
	case ir::PTXInstruction::Prmt:
	case ir::PTXInstruction::Ret:
	case ir::PTXInstruction::SelP:
	case ir::PTXInstruction::Set:
	case ir::PTXInstruction::SetP:
	case ir::PTXInstruction::Shl:
	case ir::PTXInstruction::Shr:
	case ir::PTXInstruction::SlCt:
	case ir::PTXInstruction::SubC:
	case ir::PTXInstruction::Trap:
	case ir::PTXInstruction::Vote:
	case ir::PTXInstruction::Xor:
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 4.0;
		} else {
			cost = 2.0;
		}
		break;

	// emulated integer operation in cap_1_0
	case ir::PTXInstruction::Sad:
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 32.0; // assume emulation takes 8 instructions
		} else {
			cost = 2.0;
		}
		break;

	// arithmetic instructions that can handle doubles
	case ir::PTXInstruction::Abs:
	case ir::PTXInstruction::Add:
	case ir::PTXInstruction::Cvt:
	case ir::PTXInstruction::Div: // approximate: d=a*rcp(b) (10 FMAs to calculate rcp), exact: cost more
	case ir::PTXInstruction::Fma:
	case ir::PTXInstruction::Mad:
	case ir::PTXInstruction::Max:
	case ir::PTXInstruction::Min:
	case ir::PTXInstruction::Mov:
	case ir::PTXInstruction::Mul:
	case ir::PTXInstruction::Neg:
	case ir::PTXInstruction::Rem: // has higher cost?
	case ir::PTXInstruction::Sub: {
		float baseCost = 2.0;
		if (cap == ir::PTXInstruction::Cap_1_0) {
			baseCost = 4.0;
		}

		if (type == ir::PTXOperand::f64) {
			if (cap == ir::PTXInstruction::Cap_2_0) {
				// fermi level tesla
				cost = baseCost*2.0;
			} else {
				// on geforce cards, double operations are 8 times slower
				cost = baseCost*8.0;
			}
		} else {
			cost = baseCost;
		}
		break;
	}

	// 24-bit integer instructions
	case ir::PTXInstruction::Mad24:
	case ir::PTXInstruction::Mul24:
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 4.0;
		} else {
			cost = 16.0; // assume emulation takes 8 instructions
		}
		break;

	// transcedental floating point operations
	case ir::PTXInstruction::Cos:
	case ir::PTXInstruction::Sin:
	case ir::PTXInstruction::Lg2:
	case ir::PTXInstruction::Rcp: // could be approximated using 10 FMAs by newton method
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 16.0;
		} else {
			cost = 8.0;
		}
		break;

	// more complex transcedental floating point operations
	case ir::PTXInstruction::Ex2:
	case ir::PTXInstruction::Rsqrt:
	case ir::PTXInstruction::Sqrt:
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 32.0;
		} else {
			cost = 16.0;
		}
		break;

	// memory access
	case ir::PTXInstruction::Atom:
		cost = 200.0;
		break;
	case ir::PTXInstruction::Ld:
	case ir::PTXInstruction::Ldu:
		cost = 100.0;
		break;
	case ir::PTXInstruction::Membar:
		cost = 200.0; // todo: get a more accurate estimation
		break;
	case ir::PTXInstruction::Prefetch:
	case ir::PTXInstruction::Prefetchu:
		// these instructions only exist in cap_2_0_geforce or superior
		cost = 2.0;
		break;
	case ir::PTXInstruction::St:
		cost = 50.0; // todo: get a more accurate estimation
		break;
	case ir::PTXInstruction::Tex:
	case ir::PTXInstruction::Txq:
	case ir::PTXInstruction::Suld:
	case ir::PTXInstruction::Sured:
	case ir::PTXInstruction::Sust:
	case ir::PTXInstruction::Suq:
		cost = 20.0; // todo: get a more accurate estimation
		break;

	// video instructions:
	case ir::PTXInstruction::Vabsdiff:
	case ir::PTXInstruction::Vadd:
	case ir::PTXInstruction::Vmad:
	case ir::PTXInstruction::Vmax:
	case ir::PTXInstruction::Vmin:
	case ir::PTXInstruction::Vset:
	case ir::PTXInstruction::Vshl:
	case ir::PTXInstruction::Vshr:
	case ir::PTXInstruction::Vsub:
		// these instructions only exist in cap_2_0_geforce or superior
		cost = 2.0;
		break;

	// other instructions
	case ir::PTXInstruction::Bar:
		if (cap == ir::PTXInstruction::Cap_1_0) {
			cost = 4.0;
		} else {
			cost = 2.0;
		}
		break;
	case ir::PTXInstruction::Red:
		cost = 100.0; // Cost depends on scope. Could be between 10 and 200+ .
		break;
	default:
		return 0.0;
		break;
	}

	return cost;
}

float BlockMatcher::GainMatrix::getInstructionCost(const ir::PTXInstruction& inst, ir::PTXInstruction::ComputeCapability cap)
{
	ir::PTXInstruction::Opcode op = inst.opcode;
	ir::PTXOperand::DataType type = inst.type;

	return getInstructionCost(op, type, cap);
}

void BlockMatcher::GainMatrix::printMatrix() const
{
	std::cout << std::endl << std::endl << "----------------" << std::endl;

	// first line
	std::cout << "\t";
	for (unsigned int i=1; i<width; i++) {
		std::cout << "\t" << instructions2[i-1]->toString(instructions2[i-1]->opcode);
	}
	std::cout << std::endl;

	// other lines
	for (unsigned int i=0; i<height; i++) {
		// first line of each cell: opcode and gain
		if (i>0) {
			std::cout << instructions1[i-1]->toString(instructions1[i-1]->opcode);
		}
		for (unsigned int j=0; j<width; j++) {
			std::cout << "\t" << matrix[i][j].gain;
		}
		std::cout << std::endl;

		// second line: direction
		for (unsigned int j=0; j<width; j++) {
			switch (matrix[i][j].lastStep) {
			case Match:
				std::cout << "\tmatch";
				break;
			case Horizontal:
				std::cout << "\thoriz";
				break;
			case Vertical:
				std::cout << "\tvert";
				break;
			case Substitution:
				std::cout << "\tsubst";
				break;
			}
		}
		std::cout << std::endl;

		// third line: paid branch cost?
		for (unsigned int j=0; j<width; j++) {
			if(matrix[i][j].paidBranchCost) {
				std::cout << "\tb:t";
			} else {
				std::cout << "\tb:f";
			}
		}
		std::cout << std::endl;

		// forth line: paid goto cost?
		for (unsigned int j=0; j<width; j++) {
			if(matrix[i][j].paidGotoCost) {
				std::cout << "\tg:t";
			} else {
				std::cout << "\tg:f";
			}
		}
		std::cout << std::endl;

		// fifth line: variable map size
		for (unsigned int j=0; j<width; j++) {
			std::cout << "\t" << matrix[i][j].aliasMap.size();
		}
		std::cout << std::endl << std::endl;
	}
}

BlockMatcher::BlockMatcher(ir::PTXInstruction::ComputeCapability cap)
	: deviceCapability(cap)
{}

BlockMatcher::~BlockMatcher()
{

}

float BlockMatcher::calculateUnificationGain(analysis::DataflowGraph* dfg, analysis::DataflowGraph::Block& bb1, analysis::DataflowGraph::Block& bb2, MatrixPath& path, InstructionConverter instMatcher, ir::PTXInstruction::ComputeCapability cap) {
	GainMatrix gm = BlockMatcher::GainMatrix(dfg, bb1, bb2, instMatcher, cap);
	float gain = gm.calculateMatch();
	path = gm.getPath();

	// debug
	//gm.printMatrix();

	return gain;
}
