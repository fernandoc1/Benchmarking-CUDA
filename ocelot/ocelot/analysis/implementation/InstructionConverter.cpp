/*
 * InstructionConverter.cpp
 *
 *  Created on: Jun 17, 2010
 *      Author: coutinho
 */

#include <ocelot/analysis/interface/InstructionConverter.h>
#include <ocelot/ir/interface/PTXInstruction.h>

using namespace analysis;

InstructionConverter::InstructionConverter()
{
	opcodePair oPair;
	opcodeConverterPair convPair;

	initCmpConversionTable();

	// Comparison operator conversions
	oPair.first  = ir::PTXInstruction::Set; // set tem bool op e operando opcional
	oPair.second = ir::PTXInstruction::Set;
	convPair.first  = &mustHandleConversionOperators;
	convPair.second = &mustHandleConversionOperators;
	opcodeConversions[oPair] = convPair;

	oPair.first  = ir::PTXInstruction::SetP; // setp tem  bool op, operando opcional e segundo destino opcional
	oPair.second = ir::PTXInstruction::SetP;
	convPair.first  = &mustHandleConversionOperators;
	convPair.second = &mustHandleConversionOperators;
	opcodeConversions[oPair] = convPair;
	// setp e selp podem ser convertidas em set e slct (e vice-versa)

	// add mad
	/*oPair.first  = ir::PTXInstruction::Add;
	oPair.second = ir::PTXInstruction::Mad;
	convPair.first  = &convertAdd2Mad;
	convPair.second = NULL;
	opcodeConversions[oPair] = convPair;

	oPair.first  = ir::PTXInstruction::Mad;
	oPair.second = ir::PTXInstruction::Add;
	convPair.first  = NULL;
	convPair.second = &convertAdd2Mad;
	opcodeConversions[oPair] = convPair;

	// mul mad
	oPair.first  = ir::PTXInstruction::Mul;
	oPair.second = ir::PTXInstruction::Mad;
	convPair.first  = &convertMul2Mad;
	convPair.second = NULL;
	opcodeConversions[oPair] = convPair;

	oPair.first  = ir::PTXInstruction::Mad;
	oPair.second = ir::PTXInstruction::Mul;
	convPair.first  = NULL;
	convPair.second = &convertMul2Mad;
	opcodeConversions[oPair] = convPair;

	// add mul
	oPair.first  = ir::PTXInstruction::Add;
	oPair.second = ir::PTXInstruction::Mul;
	convPair.first  = &convertAdd2Mad;
	convPair.second = &convertMul2Mad;
	opcodeConversions[oPair] = convPair;

	oPair.first  = ir::PTXInstruction::Mul;
	oPair.second = ir::PTXInstruction::Add;
	convPair.first  = &convertMul2Mad;
	convPair.second = &convertAdd2Mad;
	opcodeConversions[oPair] = convPair;
	*/
}

void InstructionConverter::initCmpConversionTable()
{
	cmpOperatorPair  cmpPair;
	cmpConverterPair convPair;

	cmpPair.first = ir::PTXInstruction::Gt;
	cmpPair.second = ir::PTXInstruction::Lt;
	convPair.first  = &convertGreater2Less;
	convPair.second = NULL;
	cmpOperatorConversions[cmpPair] = convPair;

	cmpPair.first = ir::PTXInstruction::Lt;
	cmpPair.second = ir::PTXInstruction::Gt;
	convPair.first  = NULL;
	convPair.second = &convertGreater2Less;
	cmpOperatorConversions[cmpPair] = convPair;

	cmpPair.first = ir::PTXInstruction::Ge;
	cmpPair.second = ir::PTXInstruction::Le;
	convPair.first  = &convertGreater2Less;
	convPair.second = NULL;
	cmpOperatorConversions[cmpPair] = convPair;

	cmpPair.first = ir::PTXInstruction::Le;
	cmpPair.second = ir::PTXInstruction::Ge;
	convPair.first  = NULL;
	convPair.second = &convertGreater2Less;
	cmpOperatorConversions[cmpPair] = convPair;
}

bool InstructionConverter::normalize(DataflowGraph::Instruction& output1, DataflowGraph::Instruction& output2, const DataflowGraph::Instruction& input1, const DataflowGraph::Instruction& input2)
{
	const ir::PTXInstruction* ptxInput1 = static_cast< ir::PTXInstruction* >(input1.i);
	const ir::PTXInstruction* ptxInput2 = static_cast< ir::PTXInstruction* >(input2.i);

	ir::PTXInstruction ptxOutput1;
	ir::PTXInstruction ptxOutput2;

	// check conversions
	bool conversionSuccessful = normalize(ptxOutput1, ptxOutput2, *ptxInput1, *ptxInput2);

	if (conversionSuccessful) {
		output1.label = ptxOutput1.toString();
		output1.d = input1.d;
		output1.s = input1.s;
		output1.i = new  ir::PTXInstruction(ptxOutput1);

		output2.label = ptxOutput2.toString();
		output2.d = input2.d;
		output2.s = input2.s;
		output2.i = new  ir::PTXInstruction(ptxOutput2);
	}

	return conversionSuccessful;
}

bool InstructionConverter::normalize(ir::PTXInstruction& output1, ir::PTXInstruction& output2, const ir::PTXInstruction& input1, const ir::PTXInstruction& input2)
{
	opcodePair oPair;
	oPair.first  = input1.opcode;
	oPair.second = input2.opcode;

	// if there's a conversion
	std::map<opcodePair, opcodeConverterPair>::const_iterator conv = opcodeConversions.find(oPair);
	if (conv != opcodeConversions.end()) {
		if (conv->second.first == &mustHandleConversionOperators) {
			bool status = handleComparisonOperators(output1, output2, input1, input2);
			if (status == false) {
				return false;
			}
		} else {
			// TODO: try to unify bool operations

			applyOpcodeConversion(output1, input1, conv->second.first);
			applyOpcodeConversion(output2, input2, conv->second.second);
		}
	} else {
		// maybe we didin't got a conversion because instruction are the same
		if (input1.opcode == input2.opcode) {
			// just copy
			output1 = input1;
			output2 = input2;
		} else {
			return false;
		}
	}

	// check conversions
	return checkInstructionTypes(output1, output2);
}

ir::PTXInstruction InstructionConverter::convertGreater2Less(const ir::PTXInstruction& src)
{
	ir::PTXInstruction convInst = src;

	// change comparison operator
	switch (convInst.comparisonOperator) {
	case ir::PTXInstruction::Gt:
		convInst.comparisonOperator = ir::PTXInstruction::Lt;
		break;
	case ir::PTXInstruction::Gtu:
		convInst.comparisonOperator = ir::PTXInstruction::Ltu;
		break;
	case ir::PTXInstruction::Ge:
		convInst.comparisonOperator = ir::PTXInstruction::Le;
		break;
	case ir::PTXInstruction::Geu:
		convInst.comparisonOperator = ir::PTXInstruction::Leu;
		break;
	default:
		assertM(false, "convertGreater2Less called for instruction with unhandled operator: "
				<< ir::PTXInstruction::toString(convInst.comparisonOperator));
	}

	// swap source operands
	ir::PTXOperand op = convInst.a;
	convInst.a = convInst.b;
	convInst.b = op;

	return convInst;
}

ir::PTXInstruction InstructionConverter::convertAdd2Mad(const ir::PTXInstruction& src)
{
	ir::PTXInstruction convInst = src;
	convInst.opcode = ir::PTXInstruction::Mad;

	// move operand B to C
	convInst.c = convInst.b;

	// new operand B: 1
	ir::PTXOperand::DataType newOperandBType = convInst.type;
	ir::PTXOperand constOne(newOperandBType, 1.0);
	switch (newOperandBType) {
	case ir::PTXOperand::s8:
	case ir::PTXOperand::s16:
	case ir::PTXOperand::s32:
	case ir::PTXOperand::s64:
		constOne.imm_int  = 1;
		convInst.modifier = ir::PTXInstruction::lo | convInst.modifier;
		break;
	case ir::PTXOperand::u8:
	case ir::PTXOperand::u16:
	case ir::PTXOperand::u32:
	case ir::PTXOperand::u64:
		constOne.imm_uint = 1;
		convInst.modifier = ir::PTXInstruction::lo;
		break;
	case ir::PTXOperand::f16:
	case ir::PTXOperand::f32:
	case ir::PTXOperand::f64:
		constOne.imm_float = 1.0;
		break;
	default:
		assertM(false, "convertAdd2Mad(): unhandled operand type\n");
		break;
	}

	convInst.b = constOne;
	return convInst;
}

ir::PTXInstruction InstructionConverter::convertMul2Mad(const ir::PTXInstruction& src)
{
	ir::PTXInstruction convInst = src;
	convInst.opcode = ir::PTXInstruction::Mad;

	ir::PTXOperand::DataType operandCType = convInst.type;
	if (convInst.modifier & ir::PTXInstruction::wide) {
		operandCType = ir::PTXOperand::shortToWide(convInst.type);
	}

	// add operand C: 0
	ir::PTXOperand constZero(operandCType, 0.0);
	switch (operandCType) {
	case ir::PTXOperand::s8:
	case ir::PTXOperand::s16:
	case ir::PTXOperand::s32:
	case ir::PTXOperand::s64:
		constZero.imm_int  = 0;
		break;
	case ir::PTXOperand::u8:
	case ir::PTXOperand::u16:
	case ir::PTXOperand::u32:
	case ir::PTXOperand::u64:
		constZero.imm_uint = 0;
		break;
	case ir::PTXOperand::f16:
	case ir::PTXOperand::f32:
	case ir::PTXOperand::f64:
		constZero.imm_float = 0.0;
		break;
	default:
		assertM(false, "convertAdd2Mad(): unhandled operand type\n");
		break;
	}

	convInst.c = constZero;
	return convInst;
}

ir::PTXInstruction InstructionConverter::mustHandleConversionOperators(const ir::PTXInstruction& inst)
{
	return inst;
}

bool InstructionConverter::handleComparisonOperators(ir::PTXInstruction& output1, ir::PTXInstruction& output2, const ir::PTXInstruction& input1, const ir::PTXInstruction& input2)
{
	cmpOperatorPair cmpPair;
	cmpPair.first  = input1.comparisonOperator;
	cmpPair.second = input2.comparisonOperator;

	// if there's a conversion
	std::map<cmpOperatorPair, cmpConverterPair>::const_iterator conv = cmpOperatorConversions.find(cmpPair);
	if (conv != cmpOperatorConversions.end()) {
		applyCmpOperatorConversion(output1, input1, conv->second.first);
		applyCmpOperatorConversion(output2, input2, conv->second.second);
	} else {
		// maybe we didin't got a conversion because instruction are the same
		if (input1.comparisonOperator == input2.comparisonOperator) {
			// just copy
			output1 = input1;
			output2 = input2;
		} else {
			return false;
		}
	}

	return true;
}

bool InstructionConverter::checkInstructionTypes(const ir::PTXInstruction& input1, const ir::PTXInstruction& input2) const
{
	if (input1.type != input2.type) {
		return false;
	}

	if (input1.opcode != input2.opcode) {
		return false;
	}

	// check address apace
	if (input1.opcode == ir::PTXInstruction::Ldu) {
		if ((input1.addressSpace != ir::PTXInstruction::Global)
				|| (input2.addressSpace != ir::PTXInstruction::Global)) {
			// only global address space allowed for ldu
			return false;
		}
	}
	if ((input1.opcode == ir::PTXInstruction::Ld)
			|| (input1.opcode == ir::PTXInstruction::St)) {
		if (input1.addressSpace != input2.addressSpace) {
			return false;
		}
	}

	// check .uni field
	if ((input1.opcode == ir::PTXInstruction::Bra)
			|| (input1.opcode == ir::PTXInstruction::Call)
			|| (input1.opcode == ir::PTXInstruction::Ret)) {
		if (input1.uni != input2.uni) {
			return false;
		}
	}

	// TODO: check more instruction operators

	// TODO: unify predicated instructions with non-predicated ones
	if (false == checkOperand(input1, input2, ir::PTXInstruction::PredicateGuard)) return false;
	if (false == checkOperand(input1, input2, ir::PTXInstruction::OperandA)) return false;
	if (false == checkOperand(input1, input2, ir::PTXInstruction::OperandB)) return false;
	if (false == checkOperand(input1, input2, ir::PTXInstruction::OperandC)) return false;
	if (false == checkOperand(input1, input2, ir::PTXInstruction::OperandD)) return false;
	if (false == checkOperand(input1, input2, ir::PTXInstruction::OperandQ)) return false;

	return true;
}

void InstructionConverter::printOpcdeConversionTable() const
{
	std::map<opcodePair, opcodeConverterPair>::const_iterator conv = opcodeConversions.begin();
	for (; conv != opcodeConversions.end(); conv++) {
		std::cout << ir::PTXInstruction::toString( conv->first.first ) << std::endl;
		std::cout << ir::PTXInstruction::toString( conv->first.second ) << std::endl;

		std::cout << (unsigned long) conv->second.first << std::endl;
		std::cout << (unsigned long) conv->second.second << std::endl;

		std::cout << "-------" << std::endl;
	}
}

void InstructionConverter::printCmpOperatorConversionTable() const
{
	std::map<cmpOperatorPair, cmpConverterPair>::const_iterator conv =  cmpOperatorConversions.begin();
	for (; conv != cmpOperatorConversions.end(); conv++) {
		std::cout << ir::PTXInstruction::toString( conv->first.first ) << std::endl;
		std::cout << ir::PTXInstruction::toString( conv->first.second ) << std::endl;

		std::cout << (unsigned long) conv->second.first << std::endl;
		std::cout << (unsigned long) conv->second.second << std::endl;

		std::cout << "-------" << std::endl;
	}
}

bool InstructionConverter::checkOperand(const ir::PTXInstruction& input1, const ir::PTXInstruction& input2, const ir::PTXInstruction::Operand op) const
{
	if (input1.isUsingOperand(op)) {
		if (!input2.isUsingOperand(op)) {
			return false;
		}
	} else {
		if (input2.isUsingOperand(op)) {
			return false;
		} else {
			// unused operand
			return true;
		}
	}

	// get operands to test them
	const ir::PTXOperand* operand1 = input1.getConstOperand(op);
	const ir::PTXOperand* operand2 = input2.getConstOperand(op);
	return checkOperandTypes(*operand1, *operand2);
}

bool InstructionConverter::checkOperandTypes(const ir::PTXOperand& op1, const ir::PTXOperand& op2) const
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

	if (op1.addressMode != op2.addressMode) {
		// these types are compatibles: register, immediate, special
		if ((op1.addressMode == ir::PTXOperand::Register) ||
				(op1.addressMode == ir::PTXOperand::Immediate) ||
				(op1.addressMode == ir::PTXOperand::Special)) {
			if ( !((op2.addressMode == ir::PTXOperand::Register) ||
					(op2.addressMode == ir::PTXOperand::Immediate) ||
					(op2.addressMode == ir::PTXOperand::Special)) ) {
				return false;
			}
		} else {
			// these types are compatibles: indirect, address
			if ((op1.addressMode == ir::PTXOperand::Address) ||
					(op1.addressMode == ir::PTXOperand::Indirect)) {
				if ( !((op2.addressMode == ir::PTXOperand::Address) ||
						(op2.addressMode == ir::PTXOperand::Indirect)) ) {
					return false;
				}
			} else {
				return false;
			}
		}
	}

	if (op1.type != op2.type) {
		return false;
	}

	if (op1.addressMode == ir::PTXOperand::Invalid) {
		// this function checks required operands
		assertM(false, "checkInstructionOperand() is comparing two instruction with invalid operands\n");
		return false;
	}

	// No special verification for immediates.
	// If they are different, generator will add selection instructions.

	// indirect and address operand checks
	if ((op1.addressMode == ir::PTXOperand::Address) ||
		(op1.addressMode == ir::PTXOperand::Indirect)) {

		if (op1.offset != op2.offset) {
			return false;
		}
	}

	// label
	if (op1.addressMode == ir::PTXOperand::Label) {
		if (0 != op1.identifier.compare(op2.identifier)) {
			// different labels
			return false;
		}
	}

	// no special verification for specials

	// for now vector operands can't be unified
	if ((op1.vec!=ir::PTXOperand::v1) || (op2.vec!=ir::PTXOperand::v1)) {
		// TODO: try to unify vector operands
		return false;
	}

	return true;
}
