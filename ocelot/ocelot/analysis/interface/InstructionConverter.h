/*
 * InstructionConverter.h
 *
 *  Created on: Jun 17, 2010
 *      Author: coutinho
 */

#ifndef INSTRUCTIONMATCHER_H_
#define INSTRUCTIONMATCHER_H_

#include <map>
#include <ocelot/analysis/interface/DataflowGraph.h>

namespace analysis {

/// These are the prototypes of the functions that convert instructions.
typedef ir::PTXInstruction (opcodeConverterFunc)(const ir::PTXInstruction& inst);
typedef ir::PTXInstruction (cmpOperatorConverterFunc)(const ir::PTXInstruction& inst);

/// \brief: Our converter is table based, that is, in order to find out how to
/// convert one instruction to another, we must do a lookup into a table. This
/// table maps pairs of opcodes, e.g, op1 and op2, to pairs of converters, e.g
/// op1_2_opX and op2_2_opX, so that op1_2_opX(op1) produces a new instruction
/// that is compatible with op2, but that may not be necessarily the same as
/// op2.
typedef std::pair<ir::PTXInstruction::Opcode, ir::PTXInstruction::Opcode> opcodePair;
typedef std::pair<ir::PTXInstruction::CmpOp, ir::PTXInstruction::CmpOp> cmpOperatorPair;

typedef std::pair<opcodeConverterFunc*, opcodeConverterFunc*> opcodeConverterPair;
typedef std::pair<cmpOperatorConverterFunc*, cmpOperatorConverterFunc*> cmpConverterPair;

/// This is the comparison function that we use to index the table of
/// opcode pairs.
struct ltOpcodePair {
	bool operator()(const opcodePair p1, const opcodePair p2) const
	{
		if (p1.first < p2.first) {
			return true;
		}
		if (p1.first > p2.first) {
			return false;
		}

		// first is equal
		if (p1.second < p2.second) {
			return true;
		}
		return false;
	}
};

/// This is the comparison function that we use to index the table of
/// pairs of comparison operators.
struct ltCmpOperatorPair {
	bool operator()(const cmpOperatorPair p1, const cmpOperatorPair p2) const
	{
		if (p1.first < p2.first) {
			return true;
		}
		if (p1.first > p2.first) {
			return false;
		}

		// first is equal
		if (p1.second < p2.second) {
			return true;
		}
		return false;
	}
};


class InstructionConverter {
public:

	/// Conversion tables.
	typedef std::map<opcodePair, opcodeConverterPair, ltOpcodePair> OpcodeConversionTable;
	typedef std::map<cmpOperatorPair, cmpConverterPair, ltCmpOperatorPair> CmpOperatorConversionTable;

private:

	/// This is the table that tells us how to convert one opcode to another.
	OpcodeConversionTable opcodeConversions;

	/// This is the table that tells us how to convert a comparison operator
	/// to another.
	CmpOperatorConversionTable cmpOperatorConversions;

public:
	InstructionConverter();
	virtual ~InstructionConverter() {};

	/// This method normalizes two instructions, trying to find an identity
	/// between them. If the instructions have the same opcode, then the method
	/// does nothing, otherwise it tries to find a transformation between these
	/// instructions in such a way that they will have the same opcode.
	/// 	\param output1 the normalized copy of the first instruction.
	///		\param output2 the normalized copy of the second instruction.
	///		\param input1 the first input instruction.
	///		\param input2 the second input instruction.
	///		\return returns true if a normalization is possible, and false
	///		otherwise. In case the normalization fails, then it is not possible
	///		to transform the opcode of any instruction in such a way that both
	///		will have the same opcode.
	bool normalize(
			DataflowGraph::Instruction& output1,
			DataflowGraph::Instruction& output2,
			const DataflowGraph::Instruction& input1,
			const DataflowGraph::Instruction& input2);

	bool normalize(
				ir::PTXInstruction& output1,
				ir::PTXInstruction& output2,
				const ir::PTXInstruction& input1,
				const ir::PTXInstruction& input2);

private:

	/// This function is used by the ctor to initialize the table that holds
	/// the converters of comparison operators.
	void initCmpConversionTable();

	void inline applyOpcodeConversion(ir::PTXInstruction& output, const ir::PTXInstruction& input, opcodeConverterFunc* converter)
	{
		if (converter == NULL) {
			// nothing to do, just copy
			output = input;
		} else {
			output = (*converter)(input);
		}
	}

	void inline applyCmpOperatorConversion(ir::PTXInstruction& output, const ir::PTXInstruction& input, cmpOperatorConverterFunc* converter)
	{
		if (converter == NULL) {
			// nothing to do, just copy
			output = input;
		} else {
			output = (*converter)(input);
		}
	}

	/// this function is used as a stub in the conversion table, so that once
	/// we find it, it send us to handleComparsionOperators.
	static ir::PTXInstruction mustHandleConversionOperators(const ir::PTXInstruction& inst);

	/// This function is analogous to normalize(...), but it only deals with
	/// comparison operators.
	bool handleComparisonOperators(
			ir::PTXInstruction& output1,
			ir::PTXInstruction& output2,
			const ir::PTXInstruction& input1,
			const ir::PTXInstruction& input2);

	/// Convert 'greater then' into 'less than', 'greater or equal' into 'less
	/// or equal', plus some other variations of these.
	static ir::PTXInstruction convertGreater2Less(const ir::PTXInstruction& inst);

	/// Convert add instructions into multiply and add
	static ir::PTXInstruction convertAdd2Mad(const ir::PTXInstruction& inst);

	/// Convert multiplication instructions into multiply and add
	static ir::PTXInstruction convertMul2Mad(const ir::PTXInstruction& inst);

	// instructions with optional inputs
	// instructions with optional outputs

	// mov -> add 0
	// selp a b, @p -> selp b a, @!p
	// neg -> mul -1
	// neg -> div -1
	// mul p2 -> shl
	// div p2 -> shr
	// rcp -> div 1/x

	// add -> add.cc
	// add -> addc
	// sub -> sub.cc
	// sub -> subc

	/// This function verifies the types of the instructions, to make sure that
	/// it is possible to normalize them. For instance, sum of float and
	/// sum of int are not mutually convertible.
	bool checkInstructionTypes(const ir::PTXInstruction& input1, const ir::PTXInstruction& input2) const;

	///
	bool checkOperand(const ir::PTXInstruction& input1, const ir::PTXInstruction& input2, const ir::PTXInstruction::Operand op) const;

	/// This function verifies the type of a operand, to make sure that it is
	/// possible to normalize it. For instance, sum of float and sum of int
	/// are not mutually convertible.
	bool checkOperandTypes(const ir::PTXOperand& op1, const ir::PTXOperand& op2) const;

	/// This function verifies the type of a optional operand, to make sure
	/// that it is possible to normalize it (it it's present).
	bool checkOptionalOperand(const ir::PTXOperand& op1, const ir::PTXOperand& op2) const;

	void printOpcdeConversionTable() const;
	void printCmpOperatorConversionTable() const;
};

}

#endif /* INSTRUCTIONMATCHER_H_ */
