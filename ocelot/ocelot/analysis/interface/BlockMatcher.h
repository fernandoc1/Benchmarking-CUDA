/*
 * BlockExtractor.h
 *
 *  Created on: 14/07/2010
 *      Author: coutinho
 */

#ifndef BLOCKMATCHER_H_
#define BLOCKMATCHER_H_


#include <map>
#include <list>
#include <vector>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/InstructionConverter.h>


namespace analysis {

typedef std::pair<analysis::DataflowGraph::RegisterId, analysis::DataflowGraph::RegisterId> VariableMatch;

/// This is the comparison function that we use to index the table of variable
/// aliases.
struct ltVariablePair {
	bool operator()(const VariableMatch m1, const VariableMatch m2) const
	{
		if (m1.first < m2.first) {
			return true;
		}
		if (m1.first > m2.first) {
			return false;
		}

		// first is equal
		if (m1.second < m2.second) {
			return true;
		}
		return false;
	}
};

typedef std::map<VariableMatch, analysis::DataflowGraph::RegisterId, ltVariablePair> VariableUnifications;
typedef std::map<DataflowGraph::RegisterId, DataflowGraph::RegisterId> VariableAliases;


class BlockMatcher {
public:

	typedef enum {Match, Horizontal, Vertical, Substitution} StepDirection;

	typedef std::list<StepDirection> MatrixPath;

	class GainMatrix {
	private:

		class MatrixCell {
		public:
			float gain;
			StepDirection lastStep;
			bool paidBranchCost;
			bool paidGotoCost;
			VariableUnifications aliasMap;
		};

		ir::PTXInstruction::ComputeCapability deviceCapability;

		// basic blocks that will be compared
		DataflowGraph::Block& verticalBlock;
		DataflowGraph::Block& horizontelBlock;

		unsigned int height;
		unsigned int width;

		// instruction at matrix borders
		std::vector<ir::PTXInstruction*> instructions1;
		std::vector<ir::PTXInstruction*> instructions2;

		// middle of matrix
		MatrixCell** matrix;
		MatrixCell*  cells;

		// a instruction converter to match basic block instructions
		InstructionConverter& instMatcher;

		// alternative variable names, defined by phi instructions
		VariableAliases alternativeVarNames;

		DataflowGraph::RegisterId maxRegister;

		void calculateCellGain(unsigned int row, unsigned int col);

		StepDirection getBestSquareMove(unsigned int row, unsigned int col, float& gain, bool& paidGotoCost) const;
		StepDirection getBestDiagonalMove(unsigned int row, unsigned int col, float& gain, VariableUnifications& aliases);

		float calculateSquareMoveGain(unsigned int row, unsigned int col, bool& payGotoCost, const StepDirection move) const;
		float calculateMatchGain(unsigned int row, unsigned int col, VariableUnifications& aliases);
		float calculateSubstGain(unsigned int row, unsigned int col, VariableUnifications& aliases) const;

		float getOperandUnificationCost(const ir::PTXOperand& op1, const ir::PTXOperand& op2, VariableUnifications& aliases);

		/// Returns the cost of a instruction with opcode op, operands with type
		/// opType running on a gpu with compute capability cap.
		static float getInstructionCost(ir::PTXInstruction::Opcode op, ir::PTXOperand::DataType opType = ir::PTXOperand::s32, ir::PTXInstruction::ComputeCapability cap = ir::PTXInstruction::Cap_1_0);

		/// Returns the cost of the calling instruction
		/// running on a gpu with compute capability cap.
		static float getInstructionCost(const ir::PTXInstruction& inst, ir::PTXInstruction::ComputeCapability cap = ir::PTXInstruction::Cap_1_0);

		float getOperandsCost(const ir::PTXInstruction& inst1, const ir::PTXInstruction& inst2, VariableUnifications& aliases, const ir::PTXInstruction::ComputeCapability cap = ir::PTXInstruction::Cap_1_0);

		/// Add a operand unification to aliases table
		void addOperandUnification(const ir::PTXOperand& op1, const ir::PTXOperand& op2, VariableUnifications& aliases);

		/// Return a id for a new register
		DataflowGraph::RegisterId getNewRegister()
		{
			return ++maxRegister;
		}

	public:

		GainMatrix(DataflowGraph* dfg,DataflowGraph::Block& bb1, DataflowGraph::Block& bb2, InstructionConverter& ic, ir::PTXInstruction::ComputeCapability cap);
		virtual ~GainMatrix();

		float calculateMatch();

		/// Returns the path choose by algorithm
		MatrixPath getPath() const;

		static void printPath(const MatrixPath &path) {
			MatrixPath::const_iterator it = path.begin();
			for (; it != path.end(); it++) {
				switch (*it) {
				case Match:
					std::cout << " M";
					break;
				case Horizontal:
					std::cout << " H";
					break;
				case Vertical:
					std::cout << " V";
					break;
				case Substitution:
					std::cout << " S";
					break;
				}
			}
		}

		/// prints the gain matrix for debug
		void printMatrix() const;
	};

private:

	ir::PTXInstruction::ComputeCapability deviceCapability;
	InstructionConverter instMatcher;

public:

	BlockMatcher(ir::PTXInstruction::ComputeCapability cap = ir::PTXInstruction::Cap_1_0);
	virtual ~BlockMatcher();

	static float calculateUnificationGain(DataflowGraph* dfg, DataflowGraph::Block& bb1, DataflowGraph::Block& bb2, MatrixPath& path, InstructionConverter instMatcher, ir::PTXInstruction::ComputeCapability cap);
};

}

#endif /* BLOCKMATCHER_H_ */
