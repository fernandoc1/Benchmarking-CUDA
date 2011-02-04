/*! \file PTXInstruction.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief base class for all instructions
*/

#ifndef IR_PTXINSTRUCTION_H_INCLUDED
#define IR_PTXINSTRUCTION_H_INCLUDED

#include <iostream>
#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/PTXOperand.h>

namespace ir {

class PTXInstruction;

	/*! Exception raised when operand type doesn't match instruction type. */
	class OperandTypeException: public std::exception {
	private:
		PTXInstruction* instruction; ///< Instruction that caused this exception
		PTXOperand::DataType operandType;
		PTXOperand::DataType instructionType;

	public:
		OperandTypeException()
			: instruction(NULL)
			, operandType(PTXOperand::TypeSpecifier_invalid)
			, instructionType(PTXOperand::TypeSpecifier_invalid)
		{}

		OperandTypeException(PTXInstruction* inst, PTXOperand::DataType opType, PTXOperand::DataType iType)
			: instruction(inst)
			, operandType(opType)
			, instructionType(iType)
		{}

  		virtual const char* what() const throw();
	};

	class PTXInstruction: public Instruction {
	public:
		/*! Hierarchy Level */
		enum Level {
			CtaLevel,
			GlobalLevel,
			Level_Invalid
		};

		/*!	List of opcodes for PTX instructions */
		enum Opcode {
			Abs,
			Add,
			AddC,
			And,
			Atom,
			Bar,
			Bfe,
			Bfi,
			Bfind,
			Bra,
			Brev,
			Brkpt,
			Call,
			Clz,
			CNot,
			CopySign,
			Cos,
			Cvt,
			Cvta,
			Div,
			Ex2,
			Exit,
			Fma,
			Isspacep,
			Ld,
			Ldu,
			Lg2,
			Mad24,
			Mad,
			Max,
			Membar,
			Min,
			Mov,
			Mul24,
			Mul,
			Neg,
			Not,
			Or,
			Pmevent,
			Popc,
			Prefetch,
			Prefetchu,
			Prmt,
			Rcp,
			Red,
			Rem,
			Ret,
			Rsqrt,
			Sad,
			SelP,
			Set,
			SetP,
			Shl,
			Shr,
			Sin,
			SlCt,
			Sqrt,
			St,
			Sub,
			SubC,
			Suld,
			Sured,
			Sust,
			Suq,
			TestP,
			Tex,
			Txq,
			Trap,
			Vabsdiff,
			Vadd,
			Vmad,
			Vmax,
			Vmin,
			Vset,
			Vshl,
			Vshr,
			Vsub,
			Vote,
			Xor,

			// Special instructions inserted by the analysis procedures
			Reconverge,
			Phi,
			Nop
		};

		/*!
			Modifiers for integer and floating-point instructions. The
			FP rounding-modes are mutually exclusive but compatible with
			sat.
		*/
		enum Modifier {
			hi = 1,			//< 
			lo = 2,			//<
			wide = 4,		//<
			sat = 8,		//< saturation modifier	
			rn = 32,		//< mantissa LSB rounds to nearest even
			rz = 64,		//< mantissa LSB roudns toward zero
			rm = 128,		//< mantissa LSB rounds toward negative infty
			rp = 256,		//< mantissa LSB rounds toward positive infty
			approx = 512, 	//< identify an approximate instruction
			ftz = 1024, 	//< flush to zero
			full = 2048, 	//< full division
			Modifier_invalid = 0
		};
			
		enum CarryFlag {
			None = 0,
			CC = 1
		};
		
		enum Volatility {
			Nonvolatile = 0,
			Volatile
		};
	
		enum AddressSpace {
			Reg = 0,
			SReg,
			Const,
			Global,
			Local,
			Param,
			Shared,
			Texture,
			AddressSpace_Invalid
		};

		enum AtomicOperation {
			AtomicAnd,
			AtomicOr,
			AtomicXor,
			AtomicCas,
			AtomicExch,
			AtomicAdd,
			AtomicInc,
			AtomicDec, 
			AtomicMin,
			AtomicMax,
			AtomicOperation_Invalid
		};
		
		enum ReductionOperation {
			ReductionAnd,
			ReductionXor,
			ReductionOr,
			ReductionAdd,
			ReductionInc,
			ReductionDec,
			ReductionMin,
			ReductionMax,
			ReductionOperation_Invalid
		};

		enum VoteMode {
			All,
			Any,
			Uni,
			VoteMode_Invalid
		};

		/*! comparison operator */
		enum CmpOp {
			Eq,
			Ne,
			Lt,
			Le,
			Gt,
			Ge,
			Lo,
			Ls,
			Hi,
			Hs,
			Equ,
			Neu,
			Ltu,
			Leu,
			Gtu,
			Geu,
			Num,
			Nan,
			CmpOp_Invalid
		};
		
		enum PermuteMode {
			DefaultPermute,
			ForwardFourExtract,
			BackwardFourExtract,
			ReplicateEight,
			EdgeClampLeft,
			EdgeClampRight,
			ReplicateSixteen
		};
			
		enum FloatingPointMode {
			Finite,
			Infinite,
			Number,
			NotANumber,
			Normal,
			SubNormal,
			FloatingPointMode_Invalid
		};
		
		/*! Vector operation */
		typedef PTXOperand::Vec Vec;
		
		/*! boolean operator */
		enum BoolOp {
			BoolAnd,
			BoolOr,
			BoolXor,
			BoolNop,
			BoolOp_Invalid
		};
		
		/*! geometry for textures */
		enum Geometry {
			_1d,
			_2d,
			_3d,
			Geometry_Invalid
		};

		enum ComputeCapability {
			Cap_1_0,
			Cap_2_0_geforce,
			Cap_2_0,
			Capability_Invalid
		};

		/*! instruction operands */
		enum Operand {
			PredicateGuard,
			OperandD,
			OperandA,
			OperandB,
			OperandC,
			OperandQ
		};

		/*! geometry for textures */
		enum OperandPresence {
			PresenceDenied = 0,
			PresenceOptional = 1,
			PresenceCompulsory = 2
		};

	private:

		/*! Check if operand type is compatible with instruction type.
		 *  If instruction type is not defined yet, define it. */
		void checkOperandType(const PTXOperand::DataType oprType) throw(OperandTypeException) {
			if (type == ir::PTXOperand::TypeSpecifier_invalid) {
				type = oprType;
			} else {
				if (type != oprType) {
					throw OperandTypeException(this, oprType, type);
				}
			}
		}

		/*! Verify if for that instruction, the asked operand
		    can't be used, is optional or is compulsory. */
		OperandPresence getOperandPresence(const Operand op) const;

	public:
		static std::string toString( Level );
		static std::string toString( PermuteMode );
		static std::string toString( FloatingPointMode );
		static std::string toString( Vec );
		static std::string toString( AddressSpace );
		static std::string toString( AtomicOperation );
		static std::string toString( ReductionOperation );
		static std::string toString( CmpOp );
		static std::string toString( BoolOp );
		static std::string roundingMode( Modifier );
		static std::string toString( Modifier );
		static std::string toString( Geometry );
		static std::string modifierString( unsigned int, CarryFlag = None );
		static std::string toString( VoteMode );
		static std::string toString( Opcode );
		static bool isPt( const PTXOperand& );

	public:
		//PTXInstruction( Opcode op = Nop, const PTXOperand& d = PTXOperand(),
		//	const PTXOperand& a = PTXOperand(),
		//	const PTXOperand& b = PTXOperand(),
		//	const PTXOperand& c = PTXOperand() );

		/*! Coutinho: if some place calls a PTXInstruction contructor with less
		 than two arguments, it will call my contructor, not this.
		 So my type-checking operand add methods will still work. */
		PTXInstruction( Opcode op /*= Nop*/, const PTXOperand& d /*= PTXOperand()*/,
				const PTXOperand& a = PTXOperand(),
				const PTXOperand& b = PTXOperand(),
				const PTXOperand& c = PTXOperand() );
		PTXInstruction( Opcode op = Nop );
		~PTXInstruction();

		bool operator==( const PTXInstruction& ) const;
		
		/*! Is this a valid instruction?
			
			return null string if valid, otherwise a description of why 
				not.
		*/
		std::string valid() const;

		/*! Returns the guard predicate representation for the instruction */
		std::string guard() const;

		/*! Returns a parsable string representation of the instruction */
		std::string toString() const;

		/*! prints the current instruction in stdout, to be used in gdb */
		std::string printInst() const;

		/*! \brief Clone the instruction */
		Instruction* clone( bool copy = true ) const;
		
		const PTXOperand* getPredicate() const {
			return &(this->pg);
		}
		
		const PTXInstruction::Opcode getOpcode() const {
			return this->opcode;
		}
		
		/*! Return a const pointer to the specified operand */
		const PTXOperand* getConstOperand(const ir::PTXInstruction::Operand op) const;

		/*! Return a pointer to the specified operand */
		PTXOperand* getOperand(const ir::PTXInstruction::Operand op)
		{
			return (PTXOperand*) getConstOperand(op);
		}

		/*!
		 * Set result destination. Operand's and destination type must match 
		 * (except .wide instructions).
		 * Instruction type will be the operands' type. 
		 */
		void setDestination(const ir::PTXOperand& opr);
		
		/*!
		 * Set operand A. Operands' type must match. 
		 * Instruction type will be the operands' type. 
		 */
		void setOperandA(const ir::PTXOperand& opr);
		
		/*!
		 * Set operand B. Operands' type must match. 
		 * Instruction type will be the operands' type. 
		 */ 
		void setOperandB(const ir::PTXOperand& opr);
		
		/*!
		 * Set operand C. Operands' type must match. 
		 * Instruction type will be the operands' type. 
		 */
		void setOperandC(const ir::PTXOperand& opr);
		
		/*! Set a predicate for guarding the instruction */
		void setPredicate(const ir::PTXOperand& predicate) {
			this->pg = predicate;	
		}
		
		/*! Set comparison operator for comparison instructions */
		void setComparisonOperator(PTXInstruction::CmpOp cmp) {
			this->comparisonOperator = cmp;
		}
		
		/*! Set modifier of FP operations */
		void setModifier(PTXInstruction::Modifier m) {
			this->modifier = m;
		}
		
		/*! Set the type of memory where load and store instructions will access */
		void setAddressSpace(AddressSpace s) {
			this->addressSpace = s;
		}
		
		/*! Set the mode of warp vote operations */
		void setVoteMode(VoteMode m) {
			this->vote = m;
		}
		
		/*! Set the operation of atomic instruction */
		void setAtomicOperation(AtomicOperation op) {
			this->atomicOperation = op;
		}
		
		/*! Set the type of boolean operation in setp instructions */
		void setBooleanOperator(BoolOp op) {
			this->booleanOperator = op;
		}
		
		/*! Set if a branch is uni (all threads must go to the same place) or not */
		void setBranchUni(bool isUni) {
			this->uni = isUni;
		}
		
		/*! Verify if the instruction is using a given operand */
		bool isUsingOperand(const Operand op) const;

		/*! Verify if the instruction have a predicate guard */
		bool isPredicated() const {
			if (false == isUsingOperand(PredicateGuard)) {
				return false;
			}

			// check condition
			const PTXOperand* pred = this->getPredicate();
			if (pred->condition != ir::PTXOperand::PT) {
				return true;
			}
			
			return false;
		}

		/*! Verify if the instruction is a basic block terminator */
		bool isTerminator() const {
			PTXInstruction::Opcode op = getOpcode();
			
			switch (op) {
			    case ir::PTXInstruction::Bra:	// branch - jumps to another basic block.
			    case ir::PTXInstruction::Exit:	// exit - ends the program
			    case ir::PTXInstruction::Ret:	// return - ends the function
				    return true;
			    default:
			        return false;
			}
		}
		
		/*! Verify if the instruction is a conditional branch.
		    In PTX case we considere a conditional branch any instruction that
		    can change control flow that is predicated. */
		bool isConditionalBranch() const {
			if (isPredicated()) {
				PTXInstruction::Opcode op = getOpcode();
				
				switch (op) {
				    case ir::PTXInstruction::Bra:	// branch - jumps to another basic block
				    case ir::PTXInstruction::Call:	// conditional call
				    case ir::PTXInstruction::Exit:	// exit - ends the program
				    case ir::PTXInstruction::Ret:	// return - ends the function
					    return true;
				    default:
				        return false;
				}
			}
			
			return false;
		}
		
		/*! Verify if the instruction is an unconditional branch.
		    In PTX case we consider a unconditional branch any instruction that
		    changes the control flow but isn't predicated. */
		bool isUnconditionalBranch() const {
			return !isPredicated() && getOpcode() == ir::PTXInstruction::Bra;
		}

	public:
		/*! Opcode of PTX instruction */
		Opcode opcode;

		/*! indicates data type of instruction */
		PTXOperand::DataType type;

		/*! optionally writes carry-out value to condition code register */
		CarryFlag carry;

		union {
			/*! Flag containing one or more floating-point modifiers */
			unsigned int modifier;

			/*! Comparison operator */
			CmpOp comparisonOperator;

			/*! For load and store, indicates which addressing mode to use */
			AddressSpace addressSpace;
			
			/*! For membar, the visibility level in the thread hierarchy */
			Level level;
			
			/*! Shift amount flag for bfind instructions */
			bool shiftAmount;
			
			/*! Permute mode for prmt instructions */
			PermuteMode permuteMode;
			
			/*! For call instructions, indicates a tail call */
			bool tailCall;
		};
	
		/*! If the instruction is predicated, the guard */
		PTXOperand pg;
				
		/*! Second destination register for SetP, otherwise unused */
		PTXOperand pq;
		
		/*! Indicates whether target or source is a vector or scalar */
		Vec vec;

		union {
			/*! If instruction type is atomic, select this atomic operation */
			AtomicOperation atomicOperation;
			
			/*! If the instruction is a reduction, select this operation */
			ReductionOperation reductionOperation;
			
			/* If instruction type is vote, specifies the mode of voting */
			VoteMode vote;
			
			/* For TestP instructions, specifies the floating point mode */
			FloatingPointMode floatingPointMode;
			
			/* If instruction is a branch, is it .uni */
			bool uni;

			/*! Boolean operator */
			BoolOp booleanOperator;

			/*! Indicates whether the target address space is volatile */
			Volatility volatility;
			
			/*! Geometry if this is a texture instruction */
			Geometry geometry;
			
			/*! Is this a divide full instruction? */
			bool divideFull;
			
			/*! If the instruction updates the CC, what is the CC register */
			PTXOperand::RegisterType cc;
		};

		/*! Destination operand */
		PTXOperand d;

		/*! Source operand a */
		PTXOperand a;

		/*! Source operand b */
		PTXOperand b;

		/*! Source operand c */
		PTXOperand c;

		/*  Runtime annotations 
			
			The following members are used to annotate the instruction 
				at analysis time
		*/
	public:
		/*! \brief Index of post dominator instruction at which possibly 
			divergent branches reconverge */
		int reconvergeInstruction;

		union
		{
			/*! \brief Branch target instruction index */
			int branchTargetInstruction;
			/*! \brief Context switch reentry point */
			int reentryPoint;
			/*! \brief Is this a kernel argument in the parameter space? */
			bool isArgument;
		};
		/*!	The following are used for debugging information at runtime. */
	public:
		/*! \brief The index of the statement that this instruction was 
			created from */
		unsigned int statementIndex;
		/*! \brief The program counter of the instruction */
		unsigned int pc;
	};

}

#endif

