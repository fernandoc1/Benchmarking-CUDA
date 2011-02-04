/*! \file PTXOperand.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief internal representation of a PTX operand
*/

#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/Parameter.h>

#include <hydrazine/implementation/debug.h>

#include <cassert>
#include <sstream>
#include <iomanip>

std::string ir::PTXOperand::toString( DataType type ) {
	switch( type ) {
		case s8:   return "s8";   break;
		case s16:  return "s16";  break;
		case s32:  return "s32";  break;
		case s64:  return "s64";  break;
		case u8:   return "u8";   break;
		case u16:  return "u16";  break;
		case u32:  return "u32";  break;
		case u64:  return "u64";  break;
		case b8:   return "b8";   break;
		case b16:  return "b16";  break;
		case b32:  return "b32";  break;
		case b64:  return "b64";  break;
		case f16:  return "f16";  break;
		case f32:  return "f32";  break;
		case f64:  return "f64";  break;
		case pred: return "pred"; break;
		default: break;
	}
	return "Invalid";
}

std::string ir::PTXOperand::toString( SpecialRegister reg ) {
	switch( reg ) {
		case tidX: return "%tid.x"; break;
		case tidY: return "%tid.y"; break;
		case tidZ: return "%tid.z"; break;
		case ntidX: return "%ntid.x"; break;
		case ntidY: return "%ntid.y"; break;
		case ntidZ: return "%ntid.z"; break;
		case laneId: return "%laneid"; break;
		case warpId: return "%warpid"; break;
		case warpSize: return "WARP_SZ"; break;
		case ctaIdX: return "%ctaid.x"; break;
		case ctaIdY: return "%ctaid.y"; break;
		case ctaIdZ: return "%ctaid.z"; break;
		case nctaIdX: return "%nctaid.x"; break;
		case nctaIdY: return "%nctaid.y"; break;
		case nctaIdZ: return "%nctaid.z"; break;
		case smId: return "%smid"; break;
		case nsmId: return "%nsmid"; break;
		case gridId: return "%gridid"; break;
		case clock: return "%clock"; break;
		case pm0: return "%pm0"; break;
		case pm1: return "%pm1"; break;
		case pm2: return "%pm2"; break;
		case pm3: return "%pm3"; break;
		default: break;
	}
	return "SpecialRegister_invalid";
}

std::string ir::PTXOperand::toString( AddressMode mode ) {
	switch( mode ) {
		case Register:  return "Register";  break;
		case Indirect:  return "Indirect";  break;
		case Immediate: return "Immediate"; break;
		case Address:   return "Address";   break;
		case Label:     return "Label";     break;
		case Special:   return "Special";   break;
		case BitBucket: return "BitBucket"; break;
		default: break;
	}
	return "Invalid";
}

std::string ir::PTXOperand::toString( DataType type, RegisterType reg ) {
	std::stringstream stream;
	if( type == pred ) {
		stream << "%p" << reg;
	}
	else {
		stream << "%r_" << toString( type ) << "_" << reg;
	}
	return stream.str();
}

bool ir::PTXOperand::isFloat( DataType type ) {
	bool result = false;
	switch( type ) {
		case f16: /* fall through */
		case f32: /* fall through */
		case f64: result = true;
		default: break;
	}
	return result;
}

bool ir::PTXOperand::isInt( DataType type ) {
	bool result = false;
	switch( type ) {
		case s8:  /* fall through */
		case s16: /* fall through */
		case s32: /* fall through */
		case s64: /* fall through */
		case u8:  /* fall through */
		case u16: /* fall through */
		case u32: /* fall through */
		case u64: result = true; break;
		default: break;
	}
	return result;
}

bool ir::PTXOperand::isSigned( DataType type ) {
	bool result = false;
	switch( type ) {
		case s8:  /* fall through */
		case s16: /* fall through */
		case s32: /* fall through */
		case s64: result = true; break;
		default: break;
	}
	return result;
}

unsigned int ir::PTXOperand::bytes( DataType type ) {
	assert( type != TypeSpecifier_invalid );
	switch( type ) {
		case pred: /* fall through */
		case b8:   /* fall through */
		case u8:   /* fall through */
		case s8:   return 1; break;
		case u16:  /* fall through */
		case f16:  /* fall through */
		case b16:  /* fall through */
		case s16:  return 2; break;
		case u32:  /* fall through */
		case b32:  /* fall through */
		case f32:  /* fall through */
		case s32:  return 4; break;
		case f64:  /* fall through */
		case u64:  /* fall through */
		case b64:  /* fall through */
		case s64:  return 8; break;
		default:   return 0; break;
	}
	return 0;	
}

bool ir::PTXOperand::valid( DataType destination, DataType source ) {
	switch( destination ) {
		case b64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case b32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case b16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case b8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case u64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case u32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case u16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case u8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case s64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case s32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case s16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case s8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case f64: {
			switch( source ) {
				case b64: /* fall through */
				case f64: return true; break;
				default: break;
			}
			break;
		}
		case f32: {
			switch( source ) {
				case b32: /* fall through */
				case f32: return true; break;
				default: break;
			}
			break;
		}
		case f16: {
			switch( source ) {
				case b16: /* fall through */
				case f16: return true; break;
				default: break;
			}
			break;
		}
		case pred: {
			return source == pred;
			break;
		}
		default: break;
		
	}
	return false;
}

bool ir::PTXOperand::relaxedValid( DataType instructionType, 
	DataType operand ) {
	switch( instructionType ) {
		case b64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case b32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case b16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case b8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case u64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case u32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case u16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case u8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case s64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case s32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case s16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case s8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case f64: {
			switch( operand ) {
				case b64: /* fall through */
				case f64: return true; break;
				default: break;
			}
			break;
		}
		case f32: {
			switch( operand ) {
				case b32: /* fall through */
				case f32: return true; break;
				default: break;
			}
			break;
		}
		case f16: {
			switch( operand ) {
				case b16: /* fall through */
				case f16: return true; break;
				default: break;
			}
			break;
		}
		case pred: {
			return operand == pred;
			break;
		}
		default: break;
		
	}
	return false;
}


ir::PTXOperand::PTXOperand() {
	identifier = "";
	addressMode = Invalid;
	type = PTXOperand::TypeSpecifier_invalid;
	offset = 0;
	imm_int = 0;
	imm_uint = 0;
	vec = v1;
}

ir::PTXOperand::PTXOperand(SpecialRegister r)
	: addressMode(Special)
	, special(r)
	, vec(v1)
{
	// Each special register has a predefined type.
	// Using ype definitions of PTX 1.4 ISA.
	switch (r) {
		case tidX:  /* fall through */
		case tidY:  /* fall through */
		case tidZ:  /* fall through */
		case ntidX: /* fall through */
		case ntidY: /* fall through */
		case ntidZ:
			type = u16;
			break;
		case laneId: /* fall through */
		case warpId:
			type = u32;
			break;
		case warpSize: /// ???
		case ctaIdX:  /* fall through */
		case ctaIdY:  /* fall through */
		case ctaIdZ:  /* fall through */
		case nctaIdX: /* fall through */
		case nctaIdY: /* fall through */
		case nctaIdZ: /* fall through */
			type = u16;
			break;
		case smId:   /* fall through */
		case nsmId:  /* fall through */
		case gridId: /* fall through */
		case clock:  /* fall through */
		case pm0:    /* fall through */
		case pm1:    /* fall through */
		case pm2:    /* fall through */
		case pm3:
			type = u32;
			break;
		case SpecialRegister_invalid:
			type = TypeSpecifier_invalid;
			break;
	}
}

ir::PTXOperand::PTXOperand(const std::string& l)
	: identifier(l)
	, addressMode(Label)
{
}

ir::PTXOperand::PTXOperand(AddressMode m, DataType t, RegisterType r, int o, Vec v) 
	: addressMode(m)
	, type(t)
	, offset(o)
	, reg(r)
	, vec(v) {
}

ir::PTXOperand::PTXOperand(std::string id, AddressMode am, DataType t) {
	identifier  = id;
	addressMode = am;
	type        = t;
	offset      = 0;
	imm_int     = 0;
	vec         = v1;
}

ir::PTXOperand::PTXOperand(DataType t, unsigned long long int val) {
	addressMode = Immediate;
	type        = t;
	offset      = 0;
	imm_uint    = val;
	vec         = v1;
}

ir::PTXOperand::PTXOperand(DataType t, long long int val) {
	addressMode = Immediate;
	type        = t;
	offset      = 0;
	imm_int     = val;
	vec         = v1;
}

ir::PTXOperand::PTXOperand(DataType t, double val) {
	addressMode = Immediate;
	type        = t;
	offset      = 0;
	imm_float   = val;
	vec         = v1;
}

ir::PTXOperand::PTXOperand(ir::Parameter* p) {
	identifier  = p->name;
	addressMode = Address; // parameters can be imediates too??? 
	type        = p->type;
	offset      = p->offset;
	vec         = p->vector;
}

ir::PTXOperand::~PTXOperand() {

}

/*!
	Displays a binary represetation of a 32-bit floating-point value
*/
static std::ostream & write(std::ostream &stream, float value) {
	union {
		unsigned int imm_uint;
		float value;
	} float_union;
	float_union.value = value;
	stream << "0f" << std::setw(8) << std::setfill('0') 
		<< std::hex << float_union.imm_uint << std::dec;
	return stream;
}

/*!
	Displays a binary represetation of a 64-bit floating-point value
*/
static std::ostream & write(std::ostream &stream, double value) {
	union {
		long long unsigned int imm_uint;
		double value;
	} double_union;
	double_union.value = value;
	stream << "0d" << std::setw(16) << std::setfill('0') << std::hex 
		<< double_union.imm_uint;
	return stream;
}

std::string ir::PTXOperand::toString() const {
	if( addressMode == BitBucket ) {
		return "_";
	} else if( addressMode == Indirect ) {
		std::stringstream stream;
		if( offset < 0 ) {
			if ( identifier != "" ) {
				stream << identifier;
			}
			else {
				stream << "%r" << reg;
			}
			stream << " + " << ( offset );
			return stream.str();
		} else {
			if ( identifier != "" ) {
				stream << identifier;
			}
			else {
				stream << "%r" << reg;
			}
			stream << " + " << offset;
			return stream.str();
		}
	} else if( addressMode == Address ) {
		std::stringstream stream;
		if( offset == 0 ) {
			return identifier;
		}
		else if( offset < 0 ) {
			stream << ( offset );
			return identifier + " + " + stream.str();
		} else {
			stream << offset;
			return identifier + " + " + stream.str();
		}
	} else if( addressMode == Immediate ) {
		std::stringstream stream;
		switch( type ) {
			case s8:  /* fall through */
			case s16: /* fall through */
			case s32: /* fall through */
			case s64: stream << imm_int; break;
			case u8:  /* fall through */
			case u16: /* fall through */
			case u32: /* fall through */
			case u64: /* fall through */
			case b8:  /* fall through */
			case b16: /* fall through */
			case b32: /* fall through */
			case b64: stream << imm_uint; break;
			case f16: /* fall through */
			case f32: {
				write(stream, (float)imm_float);
			} break;
			case f64: {
				write(stream, imm_float);
			} break;
			case pred: /* fall through */
			default: assertM( false, "Invalid immediate type " 
				+ PTXOperand::toString( type ) ); break;
		}
		return stream.str();
	} else if( addressMode == Special ) {
		return toString( special );
	} else if( type == pred ) {
		switch( condition ) {
			case PT: return "%pt"; break;
			case nPT: return "%pt"; break;
			default:
			{
				if( !identifier.empty() ) {
					return identifier;
				}
				else {
					std::stringstream stream;
					stream << "%p" << reg;
					return stream.str();
				}
				break;
			}
		}
	} else if( vec != v1 && !array.empty() ) {
		assert( ( vec == v2 && array.size() == 2 ) 
			|| ( vec == v4 && array.size() == 4 ) );
		std::string result = "{";
		for( Array::const_iterator fi = array.begin(); 
			fi != array.end(); ++fi ) {
			result += fi->toString();
			if( fi != --array.end() ) {
				result += ", ";
			}
		}
		return result + "}";
	}
	
	if( !identifier.empty() ) {
		return identifier;
	}
	else {
		std::stringstream stream;
		stream << "%r" << reg;
		return stream.str();
	}
}

std::string ir::PTXOperand::registerName() const {
	assert( addressMode == Indirect || addressMode == Register
		|| addressMode == BitBucket );
	
	if (addressMode == BitBucket) return "_";
	
	if( !identifier.empty() ) {
		return identifier;
	}
	else {
		std::stringstream stream;
		if(type == pred) {
			switch( condition ) {
				case PT: return "%pt"; break;
				case nPT: return "%pt"; break;
				default:
				{
					std::stringstream stream;
					stream << "%p" << reg;
					return stream.str();
					break;
				}
			}
		}
		else {
			stream << "%r" << reg;
		}
		return stream.str();
	}
}

unsigned int ir::PTXOperand::bytes() const {
	return bytes( type ) * vec;
}

bool ir::PTXOperand::equal(const PTXOperand& other) const
{
	// TODO: handle vectors
	if ((vec != ir::PTXOperand::v1)  || (other.vec != ir::PTXOperand::v1)) {
		return false;
	}

	if (addressMode != other.addressMode) {
		return false;
	}

	// address and indirect need to check offsets
	if ((addressMode == ir::PTXOperand::Address)
			|| (addressMode == ir::PTXOperand::Indirect)) {
		if (offset != other.offset) {
			return false;
		}
	}

	// addresses and labels can be identified by their identifier
	if ((addressMode == ir::PTXOperand::Address)
			|| (addressMode == ir::PTXOperand::Label)) {
		return (0 == identifier.compare(other.identifier));
	}

	// next operand types need data type check
	if (type != other.type) {
		return false;
	}

	// specials can be identified by their special field
	if (addressMode == ir::PTXOperand::Special) {
		return (special == other.special);
	}

	if (addressMode == ir::PTXOperand::Immediate) {
		switch (type) {
		case s8:
		case s16:
		case s32:
		case s64:
			return (imm_int == other.imm_int);
			break;
		case f16:
		case f32:
		case f64:
			return (imm_float == other.imm_float);
			break;
		case u8:
		case u16:
		case u32:
		case u64:
		case b8:
		case b16:
		case b32:
		case b64:
			return (imm_uint == other.imm_uint);
			break;
		default:
			assertM(false, "PTXOperand::equal() error: unhandled immediate type");
			return false;
			break;
		}
	}

	// indirect or register can be identified by their id (except predicates)
	if (reg != other.reg) {
		return false;
	}

	// extra verification for predicates
	if (type == ir::PTXOperand::pred) {
		return (condition == other.condition);
	}

	return true;
}

ir::PTXOperand::DataType ir::PTXOperand::wideToShort(ir::PTXOperand::DataType t) 
{
	switch (t){
		case ir::PTXOperand::s16: 
			return ir::PTXOperand::s8;
			break;
		case ir::PTXOperand::s32: 
			return ir::PTXOperand::s16;
			break;
		case ir::PTXOperand::s64: 
			return ir::PTXOperand::s32;
			break;
		case ir::PTXOperand::u16: 
			return ir::PTXOperand::u8;
			break;
		case ir::PTXOperand::u32: 
			return ir::PTXOperand::u16;
			break;
		case ir::PTXOperand::u64: 
			return ir::PTXOperand::u32;
			break;
		case ir::PTXOperand::f32: 
			return ir::PTXOperand::f16;
			break;
		case ir::PTXOperand::f64: 
			return ir::PTXOperand::f32;
			break;
		default:
			return ir::PTXOperand::TypeSpecifier_invalid;
			break;
	}
					
	return ir::PTXOperand::TypeSpecifier_invalid;
}

ir::PTXOperand::DataType ir::PTXOperand::shortToWide(ir::PTXOperand::DataType t)
{
	switch (t){
		case ir::PTXOperand::s8:
			return ir::PTXOperand::s16;
			break;
		case ir::PTXOperand::s16:
			return ir::PTXOperand::s32;
			break;
		case ir::PTXOperand::s32:
			return ir::PTXOperand::s64;
			break;
		case ir::PTXOperand::u8:
			return ir::PTXOperand::u16;
			break;
		case ir::PTXOperand::u16:
			return ir::PTXOperand::u32;
			break;
		case ir::PTXOperand::u32:
			return ir::PTXOperand::u64;
			break;
		case ir::PTXOperand::f16:
			return ir::PTXOperand::f32;
			break;
		case ir::PTXOperand::f32:
			return ir::PTXOperand::f64;
			break;
		default:
			return ir::PTXOperand::TypeSpecifier_invalid;
			break;
	}

	return ir::PTXOperand::TypeSpecifier_invalid;
}

void ir::PTXOperand::invertPredicateCondition()
{
	assertM(type == ir::PTXOperand::pred, "invertPredicateCondition() called for a operand that isn't a predicate");

	switch (condition) {
	case ir::PTXOperand::PT:
		condition = ir::PTXOperand::nPT;
		break;
	case ir::PTXOperand::nPT:
		condition = ir::PTXOperand::PT;
		break;
	case ir::PTXOperand::Pred:
		condition = ir::PTXOperand::InvPred;
		break;
	case ir::PTXOperand::InvPred:
		condition = ir::PTXOperand::Pred;
		break;
	default:
		assertM(false, "invertPredicateCondition(): Unhandled condition");
		break;
	}
}
