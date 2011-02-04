#include <ocelot/graphs/interface/BlockDivergenceInfo.h>
#include <fstream>
#include <assert.h>

using namespace std;
namespace graph_utils {
/*!\brief Constructor, the number of divergences cannot be higher than the number of visits to a block */
BlocksDivergenceInfo::BlockDivergence::BlockDivergence( const unsigned int &id, const unsigned long vis, const unsigned long div ) :
	_blockId(id), _visits(vis), _divergences(div){
	assert(_divergences <= _visits);
}

/*!\brief Returns if occurred divergence on the block */
bool BlocksDivergenceInfo::BlockDivergence::isDivergent() const{
	return _divergences != 0;
}

/*!\brief Returns the block id */
const ir::ControlFlowGraph::BasicBlock::Id &BlocksDivergenceInfo::BlockDivergence::block() const{
	return _blockId;
}

/*!\brief Returns number of visits to the block on the profiling results */
const unsigned long &BlocksDivergenceInfo::BlockDivergence::visits() const{
	return _visits;
}

/*!\brief Returns number of divergences on the block on the profiling results */
const unsigned long &BlocksDivergenceInfo::BlockDivergence::divergencies() const{
	return _divergences;
}

/*!\brief Returns if the block is divergent on the profiling results */
bool BlocksDivergenceInfo::isPossibleDivergent( const unsigned int &blockId ) const{
	return( _blocks.find(blockId) != _blocks.end() );
}

/*!\brief Returns if the block is divergent on the profiling results */
bool BlocksDivergenceInfo::isDivergent( const unsigned int &blockId ) const{
	set<BlockDivergence>::const_iterator block = _blocks.find(blockId);
	if( block != _blocks.end() )
		return block->isDivergent();

	return false;
}

/*!\brief Loads the profiling results text file
 * Each block has 3 number:
 * * First: Block id number, unsigned int >= 0
 * * Second: Number of times the block was visited on the profiled execution
 * * Third: Number of times that occurred divergence on the profiled execution
 *
 * More than one file can be loaded, in that case, the visits and divergence numbers of the existing data and the one on the file are summed */
bool BlocksDivergenceInfo::load( const string &fileName ){
	_blocks.clear();
	fstream in(fileName, ios_base::in);
	if( !in.is_open() ){
		return false;
	}


	while( !in.eof() ){
		unsigned int blockId = 0;
		unsigned long visits = 0;
		unsigned long diverg = 0;

		in >> blockId;
		in >> visits;
		in >> diverg;

		if( visits > 0 ){
			_blocks.insert(BlockDivergence(blockId, visits, diverg));
		}
	}
	return true;
}

/*!\brief Returns if there is profiling information for the block with a certain id */
bool BlocksDivergenceInfo::hasBlock(const unsigned int &id) const
{
	return _blocks.find(id) != _blocks.end();
}

/*!\brief Default constructor */
BlocksDivergenceInfo::BlocksDivergenceInfo(){
}

/*!\brief Constructor already loading information of a profiling results text file */
BlocksDivergenceInfo::BlocksDivergenceInfo( const string &fileName ){
	load(fileName);
}

/*!\brief Tests if the profiling data read is empty */
bool BlocksDivergenceInfo::empty() const{
	return(_blocks.empty());
}

/*!\brief Clears the profiling data read */
void BlocksDivergenceInfo::clear(){
	_blocks.clear();
}

/*!\brief Gets the first block profiling results */
BlocksDivergenceInfo::const_BlockInfo BlocksDivergenceInfo::begin() const{
	return _blocks.begin();
}

/*!\brief Gets the limit of the block profiling results */
BlocksDivergenceInfo::const_BlockInfo BlocksDivergenceInfo::end() const{
	return _blocks.end();
}

/*!\brief Searches for the profiling results of a specific block id */
BlocksDivergenceInfo::const_BlockInfo BlocksDivergenceInfo::find( const unsigned int &id ) const{
	return _blocks.find(id);
}

bool operator<( const BlocksDivergenceInfo::BlockDivergence &first, const BlocksDivergenceInfo::BlockDivergence &second ){
	return first.block() < second.block();
}

bool operator<( const unsigned int &first, const BlocksDivergenceInfo::BlockDivergence &second ){
	return first < second.block();
}

inline bool operator<( const BlocksDivergenceInfo::BlockDivergence &second, const unsigned int &first ){
	return first < second;
}

bool operator==( const BlocksDivergenceInfo::BlockDivergence &first, const BlocksDivergenceInfo::BlockDivergence &second ){
	return first.block() == second.block();
}

bool operator==( const unsigned int &first, const BlocksDivergenceInfo::BlockDivergence &second ){
	return first == second.block();
}

inline bool operator==( const BlocksDivergenceInfo::BlockDivergence &second, const unsigned int &first ){
	return first == second;
}
}
