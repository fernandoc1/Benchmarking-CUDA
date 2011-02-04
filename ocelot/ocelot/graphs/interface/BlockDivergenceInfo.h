#ifndef BLOCKDIVERGENCEINFO_H_
#define BLOCKDIVERGENCEINFO_H_
/*
 * BlockDivergenceInfo.h
 *
 *  Created on: Jul 1, 2010
 *      Author: Diogo Sampaio
 */

#include <set>
#include <string>
#include <ocelot/ir/interface/ControlFlowGraph.h>
using namespace std;
namespace graph_utils {
/*!\brief Holds divergence results of instrumented kernels */
class BlocksDivergenceInfo
{
	public:
		/*!\brief Holds divergence results of a single instructions block */
		class BlockDivergence
		{
			friend class BlocksDivergenceInfo;
			private:
				ir::ControlFlowGraph::BasicBlock::Id _blockId;
				unsigned long _visits;
				unsigned long _divergences;

			public:
				BlockDivergence( const unsigned int &id, const unsigned long vis = 0, const unsigned long div = 0 );
				bool isDivergent() const;
				const ir::ControlFlowGraph::BasicBlock::Id &block() const;
				const unsigned long &visits() const;
				const unsigned long &divergencies() const;
		};

		/*!\brief Holds blocks divergence results, ordered by the block->id(), the first number of the 3 number on each line */
		typedef set<BlockDivergence> BlockInfoSet;
		typedef BlockInfoSet::iterator BlockInfo;
		typedef BlockInfoSet::const_iterator const_BlockInfo;

	private:
		BlockInfoSet _blocks;

	public:
		bool isPossibleDivergent( const unsigned int &blockId ) const;
		bool isDivergent( const unsigned int &blockId ) const;
		bool load( const string &fileName );
		const_BlockInfo begin() const;
		const_BlockInfo end() const;
		const_BlockInfo find( const unsigned int &id ) const;
		bool hasBlock(const unsigned int &id) const;
		bool empty() const;
		void clear();
		BlocksDivergenceInfo();
		BlocksDivergenceInfo(const string &fileName);
};

bool operator<( const BlocksDivergenceInfo::BlockDivergence &first, const BlocksDivergenceInfo::BlockDivergence &second );
bool operator<( const unsigned int &first, const BlocksDivergenceInfo::BlockDivergence &second );
inline bool operator<( const BlocksDivergenceInfo::BlockDivergence &second, const unsigned int &first );
bool operator==( const BlocksDivergenceInfo::BlockDivergence &first, const BlocksDivergenceInfo::BlockDivergence &second );
bool operator==( const unsigned int &first, const BlocksDivergenceInfo::BlockDivergence &second );
inline bool operator==( const BlocksDivergenceInfo::BlockDivergence &second, const unsigned int &first );
}
#endif /* BLOCKDIVERGENCEINFO_H_ */
