#ifndef _MINERTHREAD_H_
#define _MINERTHREAD_H_

#include "types.h"
#include <boost/thread/mutex.hpp>
 
namespace Core
{
	class CBlock;

	class MinerThread
		{
		private:

			CBlock* m_pBLOCK;
            unsigned int m_GpuId;
			bool m_bBlockFound, m_bNewBlock, m_bReady, m_bBenchmark;
			LLP::Thread_t m_pTHREAD;
			volatile unsigned long long m_unHashes;
			double total_mhashes_done;
			boost::mutex m_clLock;
			int m_nThroughput;

		public:			
		
			MinerThread(unsigned int Thread_Id);
			MinerThread(const MinerThread& miner);
			MinerThread& operator=(const MinerThread& miner);
			~MinerThread();

		
			//Main Miner Thread. Bound to the class with boost. Might take some rearranging to get working with OpenCL.
			void SK1024Miner();
			
			///////////////////////////////////////////////////////////////////////////////
			//Accessors
			///////////////////////////////////////////////////////////////////////////////
			const bool				        GetIsBlockFound()	const	{ return this->m_bBlockFound; }
			const bool				        GetIsBenchmark()	const	{ return this->m_bBenchmark; }
			const bool				        GetIsNewBlock()		const	{	return this->m_bNewBlock;			}
			const bool				        GetIsReady()		const	{	return this->m_bReady;				}
			const unsigned long long 		GetHashes()			const	{	return this->m_unHashes;			}
			const int						GetThroughput()		const	{	return this->m_nThroughput;			}
			CBlock*					        GetBlock()			const	{	return this->m_pBLOCK;				}
//			GPUData*				GetGPUData()		const	{	return this->m_pGPUData;			}
			        unsigned int            GetGpuId()          const   {   return this->m_GpuId;               }
			///////////////////////////////////////////////////////////////////////////////
			//Mutators
			///////////////////////////////////////////////////////////////////////////////
			void	SetIsBlockFound(bool bFoundBlock)			{	this->m_bBlockFound = bFoundBlock;	}
			void	SetIsBenchmark(bool bBenchmark)				{	this->m_bBenchmark = bBenchmark; }
			void	SetIsNewBlock(bool bNewBlock)				{	this->m_bNewBlock = bNewBlock;		}
			void	SetIsReady(bool bReady)						{	this->m_bReady = bReady;			}
			void	SetHashes(unsigned long long unHashes)		{	this->m_unHashes = unHashes;		}
			void	SetThroughput(int nThroughput)				{	this->m_nThroughput = nThroughput; }
			void	SetBlock(CBlock* pBlock)					{	this->m_pBLOCK = pBlock; }
//			void	SetGPUData(GPUData* data)					{	this->m_pGPUData = data;			}
			void    SetGpuId(unsigned int Thread_Id)            {   this->m_GpuId = Thread_Id;          }


	};
}
#endif //_MINERTHREAD_H_