#include "MinerThread.h"
#include "CBlock.h"
#include "../hash/templates.h"	
#include "bignum.h"
#include "../util_llh.h"
#include "../miner2.h"
namespace Core
{
	MinerThread::MinerThread(unsigned int pData)
	{
		m_unNonceCount = 0;
		m_unNonces[4] = {0};
		m_unBits = 0;
		m_pBLOCK = NULL;
		m_GpuId = pData;
		m_bBlockFound =false; 
		m_unHashes = 0;
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);
		total_mhashes_done = 0;
		m_nThroughput = 0;
		m_bBenchmark = false;
		m_bShutdown = false;
	}

	MinerThread::MinerThread(const MinerThread& miner)
	{
		m_pBLOCK = miner.GetBlock();
		m_GpuId = miner.GetGpuId();
		m_bBlockFound = miner.GetIsBlockFound(); 
		m_bNewBlock = miner.GetIsNewBlock();
		m_bReady = miner.GetIsReady();
		m_unHashes = miner.GetHashes();
		m_nThroughput = miner.GetThroughput();
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);
	}

	MinerThread& MinerThread::operator=(const MinerThread& miner)
	{
		m_pBLOCK = miner.GetBlock();
		m_GpuId = miner.GetGpuId();
		m_bBlockFound = miner.GetIsBlockFound(); 
		m_bNewBlock = miner.GetIsNewBlock();
		m_bReady = miner.GetIsReady();
		m_unHashes = miner.GetHashes();
		m_nThroughput = miner.GetThroughput();
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);

		return *this;
	}

	MinerThread::~MinerThread()
	{
	}


	void MinerThread::SK1024Miner()
	{
#ifdef WIN32
		SetThreadPriority(GetCurrentThread(), 2);
#endif
		const int throughput = GetThroughput() == 0 ? (512 * 896 * 28) : GetThroughput();
		loop
		{
			try
			{
				if (m_bShutdown)
					break;

				/** Don't mine if the Connection Thread is Submitting Block or Receiving New. **/
				while(m_bNewBlock || m_bBlockFound || !m_pBLOCK)
					Sleep(10);					
		
				CBigNum target;
				target.SetCompact(m_unBits);

				if (m_bBenchmark == true)
				{
					printf("\nBenchmark Mode - Should find a TEST block every 2^(32) hashes \"On Average\"\n\n");
					target.SetCompact(0x7d00ffff); //32-bits
				}

				while(!m_bNewBlock)
				{					
					if (m_bShutdown)
						break;

					uint1024 hash;
					unsigned long long hashes=0;
					unsigned int * TheData =(unsigned int*) m_pBLOCK->GetData();

					unsigned nHeight = m_pBLOCK->GetHeight();
					uint1024 TheTarget = target.getuint1024();
					uint64_t Nonce;
					bool found = false;
					m_clLock.lock();
					{
						Nonce = m_pBLOCK->GetNonce();
						found = scanhash_sk1024(m_GpuId, TheData, TheTarget, Nonce, 1, &hashes, throughput, 896, nHeight);
						if (hashes < 0xffffffffffff)
							SetHashes(GetHashes()+hashes);

					}
					m_clLock.unlock();

					if(found)
					{
						m_bBlockFound = true;
						m_clLock.lock();
						{
							m_pBLOCK->SetNonce(Nonce);
						}
						m_clLock.unlock();
                        						
                        break;
					}

					if(Nonce >= MAX_THREADS) //max_thread==> max_nonce
					{
						m_bNewBlock = true;
						break;
					}
				}
				
			}
			catch(std::exception& e)
			{ 
				printf("ERROR: %s\n", e.what()); 
			}
		}
	}


}