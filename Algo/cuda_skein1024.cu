#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <memory.h>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if __WORDSIZE == 64
typedef unsigned long uint64_t;
#else
typedef unsigned long long uint64_t;
#endif

#include "cuda_helper.h"

extern int device_major[8];
extern int device_minor[8];

extern int device_map[8];

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__constant__ uint64_t pTarget[16];

#define ROL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))

static __constant__ uint2 uMessage[27];
static __constant__ uint2 c_hv[17];
static __constant__ uint2 skein_ks_parity = { 0x55555555, 0x55555555 };

static __constant__ uint2 t12[9] =
{
	{ 0x80, 0 },
	{ 0, 0x70000000 },
	{ 0x80, 0x70000000 },
	{ 0xd8, 0 },
	{ 0, 0xb0000000 },
	{ 0xd8, 0xb0000000 },
	{ 0x08, 0 },
	{ 0, 0xff000000 },
	{ 0x08, 0xff000000 }
};

static const uint64_t cpu_SKEIN1024_IV_1024[16] =
{
	0x5A4352BE62092156,
	0x5F6E8B1A72F001CA,
	0xFFCBFE9CA1A2CE26,
	0x6C23C39667038BCA,
	0x583A8BFCCE34EB6C,
	0x3FDBFB11D4A46A3E,
	0x3304ACFCA8300998,
	0xB2F6675FA17F0FD2,
	0x9D2599730EF7AB6B,
	0x0914A20D3DFEA9E4,
	0xCC1A9CAFA494DBD3,
	0x9828030DA0A6388C,
	0x0D339D5DAADEE3DC,
	0xFC46DE35C4E2A086,
	0x53D6E4F52E19A6D1,
	0x5663952F715D1DDD,
};
static const int cpu_ROT1024[8][8] =
{
	{ 55, 43, 37, 40, 16, 22, 38, 12 },
	{ 25, 25, 46, 13, 14, 13, 52, 57 },
	{ 33, 8, 18, 57, 21, 12, 32, 54 },
	{ 34, 43, 25, 60, 44, 9, 59, 34 },
	{ 28, 7, 47, 48, 51, 9, 35, 41 },
	{ 17, 6, 18, 25, 43, 42, 40, 15 },
	{ 58, 7, 32, 45, 19, 18, 2, 56 },
	{ 47, 49, 27, 58, 37, 48, 53, 56 }
};

static __forceinline__ __device__ void Round1024(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF,
	int r0, int r1, int r2, int r3, int r4, int r5, int r6, int r7) {
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2(p1, r0) ^ p0;
	p3 = ROL2(p3, r1) ^ p2;
	p5 = ROL2(p5, r2) ^ p4;
	p7 = ROL2(p7, r3) ^ p6;
	p9 = ROL2(p9, r4) ^ p8;
	pB = ROL2(pB, r5) ^ pA;
	pD = ROL2(pD, r6) ^ pC;
	pF = ROL2(pF, r7) ^ pE;
}

static __forceinline__ __host__ void Round1024_host(uint64_t &p0, uint64_t &p1, uint64_t &p2, uint64_t &p3, uint64_t &p4, uint64_t &p5, uint64_t &p6, uint64_t &p7,
	uint64_t &p8, uint64_t &p9, uint64_t &pA, uint64_t &pB, uint64_t &pC, uint64_t &pD, uint64_t &pE, uint64_t &pF, int ROT)
{
	p0 += p1;
	p1 = ROL64(p1, cpu_ROT1024[ROT][0]);
	p1 ^= p0;
	p2 += p3;
	p3 = ROL64(p3, cpu_ROT1024[ROT][1]);
	p3 ^= p2;
	p4 += p5;
	p5 = ROL64(p5, cpu_ROT1024[ROT][2]);
	p5 ^= p4;
	p6 += p7;
	p7 = ROL64(p7, cpu_ROT1024[ROT][3]);
	p7 ^= p6;
	p8 += p9;
	p9 = ROL64(p9, cpu_ROT1024[ROT][4]);
	p9 ^= p8;
	pA += pB;
	pB = ROL64(pB, cpu_ROT1024[ROT][5]);
	pB ^= pA;
	pC += pD;
	pD = ROL64(pD, cpu_ROT1024[ROT][6]);
	pD ^= pC;
	pE += pF;
	pF = ROL64(pF, cpu_ROT1024[ROT][7]);
	pF ^= pE;
}


uint64_t *d_sknounce[8];
uint64_t *d_SKNonce[8];

__device__ __forceinline__
uint2 ROL8(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x2107);
	return result;
}

__device__ __forceinline__
uint2 ROR8(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x4321);
	result.y = __byte_perm(a.y, a.x, 0x4321);
	return result;
}

__constant__ uint2 keccak_round_constants[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

static void __forceinline__ __device__ keccak_1600(uint2 *s)
{
	uint2 bc[5], tmpxor[5], tmp1, tmp2;

	#pragma unroll 2
	for (int i = 0; i < 24; i++)
	{
		#pragma unroll
		for (uint32_t x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0] ^= keccak_round_constants[i];
	}
}

static __forceinline__ __device__ void Round1024_0(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 55) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 43) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 37) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 40) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 16) ^ p8;
	pA += pB;
	pB = ROL2(pB, 22) ^ pA;
	pC += pD;
	pD = ROL2(pD, 38) ^ pC;
	pE += pF;
	pF = ROL2(pF, 12) ^ pE;
}

static __forceinline__ __device__ void Round1024_1(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 25) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 25) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 46) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 13) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 14) ^ p8;
	pA += pB;
	pB = ROL2(pB, 13) ^ pA;
	pC += pD;
	pD = ROL2(pD, 52) ^ pC;
	pE += pF;
	pF = ROL2(pF, 57) ^ pE;
}

static __forceinline__ __device__ void Round1024_2(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 33) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 8) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 18) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 57) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 21) ^ p8;
	pA += pB;
	pB = ROL2(pB, 12) ^ pA;
	pC += pD;
	pD = ROL2(pD, 32) ^ pC;
	pE += pF;
	pF = ROL2(pF, 54) ^ pE;
}

static __forceinline__ __device__ void Round1024_3(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 34) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 43) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 25) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 60) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 44) ^ p8;
	pA += pB;
	pB = ROL2(pB, 9) ^ pA;
	pC += pD;
	pD = ROL2(pD, 59) ^ pC;
	pE += pF;
	pF = ROL2(pF, 34) ^ pE;
}

static __forceinline__ __device__ void Round1024_4(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 28) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 7) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 47) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 48) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 51) ^ p8;
	pA += pB;
	pB = ROL2(pB, 9) ^ pA;
	pC += pD;
	pD = ROL2(pD, 35) ^ pC;
	pE += pF;
	pF = ROL2(pF, 41) ^ pE;
}

static __forceinline__ __device__ void Round1024_5(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 17) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 6) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 18) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 25) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 43) ^ p8;
	pA += pB;
	pB = ROL2(pB, 42) ^ pA;
	pC += pD;
	pD = ROL2(pD, 40) ^ pC;
	pE += pF;
	pF = ROL2(pF, 15) ^ pE;
}

static __forceinline__ __device__ void Round1024_6(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 58) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 7) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 32) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 45) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 19) ^ p8;
	pA += pB;
	pB = ROL2(pB, 18) ^ pA;
	pC += pD;
	pD = ROL2(pD, 2) ^ pC;
	pE += pF;
	pF = ROL2(pF, 56) ^ pE;
}

static __forceinline__ __device__ void Round1024_7(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
	p1 = ROL2(p1, 47) ^ p0;
	p2 += p3;
	p3 = ROL2(p3, 49) ^ p2;
	p4 += p5;
	p5 = ROL2(p5, 27) ^ p4;
	p6 += p7;
	p7 = ROL2(p7, 58) ^ p6;
	p8 += p9;
	p9 = ROL2(p9, 37) ^ p8;
	pA += pB;
	pB = ROL2(pB, 48) ^ pA;
	pC += pD;
	pD = ROL2(pD, 53) ^ pC;
	pE += pF;
	pF = ROL2(pF, 56) ^ pE;
}

//__launch_bounds__(512)
//__launch_bounds__(576)
//__launch_bounds__(640)
//__launch_bounds__(704)
//__launch_bounds__(768)
//__launch_bounds__(832)
//__launch_bounds__(896)
//__launch_bounds__(1024)
__launch_bounds__(896) /* 896 performs best */  //Should match parameter in MinerThread.cpp
__global__ void  skein1024_gpu_hash_35(int threads, uint64_t startNonce, uint64_t *resNounce)
{
	//GTX 1060 SC
	//240 MH/s for Skein (2 rounds)
	//153 MH/s for Keccak (3 rounds), 181 MH/s for Prod (3 rounds).

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint64_t nonce = startNonce + (uint64_t)thread;

		__align__(16) uint2 h[17];
		__align__(16) uint2 t[3];
		__align__(16) uint2 p[16];
		__align__(16) uint2 state[25];
		__align__(16) uint2 tempnonce = vectorize(nonce);

		

		p[0] = uMessage[16] + c_hv[0];
		p[1] = uMessage[17] + c_hv[1];
		p[2] = uMessage[18] + c_hv[2];
		p[3] = uMessage[19] + c_hv[3];
		p[4] = uMessage[20] + c_hv[4];
		p[5] = uMessage[21] + c_hv[5];
		p[6] = uMessage[22] + c_hv[6];
		p[7] = uMessage[23] + c_hv[7];
		p[8] = uMessage[24] + c_hv[8];
		p[9] = uMessage[25] + c_hv[9];

		

		p[10] = tempnonce + c_hv[10];

		t[0] = t12[3]; // ptr  
		t[1] = t12[4]; // etype
		t[2] = t12[5];

		p[11] = c_hv[11];
		p[12] = c_hv[12];
		p[13] = c_hv[13] + t[0];
		p[14] = c_hv[14] + t[1];
		p[15] = c_hv[15];

		//========================================================================================

		#pragma unroll
		for (int i = 1; i < 21; i += 2)
		{
			Round1024_0(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], 0);
			Round1024_1(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1], 1);
			Round1024_2(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9], 2);
			Round1024_3(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7], 3);

			p[0] += c_hv[(i + 0) % 17];
			p[1] += c_hv[(i + 1) % 17];
			p[2] += c_hv[(i + 2) % 17];
			p[3] += c_hv[(i + 3) % 17];
			p[4] += c_hv[(i + 4) % 17];
			p[5] += c_hv[(i + 5) % 17];
			p[6] += c_hv[(i + 6) % 17];
			p[7] += c_hv[(i + 7) % 17];
			p[8] += c_hv[(i + 8) % 17];
			p[9] += c_hv[(i + 9) % 17];
			p[10] += c_hv[(i + 10) % 17];
			p[11] += c_hv[(i + 11) % 17];
			p[12] += c_hv[(i + 12) % 17];
			p[13] += c_hv[(i + 13) % 17] + t[(i + 0) % 3];
			p[14] += c_hv[(i + 14) % 17] + t[(i + 1) % 3];
			p[15] += c_hv[(i + 15) % 17] + make_uint2(i, 0);


			Round1024_4(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], 4);
			Round1024_5(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1], 5);
			Round1024_6(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9], 6);
			Round1024_7(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7], 7);

			p[0] += c_hv[(i + 1) % 17];
			p[1] += c_hv[(i + 2) % 17];
			p[2] += c_hv[(i + 3) % 17];
			p[3] += c_hv[(i + 4) % 17];
			p[4] += c_hv[(i + 5) % 17];
			p[5] += c_hv[(i + 6) % 17];
			p[6] += c_hv[(i + 7) % 17];
			p[7] += c_hv[(i + 8) % 17];
			p[8] += c_hv[(i + 9) % 17];
			p[9] += c_hv[(i + 10) % 17];
			p[10] += c_hv[(i + 11) % 17];
			p[11] += c_hv[(i + 12) % 17];
			p[12] += c_hv[(i + 13) % 17];
			p[13] += c_hv[(i + 14) % 17] + t[(i + 1) % 3];
			p[14] += c_hv[(i + 15) % 17] + t[(i + 2) % 3];
			p[15] += c_hv[(i + 16) % 17] + make_uint2(i + 1, 0);
		}

		p[0] ^= uMessage[16];
		p[1] ^= uMessage[17];
		p[2] ^= uMessage[18];
		p[3] ^= uMessage[19];
		p[4] ^= uMessage[20];
		p[5] ^= uMessage[21];
		p[6] ^= uMessage[22];
		p[7] ^= uMessage[23];
		p[8] ^= uMessage[24];
		p[9] ^= uMessage[25];
		p[10] ^= tempnonce;

		h[0] = p[0];
		h[1] = p[1];
		h[2] = p[2];
		h[3] = p[3];
		h[4] = p[4];
		h[5] = p[5];
		h[6] = p[6];
		h[7] = p[7];
		h[8] = p[8];
		h[9] = p[9];
		h[10] = p[10];
		h[11] = p[11];
		h[12] = p[12];
		h[13] = p[13];
		h[14] = p[14];
		h[15] = p[15];
		h[16] = skein_ks_parity;

		#pragma unroll
		for (int i = 0; i<16; i++) h[16] ^= p[i];

		t[0] = t12[6];
		t[1] = t12[7];
		t[2] = t12[8];

		p[13] += t[0];
		p[14] += t[1];

		//========================================================================================

		#pragma unroll
		for (int i = 1; i < 21; i += 2)
		{
			Round1024_0(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], 0);
			Round1024_1(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1], 1);
			Round1024_2(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9], 2);
			Round1024_3(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7], 3);

			p[0] += h[(i + 0) % 17];
			p[1] += h[(i + 1) % 17];
			p[2] += h[(i + 2) % 17];
			p[3] += h[(i + 3) % 17];
			p[4] += h[(i + 4) % 17];
			p[5] += h[(i + 5) % 17];
			p[6] += h[(i + 6) % 17];
			p[7] += h[(i + 7) % 17];
			p[8] += h[(i + 8) % 17];
			p[9] += h[(i + 9) % 17];
			p[10] += h[(i + 10) % 17];
			p[11] += h[(i + 11) % 17];
			p[12] += h[(i + 12) % 17];
			p[13] += h[(i + 13) % 17] + t[(i + 0) % 3];
			p[14] += h[(i + 14) % 17] + t[(i + 1) % 3];
			p[15] += h[(i + 15) % 17] + make_uint2(i, 0);


			Round1024_4(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], 4);
			Round1024_5(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1], 5);
			Round1024_6(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9], 6);
			Round1024_7(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7], 7);

			p[0] += h[(i + 1) % 17];
			p[1] += h[(i + 2) % 17];
			p[2] += h[(i + 3) % 17];
			p[3] += h[(i + 4) % 17];
			p[4] += h[(i + 5) % 17];
			p[5] += h[(i + 6) % 17];
			p[6] += h[(i + 7) % 17];
			p[7] += h[(i + 8) % 17];
			p[8] += h[(i + 9) % 17];
			p[9] += h[(i + 10) % 17];
			p[10] += h[(i + 11) % 17];
			p[11] += h[(i + 12) % 17];
			p[12] += h[(i + 13) % 17];
			p[13] += h[(i + 14) % 17] + t[(i + 1) % 3];
			p[14] += h[(i + 15) % 17] + t[(i + 2) % 3];
			p[15] += h[(i + 16) % 17] + make_uint2(i + 1, 0);
		}

		//========================================================================================

		state[0] = p[0];
		state[1] = p[1];
		state[2] = p[2];
		state[3] = p[3];
		state[4] = p[4];
		state[5] = p[5];
		state[6] = p[6];
		state[7] = p[7];
		state[8] = p[8];

		#pragma unroll
		for (int i = 9; i<25; i++) state[i] = make_uint2(0, 0);

		keccak_1600(state);

		state[0] ^= p[9];
		state[1] ^= p[10];
		state[2] ^= p[11];
		state[3] ^= p[12];
		state[4] ^= p[13];
		state[5] ^= p[14];
		state[6] ^= p[15];
		state[7] ^= vectorize(0x05);
		state[8] ^= vectorize(1ULL << 63);

		keccak_1600(state);
		keccak_1600(state);

		if (devectorize(state[6]) <= pTarget[15]) *resNounce = nonce;
	}
}

__host__ void skein1024_cpu_init(int thr_id, int threads)
{
}

__host__ uint64_t skein1024_cpu_hash(int thr_id, int threads, uint64_t startNounce, int order, int threadsperblock)
{
	uint64_t result = 0xffffffffffffffff;
	cudaMemset(d_SKNonce[thr_id], 0xff, sizeof(uint64_t));

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	skein1024_gpu_hash_35 << <grid, block >> >(threads, startNounce, d_SKNonce[thr_id]);
	cudaMemcpy(d_sknounce[thr_id], d_SKNonce[thr_id], sizeof(uint64_t), cudaMemcpyDeviceToHost);

	MyStreamSynchronize(NULL, order, thr_id);

	result = *d_sknounce[thr_id];
	return result;
}

__host__ void skein1024_setBlock(void *pdata)
{
	uint2 hv[17];
	uint64_t t[3];
	uint64_t h[17];
	uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;

	uint64_t cpu_skein_ks_parity = 0x5555555555555555;
	h[16] = cpu_skein_ks_parity;
	for (int i = 0; i<16; i++) {
		h[i] = cpu_SKEIN1024_IV_1024[i];
		h[16] ^= h[i];
	}
	uint64_t* alt_data = (uint64_t*)pdata;
	/////////////////////// round 1 //////////////////////////// should be on cpu => constant on gpu
	p0 = alt_data[0];
	p1 = alt_data[1];
	p2 = alt_data[2];
	p3 = alt_data[3];
	p4 = alt_data[4];
	p5 = alt_data[5];
	p6 = alt_data[6];
	p7 = alt_data[7];
	p8 = alt_data[8];
	p9 = alt_data[9];
	p10 = alt_data[10];
	p11 = alt_data[11];
	p12 = alt_data[12];
	p13 = alt_data[13];
	p14 = alt_data[14];
	p15 = alt_data[15];
	t[0] = 0x80; // ptr  
	t[1] = 0x7000000000000000; // etype
	t[2] = 0x7000000000000080;

	p0 += h[0];
	p1 += h[1];
	p2 += h[2];
	p3 += h[3];
	p4 += h[4];
	p5 += h[5];
	p6 += h[6];
	p7 += h[7];
	p8 += h[8];
	p9 += h[9];
	p10 += h[10];
	p11 += h[11];
	p12 += h[12];
	p13 += h[13] + t[0];
	p14 += h[14] + t[1];
	p15 += h[15];

	for (int i = 1; i < 21; i += 2)
	{
		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 1);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 2);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 3);

		p0 += h[(i + 0) % 17];
		p1 += h[(i + 1) % 17];
		p2 += h[(i + 2) % 17];
		p3 += h[(i + 3) % 17];
		p4 += h[(i + 4) % 17];
		p5 += h[(i + 5) % 17];
		p6 += h[(i + 6) % 17];
		p7 += h[(i + 7) % 17];
		p8 += h[(i + 8) % 17];
		p9 += h[(i + 9) % 17];
		p10 += h[(i + 10) % 17];
		p11 += h[(i + 11) % 17];
		p12 += h[(i + 12) % 17];
		p13 += h[(i + 13) % 17] + t[(i + 0) % 3];
		p14 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p15 += h[(i + 15) % 17] + (uint64_t)i;

		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 4);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 5);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 6);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 7);

		p0 += h[(i + 1) % 17];
		p1 += h[(i + 2) % 17];
		p2 += h[(i + 3) % 17];
		p3 += h[(i + 4) % 17];
		p4 += h[(i + 5) % 17];
		p5 += h[(i + 6) % 17];
		p6 += h[(i + 7) % 17];
		p7 += h[(i + 8) % 17];
		p8 += h[(i + 9) % 17];
		p9 += h[(i + 10) % 17];
		p10 += h[(i + 11) % 17];
		p11 += h[(i + 12) % 17];
		p12 += h[(i + 13) % 17];
		p13 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p14 += h[(i + 15) % 17] + t[(i + 2) % 3];
		p15 += h[(i + 16) % 17] + (uint64_t)(i + 1);

	}

	h[0] = p0^alt_data[0];
	h[1] = p1^alt_data[1];
	h[2] = p2^alt_data[2];
	h[3] = p3^alt_data[3];
	h[4] = p4^alt_data[4];
	h[5] = p5^alt_data[5];
	h[6] = p6^alt_data[6];
	h[7] = p7^alt_data[7];
	h[8] = p8^alt_data[8];
	h[9] = p9^alt_data[9];
	h[10] = p10^alt_data[10];
	h[11] = p11^alt_data[11];
	h[12] = p12^alt_data[12];
	h[13] = p13^alt_data[13];
	h[14] = p14^alt_data[14];
	h[15] = p15^alt_data[15];
	h[16] = cpu_skein_ks_parity;
	for (int i = 0; i<16; i++) { h[16] ^= h[i]; }
	for (int i = 0; i<17; i++) { hv[i] = lohi_host(h[i]); } //will slow down things


	uint2 cpu_Message[27];
	for (int i = 0; i<27; i++) { cpu_Message[i] = lohi_host(alt_data[i]); } //might slow down things

	cudaMemcpyToSymbol(c_hv, hv, sizeof(hv), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(uMessage, cpu_Message, sizeof(cpu_Message), 0, cudaMemcpyHostToDevice);
}

__host__ void sk1024_keccak_cpu_init(int thr_id, int threads)
{
	cudaMalloc(&d_SKNonce[thr_id], sizeof(uint64_t));
	cudaMallocHost(&d_sknounce[thr_id], 1 * sizeof(uint64_t));
}


__host__ void sk1024_set_Target(const void *ptarget)
{
	cudaMemcpyToSymbol(pTarget, ptarget, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

