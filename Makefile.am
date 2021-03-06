
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

AUTOMAKE_OPTIONS = subdir-objects

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

bin_PROGRAMS	= skminer

skminer_SOURCES		= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
    hash/CBlock.cpp hash/Connection.cpp hash/DDOS_Filter.cpp \
    hash/DDOS_Score.cpp    hash/Keccak-compact64.cpp   hash/KeccakDuplex.cpp  hash/KeccakHash.cpp  hash/KeccakSponge.cpp \
    hash/Miner.cpp hash/MinerThread.cpp   hash/Outbound.cpp hash/Packet.cpp hash/skein.cpp hash/skein_block.cpp \
    hash/Timer.cpp \
    bignum.h  compat\inttypes.h compat\unistd.h  core.h cpuminer-config.h cuda_helper.h  \
    hash/brg_endian.h     hash/brg_types.h     hash/CBlock.h     hash/Connection.h     hash/crypto_hash.h \
    hash/DDOS_Filter.h     hash/DDOS_Score.h     hash/KeccakDuplex.h     hash/KeccakF-1600-interface.h  \
    hash/KeccakHash.h     hash/KeccakSponge.h     hash/Miner.h     hash/MinerThread.h     hash/Outbound.h \
    hash/Packet.h     hash/skein.h     hash/skein_iv.h     hash/skein_port.h     hash/templates.h \
    hash/Timer.h     hash/uint1024.h       miner2.h     types.h     uint256.h     util_llh.h \
    cpu-miner.cpp util.cpp core.cpp\
	heavy/heavy.cu Algo/cuda_skein1024.cu Algo/sk1024.cu Algo/sk1024_keccak.cu
			  

skminer_LDFLAGS		=  @CUDA_LDFLAGS@
skminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@  @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ -L/usr/local/lib -L/usr/local/cuda/lib64 -lboost_filesystem -lboost_system -lboost_program_options -lboost_thread
skminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ -std=c++11 -L/usr/local/boost -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME -DBOOST_THREAD_USE_LIB -DBOOST_SPIRIT_THREADSAFE

# we're now targeting all major compute architectures within one binary.
# require cuda 6.5 or higher
.cu.o:
	$(NVCC) @CFLAGS@ -I . -Xptxas "-v" -gencode=arch=compute_52,code=\"sm_52,compute_52\" -gencode=arch=compute_61,code=\"sm_61,compute_61\"  --ptxas-options=-v --maxrregcount 80 $(JANSSON_INCLUDES) -o $@ -c $<
