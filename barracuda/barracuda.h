
/*
 * barracuda.h
 *
 *  Created on: 8 Jun 2012
 *      Author: yhbl2
 */


#ifndef BARRACUDA_H_
#define BARRACUDA_H_

#include <stdint.h>
#include "bwt.h"
#include "bwtaln.h"

#ifdef __cplusplus
extern "C" {
#endif
	///////////////////////////////////////////////////////////////
	// Begin CUDA ALN
	///////////////////////////////////////////////////////////////

	int detect_cuda_device();

	int bwa_deviceQuery();

	void aln_quicksort2(bwt_aln1_t *aln, int m, int n);

	double diff_in_seconds(struct timeval *finishtime, struct timeval * starttime);

	gap_opt_t *gap_init_bwaopt(barracuda_gap_opt_t * opt);

	barracuda_gap_opt_t *gap_init_opt();

	bwa_seq_t *bwa_read_seq(
			bwa_seqio_t *seq,
			unsigned int n_needed,
			unsigned int *n,
			int is_comp,
			int mid);

	unsigned long long copy_bwts_to_cuda_memory(
			const char * prefix,
			unsigned int ** bwt,
			unsigned int ** rbwt,
			int mem_available,
			bwtint_t* forward_seq_len,
			bwtint_t* backward_seq_len);

	void barracuda_bwa_aln_core(const char *prefix,
			const char *fn_fa,
			barracuda_gap_opt_t *opt);

	int bwa_read_seq_one_half_byte (
			bwa_seqio_t *bs,
			unsigned char * half_byte_array,
			unsigned int start_index,
			unsigned short * length,
			int mid);

	int bwa_read_seq_one (bwa_seqio_t *bs,
			unsigned char * byte_array,
			unsigned short * length);

	void cuda_alignment_core(const char *prefix,
			bwa_seqio_t *ks,
			barracuda_gap_opt_t *opt);


	///////////////////////////////////////////////////////////////
	// End CUDA ALN
	///////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////
	// Begin CUDA SAMSE
	///////////////////////////////////////////////////////////////

	void launch_bwa_cal_pac_pos_cuda(
		const char *prefix,
		int n_seqs,
		bwa_seq_t *seqs,
		int max_mm,
		float fnr,
		int device);

	int prepare_bwa_cal_pac_pos_cuda1(
	    unsigned int **global_bwt,
	    unsigned int **global_rbwt,
	    const char *prefix,
	    bwtint_t **bwt_sa_de,
	    bwtint_t **rbwt_sa_de,
	    const bwt_t *bwt,
	    const bwt_t *rbwt,
	    const int *g_log_n_ho,
	    int **g_log_n_de,
	    const int g_log_n_len,
	    int device);

	void prepare_bwa_cal_pac_pos_cuda2(int n_seqs_max);

	void free_bwa_cal_pac_pos_cuda1(
	    unsigned int *global_bwt,
	    unsigned int *global_rbwt,
	    bwtint_t *bwt_sa_de,
	    bwtint_t *rbwt_sa_de,
	    int *g_log_n_de);

	void free_bwa_cal_pac_pos_cuda2();

	///////////////////////////////////////////////////////////////
	// End CUDA SAMSE
	///////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif /* BARRACUDA_H_ */
