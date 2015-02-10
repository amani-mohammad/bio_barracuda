#ifndef BWTGAP_H_
#define BWTGAP_H_

#include "bwt.h"
#include "bwtaln.h"

#define MAX_BLOCK_ENTRIES_SIZE 511
#define MAX_SCORE_ENTRIES_SIZE 100

typedef struct { // recursion stack
	//unsigned int length:16, last_diff_pos:16; // info : a<<9 | score & 0x1FF
	//unsigned int n_mm:8, n_gapo:8, n_gape:8, state:8;
	unsigned int length;
	unsigned int last_diff_pos;
	unsigned char n_mm;
	unsigned char n_gapo;
	unsigned char n_gape;
	unsigned char state;
	unsigned int k; // (k,l) is the SA region of [i,n-1]
	unsigned int l;
	unsigned int score;
} gap_entry_t;

typedef struct {
	int previous_block_index;
	gap_entry_t entries[MAX_BLOCK_ENTRIES_SIZE];
	//previous GAP block;
} gap_block_t;

typedef struct {
	int n_entries;
	int m_entries;
	gap_entry_t * stack;
} gap_stack1_t;

typedef struct {
	unsigned int n_entries;
	unsigned short cur_block_entries_size;
	unsigned short cur_block_index;
} gap_stack2_t;

typedef struct {
	int n_stacks, best, n_entries;
	gap_stack1_t * stacks;
} gap_stack_t;

typedef struct {
	int n_stacks, best, n_entries;
	//gap_stack2_t stacks[MAX_SCORE_ENTRIES_SIZE];
} gap_stack_cuda_t;

#define STATE_M 0
#define STATE_I 1
#define STATE_D 2

#define aln_score(m,o,e,p) ((m)*(p)->s_mm + (o)*(p)->s_gapo + (e)*(p)->s_gape)
#define aln_score2(m,o,e,p) ((m)*(p).s_mm + (o)*(p).s_gapo + (e)*(p).s_gape)

#ifdef __cplusplus
extern "C" {
#endif

	gap_stack_t *gap_init_stack(int max_mm, int max_gapo, int max_gape, const barracuda_gap_opt_t *opt);
	void gap_destroy_stack(gap_stack_t *stack);
	barracuda_aln1_t *bwt_match_gap(bwt_t *const bwt[2], int len, const ubyte_t *seq[2], bwt_width_t *w[2],
							  bwt_width_t *seed_w[2], const barracuda_gap_opt_t *opt, int *_n_aln, gap_stack_t *stack);

	void dfs_match(bwt_t *const bwts[2], int len, const ubyte_t *seq[2], bwt_width_t *w[2], bwt_width_t *seed_w[2], const barracuda_gap_opt_t *opt, alignment_store_t *aln);

	void bwa_aln2seq(int n_aln, const barracuda_aln1_t *aln, bwa_seq_t *s);

#ifdef __cplusplus
}
#endif

#endif
