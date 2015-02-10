/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwape.c  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2012, Brian Lam and Simon Lam

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 3
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   This program is based on a modified version of BWA 0.4.9

*/

#define PACKAGE_VERSION "0.6.2 beta"

#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "bwtaln.h"
#include "kvec.h"
#include "bntseq.h"
#include "utils.h"
#include "stdaln.h"
#include "barracuda.h"

#ifdef HAVE_PTHREAD
#define THREAD_BLOCK_SIZE 1024
#include <pthread.h>
static pthread_mutex_t g_seq_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

#define ENABLE_N_MULTI 0

typedef struct {
	int n;
	bwtint_t *a;
} poslist_t;

typedef struct {
	double avg, std, ap_prior;
	bwtint_t low, high, high_bayesian;
} isize_info_t;


#include "khash.h"
KHASH_MAP_INIT_INT64(64, poslist_t)

#include "ksort.h"
KSORT_INIT_GENERIC(uint64_t)

typedef struct {
	kvec_t(uint64_t) arr;
	kvec_t(uint64_t) pos[2];
	kvec_t(barracuda_aln1_t) aln[2];
} pe_data_t;

#define MIN_HASH_WIDTH 1000

static int g_log_n[256];
static kh_64_t *g_hash;


void bwa_aln2seq(int n_aln, const barracuda_aln1_t *aln, bwa_seq_t *s);
void bwa_refine_gapped(int tid, int n_threads, const bntseq_t *bns, int n_seqs, bwa_seq_t *seqs, ubyte_t *_pacseq, bntseq_t *ntbns);
int bwa_approx_mapQ(const bwa_seq_t *p, int mm);
void bwa_print_sam1(const bntseq_t *bns, bwa_seq_t *p, const bwa_seq_t *mate, int mode, int max_top2);
bntseq_t *bwa_open_nt(const char *prefix);
void bwa_print_sam_SQ(const bntseq_t *bns);
void bwa_print_sam_PG();


//TODO: These following 4 functions are temporary and should be integrated bwase
#if ENABLE_N_MULTI == 1
void pe_swap(barracuda_aln1_t *x, barracuda_aln1_t *y)
{
   barracuda_aln1_t temp;
   temp = *x;
   *x = *y;
   *y = temp;
}

int pe_choose_pivot(int i,int j)
{
   return((i+j) /2);
}


void pe_quicksort(barracuda_aln1_t *aln, int m, int n)
//This function sorts the alignment array from barracuda to make it compatible with SAMSE/SAMPE cores
{
	int key,i,j,k;

	if (m < n)
	{
	      k = pe_choose_pivot(m, n);
	      pe_swap(&aln[m],&aln[k]);
	      key = aln[m].score;
	      i = m+1;
	      j = n;
	      while(i <= j)
	      {
	         while((i <= n) && (aln[i].score <= key))
	                i++;
	         while((j >= m) && (aln[j].score > key))
	                j--;
	         if(i < j)
	                pe_swap(&aln[i],&aln[j]);
	      }
	      // swap two elements
	      pe_swap(&aln[m],&aln[j]);
	      // recursively sort the lesser lists
	      pe_quicksort(aln, m, j-1);
	      pe_quicksort(aln, j+1, n);
	 }
}

void bwa_aln2seq_core(int n_aln, barracuda_aln1_t *aln, bwa_seq_t *s, int set_main, int n_multi)
{
	int i, cnt, best;
	if (n_aln == 0) {
		s->type = BWA_TYPE_NO_MATCH;
		s->c1 = s->c2 = 0;
		return;
	}

	// add quicksort
	if(n_aln > 1)
	{
		pe_quicksort(aln, 0, n_aln-1);
	}

	if (set_main) {
		best = aln[0].score;
		for (i = cnt = 0; i < n_aln; ++i) {
			const barracuda_aln1_t *p = aln + i;
			if (p->score > best) break;
			if (drand48() * (p->l - p->k + 1 + cnt) > (double)cnt) {
				s->n_mm = p->n_mm; s->n_gapo = p->n_gapo; s->n_gape = p->n_gape; s->strand = p->a;
				s->score = p->score;
				s->sa = p->k + (bwtint_t)((p->l - p->k + 1) * drand48());
			}
			cnt += p->l - p->k + 1;
		}
		s->c1 = cnt;
		for (; i < n_aln; ++i) cnt += aln[i].l - aln[i].k + 1;
		s->c2 = cnt - s->c1;
		s->type = s->c1 > 1? BWA_TYPE_REPEAT : BWA_TYPE_UNIQUE;
	}

	if (n_multi) {
		int k, rest, n_occ, z = 0;
		for (k = n_occ = 0; k < n_aln; ++k) {
			const barracuda_aln1_t *q = aln + k;
			n_occ += q->l - q->k + 1;
		}
		if (s->multi) free(s->multi);
		if (n_occ > n_multi + 1) { // if there are too many hits, generate none of them
			s->multi = 0; s->n_multi = 0;
			return;
		}
		/* The following code is more flexible than what is required
		 * here. In principle, due to the requirement above, we can
		 * simply output all hits, but the following samples "rest"
		 * number of random hits. */
		rest = n_occ > n_multi + 1? n_multi + 1 : n_occ; // find one additional for ->sa
		s->multi = calloc(rest, sizeof(bwt_multi1_t));
		for (k = 0; k < n_aln; ++k) {
			const barracuda_aln1_t *q = aln + k;
			if (q->l - q->k + 1 <= rest) {
				bwtint_t l;
				for (l = q->k; l <= q->l; ++l) {
					s->multi[z].pos = l;
					s->multi[z].gap = q->n_gapo + q->n_gape;
					s->multi[z].mm = q->n_mm;
					s->multi[z++].strand = q->a;
				}
				rest -= q->l - q->k + 1;
			} else { // Random sampling (http://code.activestate.com/recipes/272884/). In fact, we never come here.
				int j, i, k;
				for (j = rest, i = q->l - q->k + 1, k = 0; j > 0; --j) {
					double p = 1.0, x = drand48();
					while (x < p) p -= p * j / (i--);
					s->multi[z].pos = q->l - i;
					s->multi[z].gap = q->n_gapo + q->n_gape;
					s->multi[z].mm = q->n_mm;
					s->multi[z++].strand = q->a;
				}
				rest = 0;
				break;
			}
		}
		s->n_multi = z;
		for (k = z = 0; k < s->n_multi; ++k)
			if (s->multi[k].pos != s->sa)
				s->multi[z++] = s->multi[k];
		s->n_multi = z < n_multi? z : n_multi;
	}
}
#endif

pe_opt_t *bwa_init_pe_opt()
{
	pe_opt_t *po;
	po = (pe_opt_t*)calloc(1, sizeof(pe_opt_t));
	po->max_isize = 500;
	po->force_isize = 0;
	po->max_occ = 100000;
	po->n_multi = 0; //changed from (currently disabled) 3;
	po->N_multi = 0; //changed from (currently disabled) 10;
	po->type = BWA_PET_STD;
	po->is_sw = 1;
	po->ap_prior = 1e-5;
	po->thread = -1;
	return po;
}

static inline uint64_t hash_64(uint64_t key)
{
	key += ~(key << 32);
	key ^= (key >> 22);
	key += ~(key << 13);
	key ^= (key >> 8);
	key += (key << 3);
	key ^= (key >> 15);
	key += ~(key << 27);
	key ^= (key >> 31);
	return key;
}

// for normal distribution, this is about 3std
#define OUTLIER_BOUND 2.0

static int infer_isize(int n_seqs, bwa_seq_t *seqs[2], isize_info_t *ii, double ap_prior, int64_t L)
{
	uint64_t x, *isizes, n_ap = 0;
	int n, i, tot, p25, p75, p50, max_len = 1, tmp;
	double skewness = 0.0, kurtosis = 0.0, y;

	ii->avg = ii->std = -1.0;
	ii->low = ii->high = ii->high_bayesian = 0;
	isizes = (uint64_t*)calloc(n_seqs, 8);
	for (i = 0, tot = 0; i != n_seqs; ++i) {
		bwa_seq_t *p[2];
		p[0] = seqs[0] + i; p[1] = seqs[1] + i;
		if (p[0]->mapQ >= 20 && p[1]->mapQ >= 20) {
			x = (p[0]->pos < p[1]->pos)? p[1]->pos + p[1]->len - p[0]->pos : p[0]->pos + p[0]->len - p[1]->pos;
			if (x < 100000) isizes[tot++] = x;
		}
		if (p[0]->len > max_len) max_len = p[0]->len;
		if (p[1]->len > max_len) max_len = p[1]->len;
	}
	if (tot < 20) {
		fprintf(stderr, "  [infer_isize] fail to infer insert size: too few good pairs\n");
		free(isizes);
		return -1;
	}
	ks_introsort(uint64_t, tot, isizes);
	p25 = isizes[(int)(tot*0.25 + 0.5)];
	p50 = isizes[(int)(tot*0.50 + 0.5)];
	p75 = isizes[(int)(tot*0.75 + 0.5)];
	tmp  = (int)(p25 - OUTLIER_BOUND * (p75 - p25) + .499);
	ii->low = tmp > max_len? tmp : max_len; // ii->low is unsigned
	ii->high = (int)(p75 + OUTLIER_BOUND * (p75 - p25) + .499);
	for (i = 0, x = n = 0; i < tot; ++i)
		if (isizes[i] >= ii->low && isizes[i] <= ii->high)
			++n, x += isizes[i];
	ii->avg = (double)x / n;
	for (i = 0; i < tot; ++i) {
		if (isizes[i] >= ii->low && isizes[i] <= ii->high) {
			double tmp = (isizes[i] - ii->avg) * (isizes[i] - ii->avg);
			ii->std += tmp;
			skewness += tmp * (isizes[i] - ii->avg);
			kurtosis += tmp * tmp;
		}
	}
	kurtosis = kurtosis/n / (ii->std / n * ii->std / n) - 3;
	ii->std = sqrt(ii->std / n); // it would be better as n-1, but n is usually very large
	skewness = skewness / n / (ii->std * ii->std * ii->std);
	for (y = 1.0; y < 10.0; y += 0.01)
		if (.5 * erfc(y / M_SQRT2) < ap_prior / L * (y * ii->std + ii->avg)) break;
	ii->high_bayesian = (bwtint_t)(y * ii->std + ii->avg + .499);
	for (i = 0; i < tot; ++i)
		if (isizes[i] > ii->high_bayesian) ++n_ap;
	ii->ap_prior = .01 * (n_ap + .01) / tot;
	if (ii->ap_prior < ap_prior) ii->ap_prior = ap_prior;
	free(isizes);
	fprintf(stderr, "  [infer_isize] (25, 50, 75) percentile: (%d, %d, %d)\n", p25, p50, p75);
	if (isnan(ii->std) || p75 > 100000) {
		ii->low = ii->high = ii->high_bayesian = 0; ii->avg = ii->std = -1.0;
		fprintf(stderr, "  [infer_isize] fail to infer insert size: weird pairing\n");
		return -1;
	}
	for (y = 1.0; y < 10.0; y += 0.01)
		if (.5 * erfc(y / M_SQRT2) < ap_prior / L * (y * ii->std + ii->avg)) break;
	ii->high_bayesian = (bwtint_t)(y * ii->std + ii->avg + .499);
	fprintf(stderr, "  [infer_isize] low and high boundaries: %d and %d for estimating avg and std\n", ii->low, ii->high);
	fprintf(stderr, "  [infer_isize] inferred external isize from %d pairs: %.3lf +/- %.3lf\n", n, ii->avg, ii->std);
	fprintf(stderr, "  [infer_isize] skewness: %.3lf; kurtosis: %.3lf; ap_prior: %.2e\n", skewness, kurtosis, ii->ap_prior);
	fprintf(stderr, "  [infer_isize] inferred maximum insert size: %d (%.2lf sigma)\n", ii->high_bayesian, y);
	return 0;
}

static int pairing(bwa_seq_t *p[2], pe_data_t *d, const pe_opt_t *opt, int s_mm, const isize_info_t *ii)
{
	int i, j, o_n, subo_n, cnt_chg = 0;
	uint64_t last_pos[2][2], o_pos[2], subo_score, o_score;

	// here v>=u. When ii is set, we check insert size with ii; otherwise with opt->max_isize
#define __pairing_aux(u,v) do {											\
		bwtint_t l = ((v)>>32) + p[(v)&1]->len - ((u)>>32);				\
		if ((u) != (uint64_t)-1 && (v)>>32 > (u)>>32					\
			&& ((ii->high && l >= ii->low && l <= ii->high) || (ii->high == 0 && l <= opt->max_isize))) \
		{																\
			uint64_t s = d->aln[(v)&1].a[(uint32_t)(v)>>1].score + d->aln[(u)&1].a[(uint32_t)(u)>>1].score; \
			s *= 10;													\
			if (ii->high) s += (int)(-4.343 * log(.5 * erfc(M_SQRT1_2 * fabs(l - ii->avg) / ii->std)) + .499); \
			s = s<<32 | (uint32_t)hash_64((u)>>32<<32 | (v)>>32);		\
			if (s>>32 == o_score>>32) ++o_n;							\
			else if (s>>32 < o_score>>32) { subo_n += o_n; o_n = 1; }	\
			else ++subo_n;												\
			if (s < o_score) subo_score = o_score, o_score = s, o_pos[(u)&1] = (u), o_pos[(v)&1] = (v); \
			else if (s < subo_score) subo_score = s;					\
		}																\
	} while (0)

#define __pairing_aux2(q, w) do {										\
		(q)->extra_flag |= SAM_FPP;										\
		if ((q)->pos != (w)>>32) {										\
			const barracuda_aln1_t *r = d->aln[(w)&1].a + ((uint32_t)(w)>>1);	\
			(q)->n_mm = r->n_mm; (q)->n_gapo = r->n_gapo; (q)->n_gape = r->n_gape; (q)->strand = r->a; \
			(q)->score = r->score; (q)->mapQ = mapQ_p;					\
			(q)->pos = (w)>>32;											\
			if ((q)->mapQ > 0) ++cnt_chg;								\
		}																\
	} while (0)

	o_score = subo_score = (uint64_t)-1;
	o_n = subo_n = 0;
	ks_introsort(uint64_t, d->arr.n, d->arr.a);
	for (j = 0; j < 2; ++j) last_pos[j][0] = last_pos[j][1] = (uint64_t)-1;
	if (opt->type == BWA_PET_STD) {
		for (i = 0; i < d->arr.n; ++i) {
			uint64_t x = d->arr.a[i];
			int strand = d->aln[x&1].a[(uint32_t)x>>1].a;
			if (strand == 1) { // reverse strand, then check
				int y = 1 - (x&1);
				__pairing_aux(last_pos[y][1], x);
				__pairing_aux(last_pos[y][0], x);
			} else { // forward strand, then push
				last_pos[x&1][0] = last_pos[x&1][1];
				last_pos[x&1][1] = x;
			}
		}
	} else if (opt->type == BWA_PET_SOLID) {
		for (i = 0; i < d->arr.n; ++i) {
			uint64_t x = d->arr.a[i];
			int strand = d->aln[x&1].a[(uint32_t)x>>1].a;
			if ((strand^x)&1) { // push
				int y = 1 - (x&1);
				__pairing_aux(last_pos[y][1], x);
				__pairing_aux(last_pos[y][0], x);
			} else { // check
				last_pos[x&1][0] = last_pos[x&1][1];
				last_pos[x&1][1] = x;
			}
		}
	} else {
		fprintf(stderr, "[paring] not implemented yet!\n");
		exit(1);
	}
	// set pairing
	//fprintf(stderr, "[%d, %d, %d, %d]\n", d->arr.n, (int)(o_score>>32), (int)(subo_score>>32), o_n);
	if (o_score != (uint64_t)-1) {
		int mapQ_p = 0; // this is the maximum mapping quality when one end is moved
		//fprintf(stderr, "%d, %d\n", o_n, subo_n);
		if (o_n == 1) {
			if (subo_score == (uint64_t)-1) mapQ_p = 29; // no sub-optimal pair
			else if ((subo_score>>32) - (o_score>>32) > s_mm * 10) mapQ_p = 23; // poor sub-optimal pair
			else {
				int n = subo_n > 255? 255 : subo_n;
				mapQ_p = ((subo_score>>32) - (o_score>>32)) / 2 - g_log_n[n];
				if (mapQ_p < 0) mapQ_p = 0;
			}
		}
		if (p[0]->pos == o_pos[0]>>32 && p[1]->pos == o_pos[1]>>32) { // both ends not moved
			if (p[0]->mapQ > 0 && p[1]->mapQ > 0) {
				int mapQ = p[0]->mapQ + p[1]->mapQ;
				if (mapQ > 60) mapQ = 60;
				p[0]->mapQ = p[1]->mapQ = mapQ;
			} else {
				if (p[0]->mapQ == 0) p[0]->mapQ = (mapQ_p + 7 < p[1]->mapQ)? mapQ_p + 7 : p[1]->mapQ;
				if (p[1]->mapQ == 0) p[1]->mapQ = (mapQ_p + 7 < p[0]->mapQ)? mapQ_p + 7 : p[0]->mapQ;
			}
		} else if (p[0]->pos == o_pos[0]>>32) { // [1] moved
			p[1]->seQ = 0; p[1]->mapQ = p[0]->mapQ;
			if (p[1]->mapQ > mapQ_p) p[1]->mapQ = mapQ_p;
		} else if (p[1]->pos == o_pos[1]>>32) { // [0] moved
			p[1]->seQ = 0; p[0]->mapQ = p[1]->mapQ;
			if (p[0]->mapQ > mapQ_p) p[0]->mapQ = mapQ_p;
		} else { // both ends moved
			p[0]->seQ = p[1]->seQ = 0;
			mapQ_p -= 20;
			if (mapQ_p < 0) mapQ_p = 0;
			p[0]->mapQ = p[1]->mapQ = mapQ_p;
		}
		__pairing_aux2(p[0], o_pos[0]);
		__pairing_aux2(p[1], o_pos[1]);
	}
	return cnt_chg;
}

typedef struct {
	kvec_t(barracuda_aln1_t) aln;
} aln_buf_t;


void posix_pe(int n_seqs, bwa_seq_t *seqs[2], aln_buf_t* buf[2], /*pe_data_t *d, */const pe_opt_t *opt, const bwt_t *bwt[2], const isize_info_t *ii, const barracuda_gap_opt_t *gopt, int tid, int n_threads)
{
	int i,j;
	int cnt_chg = 0;
	pe_data_t *d;
	d = (pe_data_t*)calloc(1, sizeof(pe_data_t));

	//fprintf(stderr,"printing from thread %d\n",tid);

	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p[2];
		for (j = 0; j < 2; ++j) {
			p[j] = seqs[j] + i;
			kv_copy(barracuda_aln1_t, d->aln[j], buf[j][i].aln);
		}

		#ifdef HAVE_PTHREAD
			if (n_threads > 1) {
				pthread_mutex_lock(&g_seq_lock);
				if (p[0]->tid < 0) { // unassigned
					int j;
					for (j = i; j < n_seqs && j < i + THREAD_BLOCK_SIZE; ++j){
						seqs[0][j].tid = tid;
						seqs[1][j].tid = tid;
					}
				} else if (p[0]->tid != tid) {
					pthread_mutex_unlock(&g_seq_lock);
					continue;
				}
				pthread_mutex_unlock(&g_seq_lock);

			}
		#endif


		if ((p[0]->type == BWA_TYPE_UNIQUE || p[0]->type == BWA_TYPE_REPEAT)
			&& (p[1]->type == BWA_TYPE_UNIQUE || p[1]->type == BWA_TYPE_REPEAT))
		{ // only when both ends mapped
			uint64_t x;
			int j, k, n_occ[2];
			for (j = 0; j < 2; ++j) {
				n_occ[j] = 0;
				for (k = 0; k < d->aln[j].n; ++k)
					n_occ[j] += d->aln[j].a[k].l - d->aln[j].a[k].k + 1;
			}
			if (n_occ[0] > opt->max_occ || n_occ[1] > opt->max_occ) continue;
			d->arr.n = 0;
			for (j = 0; j < 2; ++j) {
				for (k = 0; k < d->aln[j].n; ++k) {
					barracuda_aln1_t *r = d->aln[j].a + k;
					bwtint_t l;
					if (r->l - r->k + 1 >= MIN_HASH_WIDTH) { // then check hash table
						uint64_t key = (uint64_t)r->k<<32 | r->l;
						int ret;
						khint_t iter = kh_put(64, g_hash, key, &ret);
						if (ret) { // not in the hash table; ret must equal 1 as we never remove elements
							poslist_t *z = &kh_val(g_hash, iter);
							z->n = r->l - r->k + 1;
							z->a = (bwtint_t*)malloc(sizeof(bwtint_t) * z->n);
							for (l = r->k; l <= r->l; ++l)
								z->a[l - r->k] = r->a? bwt_sa(bwt[0], l) : bwt[1]->seq_len - (bwt_sa(bwt[1], l) + p[j]->len);
						}
						for (l = 0; l < kh_val(g_hash, iter).n; ++l) {
							x = kh_val(g_hash, iter).a[l];
							x = x<<32 | k<<1 | j;
							kv_push(uint64_t, d->arr, x);
						}
					} else { // then calculate on the fly
						for (l = r->k; l <= r->l; ++l) {
							x = r->a? bwt_sa(bwt[0], l) : bwt[1]->seq_len - (bwt_sa(bwt[1], l) + p[j]->len);
							x = x<<32 | k<<1 | j;
							kv_push(uint64_t, d->arr, x);
						}
					}
				}
			}
			cnt_chg += pairing(p, d, opt, gopt->s_mm, ii);
		}
#if ENABLE_N_MULTI == 1
		if (opt->N_multi || opt->n_multi) {
			for (j = 0; j < 2; ++j) {
				if (p[j]->type != BWA_TYPE_NO_MATCH) {
					int k;
					if (!(p[j]->extra_flag&SAM_FPP) && p[1-j]->type != BWA_TYPE_NO_MATCH) {
						bwa_aln2seq_core(d->aln[j].n, d->aln[j].a, p[j], 0, p[j]->c1+p[j]->c2-1 > opt->N_multi? opt->n_multi : opt->N_multi);
					} else bwa_aln2seq_core(d->aln[j].n, d->aln[j].a, p[j], 0, opt->n_multi);
					for (k = 0; k < p[j]->n_multi; ++k) {
						bwt_multi1_t *q = p[j]->multi + k;
						q->pos = q->strand? bwt_sa(bwt[0], q->pos) : bwt[1]->seq_len - (bwt_sa(bwt[1], q->pos) + p[j]->len);
					}
				}
			}
		}

#endif
	}

	kv_destroy(d->arr);
	kv_destroy(d->pos[0]); kv_destroy(d->pos[1]);
	kv_destroy(d->aln[0]); kv_destroy(d->aln[1]);
	free(d);

	fprintf(stderr, "  [posix_pe] Thread %d: Change of coordinates in %d alignments.\n", tid, cnt_chg);

}

//POSIX Multithreading for posix_pe and posix_se

#ifdef HAVE_PTHREAD

typedef struct {
	int tid;
	int n_seqs;
	bwa_seq_t *seqs[2];
	const bwt_t *bwt[2];
	aln_buf_t *buf[2];
	int n_threads;
	const pe_opt_t *opt;
	const barracuda_gap_opt_t *gopt;
	isize_info_t *ii;

} thread_sampe_pe_aux_t;

static void *sampe_pe_worker(void *data)
{
	thread_sampe_pe_aux_t *d = (thread_sampe_pe_aux_t*)data;
	posix_pe(d->n_seqs, d->seqs, d->buf, /*d->d,*/ d->opt, d->bwt, d->ii, d->gopt, d->tid, d->n_threads);
	return 0;
}

#endif //HAVE_PTHREAD

//POSIX SE
void posix_se(int n_seqs, bwa_seq_t *seqs[2], const bwt_t *bwt[2], aln_buf_t *buf[2], const barracuda_gap_opt_t *gopt, int tid, int n_threads)
{
	int i, j;

	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p[2];
		p[0] = seqs[0] + i;
		p[1] = seqs[1] + i;

		#ifdef HAVE_PTHREAD
			if (n_threads > 1) {
				pthread_mutex_lock(&g_seq_lock);
				if (p[0]->tid < 0) { // unassigned
					int j;
					for (j = i; j < n_seqs && j < i + THREAD_BLOCK_SIZE; ++j){
						seqs[0][j].tid = tid;
						seqs[1][j].tid = tid;
					}
				} else if (p[0]->tid != tid) {
					pthread_mutex_unlock(&g_seq_lock);
					continue;
				}
				pthread_mutex_unlock(&g_seq_lock);

			}
		#endif

		for (j = 0; j < 2; ++j) {
			unsigned int n_aln = buf[j][i].aln.n;
			//p[j]->n_multi = 0;
			p[j]->extra_flag |= SAM_FPD | (j == 0? SAM_FR1 : SAM_FR2);
			// generate SE alignment and mapping quality
			bwa_aln2seq(n_aln, buf[j][i].aln.a, p[j]);
			if (p[j]->type == BWA_TYPE_UNIQUE || p[j]->type == BWA_TYPE_REPEAT) {
				int max_diff = gopt->fnr > 0.0? bwa_cal_maxdiff(p[j]->len, BWA_AVG_ERR, gopt->fnr) : gopt->max_diff;
				p[j]->pos = p[j]->strand? bwt_sa(bwt[0], p[j]->sa)
					: bwt[1]->seq_len - (bwt_sa(bwt[1], p[j]->sa) + p[j]->len);
				p[j]->seQ = p[j]->mapQ = bwa_approx_mapQ(p[j], max_diff);
			}
		}
	}

	//fprintf(stderr, "  [posix_se] Thread %d: finished single end conversions.\n", tid);

}
#ifdef HAVE_PTHREAD

static void *sampe_se_worker(void *data)
{
	thread_sampe_pe_aux_t *d = (thread_sampe_pe_aux_t*)data;
	posix_se(d->n_seqs, d->seqs, d->bwt, d->buf, d->gopt, d->tid, d->n_threads);
	return 0;
}

#endif //HAVE_PTHREAD

int bwa_cal_pac_pos_pe(const char *prefix, int n_seqs, bwa_seq_t *seqs[2], FILE *fp_sa[2], isize_info_t *ii,
					   const pe_opt_t *opt, const barracuda_gap_opt_t *gopt, const isize_info_t *last_ii, const bwt_t *bwt[2], int n_threads)
{
	int i, j;
	pe_data_t *d;
	d = (pe_data_t*)calloc(1, sizeof(pe_data_t));
	aln_buf_t *buf[2];
	buf[0] = (aln_buf_t*)calloc(n_seqs, sizeof(aln_buf_t));
	buf[1] = (aln_buf_t*)calloc(n_seqs, sizeof(aln_buf_t));

	//read data from sai files
	for (i = 0; i != n_seqs; ++i) {
		for (j = 0; j < 2; ++j) {
			unsigned int n_aln;
			fread(&n_aln, 4, 1, fp_sa[j]);
			if (n_aln > kv_max(d->aln[j]))
				kv_resize(barracuda_aln1_t, d->aln[j], n_aln);
			d->aln[j].n = n_aln;
			fread(d->aln[j].a, sizeof(barracuda_aln1_t), n_aln, fp_sa[j]);
			kv_copy(barracuda_aln1_t, buf[j][i].aln, d->aln[j]); // backup d->aln[j]
		}
	}

	// SE
	#ifdef HAVE_PTHREAD
	if (n_threads <= 1) { // no multi-threading at all
		posix_se(n_seqs, seqs, bwt, buf, gopt, 0, 0);
	} else {
		pthread_t *tid;
		pthread_attr_t attr;
		thread_sampe_pe_aux_t *data;
		int j;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		data = (thread_sampe_pe_aux_t*)calloc(n_threads, sizeof(thread_sampe_pe_aux_t));
		tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
		for (j = 0; j < n_threads; ++j) {
			data[j].tid = j;
			data[j].n_seqs = n_seqs;
			data[j].seqs[0] = seqs[0];
			data[j].seqs[1] = seqs[1];
			data[j].gopt = gopt;
			data[j].bwt[0] = bwt[0];
			data[j].bwt[1] = bwt[1];
			data[j].buf[0] = buf[0];
			data[j].buf[1] = buf[1];
			data[j].n_threads = n_threads;
			pthread_create(&tid[j], &attr, sampe_se_worker, data + j);
		}
		for (j = 0; j < n_threads; ++j) pthread_join(tid[j], 0);
		free(data); free(tid);
	}
	#else
	posix_se(n_seqs, seqs, bwt, buf, gopt, 0, 0);
	#endif
		//posix_se(n_seqs, seqs, bwt, buf, gopt, 0, 0);

	//	fprintf(stderr, "SE %.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC); t = clock();

	// infer isize
	infer_isize(n_seqs, seqs, ii, opt->ap_prior, bwt[0]->seq_len);
	if (ii->avg < 0.0 && last_ii->avg > 0.0) *ii = *last_ii;
	if (opt->force_isize) {
		fprintf(stderr, " [%s] discard insert size estimate as user's request.\n", __func__);
		ii->low = ii->high = 0; ii->avg = ii->std = -1.0;
	}
//	fprintf(stderr, "infer isize %.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC); t = clock();

	// PE
	#ifdef HAVE_PTHREAD
	if (n_threads <= 1) { // no multi-threading at all
		posix_pe(n_seqs, seqs, buf,/* d,*/ opt, bwt, ii, gopt, 0, 0);
	} else {
		pthread_t *tid;
		pthread_attr_t attr;
		thread_sampe_pe_aux_t *data;
		int j;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		data = (thread_sampe_pe_aux_t*)calloc(n_threads, sizeof(thread_sampe_pe_aux_t));
		tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
		for (j = 0; j < n_threads; ++j) {
			data[j].tid = j;
			data[j].n_seqs = n_seqs;
			data[j].seqs[0] = seqs[0];
			data[j].seqs[1] = seqs[1];
			data[j].opt = opt;
			data[j].gopt = gopt;
			data[j].bwt[0] = bwt[0];
			data[j].bwt[1] = bwt[1];
			data[j].buf[0] = buf[0];
			data[j].buf[1] = buf[1];
			data[j].ii = ii;
			data[j].n_threads = n_threads;
			pthread_create(&tid[j], &attr, sampe_pe_worker, data + j);
		}
		for (j = 0; j < n_threads; ++j) pthread_join(tid[j], 0);
		free(data); free(tid);
	}
	#else
	posix_pe(n_seqs, seqs, buf, opt, bwt, ii, gopt, 0, 0);
	#endif

	// free
	for (i = 0; i < n_seqs; ++i) {
		kv_destroy(buf[0][i].aln);
		kv_destroy(buf[1][i].aln);
	}
	free(buf[0]); free(buf[1]);
	kv_destroy(d->arr);
	kv_destroy(d->pos[0]); kv_destroy(d->pos[1]);
	kv_destroy(d->aln[0]); kv_destroy(d->aln[1]);
	free(d);


	return 0;
}

#define SW_MIN_MATCH_LEN 20
#define SW_MIN_MAPQ 17

// cnt = n_mm<<16 | n_gapo<<8 | n_gape
uint16_t *bwa_sw_core(bwtint_t l_pac, const ubyte_t *pacseq, int len, const ubyte_t *seq, int64_t *beg, int reglen,
					  int *n_cigar, uint32_t *_cnt)
{
	uint16_t *cigar = 0;
	ubyte_t *ref_seq;
	bwtint_t k, x, y, l;
	int path_len;
	AlnParam ap = aln_param_bwa;
	path_t *path, *p;

	// check whether there are too many N's
	if (reglen < SW_MIN_MATCH_LEN || (int64_t)l_pac - *beg < len) return 0;
	for (k = 0, x = 0; k < len; ++k)
		if (seq[k] >= 4) ++x;
	if ((float)x/len >= 0.25 || len - x < SW_MIN_MATCH_LEN) return 0;

	// get reference subsequence
	ref_seq = (ubyte_t*)calloc(reglen, 1);
	for (k = *beg, l = 0; k < *beg + reglen && k < l_pac; ++k)
		ref_seq[l++] = pacseq[k>>2] >> ((~k&3)<<1) & 3;
	path = (path_t*)calloc(l+len, sizeof(path_t));

	// do alignment
	aln_local_core(ref_seq, l, (ubyte_t*)seq, len, &ap, path, &path_len, 1, 0);
	cigar = aln_path2cigar(path, path_len, n_cigar);

	// check whether the alignment is good enough
	for (k = 0, x = y = 0; k < *n_cigar; ++k) {
		uint16_t c = cigar[k];
		if (c>>14 == FROM_M) x += c&0x3fff, y += c&0x3fff;
		else if (c>>14 == FROM_D) x += c&0x3fff;
		else y += c&0x3fff;
	}
	if (x < SW_MIN_MATCH_LEN || y < SW_MIN_MATCH_LEN) { // not good enough
		free(path); free(cigar);
		*n_cigar = 0;
		return 0;
	}

	{ // update cigar and coordinate;
		int start, end;
		p = path + path_len - 1;
		*beg += (p->i? p->i : 1) - 1;
		start = (p->j? p->j : 1) - 1;
		end = path->j;
		cigar = (uint16_t*)realloc(cigar, 2 * (*n_cigar + 2));
		if (start) {
			memmove(cigar + 1, cigar, 2 * (*n_cigar));
			cigar[0] = 3<<14 | start;
			++(*n_cigar);
		}
		if (end < len) {
			cigar[*n_cigar] = 3<<14 | (len - end);
			++(*n_cigar);
		}
	}

	{ // set *cnt
		int n_mm, n_gapo, n_gape;
		n_mm = n_gapo = n_gape = 0;
		p = path + path_len - 1;
		x = p->i? p->i - 1 : 0; y = p->j? p->j - 1 : 0;
		for (k = 0; k < *n_cigar; ++k) {
			uint16_t c = cigar[k];
			if (c>>14 == FROM_M) {
				for (l = 0; l < (c&0x3fff); ++l)
					if (ref_seq[x+l] < 4 && seq[y+l] < 4 && ref_seq[x+l] != seq[y+l]) ++n_mm;
				x += c&0x3fff, y += c&0x3fff;
			} else if (c>>14 == FROM_D) {
				x += c&0x3fff, ++n_gapo, n_gape += (c&0x3fff) - 1;
			} else if (c>>14 == FROM_I) {
				y += c&0x3fff, ++n_gapo, n_gape += (c&0x3fff) - 1;
			}
		}
		*_cnt = (uint32_t)n_mm<<16 | n_gapo<<8 | n_gape;
	}

	free(ref_seq); free(path);
	return cigar;
}

void bwa_paired_sw(const bntseq_t *bns, int n_seqs, bwa_seq_t *seqs[2], const pe_opt_t *popt, const isize_info_t *ii, ubyte_t *pacseq, int tid, int n_threads)
{
	//ubyte_t *pacseq;
	int i;
	uint64_t x, n;



	if (!popt->is_sw || ii->avg < 0.0) return;

	// perform mate alignment
	for (i = 0, x = n = 0; i != n_seqs; ++i) {
		bwa_seq_t *p[2];

		int is_first = 1;
		p[0] = seqs[0] + i; p[1] = seqs[1] + i;

	#ifdef HAVE_PTHREAD
		if (n_threads > 1) {
			pthread_mutex_lock(&g_seq_lock);
			if (p[0]->tid < 0) { // unassigned
				int j;
				for (j = i; j < n_seqs && j < i + THREAD_BLOCK_SIZE; ++j){
					seqs[0][j].tid = tid;
					seqs[1][j].tid = tid;
				}
			} else if (p[0]->tid != tid) {
				pthread_mutex_unlock(&g_seq_lock);
				continue;
			}
			pthread_mutex_unlock(&g_seq_lock);

		}
	#endif

		if ((p[0]->mapQ >= 20 && p[1]->type == BWA_TYPE_NO_MATCH) || (p[1]->mapQ >= 20 && p[0]->type == BWA_TYPE_NO_MATCH)) {
			++n;
			if (p[0]->type == BWA_TYPE_NO_MATCH) {
				p[0] = seqs[1] + i; p[1] = seqs[0] + i; // swap s.t p[0] is the mapped read
				is_first = 0;
			}
			if (popt->type == BWA_PET_STD || popt->type == BWA_PET_SOLID) {
				int64_t beg, end; // this is the start and end of the region
				ubyte_t *seq;
				uint32_t cnt;

#define __set_rght_coor(_a, _b) do {									\
					(_a) = p[0]->pos + ii->avg - 3 * ii->std - p[1]->len * 1.5; \
					(_b) = (_a) + 6 * ii->std + 2 * p[1]->len;			\
					if ((_a) < p[0]->pos + p[0]->len) (_a) = p[0]->pos + p[0]->len; \
					if ((_b) > bns->l_pac) (_b) = bns->l_pac;			\
				} while (0)

#define __set_left_coor(_a, _b) do {									\
					(_a) = p[0]->pos + p[0]->len - ii->avg - 3 * ii->std - p[1]->len * 0.5; \
					(_b) = (_a) + 6 * ii->std + 2 * p[1]->len;			\
					if ((_a) < 0) (_a) = 0;								\
					if ((_b) > p[0]->pos) (_b) = p[0]->pos;				\
				} while (0)

				if (popt->type == BWA_PET_STD) {
					if (p[0]->strand == 0) { // the mate is on the reverse strand and has larger coordinate
						__set_rght_coor(beg, end);
						seq = p[1]->rseq;
					} else { // the mate is on forward stand and has smaller coordinate
						__set_left_coor(beg, end);
						seq = p[1]->seq;
						seq_reverse(p[1]->len, seq, 0); // because ->seq is reversed
					}
				} else { // popt->type == BWA_PET_SOLID
					if (p[0]->strand == 0) {
						if (!is_first) __set_left_coor(beg, end);
						else __set_rght_coor(beg, end);
						seq = p[1]->rseq;
						seq_reverse(p[1]->len, seq, 0); // because ->seq is reversed
					} else {
						if (!is_first) __set_rght_coor(beg, end);
						else __set_left_coor(beg, end);
						seq = p[1]->seq;
					}
				}

				p[1]->cigar = bwa_sw_core(bns->l_pac, pacseq, p[1]->len, seq, &beg, end - beg, &p[1]->n_cigar, &cnt);
				if (p[1]->cigar) { // the SW alignment is good enough
					++x;
					p[1]->type = BWA_TYPE_MATESW;
					p[1]->pos = beg;
					p[1]->mapQ = p[0]->mapQ;
					p[1]->seQ = p[0]->seQ;
					p[1]->strand = (popt->type == BWA_PET_STD)? 1 - p[0]->strand : p[0]->strand;
					p[1]->n_mm = cnt>>16; p[1]->n_gapo = cnt>>8&0xff; p[1]->n_gape = cnt&0xff;
					p[1]->extra_flag |= SAM_FPP;
					p[0]->extra_flag |= SAM_FPP;
				}
				if (popt->type == BWA_PET_STD) {
					if (p[0]->strand) seq_reverse(p[1]->len, seq, 0); // reverse it back
				} else {
					if (p[0]->strand == 0) seq_reverse(p[1]->len, seq, 0);
				}
			} else {
				fprintf(stderr, "  [bwa_paired_sw] not implemented!\n");
				exit(1);
			}
		}
	}

	fprintf(stderr, "  [bwa_paired_sw] Thread %d: %lld reads aligned out of %lld candidates.\n",
			tid,(long long)x, (long long)n);

	return;
}

//POSIX Multithreading for paired_sw

#ifdef HAVE_PTHREAD

typedef struct {
	int tid;
	int n_seqs;
	bwa_seq_t *seqs[2];
	bntseq_t *bns;
	//bwt_t *bwt[2];
	int n_threads;
	const pe_opt_t *popt;
	ubyte_t *pacseq;
	isize_info_t *ii;
} thread_sampe_paired_sw_aux_t;

static void *sampe_paired_sw_worker(void *data)
{
	thread_sampe_paired_sw_aux_t *d = (thread_sampe_paired_sw_aux_t*)data;
	//bwa_refine_gapped(d->tid, d->n_threads, d->bns, d->n_seqs, d->seqs, 0, d->ntbns);
	bwa_paired_sw(d->bns, d->n_seqs, d->seqs, d->popt, d->ii, d->pacseq, d->tid, d->n_threads);
	return 0;
}
#endif // HAVE_PTHREAD




void bwa_sai2sam_pe_core(const char *prefix, char *const fn_sa[2], char *const fn_fa[2], pe_opt_t *popt)
{
	int i, j, n_seqs, tot_seqs = 0;
	bwa_seq_t *seqs[2];
	bwa_seqio_t *ks[2];
	bntseq_t *bns, *ntbns = 0;
	FILE *fp_sa[2];
	barracuda_gap_opt_t opt;
	khint_t iter;
	isize_info_t last_ii; // this is for the last batch of reads

	// For timing purpose only
	struct timeval start, end;
	double time_used = 0, total_time_used = 0;


	#define BATCH_SIZE 0x80000

	// initialization

	for (i = 1; i != 256; ++i) g_log_n[i] = (int)(4.343 * log(i) + 0.5);
	bns = bns_restore(prefix);
	srand48(bns->seed);
	for (i = 0; i < 2; ++i) {
		ks[i] = bwa_seq_open(fn_fa[i]);
		fp_sa[i] = xopen(fn_sa[i], "r");
	}
	g_hash = kh_init(64);
	last_ii.avg = -1.0;

	// load forward SA - original from bwa_cal_pac_pos_pe, moved here to save disk I/O for big bwts
	bwt_t *bwt[2];
	char str[1024];

	// open bwts added by brian
	fprintf(stderr,"[sampe_core] Loading BWTs, please wait..");
	// load forward & reverse SA
	gettimeofday (&start, NULL);

	strcpy(str, prefix); strcat(str, ".bwt");  bwt[0] = bwt_restore_bwt(str);
	strcpy(str, prefix); strcat(str, ".sa"); bwt_restore_sa(str, bwt[0]);
	strcpy(str, prefix); strcat(str, ".rbwt"); bwt[1] = bwt_restore_bwt(str);
	strcpy(str, prefix); strcat(str, ".rsa"); bwt_restore_sa(str, bwt[1]);

	gettimeofday (&end, NULL);
	time_used = diff_in_seconds(&end,&start);
	total_time_used += time_used;

	fprintf(stderr, "Done! \n[sampe_core] Time used: %0.2fs\n", time_used);

	//print SAM header
	bwa_print_sam_SQ(bns);
	bwa_print_sam_PG();

	int n_threads = 0;

	//POSIX multithreading
	#ifdef HAVE_PTHREAD

		int max_threads = sysconf( _SC_NPROCESSORS_ONLN );

	if(popt->thread > 0)
	{
		n_threads = (popt->thread > max_threads)? max_threads : popt->thread ;
	}else if (max_threads >= 2)
	{
		n_threads = (max_threads <= 6)? max_threads : 6 ;
	}else
	{
		n_threads = 1;
	}
	if(n_threads > 8) n_threads = 8;
	fprintf(stderr, "[sampe_core] Running with %d threads\n", n_threads);
	#endif


	// core loop
	fread(&opt, sizeof(barracuda_gap_opt_t), 1, fp_sa[0]);
	fread(&opt, sizeof(barracuda_gap_opt_t), 1, fp_sa[1]);
	if (!(opt.mode & BWA_MODE_COMPREAD)) {
		popt->type = BWA_PET_SOLID;
		ntbns = bwa_open_nt(prefix);
	}


	while ((seqs[0] = bwa_read_seq(ks[0], BATCH_SIZE, &n_seqs, opt.mode & BWA_MODE_COMPREAD, opt.mid)) != 0) {
		//int cnt_chg;
		isize_info_t ii;
		ubyte_t *pacseq = 0;

		seqs[1] = bwa_read_seq(ks[1], BATCH_SIZE, &n_seqs, opt.mode & BWA_MODE_COMPREAD, opt.mid);
		tot_seqs += n_seqs;

		gettimeofday (&start, NULL);

		fprintf(stderr, "[sampe_core] Mapping SA coordinates to linear space, please wait... \n");
		//cnt_chg = bwa_cal_pac_pos_pe(prefix, n_seqs, seqs, fp_sa, &ii, popt, &opt, &last_ii, bwt);
		bwa_cal_pac_pos_pe(prefix, n_seqs, seqs, fp_sa, &ii, popt, &opt, &last_ii, bwt, n_threads);

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;

		fprintf(stderr, "[sampe_core] Time used: %.2f sec\n", time_used);

		//fprintf(stderr, "[sampe_core] Change of coordinates in %d alignments.\n", cnt_chg);
		fprintf(stderr, "[sampe_core] Aligning unmapped mate pairs...\n");

		// load reference sequence
		pacseq = (ubyte_t*)calloc(bns->l_pac/4+1, 1);
		rewind(bns->fp_pac);
		fread(pacseq, 1, bns->l_pac/4+1, bns->fp_pac);

		gettimeofday (&start, NULL);

		#ifdef HAVE_PTHREAD
		if (n_threads <= 1) { // no multi-threading at all
			bwa_paired_sw(bns, n_seqs, seqs, popt, &ii, pacseq, 0, 0);
		} else {
			pthread_t *tid;
			pthread_attr_t attr;
			thread_sampe_paired_sw_aux_t *data;
			int j;
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
			data = (thread_sampe_paired_sw_aux_t*)calloc(n_threads, sizeof(thread_sampe_paired_sw_aux_t));
			tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
			for (j = 0; j < n_threads; ++j) {
				data[j].tid = j;
				data[j].n_seqs = n_seqs;
				data[j].seqs[0] = seqs[0];
				data[j].seqs[1] = seqs[1];
				data[j].popt = popt;
				data[j].bns = bns;
				data[j].pacseq = pacseq;
				data[j].ii = &ii;
				data[j].n_threads = n_threads;
				pthread_create(&tid[j], &attr, sampe_paired_sw_worker, data + j);
			}
			for (j = 0; j < n_threads; ++j) pthread_join(tid[j], 0);
			free(data); free(tid);
		}
		#else
		bwa_paired_sw(bns, n_seqs, seqs, popt, &ii, pacseq, 0, 0);
		#endif


		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;

		fprintf(stderr, "[sampe_core] Time Used: %.2f sec\n", time_used);


		gettimeofday (&start, NULL);

		fprintf(stderr, "[sampe_core] Refining gapped alignments...\n");
		for (j = 0; j < 2; ++j)
			bwa_refine_gapped(0,0, bns, n_seqs, seqs[j], pacseq, ntbns);

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;

		fprintf(stderr, "[sampe_core] Time used: %.2f sec\n", time_used);

		free(pacseq);


		gettimeofday (&start, NULL);

		fprintf(stderr, "[sampe_core] Printing alignments... \n");
		for (i = 0; i < n_seqs; ++i) {
			bwa_print_sam1(bns, seqs[0] + i, seqs[1] + i, opt.mode, opt.max_top2);
			bwa_print_sam1(bns, seqs[1] + i, seqs[0] + i, opt.mode, opt.max_top2);
		}


		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;

		fprintf(stderr, "[sampe_core] Time used: %.2f sec\n", time_used);

		for (j = 0; j < 2; ++j)
			bwa_free_read_seq(n_seqs, seqs[j]);
		fprintf(stderr, "[sampe_core] %d sequences have been processed.\n", tot_seqs);
		last_ii = ii;


	}

	fprintf(stderr, "\n[sampe_core] Done!\n[sampe_core] Total no. of sequences: %d\n[sampe_core] Total program time: %0.2fs (%0.2f sequences/sec)\n", tot_seqs, total_time_used, (float) tot_seqs/total_time_used);


	// destroy
	bwt_destroy(bwt[0]); bwt_destroy(bwt[1]); // move to here to save disk I/O
	bns_destroy(bns);
	if (ntbns) bns_destroy(ntbns);
	for (i = 0; i < 2; ++i) {
		bwa_seq_close(ks[i]);
		fclose(fp_sa[i]);
	}
	for (iter = kh_begin(g_hash); iter != kh_end(g_hash); ++iter)
		if (kh_exist(g_hash, iter)) free(kh_val(g_hash, iter).a);
	kh_destroy(64, g_hash);


}

int bwa_sai2sam_pe(int argc, char *argv[])
{

	fprintf(stderr, "Barracuda, Version %s\n", PACKAGE_VERSION);

	extern char *bwa_rg_line, *bwa_rg_id;
	extern int bwa_set_rg(const char *s);
	int c;
	pe_opt_t *popt;
	popt = bwa_init_pe_opt();


	while ((c = getopt(argc, argv, "a:o:sPn:N:t:c:f:Ar:")) >= 0) {
		switch (c) {
		case 'r':
			if (bwa_set_rg(optarg) < 0) {
				fprintf(stderr, "[%s] ERROR!! malformated @RG line! Aborting...\n", __func__);
				return 1;
			}
			break;
		case 'a': popt->max_isize = atoi(optarg); break;
		case 'o': popt->max_occ = atoi(optarg); break;
		case 's': popt->is_sw = 0; break;
//		case 'P': popt->is_preload = 1; break;
//		case 'n': popt->n_multi = atoi(optarg); break;
//		case 'N': popt->N_multi = atoi(optarg); break;
		case 'c': popt->ap_prior = atof(optarg); break;
//		case 'f': xreopen(optarg, "w", stdout); break;
		case 't': popt->thread = atoi(optarg); break;
		case 'A': popt->force_isize = 1; break;
		default: return 1;
		}
	}

	if (optind + 5 > argc) {
		fprintf(stderr, "\nSAM paired-end alignment output module\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Usage:   barracuda sampe [options] <reference.fa> <in1.sai> <in2.sai> <in1.fastq> <in2.fastq>\n\n");
		fprintf(stderr, "Options: -a INT   maximum insert size [%d]\n", popt->max_isize);
		fprintf(stderr, "         -o INT   maximum occurrences for one end [%d]\n", popt->max_occ);
//		fprintf(stderr, "         -n INT   maximum hits to output for paired reads [%d]\n", popt->n_multi);
//		fprintf(stderr, "         -N INT   maximum hits to output for discordant pairs [%d]\n", popt->N_multi);
		fprintf(stderr, "         -c FLOAT prior of chimeric rate (lower bound) [%.1le]\n", popt->ap_prior);
//      fprintf(stderr, "         -f FILE  sam file to output results to [stdout]\n");
		fprintf(stderr, "         -r STR   read group header line such as `@RG\\tID:foo\\tSM:bar' [null]\n");
//		fprintf(stderr, "         -P       preload index into memory (for base-space reads only)\n");
		fprintf(stderr, "         -t       Specify the number of CPU threads to use, default: [AUTO-DETECT]\n");
		fprintf(stderr, "         -s       disable Smith-Waterman for the unmapped mate\n");
		fprintf(stderr, "         -A       disable insert size estimate (force -s)\n\n");
		fprintf(stderr, "Notes: 1. For SOLiD reads, <in1.fq> corresponds R3 reads and <in2.fq> to F3.\n");
		fprintf(stderr, "       2. For reads shorter than 30bp, applying a smaller -o is recommended to\n");
		fprintf(stderr, "          to get a sensible speed at the cost of pairing accuracy.\n");
		fprintf(stderr, "\n");
		return 1;
	}
	bwa_sai2sam_pe_core(argv[optind], argv + optind + 1, argv + optind+3, popt);
	free(bwa_rg_line); free(bwa_rg_id);
	free(popt);
	return 0;
}

