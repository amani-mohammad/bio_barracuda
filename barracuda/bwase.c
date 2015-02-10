/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwase.c  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2012, University of Cambridge Metabolic Research Labs.
   Contributers: Dag Lyberg, Simon Lam and Brian Lam

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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "stdaln.h"
#include "bwtaln.h"
#include "bntseq.h"
#include "utils.h"
#include "kstring.h"
#include "barracuda.h"

#ifdef HAVE_PTHREAD
#define THREAD_BLOCK_SIZE 1024
#include <pthread.h>
static pthread_mutex_t g_seq_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

// the read batch size, currently set at 524288
#define BATCH_SIZE 0x80000

char *bwa_rg_line, *bwa_rg_id;

static int g_log_n[256];

void bwa_print_sam_PG();

void swap(barracuda_aln1_t *x, barracuda_aln1_t *y)
{
   barracuda_aln1_t temp;
   temp = *x;
   *x = *y;
   *y = temp;
}

int choose_pivot(int i,int j)
{
   return((i+j) /2);
}

void aln_quicksort(barracuda_aln1_t *aln, int m, int n)
//This function sorts the alignment array from barracuda to make it compatible with SAMSE/SAMPE cores
{
	int key,i,j,k;

	if (m < n)
	{
	      k = choose_pivot(m, n);
	      swap(&aln[m],&aln[k]);
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
	                swap(&aln[i],&aln[j]);
	      }
	      // swap two elements
	      swap(&aln[m],&aln[j]);
	      // recursively sort the lesser lists
	      aln_quicksort(aln, m, j-1);
	      aln_quicksort(aln, j+1, n);
	 }
}

void bwa_aln2seq(int n_aln, barracuda_aln1_t *aln, bwa_seq_t *s)
//This function is modified so that it would be compatible with barracuda raw output.
//It is compatible with the original bwa output.
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
		aln_quicksort(aln, 0, n_aln-1);
	}

	best = aln[0].score;

	for (i = cnt = 0; i < n_aln; ++i) {
		const barracuda_aln1_t *p = aln + i;
		if (p->score > best) break;
		if (drand48() * (p->l - p->k + 1) > (double)cnt) {
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
//Original
/*
void bwa_aln2seq(int n_aln, const barracuda_aln1_t *aln, bwa_seq_t *s)
{
	int i, cnt, best;
	if (n_aln == 0) {
		s->type = BWA_TYPE_NO_MATCH;
		s->c1 = s->c2 = 0;
		return;
	}
	best = aln[0].score;
	for (i = cnt = 0; i < n_aln; ++i) {
		const barracuda_aln1_t *p = aln + i;
		if (p->score > best) break;
		if (drand48() * (p->l - p->k + 1) > (double)cnt) {
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
*/
int bwa_approx_mapQ(const bwa_seq_t *p, int mm)
{
	int n;
	if (p->c1 == 0) return 23;
	if (p->c1 > 1) return 0;
	if (p->n_mm == mm) return 25;
	if (p->c2 == 0) return 37;
	n = (p->c2 >= 255)? 255 : p->c2;
	return (23 < g_log_n[n])? 0 : 23 - g_log_n[n];
}

void bwa_cal_pac_pos(const char *prefix, int n_seqs, bwa_seq_t *seqs, int max_mm, float fnr, bwt_t * bwt, bwt_t * rbwt)
//this is the most time consuming step
//A little change is done here to put bwts permanant in memory in caller function so to reduce disk I/O
//The only drawback is that more memory is needed for this function (4GB memory needed for full human genome)
{
	int i;
	//char str[1024];
	//bwt_t * bwt;

	//int nfwd = 0, nrev = 0;

	// load forward SA
	//strcpy(str, prefix); strcat(str, ".bwt");  bwt = bwt_restore_bwt(str);
	//strcpy(str, prefix); strcat(str, ".sa"); bwt_restore_sa(str, bwt);
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p = seqs + i;
		int max_diff = fnr > 0.0? bwa_cal_maxdiff(p->len, BWA_AVG_ERR, fnr) : max_mm;
		if ((p->type == BWA_TYPE_UNIQUE || p->type == BWA_TYPE_REPEAT) && p->strand) { // reverse strand only
			p->pos = bwt_sa(bwt, p->sa);
			p->seQ = p->mapQ = bwa_approx_mapQ(p, max_diff);
			//nfwd ++;
		}
	}
	//bwt_destroy(bwt);
	// load reverse BWT and SA
	//strcpy(str, prefix); strcat(str, ".rbwt"); bwt = bwt_restore_bwt(str);
	//strcpy(str, prefix); strcat(str, ".rsa"); bwt_restore_sa(str, bwt);
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p = seqs + i;
		int max_diff = fnr > 0.0? bwa_cal_maxdiff(p->len, BWA_AVG_ERR, fnr) : max_mm;
		if ((p->type == BWA_TYPE_UNIQUE || p->type == BWA_TYPE_REPEAT) && !p->strand) { // forward strand only
			/* NB: For gapped alignment, p->pos may not be correct,
			 *     which will be fixed in refine_gapped_core(). This
			 *     line also determines the way "x" is calculated in
			 *     refine_gapped_core() when (ext < 0 && is_end == 0). */
			p->pos = bwt->seq_len - (bwt_sa(rbwt, p->sa) + p->len);
			p->seQ = p->mapQ = bwa_approx_mapQ(p, max_diff);
			//nrev ++;
		}
	}
	//fprintf(stderr,"fwdcount: %d, revcount %d\n", nfwd, nrev);
	//bwt_destroy(bwt);
}

/* is_end_correct == 1 if (*pos+len) gives the correct coordinate on
 * forward strand. This happens when p->pos is calculated by
 * bwa_cal_pac_pos(). is_end_correct==0 if (*pos) gives the correct
 * coordinate. This happens only for color-converted alignment. */
static uint16_t *refine_gapped_core(bwtint_t l_pac, const ubyte_t *pacseq, int len, const ubyte_t *seq, bwtint_t *_pos,
									int ext, int *n_cigar, int is_end_correct)
{
	uint16_t *cigar = 0;
	ubyte_t *ref_seq;
	int l = 0, path_len, ref_len;
	AlnParam ap = aln_param_bwa;
	path_t *path;
	int64_t k, __pos = *_pos > l_pac? (int64_t)((int32_t)*_pos) : *_pos;

	ref_len = len + abs(ext);
	if (ext > 0) {
		ref_seq = (ubyte_t*)calloc(ref_len, 1);
		for (k = __pos; k < __pos + ref_len && k < l_pac; ++k)
			ref_seq[l++] = pacseq[k>>2] >> ((~k&3)<<1) & 3;
	} else {
		int64_t x = __pos + (is_end_correct? len : ref_len);
		ref_seq = (ubyte_t*)calloc(ref_len, 1);
		for (l = 0, k = x - ref_len > 0? x - ref_len : 0; k < x && k < l_pac; ++k)
			ref_seq[l++] = pacseq[k>>2] >> ((~k&3)<<1) & 3;
	}
	path = (path_t*)calloc(l+len, sizeof(path_t));

	aln_global_core(ref_seq, l, (ubyte_t*)seq, len, &ap, path, &path_len);
	cigar = aln_path2cigar(path, path_len, n_cigar);

	if (cigar[*n_cigar-1]>>14 == FROM_D) --(*n_cigar); // deletion at the 3'-end
	if (cigar[0]>>14 == FROM_D) { // deletion at the 5'-end
		for (k = 0; k < *n_cigar - 1; ++k) cigar[k] = cigar[k+1];
		--(*n_cigar);
	}

	if (ext < 0 && is_end_correct) { // fix coordinate for reads mapped on the forward strand
		for (l = k = 0; k < *n_cigar; ++k) {
			if (cigar[k]>>14 == FROM_D) l -= cigar[k]&0x3fff;
			else if (cigar[k]>>14 == FROM_I) l += cigar[k]&0x3fff;
		}
		__pos += l;
	}

	// change "I" at either end of the read to S. just in case. This should rarely happen...
	if (cigar[*n_cigar-1]>>14 == FROM_I) cigar[*n_cigar-1] = 3<<14 | (cigar[*n_cigar-1]&0x3fff);
	if (cigar[0]>>14 == FROM_I) {
		cigar[0] = 3<<14 | (cigar[0]&0x3fff);
		__pos += cigar[0]&0x3fff;
	}

	*_pos = (bwtint_t)__pos;
	free(ref_seq); free(path);
	return cigar;
}

char *bwa_cal_md1(int n_cigar, uint16_t *cigar, int len, bwtint_t pos, ubyte_t *seq,
				  bwtint_t l_pac, ubyte_t *pacseq, kstring_t *str)
{
	bwtint_t x, y;
	int z, u, c;
	str->l = 0; // reset
	x = pos; y = 0;
	if (cigar) {
		int k, l;
		for (k = u = 0; k < n_cigar; ++k) {
			l = cigar[k]&0x3fff;
			if (cigar[k]>>14 == FROM_M) {
				for (z = 0; z < l && x+z < l_pac; ++z) {
					c = pacseq[(x+z)>>2] >> ((~(x+z)&3)<<1) & 3;
					if (c > 3 || seq[y+z] > 3 || c != seq[y+z]) {
						ksprintf(str, "%d", u);
						kputc("ACGTN"[c], str);
						u = 0;
					} else ++u;
				}
				x += l; y += l;
			} else if (cigar[k]>>14 == FROM_I || cigar[k]>>14 == 3) {
				y += l;
			} else if (cigar[k]>>14 == FROM_D) {
				ksprintf(str, "%d", u);
				kputc('^', str);
				for (z = 0; z < l && x+z < l_pac; ++z)
					kputc("ACGT"[pacseq[(x+z)>>2] >> ((~(x+z)&3)<<1) & 3], str);
				u = 0;
				x += l;
			}
		}
	} else { // no gaps
		for (z = u = 0; z < (bwtint_t)len; ++z) {
			c = pacseq[(x+z)>>2] >> ((~(x+z)&3)<<1) & 3;
			if (c > 3 || seq[y+z] > 3 || c != seq[y+z]) {
				ksprintf(str, "%d", u);
				kputc("ACGTN"[c], str);
				u = 0;
			} else ++u;
		}
	}
	ksprintf(str, "%d", u);
	return strdup(str->s);
}

void bwa_refine_gapped(int tid, int n_threads, const bntseq_t *bns, int n_seqs, bwa_seq_t *seqs, ubyte_t *_pacseq, bntseq_t *ntbns)
{
	ubyte_t *pacseq, *ntpac = 0;
	int i;
	kstring_t *str;

	if (ntbns) { // in color space
		ntpac = (ubyte_t*)calloc(ntbns->l_pac/4+1, 1);
		rewind(ntbns->fp_pac);
		fread(ntpac, 1, ntbns->l_pac/4 + 1, ntbns->fp_pac);
	}

	if (!_pacseq) {
		pacseq = (ubyte_t*)calloc(bns->l_pac/4+1, 1);
		rewind(bns->fp_pac);
		fread(pacseq, 1, bns->l_pac/4+1, bns->fp_pac);
	} else pacseq = _pacseq;

	//fprintf(stderr, "Running with %d threads, printed from thread ID: %d, no of sequences: %d\n", n_threads, tid, n_seqs);

	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *s = seqs + i;

		#ifdef HAVE_PTHREAD
		if (n_threads > 1) {
			pthread_mutex_lock(&g_seq_lock);
			if (s->tid < 0) { // unassigned
				int j;
				for (j = i; j < n_seqs && j < i + THREAD_BLOCK_SIZE; ++j)
					seqs[j].tid = tid;
			} else if (s->tid != tid) {
				pthread_mutex_unlock(&g_seq_lock);
				continue;
			}
			pthread_mutex_unlock(&g_seq_lock);

		}
		#endif
		seq_reverse(s->len, s->seq, 0); // IMPORTANT: s->seq is reversed here!!!
		if (s->type == BWA_TYPE_NO_MATCH || s->type == BWA_TYPE_MATESW || s->n_gapo == 0) continue;
		s->cigar = refine_gapped_core(bns->l_pac, pacseq, s->len, s->strand? s->rseq : s->seq, &s->pos,
									  (s->strand? 1 : -1) * (s->n_gapo + s->n_gape), &s->n_cigar, 1);
	}

	if (ntbns) { // in color space
		for (i = 0; i < n_seqs; ++i) {
			bwa_seq_t *s = seqs + i;
			bwa_cs2nt_core(s, bns->l_pac, ntpac);
			if (s->type != BWA_TYPE_NO_MATCH && s->cigar) { // update cigar again
				free(s->cigar);
				s->cigar = refine_gapped_core(bns->l_pac, ntpac, s->len, s->strand? s->rseq : s->seq, &s->pos,
											  (s->strand? 1 : -1) * (s->n_gapo + s->n_gape), &s->n_cigar, 0);
			}
		}
	}

	// generate MD tag
	str = (kstring_t*)calloc(1, sizeof(kstring_t));
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *s = seqs + i;
		if (s->type != BWA_TYPE_NO_MATCH)
			s->md = bwa_cal_md1(s->n_cigar, s->cigar, s->len, s->pos, s->strand? s->rseq : s->seq,
								bns->l_pac, ntbns? ntpac : pacseq, str);
	}
	free(str->s); free(str);

	if (!_pacseq) free(pacseq);
	free(ntpac);
}

static int64_t pos_end(const bwa_seq_t *p)
{
	if (p->cigar) {
		int j;
		int64_t x = p->pos;
		for (j = 0; j != p->n_cigar; ++j) {
			int op = p->cigar[j]>>14;
			if (op == 0 || op == 2) x += p->cigar[j]&0x3fff;
		}
		return x;
	} else return p->pos + p->len;
}

static int64_t pos_5(const bwa_seq_t *p)
{
	if (p->type != BWA_TYPE_NO_MATCH)
		return p->strand? pos_end(p) : p->pos;
	return -1;
}

void bwa_print_sam1(const bntseq_t *bns, bwa_seq_t *p, const bwa_seq_t *mate, int mode, int max_top2)
{
	int j;
	if (p->type != BWA_TYPE_NO_MATCH || (mate && mate->type != BWA_TYPE_NO_MATCH)) {
		int seqid, nn, am = 0, flag = p->extra_flag;
		char XT;
		ubyte_t *s;

		if (p->type == BWA_TYPE_NO_MATCH) {
			p->pos = mate->pos;
			p->strand = mate->strand;
			flag |= SAM_FSU;
			j = 1;
		} else j = pos_end(p) - p->pos; // j is the length of the reference in the alignment

		s = p->strand? p->rseq : p->seq;

		// get seqid
		nn = bns_coor_pac2real(bns, p->pos, j, &seqid);

		// update flag and print it
		if (p->strand) flag |= SAM_FSR;
		if (mate) {
			if (mate->type != BWA_TYPE_NO_MATCH) {
				if (mate->strand) flag |= SAM_FMR;
			} else flag |= SAM_FMU;
		}
		printf("%s\t%d\t%s\t", p->name, flag, bns->anns[seqid].name);
		printf("%d\t%d\t", (int)(p->pos - bns->anns[seqid].offset + 1), p->mapQ);

		// print CIGAR
		if (p->cigar) {
			for (j = 0; j != p->n_cigar; ++j)
				printf("%d%c", p->cigar[j]&0x3fff, "MIDS"[p->cigar[j]>>14]);
		} else printf("%dM", p->len);

		// print mate coordinate
		if (mate && mate->type != BWA_TYPE_NO_MATCH) {
			int m_seqid, m_is_N;
			long long isize;
			am = mate->seQ < p->seQ? mate->seQ : p->seQ; // smaller single-end mapping quality
			// redundant calculation here, but should not matter too much
			m_is_N = bns_coor_pac2real(bns, mate->pos, mate->len, &m_seqid);
			printf("\t%s\t", (seqid == m_seqid)? "=" : bns->anns[m_seqid].name);
			isize = (seqid == m_seqid)? pos_5(mate) - pos_5(p) : 0;
			if (p->type == BWA_TYPE_NO_MATCH) isize = 0;
			printf("%d\t%lld\t", (int)(mate->pos - bns->anns[m_seqid].offset + 1), isize);
		} else printf("\t=\t%d\t0\t", (int)(p->pos - bns->anns[seqid].offset + 1));

		// print sequence and quality
		for (j = 0; j != p->len; ++j) putchar("ACGTN"[(int)s[j]]);
		putchar('\t');
		if (p->qual) {
			if (p->strand)
				seq_reverse(p->len, p->qual, 0); // reverse quality
			//printf("%s", p->qual); //changed to fix a bug when length = 72bp
			int i;
			for (i = 0; i != (p->len); ++i)	printf("%c",p->qual[i]);
		} else printf("*");

		if (bwa_rg_id) printf("\tRG:Z:%s", bwa_rg_id);

		if (p->type != BWA_TYPE_NO_MATCH) {
			// calculate XT tag
			XT = "NURM"[p->type];
			if (nn > 10) XT = 'N';
			// print tags
			printf("\tXT:A:%c\t%s:i:%d", XT, (mode & BWA_MODE_COMPREAD)? "NM" : "CM", p->n_mm + p->n_gapo);
			if (nn) printf("\tXN:i:%d", nn);
			if (mate) printf("\tSM:i:%d\tAM:i:%d", p->seQ, am);
			if (p->type != BWA_TYPE_MATESW) { // X0 and X1 are not available for this type of alignment
				printf("\tX0:i:%d", p->c1);
				if (p->c1 <= max_top2) printf("\tX1:i:%d", p->c2);
			}
			printf("\tXM:i:%d\tXO:i:%d\tXG:i:%d", p->n_mm, p->n_gapo, p->n_gapo+p->n_gape);
			if (p->md) printf("\tMD:Z:%s", p->md);
		}
		putchar('\n');
	} else { // this read has no match
		ubyte_t *s = p->strand? p->rseq : p->seq;
		printf("%s\t%d\t*\t0\t0\t*\t*\t0\t0\t", p->name, p->extra_flag|SAM_FSU);
		for (j = 0; j != p->len; ++j) putchar("ACGTN"[(int)s[j]]);
		putchar('\t');
		if (p->qual) {
			if (p->strand) seq_reverse(p->len, p->qual, 0); // reverse quality
			//printf("%s", p->qual); //changed to fix a bug when length = 72bp
			int i;
			for (i = 0; i != (p->len); ++i)	printf("%c",p->qual[i]);
		} else printf("*");
		if (bwa_rg_id) printf("\tRG:Z:%s", bwa_rg_id);
		putchar('\n');
	}
}

bntseq_t *bwa_open_nt(const char *prefix)
{
	bntseq_t *ntbns;
	char *str;
	str = (char*)calloc(strlen(prefix) + 10, 1);
	strcat(strcpy(str, prefix), ".nt");
	ntbns = bns_restore(str);
	free(str);
	return ntbns;
}
/*
double diff_in_seconds(struct timeval *finishtime, struct timeval * starttime)
{
	double sec;
	sec=(finishtime->tv_sec-starttime->tv_sec);
	sec+=(finishtime->tv_usec-starttime->tv_usec)/1000000.0;
	return sec;
}*/

//print header

void bwa_print_sam_SQ(const bntseq_t *bns)
{
	int i;
	for (i = 0; i < bns->n_seqs; ++i)
		printf("@SQ\tSN:%s\tLN:%d\n", bns->anns[i].name, bns->anns[i].len);
	if (bwa_rg_line) printf("%s\n", bwa_rg_line);
}

char *bwa_escape(char *s)
{
	char *p, *q;
	for (p = q = s; *p; ++p) {
		if (*p == '\\') {
			++p;
			if (*p == 't') *q++ = '\t';
			else if (*p == 'n') *q++ = '\n';
			else if (*p == 'r') *q++ = '\r';
			else if (*p == '\\') *q++ = '\\';
		} else *q++ = *p;
	}
	*q = '\0';
	return s;
}

int bwa_set_rg(const char *s)
{
	char *p, *q, *r;
	if (strstr(s, "@RG") != s) return -1;
	if (bwa_rg_line) free(bwa_rg_line);
	if (bwa_rg_id) free(bwa_rg_id);
	bwa_rg_line = strdup(s);
	bwa_rg_id = 0;
	bwa_escape(bwa_rg_line);
	p = strstr(bwa_rg_line, "\tID:");
	if (p == 0) return -1;
	p += 4;
	for (q = p; *q && *q != '\t' && *q != '\n'; ++q);
	bwa_rg_id = calloc(q - p + 1, 1);
	for (q = p, r = bwa_rg_id; *q && *q != '\t' && *q != '\n'; ++q)
		*r++ = *q;
	return 0;
}

//POSIX Multithreading for refine_gapped

#ifdef HAVE_PTHREAD
/*typedef struct {
	int id;
	bwa_seq_t * sequence;
} thread_data_t;*/

typedef struct {
	int tid;
	int n_seqs;
	bwa_seq_t *seqs;
	bntseq_t *bns;
	bntseq_t *ntbns;
	int n_threads;
} thread_samse_aux_t;

static void *samse_worker(void *data)
{
	thread_samse_aux_t *d = (thread_samse_aux_t*)data;
	bwa_refine_gapped(d->tid, d->n_threads, d->bns, d->n_seqs, d->seqs, 0, d->ntbns);
	return 0;
}
#endif // HAVE_PTHREAD


void bwa_sai2sam_se_core(const char *prefix, const char *fn_sa, const char *fn_fa, const int useCPU, int device)
//This is modified to load both the bwt & rbwt into memory here rather than in bwa_cal_pac_pos
//Adv is this reduces disk I/O (much faster)
{


	int i, n_seqs, tot_seqs = 0, m_aln;
	barracuda_aln1_t *aln = 0;
	bwa_seq_t *seqs;
	bwa_seqio_t *ks;
	//clock_t t;
	bntseq_t *bns, *ntbns = 0;
	FILE *fp_sa;
	barracuda_gap_opt_t opt;

	// For timing purpose only
	struct timeval start, end;
	double time_used;
	gettimeofday (&start, NULL);
	// initialization
	for (i = 1; i != 256; ++i) g_log_n[i] = (int)(4.343 * log(i) + 0.5);
	bns = bns_restore(prefix);
	srand48(bns->seed);
	ks = bwa_seq_open(fn_fa);

	fp_sa = xopen(fn_sa, "r");
	char str[1024];
	bwt_t *bwt, *rbwt;

	//POSIX multithreading

	#ifdef HAVE_PTHREAD
		int max_threads = sysconf( _SC_NPROCESSORS_ONLN );
		int n_threads = 0;
		if (max_threads >= 2)
		{
			n_threads = 2;
		}else
		{
			n_threads = 1;
		}
		//fprintf(stderr,"No of threads available %d, using %d threads\n", max_threads, n_threads);
	#endif

		// open bwts added by brian
		fprintf(stderr,"[samse_core] Loading BWTs, please wait..");
		// load forward & reverse SA
		gettimeofday (&start, NULL);

		strcpy(str, prefix); strcat(str, ".bwt");  bwt = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".sa"); bwt_restore_sa(str, bwt);
		strcpy(str, prefix); strcat(str, ".rbwt"); rbwt = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".rsa"); bwt_restore_sa(str, rbwt);

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);

		fprintf(stderr, "Done! \n[samse_core] Time used: %0.2fs\n", time_used);

	// core loop
	m_aln = 0;
	fread(&opt, sizeof(barracuda_gap_opt_t), 1, fp_sa);
	if (!(opt.mode & BWA_MODE_COMPREAD)) // in color space; initialize ntpac
		ntbns = bwa_open_nt(prefix);

	if (opt.mid > 0) fprintf(stderr,"[samse_core] Option (-s %d) used for alignment, dropping the first %d bases from 5' ends.\n", opt.mid, opt.mid);

	bwa_print_sam_SQ(bns); //print header - new from bwa 0.5.7
	bwa_print_sam_PG();

    // bwt occurrence array in GPU
    unsigned int *global_bwt = 0;
    // rbwt occurrence array in GPU
    unsigned int *global_rbwt = 0;
    bwtint_t *bwt_sa_de = 0;
    bwtint_t *rbwt_sa_de = 0;
    int *g_log_n_de;
    const int g_log_n_len = 256;
    int n_seqs_max = 0;

    if(!useCPU)
    {
    	fprintf(stderr,"[samse_core] Running CUDA Mode\n");

		int num_devices;
		cudaGetDeviceCount(&num_devices);

		if (!num_devices)
		{
			fprintf(stderr,"[samse_core] Cannot find a suitable CUDA device! aborting!\n");
			return;
		}

    	if (device < 0)
    	{
    		device =  detect_cuda_device();
    	}else
    	{
    		fprintf(stderr,"[samse_core] Using specified CUDA device %d.\n",device);
    	}


        int success = prepare_bwa_cal_pac_pos_cuda1(
            &global_bwt,
            &global_rbwt,
            prefix,
            &bwt_sa_de,
            &rbwt_sa_de,
            bwt,
            rbwt,
            &g_log_n,
            &g_log_n_de,
            g_log_n_len,
            device);

        if (!success) return;

        n_seqs_max = BATCH_SIZE;

        prepare_bwa_cal_pac_pos_cuda2(n_seqs_max);

    	//fprintf(stderr,"[samse_debug] Freeing memory\n");
		bwt_destroy(bwt);
		bwt_destroy(rbwt);

    }else
    {
    	fprintf(stderr,"[samse_core] Running CPU Mode\n");
        n_seqs_max = 0x40000;
    }

	fprintf(stderr,"[samse_core] Mapping SA coordinates to linear space, please wait... \n");
    fprintf(stderr,"[samse_core] Processing %u sequences at a time.\n[samse_core] ", n_seqs_max);


	while ((seqs = bwa_read_seq(ks, n_seqs_max, &n_seqs, (opt.mode & BWA_MODE_COMPREAD), opt.mid)) != 0) {
		tot_seqs += n_seqs;
		//t = clock();

		// read alignments
		for (i = 0; i < n_seqs; ++i) {
			bwa_seq_t *p = seqs + i;
			int n_aln;
			fread(&n_aln, 4, 1, fp_sa);
			if (n_aln > m_aln) {
				m_aln = n_aln;
				aln = (barracuda_aln1_t*)realloc(aln, sizeof(barracuda_aln1_t) * m_aln);
			}
			fread(aln, sizeof(barracuda_aln1_t), n_aln, fp_sa);
			bwa_aln2seq(n_aln, aln, p);
		}

		if (useCPU)
		{
			bwa_cal_pac_pos(prefix, n_seqs, seqs, opt.max_diff, opt.fnr, bwt, rbwt); // forward bwt will be destroyed here
			fprintf(stderr,".");
		}else if(!useCPU)
		{
			launch_bwa_cal_pac_pos_cuda(
			    prefix,
			    n_seqs,
			    seqs,
			    opt.max_diff,
			    opt.fnr,
			    device);

			fprintf(stderr,".");
		}
		//fprintf(stderr, "%.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC); t = clock();

			#ifdef HAVE_PTHREAD

				pthread_t *tid;
				pthread_attr_t attr;
				thread_samse_aux_t *data;
				int j;
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				data = (thread_samse_aux_t*)calloc(n_threads, sizeof(thread_samse_aux_t));
				tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
				for (j = 0; j < n_threads; ++j) {
					data[j].tid = j;
					data[j].bns = bns;
					data[j].ntbns = ntbns;
					data[j].n_seqs = n_seqs;
					data[j].seqs = seqs;
					data[j].n_threads = n_threads;
					pthread_create(&tid[j], &attr, samse_worker, data + j);
				}
				for (j = 0; j < n_threads; ++j) pthread_join(tid[j], 0);
				free(data); free(tid);

			#else
				bwa_refine_gapped(0,0, bns, n_seqs, seqs, 0, ntbns);
			#endif


		//fprintf(stderr, "[bwa_samse_core] refine gapped alignments... ");
		//bwa_refine_gapped(0, n_threads, bns, n_seqs, seqs, 0, ntbns);
		fprintf(stderr,".");

		//fprintf(stderr, "[bwa_samse_core] print alignments... ");
		for (i = 0; i < n_seqs; ++i)
			bwa_print_sam1(bns, seqs + i, 0, opt.mode, opt.max_top2);
		//fprintf(stderr, "%.2f sec\n", (float)(clock() - t) / CLOCKS_PER_SEC); t = clock();
		fprintf(stderr,".");


		bwa_free_read_seq(n_seqs, seqs);
	}

	if(!useCPU)
	{
		free_bwa_cal_pac_pos_cuda1(global_bwt,global_rbwt,bwt_sa_de,rbwt_sa_de,g_log_n_de);
		free_bwa_cal_pac_pos_cuda2();

	}

    ///////////////////////////////////////////////////////////
    // End: finish cuda_bwa_cal_pac_pos()
    ///////////////////////////////////////////////////////////
	gettimeofday (&end, NULL);
	time_used = diff_in_seconds(&end,&start);

	fprintf(stderr, "\n[samse_core] Done!\n[samse_core] Total no. of sequences: %d\n[samse_core] Total program time: %0.2fs (%0.2f sequences/sec)\n", tot_seqs, time_used, (float) tot_seqs/time_used);

	// destroy
	bwa_seq_close(ks);
	if (ntbns) bns_destroy(ntbns);
	bns_destroy(bns);

	if(useCPU)
	{
		bwt_destroy(bwt);
		bwt_destroy(rbwt);
	}
	fclose(fp_sa);
	free(aln);
}

static void print_aln_simple(bwt_t *const bwt[2], const bntseq_t *bns, const barracuda_aln1_t *q, int len, bwtint_t k)
{
	bwtint_t pos;
	int seqid, is_N;
	pos = q->a? bwt_sa(bwt[0], k) : bwt[1]->seq_len - (bwt_sa(bwt[1], k) + len);
	if (pos > bwt[1]->seq_len) pos = 0; // negative
	is_N = bns_coor_pac2real(bns, pos, len, &seqid);
	printf("%s\t%c%d\t%d\n", bns->anns[seqid].name, "+-"[q->a], (int)(pos - bns->anns[seqid].offset + 1),
		   q->n_mm + q->n_gapo + q->n_gape);
}

void bwa_print_all_hits(const char *prefix, const char *fn_sa, const char *fn_fa, int max_occ)
{
	int i, n_seqs, tot_seqs = 0, m_aln;
	barracuda_aln1_t *aln = 0;
	bwa_seq_t *seqs;
	bwa_seqio_t *ks;
	bntseq_t *bns;
	FILE *fp_sa;
	barracuda_gap_opt_t opt;
	bwt_t *bwt[2];

	bns = bns_restore(prefix);
	srand48(bns->seed);
	ks = bwa_seq_open(fn_fa);
	fp_sa = xopen(fn_sa, "r");

	fprintf(stderr,"[samse_core] Now coverting SA coordinates to linear space, please wait..\n");

	{ // load BWT
		char *str = (char*)calloc(strlen(prefix) + 10, 1);
		strcpy(str, prefix); strcat(str, ".bwt");  bwt[0] = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".sa"); bwt_restore_sa(str, bwt[0]);
		strcpy(str, prefix); strcat(str, ".rbwt"); bwt[1] = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".rsa"); bwt_restore_sa(str, bwt[1]);
		free(str);
	}

	m_aln = 0;
	fread(&opt, sizeof(barracuda_gap_opt_t), 1, fp_sa);
	while ((seqs = bwa_read_seq(ks, 0x40000, &n_seqs, opt.mode & BWA_MODE_COMPREAD, opt.mid)) != 0) {
		tot_seqs += n_seqs;
		for (i = 0; i < n_seqs; ++i) {
			bwa_seq_t *p = seqs + i;
			int n_aln, n_occ, k, rest, len;
			len = p->len;
			fread(&n_aln, 4, 1, fp_sa);
			if (n_aln > m_aln) {
				m_aln = n_aln;
				aln = (barracuda_aln1_t*)realloc(aln, sizeof(barracuda_aln1_t) * m_aln);
			}
			fread(aln, sizeof(barracuda_aln1_t), n_aln, fp_sa);
			for (k = n_occ = 0; k < n_aln; ++k) {
				const barracuda_aln1_t *q = aln + k;
				n_occ += q->l - q->k + 1;
			}
			rest = n_occ > max_occ? max_occ : n_occ;
			printf(">%s %d %d\n", p->name, rest, n_occ);
			for (k = 0; k < n_aln; ++k) {
				const barracuda_aln1_t *q = aln + k;
				if (q->l - q->k + 1 <= rest) {
					bwtint_t l;
					for (l = q->k; l <= q->l; ++l)
						print_aln_simple(bwt, bns, q, len, l);
					rest -= q->l - q->k + 1;
				} else { // See also: http://code.activestate.com/recipes/272884/
					int j, i, k;
					for (j = rest, i = q->l - q->k + 1, k = 0; j > 0; --j) {
						double p = 1.0, x = drand48();
						while (x < p) p -= p * j / (i--);
						print_aln_simple(bwt, bns, q, len, q->l - i);
					}
					rest = 0;
					break;
				}
			}
		}
		bwa_free_read_seq(n_seqs, seqs);
	}
	fprintf(stderr,"[samse_core] Done..\n");
	bwt_destroy(bwt[0]); bwt_destroy(bwt[1]);
	bwa_seq_close(ks);
	bns_destroy(bns);
	fclose(fp_sa);
	free(aln);
}

int bwa_sai2sam_se(int argc, char *argv[])
{

	fprintf(stderr, "Barracuda, Version %s\n", PACKAGE_VERSION);

	int useCPU = 0, selectedDevice = -1;
	int c = 0, n_occ = 1;
	while ((c = getopt(argc, argv, "htC:n:r:")) >= 0) {
		switch (c) {
		case 'h': break;
		case 't': useCPU = 1; break;
		case 'C': selectedDevice = atoi(optarg); break;
		case 'n': n_occ = atoi(optarg); break;
		case 'r':
					if (bwa_set_rg(optarg) < 0) {
						fprintf(stderr, "[%s] ERROR!! malformated @RG line! Aborting...\n", __func__);
						return 1;
					}
					break;
		default: return 1;
		}
	}

	if (optind + 3 > argc) {
		fprintf(stderr, "\nSAM single-end alignment output module");
		fprintf(stderr, "\n\n");
		fprintf(stderr, "Usage:\n          barracuda samse [options] <reference.fa> <reads.sai> <reads.fastq>\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Options: \n");
		fprintf(stderr, "         -t 	  Run in CPU mode. [default: CUDA mode]\n");
		fprintf(stderr, "         -C NUM   Specify which CUDA device to use. [default: auto-detect] \n");
		fprintf(stderr, "         -r STR   read group header line such as `@RG\\tID:foo\\tSM:bar' [null]\n");
		fprintf(stderr, "         -n NUM   Maximum number of alignments to [non-SAM] output in the XA tag for single-end reads \n");
		fprintf(stderr, "                  If a read has more than INT hits, the XA tag will not be included.\n");
		fprintf(stderr, "                  [default: %d]\n\n", 1);

		return 1;
	}
	if (n_occ > 1) bwa_print_all_hits(argv[optind], argv[optind+1], argv[optind+2], n_occ);
	else
	{
		//fprintf(stderr,"selected device %d\n", selectedDevice);
		bwa_sai2sam_se_core(argv[optind], argv[optind+1], argv[optind+2], useCPU, selectedDevice);
	}
	free(bwa_rg_line); free(bwa_rg_id);
	return 0;
}
