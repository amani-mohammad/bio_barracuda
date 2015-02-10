/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwtaln.c  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2012, University of Cambridge Metabolic Research Labs.
   Contributers: Petr Klus, Dag Lyberg, Simon Lam and Brian Lam

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 3
   of the License, or (at your option) any later version.

   This program is distribut ed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   This program is based on a modified version of BWA 0.4.9

*/

#define PACKAGE_VERSION "0.6.2 beta"
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include "bwtaln.h"
#include "bwtgap.h"
#include "utils.h"
#include "barracuda.h"

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//The followings are for DEBUG only
#define OUTPUT_ALIGNMENTS 1 // should leave ON for outputting alignment
#define STDOUT_STRING_RESULT 0 // output alignment in text format (in SA coordinates, not compatible with SAM output modules(samse/pe)
#define STDOUT_BINARY_RESULT 1 //output alignment for samse/sampe (leave ON)


// how much debugging information shall the kernel output? kernel output only works for fermi and above
#define DEBUG_LEVEL 0
#define MAX_SEED_LENGTH 50 // not tested beyond 50

//////////////////////////////////////////////////////////////////////////////////////////////
//Line below is for BWA CPU MODE

#ifdef HAVE_PTHREAD
#define THREAD_BLOCK_SIZE 1024
#include <pthread.h>
static pthread_mutex_t g_seq_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

// width must be filled as zero
static int bwt_cal_width(const bwt_t *rbwt, int len, const ubyte_t *str, bwt_width_t *width)
{
	bwtint_t k, l, ok, ol;
	int i, bid;
	bid = 0;
	k = 0; l = rbwt->seq_len;

	for (i = 0; i < len; ++i) {
		ubyte_t c = str[i];
		if (c < 4) {
			bwt_2occ(rbwt, k - 1, l, c, &ok, &ol);
			k = rbwt->L2[c] + ok + 1;
			l = rbwt->L2[c] + ol;
		}
		if (k > l || c > 3) { // then restart
			k = 0;
			l = rbwt->seq_len;
			++bid;
		}
		width[i].w = l - k + 1;
		width[i].bid = bid;
	}
	width[len].w = 0;
	width[len].bid = ++bid;

	return bid;
}

static void bwa_cal_sa_reg_gap(int tid, bwt_t *const bwt[2], int no_of_sequences, bwa_seq_t *seqs, const barracuda_gap_opt_t *opt)
{
	int i, max_l = 0, max_len;
	gap_stack_t *stack;
	bwt_width_t *w[2], *seed_w[2];
	const ubyte_t *seq[2];
	barracuda_gap_opt_t local_opt = *opt;

	// initiate priority stack
	for (i = max_len = 0; i != no_of_sequences; ++i)
		if (seqs[i].len > max_len) max_len = seqs[i].len;
	if (opt->fnr > 0.0) local_opt.max_diff = bwa_cal_maxdiff(max_len, BWA_AVG_ERR, opt->fnr);
	if (local_opt.max_diff < local_opt.max_gapo) local_opt.max_gapo = local_opt.max_diff;
	stack = gap_init_stack(local_opt.max_diff, local_opt.max_gapo, local_opt.max_gape, &local_opt);

	seed_w[0] = (bwt_width_t*)calloc(opt->seed_len+1, sizeof(bwt_width_t));
	seed_w[1] = (bwt_width_t*)calloc(opt->seed_len+1, sizeof(bwt_width_t));
	w[0] = w[1] = 0;

	//fprintf(stderr, "Running with %d threads, printed from thread ID: %d, no of sequences: %d\n", opt->n_threads, tid, no_of_sequences);

	for (i = 0; i != no_of_sequences; ++i) {
		bwa_seq_t *p = seqs + i;
		#ifdef HAVE_PTHREAD
		if (opt->n_threads > 1) {
			pthread_mutex_lock(&g_seq_lock);
			if (p->tid < 0) { // unassigned
				int j;
				for (j = i; j < no_of_sequences && j < i + THREAD_BLOCK_SIZE; ++j)
					seqs[j].tid = tid;
			} else if (p->tid != tid) {
				pthread_mutex_unlock(&g_seq_lock);
				continue;
			}
			pthread_mutex_unlock(&g_seq_lock);

		}
		#endif
		p->sa = 0; p->type = BWA_TYPE_NO_MATCH; p->c1 = p->c2 = 0; p->n_aln = 0; p->aln = 0;
		seq[0] = p->seq; seq[1] = p->rseq;
		if (max_l < p->len) {
			max_l = p->len;
			w[0] = (bwt_width_t*)calloc(max_l + 1, sizeof(bwt_width_t));
			w[1] = (bwt_width_t*)calloc(max_l + 1, sizeof(bwt_width_t));
		}
		//fprintf(stderr, "length:%d\n", p->len);
		bwt_cal_width(bwt[0], p->len, seq[0], w[0]);
		bwt_cal_width(bwt[1], p->len, seq[1], w[1]);
		if (opt->fnr > 0.0) local_opt.max_diff = bwa_cal_maxdiff(p->len, BWA_AVG_ERR, opt->fnr);
		if (opt->seed_len >= p->len) local_opt.seed_len = 0x7fffffff;
		if (p->len > opt->seed_len) {
			bwt_cal_width(bwt[0], opt->seed_len, seq[0] + (p->len - opt->seed_len), seed_w[0]);
			bwt_cal_width(bwt[1], opt->seed_len, seq[1] + (p->len - opt->seed_len), seed_w[1]);
		}

		//printf("seed length: %d", opt->seed_len);

		#if DEBUG_LEVEL > 6
			for (int x = 0; x<p->len; x++) {
				//printf(".%d",seq[x]);
			}

			// print out the widths and bids
			printf("\n");
			for (int x = 0; x<p->len; x++) {
				printf("%i,",w[0][x].w);
			}
			printf("\n");
			for (int x = 0; x<opt->seed_len; x++) {
				printf("%i,",seed_w[0][x].w);
			}
			printf("\n");
			for (int x = 0; x<p->len; x++) {
				printf("%d;",w[0][x].bid);
			}
			printf("\n");
			for (int x = 0; x<opt->seed_len; x++) {
				printf("%d;",seed_w[0][x].bid);
			}
		#endif



		// core function
		p->aln = bwt_match_gap(bwt, p->len, seq, w, p->len <= opt->seed_len? 0 : seed_w, &local_opt, &p->n_aln, stack);
		// store the alignment


		free(p->name); free(p->seq); free(p->rseq); free(p->qual);
		p->name = 0; p->seq = p->rseq = p->qual = 0;
	}
	free(seed_w[0]); free(seed_w[1]);
	free(w[0]); free(w[1]);
	gap_destroy_stack(stack);
}

#ifdef HAVE_PTHREAD
/*
typedef struct {
	int id;
	bwa_seq_t * sequence;
} thread_data_t;
*/
typedef struct {
	int tid;
	bwt_t *bwt[2];
	int n_seqs;
	bwa_seq_t *seqs;
	const barracuda_gap_opt_t *opt;
} thread_aux_t;

static void *worker(void *data)
{
	thread_aux_t *d = (thread_aux_t*)data;
	bwa_cal_sa_reg_gap(d->tid, d->bwt, d->n_seqs, d->seqs, d->opt);
	return 0;
}
#endif // HAVE_PTHREAD


void barracuda_bwa_aln_core(const char *prefix, const char *fn_fa, barracuda_gap_opt_t *opt)
//Main alignment module caller
//calls cuda_alignment_core for CUDA kernels
//contains CPU code for legacy BWA runs

{
	bwa_seqio_t *ks;
	unsigned int no_of_sequences = 0;

	// total number of sequences read
	unsigned long long total_no_of_sequences = 0;

	// initialization
	ks = bwa_seq_open(fn_fa);

	// For timing purpose only
	struct timeval start, end;
	double time_used;
	double total_time_used = 0, total_calculation_time_used = 0;

#if STDOUT_BINARY_RESULT == 1

	if (opt->no_header) {
		fprintf(stderr, "[aln_core] Not outputting header for multiple instances\n");
	}
	else
	{
		//output bwa compatible .sai if specified in '-b' option
		if(!opt->bwa_output)
		{
			fwrite(opt, sizeof(barracuda_gap_opt_t), 1, stdout);
		}else
		{
			fprintf(stderr,"[aln_core] Outputing in bwa v0.5.x-compatible file format\n");
			gap_opt_t * bwaopt = gap_init_bwaopt(opt);
			fwrite(bwaopt, sizeof(gap_opt_t), 1, stdout);
		}
	}


#endif

	if(opt->n_threads == -1) // CUDA MODE
	{
		cuda_alignment_core(prefix, ks, opt);
	}
	else if (opt->n_threads > 0) //CPU MODE
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CPU mode starts here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		fprintf(stderr,"[aln_core] Running BWA mode with %d threads\n", opt->n_threads);

		if (opt->max_entries < 0 )
		{
			opt->max_entries = 2000000;
		}

		bwt_t * bwt[2];
		bwa_seq_t * seqs;

		total_time_used = 0;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy bwt occurrences array to from HDD to CPU memory
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		// load BWT to main memory
		char *str = (char*)calloc(strlen(prefix) + 10, 1);
		strcpy(str, prefix); strcat(str, ".bwt");  bwt[0] = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".rbwt"); bwt[1] = bwt_restore_bwt(str);
		free(str);
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		fprintf(stderr, "[bwa_aln_core] Finished loading reference sequence in %0.2fs.\n", time_used );


		//main loop
		gettimeofday (&start, NULL);
		while ((seqs = bwa_read_seq(ks, 0x40000, &no_of_sequences, opt->mode & BWA_MODE_COMPREAD, opt->mid)) != 0)
		{
			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_time_used+=time_used;
			//fprintf(stderr, "Finished loading sequence reads to memory, time taken: %0.2fs, %d sequences (%.2f sequences/sec)\n", time_used, no_of_sequences, no_of_sequences/time_used);

			gettimeofday (&start, NULL);
			fprintf(stderr, "[bwa_aln_core] calculate SA coordinate... \n");
			#ifdef HAVE_PTHREAD
			if (opt->n_threads <= 1) { // no multi-threading at all
				bwa_cal_sa_reg_gap(0, bwt, no_of_sequences, seqs, opt);
			} else {
				pthread_t *tid;
				pthread_attr_t attr;
				thread_aux_t *data;
				int j;
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				data = (thread_aux_t*)calloc(opt->n_threads, sizeof(thread_aux_t));
				tid = (pthread_t*)calloc(opt->n_threads, sizeof(pthread_t));
				for (j = 0; j < opt->n_threads; ++j) {
					data[j].tid = j;
					data[j].bwt[0] = bwt[0];
					data[j].bwt[1] = bwt[1];
					data[j].n_seqs = no_of_sequences; data[j].seqs = seqs; data[j].opt = opt;
					pthread_create(&tid[j], &attr, worker, data + j);
				}
				for (j = 0; j < opt->n_threads; ++j) pthread_join(tid[j], 0);
				free(data); free(tid);
			}
			#else
			bwa_cal_sa_reg_gap(0, bwt, no_of_sequences, seqs, opt);
			#endif

			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			//fprintf(stderr, "Finished!  Time taken: %0.2fs  %d sequences analyzed.\n(%.2f sequences/sec)\n\n", time_used, no_of_sequences, no_of_sequences/time_used );
			#if OUTPUT_ALIGNMENTS == 1
			gettimeofday (&start, NULL);
			fprintf(stderr, "[bwa_aln_core] Writing to the disk... \n");



			#if STDOUT_BINARY_RESULT == 1

			if(opt->bwa_output)
			{
				int i, j;
				//fprintf(stderr, "[aln_debug] Outputing in bwa-compatible file format... ");
				for (i = 0; i < no_of_sequences; ++i) {
						bwa_seq_t *p = seqs + i;
						bwt_aln1_t * output;
						output = (bwt_aln1_t*)malloc(p->n_aln*sizeof(bwt_aln1_t));

						for (j = 0; j < p->n_aln; j++)
						{
							barracuda_aln1_t * temp = p->aln + j;
							bwt_aln1_t * temp_out = output + j;
							temp_out->a = temp->a;
							temp_out->k = temp->k;
							temp_out->l = temp->l;
							temp_out->n_mm = temp->n_mm;
							temp_out->n_gapo = temp->n_gapo;
							temp_out->n_gape = temp->n_gape;
						}

						fwrite(&p->n_aln, 4, 1, stdout);
						if (p->n_aln) fwrite(output, sizeof(bwt_aln1_t), p->n_aln, stdout);
				}
			}else
			{
				int i;
				for (i = 0; i < no_of_sequences; ++i) {
					bwa_seq_t *p = seqs + i;

					fwrite(&p->n_aln, 4, 1, stdout);
					if (p->n_aln) fwrite(p->aln, sizeof(barracuda_aln1_t), p->n_aln, stdout);
				}
			}
			#endif // STDOUT_BINARY_RESULT == 1


			#if STDOUT_STRING_RESULT == 1

			for (int i = 0; i < no_of_sequences; ++i)
			{
				bwa_seq_t *p = &seqs[i];
				if (p->n_aln > 0)
				{
							printf("Sequence %d, no of alignments: %d\n", i, p->n_aln);
							for (int j = 0; j < p->n_aln && j < MAX_NO_OF_ALIGNMENTS; j++)
								{
									barracuda_aln1_t * temp = p->aln + j;
									printf("  Aligned read %d, ",j+1);
									printf("a: %d, ", temp->a);
									printf("n_mm: %d, ", temp->n_mm);
									printf("n_gape: %d, ", temp->n_gape);
									printf("n_gapo: %d, ", temp->n_gapo);
									printf("k: %u, ", temp->k);
									printf("l: %u,", temp->l);
									printf("score: %u\n", temp->score);
								}

				}
			}
			#endif



			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			//fprintf(stderr, "%0.2f sec\n", time_used);
			#endif // OUTPUT_ALIGNMENTS == 1

			gettimeofday (&start, NULL);
			bwa_free_read_seq(no_of_sequences, seqs);
			total_no_of_sequences += no_of_sequences;
			fprintf(stderr, "[bwa_aln_core] %u sequences have been processed.\n", (unsigned int)total_no_of_sequences);
		}
		fprintf(stderr, "[bwa_aln_core] Total no. of sequences: %u \n", (unsigned int)total_no_of_sequences );
		fprintf(stderr, "[bwa_aln_core] Total compute time: %0.2fs, total program time: %0.2fs.\n", total_calculation_time_used, total_time_used);

		// destroy
		bwt_destroy(bwt[0]); bwt_destroy(bwt[1]);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CPU mode ends here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
	bwa_seq_close(ks);

	return;
} //bwa_aln_core ends

int bwa_aln(int argc, char *argv[])
{
	int c, opte = -1;
	barracuda_gap_opt_t *opt;
	//unsigned char cuda_opt = 0;

	fprintf(stderr, "Barracuda, Version %s\n",PACKAGE_VERSION);
	opt = gap_init_opt();
	while ((c = getopt(argc, argv, "s:a:C:n:o:e:i:d:l:k:cbfFLR:m:t:NM:O:E:z")) >= 0) {
		switch (c) {
		case 'a': opt->max_aln = atoi(optarg); break;
		case 's': opt->mid = atoi(optarg); break;
		case 'C': opt->cuda_device = atoi(optarg); break;
		case 'n':
			if (strstr(optarg, ".")) opt->fnr = atof(optarg), opt->max_diff = -1;
			else opt->max_diff = atoi(optarg), opt->fnr = -1.0;
			break;
		case 'o': opt->max_gapo = atoi(optarg); break;
		case 'e': opte = atoi(optarg); break;
		case 'M': opt->s_mm = atoi(optarg); break;
		case 'O': opt->s_gapo = atoi(optarg); break;
		case 'E': opt->s_gape = atoi(optarg); break;
		case 'd': opt->max_del_occ = atoi(optarg); break;
		case 'i': opt->indel_end_skip = atoi(optarg); break;
		case 'l': opt->seed_len = atoi(optarg); break;
		case 'k': opt->max_seed_diff = atoi(optarg); break;
		case 'm': opt->max_entries = atoi(optarg); break;
		case 't': opt->n_threads = atoi(optarg); break;
		case 'L': opt->mode |= BWA_MODE_LOGGAP; break;
		case 'R': opt->max_top2 = atoi(optarg); break;
		case 'c': opt->mode &= ~BWA_MODE_COMPREAD; break;
		case 'N': opt->mode |= BWA_MODE_NONSTOP; opt->max_top2 = 0x7fffffff; break;
		case 'b': opt->bwa_output = 1; break;
		case 'f': opt->fast = 0; break;
		case 'F': opt->fast = 1; break;
		case 'z': opt->no_header = 1;break;
		default: return 1;
		}
	}
	if (opte > 0) {
		opt->max_gape = opte;
		opt->mode &= ~BWA_MODE_GAPE;
	}

	if (optind + 2 > argc) {
		fprintf(stderr, "\nBWT alignment module using NVIDIA CUDA Graphics Cards");
		fprintf(stderr, "\n\n");

		fprintf(stderr, "Usage:   \n");
		fprintf(stderr, "         barracuda aln [options] <reference.fa> <reads.fastq>\n");
		fprintf(stderr, "\n");

		fprintf(stderr, "Options: \n");
		fprintf(stderr, "         -n NUM  max #diff (int) or missing prob under %.2f err rate (float) [default: %.2f]\n",BWA_AVG_ERR, opt->fnr);
		fprintf(stderr, "         -o INT  maximum number or fraction of gap opens [default: %d]\n", opt->max_gapo);
		fprintf(stderr, "         -e INT  maximum number of gap extensions, -1 for disabling long gaps [default: -1]\n");
		fprintf(stderr, "         -i INT  do not put an indel within INT bp towards the ends [default: %d]\n", opt->indel_end_skip);
		fprintf(stderr, "         -d INT  maximum occurrences for extending a long deletion [default: %d]\n", opt->max_del_occ);
		fprintf(stderr, "         -l INT  seed length [default: %d, maximum: %d]\n", opt->seed_len, MAX_SEED_LENGTH);
		fprintf(stderr, "         -k INT  maximum differences in the seed [%d]\n", opt->max_seed_diff);
		fprintf(stderr, "         -m INT  maximum no of loops/entries for matching [default: 150K for CUDA, 2M for BWA]\n");
		fprintf(stderr, "         -M INT  mismatch penalty [default: %d]\n", opt->s_mm);
		fprintf(stderr, "         -O INT  gap open penalty [default: %d]\n", opt->s_gapo);
		fprintf(stderr, "         -E INT  gap extension penalty [default: %d]\n", opt->s_gape);
		fprintf(stderr, "         -L      log-scaled gap penalty for long deletions\n");
		fprintf(stderr, "         -R INT  stop searching when >INT equally best hits are found [default: %d]\n", opt->max_top2);
		fprintf(stderr, "         -c      reverse but not complement input sequences for colour space reads\n");
		fprintf(stderr, "         -N      non-iterative mode: search for all n-difference hits.\n");
		fprintf(stderr, "                 CAUTION this is extremely slow\n");
//		fprintf(stderr, "         -s INT  Skip the first INT 5' bases for alignment.\n");
		fprintf(stderr, "         -t INT  revert to original BWA with INT threads [default: %d]\n", opt->n_threads);
		fprintf(stderr, "                 cannot use with -C, -F or -a\n");
		fprintf(stderr, "         -b      Output SA coordinates in BWA 0.5.x readable sai format.\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "CUDA only options:\n");
		fprintf(stderr, "         -C INT  Specify which CUDA device to use. [default: auto-detect] \n");
		fprintf(stderr, "         -a INT  maximum number of alignments on each strand, max: 20 [default: %d]\n", opt->max_aln);
		fprintf(stderr, "         -f      disable ungapped seed alignment (fast mode). [default: enabled]\n");
		fprintf(stderr, "\n");
		return 1;
	}

	if (opt->seed_len > MAX_SEED_LENGTH)
	{
		fprintf(stderr,"[aln_core] Warning, seed length cannot be longer than %d!\n", MAX_SEED_LENGTH);
		return 0;
	}
	if (!opt->n_threads)
	{
		fprintf(stderr, "Error in option (t): No. of threads cannot be 0!\n");
		return 0;
	}
	if (!opt->max_aln)
		{
			fprintf(stderr, "Error in option (a): Max. no. of alignments cannot be 0!\n");
			return 0;
		}

	if (opt->fnr > 0.0) {
		int i, k;
		for (i = 20, k = 0; i <= 150; ++i) {
			int l = bwa_cal_maxdiff(i, BWA_AVG_ERR, opt->fnr);
			if (l != k) fprintf(stderr, "[aln_core] %dbp reads: max_diff = %d\n", i, l);
			k = l;

		}
	}

	barracuda_bwa_aln_core(argv[optind], argv[optind+1], opt);

	free(opt);
	return 0;
}
//////////////////////////////////////////
// End of ALN_CORE
//////////////////////////////////////////
