/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwaseqio.c  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2010, Brian Lam and Simon Lam

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

#include <zlib.h>
#include "bwtaln.h"
#include "utils.h"

#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

extern unsigned char nst_nt4_table[256];

struct __bwa_seqio_t {
	kseq_t *ks;
};

bwa_seqio_t *bwa_seq_open(const char *fn)
{
	gzFile fp;
	bwa_seqio_t *bs;
	bs = (bwa_seqio_t*)calloc(1, sizeof(bwa_seqio_t));
	fp = xzopen(fn, "r");
	bs->ks = kseq_init(fp);
	return bs;
}

void bwa_seq_close(bwa_seqio_t *bs)
{
	if (bs == 0) return;
	gzclose(bs->ks->f->f);
	kseq_destroy(bs->ks);
	free(bs);
}

void seq_reverse(int len, ubyte_t *seq, int is_comp)
{
	int i;
	if (is_comp) {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			if (tmp < 4) tmp = 3 - tmp;
			seq[len-1-i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
			seq[i] = tmp;
		}
		if (len&1) seq[i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
	} else {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			seq[len-1-i] = seq[i]; seq[i] = tmp;
		}
	}
}

#define write_to_half_byte_array(array,index,data) \
	(array)[(index)>>1]=(unsigned char)(((index)&0x1)?(((array)[(index)>>1]&0xF0)|((data)&0x0F)):(((data)<<4)|((array)[(index)>>1]&0x0F)))


// read one sequence (reversed) from fastq file to half_byte_array (i.e. 4bit for one base pair )
int bwa_read_seq_one_half_byte (bwa_seqio_t *bs, unsigned char * half_byte_array, unsigned int start_index, unsigned short * length, int mid)
{
	kseq_t *seq = bs->ks;
	int len, i, mided_len;

	if (((len = kseq_read(seq)) >= 0) && (len > mid)) // added to process only when len is longer than mid tag
	{
		//To cut the length of the sequence
		if ( len > MAX_READ_LENGTH) len = MAX_READ_LENGTH;

		mided_len = len - mid;

		for (i = 0; i < mided_len; i++)
		{
			write_to_half_byte_array(half_byte_array,start_index+i,nst_nt4_table[(int)seq->seq.s[len-i-1]]);
		}

		*length = mided_len;
	}
	else
	{
		*length = 0;
	}

	return len;
}

// read one sequence from fastq file to half_byte_array (i.e. 4bit for one base pair )
int bwa_read_seq_one (bwa_seqio_t *bs, unsigned char * byte_array, unsigned short * length)
{
	kseq_t *seq = bs->ks;
	int len, i;

	if ((len = kseq_read(seq)) >= 0)
	{
		for (i = 0; i < len; i++)
			byte_array[i] = nst_nt4_table[(int)seq->seq.s[len-i-1]];
		*length = len;
	}
	return len;
}


bwa_seq_t *bwa_read_seq(bwa_seqio_t *bs, unsigned int n_needed, unsigned int *n, int is_comp, int mid)
{
	bwa_seq_t *seqs, *p;
	kseq_t *seq = bs->ks;
	int n_seqs, l, i; //, j;

	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(n_needed, sizeof(bwa_seq_t));
	while ((l = kseq_read(seq)) >= 0) {
		if ( l > MAX_READ_LENGTH ) l = MAX_READ_LENGTH; //put a limit on sequence length
		p = &seqs[n_seqs++];
		p->tid = -1; // no assigned to a thread
		p->qual = 0;
		//p->len = l;
		p->len = l - mid;
		p->seq = (ubyte_t*)calloc(p->len, 1);
		for (i = 0; i != p->len; ++i)
			p->seq[i] = nst_nt4_table[(int)seq->seq.s[i+mid]];
		p->rseq = (ubyte_t*)calloc(p->len, 1);
		memcpy(p->rseq, p->seq, p->len);
		seq_reverse(p->len, p->seq, 0); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		seq_reverse(p->len, p->rseq, is_comp);

		/*

		printf("Options is_comp = opt->mode & BWA_MODE_COMPREAD is %d\n", is_comp);

		printf("Forward Sequence:");

		int j;

		for ( j = 0; j < p->len; ++j)
		{
			printf("%d", p->seq[j]);

		}
		printf("\nReverse Sequence:");

		for (j = 0; j < p->len; ++j)
		{
			printf("%d", p->rseq[j]);

		}
		printf("\n");

		*/

		p->name = strdup((const char*)seq->name.s);
		{ // trim /[12]$
			int t = strlen(p->name);
			if (t > 2 && p->name[t-2] == '/' && (p->name[t-1] == '1' || p->name[t-1] == '2')) p->name[t-2] = '\0';
		}
		if (seq->qual.l) // copy quality
		{
			//p->qual = (ubyte_t*)strdup((char*)seq->qual.s);
			p->qual = (ubyte_t*)calloc(p->len, 1);
			int i;
			for (i = 0; i != p->len; ++i)
			{
				p->qual[i] = seq->qual.s[mid+i];
			}
		}

		if (n_seqs == n_needed) break;
	}
	*n = n_seqs;
	if (n_seqs == 0) {
		free(seqs);
		return 0;
	}
	return seqs;
}

void bwa_free_read_seq(int n_seqs, bwa_seq_t *seqs)
{
	int i;
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p = seqs + i;
		free(p->name);
		free(p->seq); free(p->rseq); free(p->qual); free(p->aln); free(p->md);
		free(p->cigar);
	}
	free(seqs);
}
