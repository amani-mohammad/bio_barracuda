/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Heng Li <lh3@sanger.ac.uk> */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "utils.h"
#include "bwt.h"

// generate table for counting 11,10,01,00(pattern) for 8 bits number
//     table    [no#     ] = representation ( # of count-pattern, . , . , . )
// ---------------------------------------------------------------------------
// e.g cnt_table[11111111] = 0x04000000     ( 4-11, 0-10, 0-01, 0-00 )
//     cnt_table[00100001] = 0x00010102     ( 0-11, 1-10, 1-01, 2-00 )
//     cnt_table[00000001] = 0x00000103     ( 0-11, 0-10, 1-01, 3-00 )
void bwt_gen_cnt_table(bwt_t *bwt)
{
	int i, j;
	for (i = 0; i != 256; ++i) {
		uint32_t x = 0;
		for (j = 0; j != 4; ++j)
			x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);
		bwt->cnt_table[i] = x;
	}
}

// bwt calculate suffix array interval
// bwt->bwt and bwt->occ must be precalculated
void bwt_cal_sa(bwt_t *bwt, int intv)
{
	bwtint_t isa, sa, i; // S(isa) = sa

	xassert(bwt->bwt, "bwt_t::bwt is not initialized.");

	if (bwt->sa) free(bwt->sa);
	bwt->sa_intv = intv;
	bwt->n_sa = (bwt->seq_len + intv) / intv;
	bwt->sa = (bwtint_t*)calloc(bwt->n_sa, sizeof(bwtint_t));
	// calculate SA value
	isa = 0; sa = bwt->seq_len;
	for (i = 0; i < bwt->seq_len; ++i) {
		if (isa % intv == 0) bwt->sa[isa/intv] = sa;
		--sa;
		isa = bwt_invPsi(bwt, isa);
	}
	if (isa % intv == 0) bwt->sa[isa/intv] = sa;
	bwt->sa[0] = (bwtint_t)-1; // before this line, bwt->sa[0] = bwt->seq_len
}

bwtint_t bwt_sa(const bwt_t *bwt, bwtint_t k)
{
	bwtint_t sa = 0;
	while (k % bwt->sa_intv != 0) {
		++sa;
		k = bwt_invPsi(bwt, k);
	}
	/* without setting bwt->sa[0] = -1, the following line should be
	   changed to (sa + bwt->sa[k/bwt->sa_intv]) % (bwt->seq_len + 1) */
	return sa + bwt->sa[k/bwt->sa_intv];
}

// with y of 64 bits which is a string sequence encoded with 2 bits per alphabet,
// count the number of occurrence of c ( one pattern of 2 bits alphabet ) in y
static inline int __occ_aux(uint64_t y, int c)
{
	// 2 bits per alphabet, 00b,01b,10b,11b,
	// after this operation the only position with pattern c will left as 1
	// all other pattern will become 0.
	// reduce nucleotide counting to bits counting

	y = ((c&2)? y : ~y) >> 1 & ((c&1)? y : ~y) & 0x5555555555555555ull;

	// count the number of 1s in y, which the occurrence of c in y
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

static unsigned int order = 0;

// return occurrence of c in bwt with k smallest suffix
inline bwtint_t bwt_occ(const bwt_t *bwt, bwtint_t k, ubyte_t c)
{
	bwtint_t n, l, j;
	uint32_t *p;

	if (k == bwt->seq_len) return bwt->L2[c+1] - bwt->L2[c];
	if (k == (bwtint_t)(-1)) return 0;
	if (k >= bwt->primary) --k; // because $ is not in bwt

	// retrieve Occ at k/OCC_INTERVAL
	n = (p = bwt_occ_intv(bwt, k))[c];
	//printf ( "%0.5d %x\n", order++, p );
	p += 4; // jump to the start of the first BWT cell

	// calculate Occ up to the last k/32
	j = k >> 5 << 5;
	for (l = k/OCC_INTERVAL*OCC_INTERVAL; l < j; l += 32, p += 2)
		n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);

	// calculate Occ
	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
	if (c == 0) n -= ~k&31; // corrected for the masked bits

	return n;
}

// find the number of occurrence of c in bwt for suffix position k and l, the occurrence count is stored
// in ok for position k and ol for position l
// an analogy to bwt_occ() but more efficient, requiring k <= l
inline void bwt_2occ_modified(const bwt_t *bwt, unsigned int * in, ubyte_t c, unsigned int *out)
{
	bwt_2occ(bwt, in[0], in[1], c, &out[0], &out[1]);
}

 void bwt_2occ(const bwt_t *bwt, bwtint_t k, bwtint_t l, ubyte_t c, bwtint_t *ok, bwtint_t *ol)
{
	bwtint_t _k, _l;
	if (k == l) {
		*ok = *ol = bwt_occ(bwt, k, c);
		return;
	}
	_k = (k >= bwt->primary)? k-1 : k;
	_l = (l >= bwt->primary)? l-1 : l;
	if (_l/OCC_INTERVAL != _k/OCC_INTERVAL || k == (bwtint_t)(-1) || l == (bwtint_t)(-1)) {
		*ok = bwt_occ(bwt, k, c);
		*ol = bwt_occ(bwt, l, c);
	} else {
		bwtint_t m, n, i, j;
		uint32_t *p;
		if (k >= bwt->primary) --k;
		if (l >= bwt->primary) --l;
		n = (p = bwt_occ_intv(bwt, k))[c];
		//printf ( "%x\n", k );
		p += 4;
		// calculate *ok
		j = k >> 5 << 5;
		//int order = 0;
		for (i = k/OCC_INTERVAL*OCC_INTERVAL; i < j; i += 32, p += 2)
		{
			//printf( "order %d %d %d\n", order++,(i&0x60)>>5, (j&0x60)>>5 );
			n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
		}
		m = n;
		n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
		if (c == 0) n -= ~k&31; // corrected for the masked bits
		*ok = n;
		// calculate *ol
		j = l >> 5 << 5;
		for (; i < j; i += 32, p += 2)
			m += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
		m += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~l&31)<<1)) - 1), c);
		if (c == 0) m -= ~l&31; // corrected for the masked bits
		*ol = m;
	}
}

#define __occ_aux4(bwt, b)											\
	((bwt)->cnt_table[(b)&0xff] + (bwt)->cnt_table[(b)>>8&0xff]		\
	 + (bwt)->cnt_table[(b)>>16&0xff] + (bwt)->cnt_table[(b)>>24])

// return occurrence of four alphabet in cnt[0],cnt[1],cnt[2],cnt[3] in bwt with k smallest suffix
inline void bwt_occ4(const bwt_t *bwt, bwtint_t k, bwtint_t cnt[4])
{
	bwtint_t l, j, x;
	uint32_t *p;
	if (k == (bwtint_t)(-1)) {
		memset(cnt, 0, 4 * sizeof(bwtint_t));
		return;
	}
	if (k >= bwt->primary) --k; // because $ is not in bwt
	p = bwt_occ_intv(bwt, k);
	memcpy(cnt, p, 16);
	p += 4;
	j = k >> 4 << 4;
	for (l = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; l < j; l += 16, ++p)
		x += __occ_aux4(bwt, *p);
	x += __occ_aux4(bwt, *p & ~((1U<<((~k&15)<<1)) - 1)) - (~k&15);
	cnt[0] += x&0xff; cnt[1] += x>>8&0xff; cnt[2] += x>>16&0xff; cnt[3] += x>>24;
}

// return occurrence of four alphabet in cntk[0],cntk[1],cntk[2],cntk[3] in bwt with k smallest suffix
// return occurrence of four alphabet in cntl[0],cntl[1],cntl[2],cntl[3] in bwt with l smallest suffix
// an analogy to bwt_occ4() but more efficient, requiring k <= l
inline void bwt_2occ4(const bwt_t *bwt, bwtint_t k, bwtint_t l, bwtint_t cntk[4], bwtint_t cntl[4])
{
	bwtint_t _k, _l;
	if (k == l) {
		bwt_occ4(bwt, k, cntk);
		memcpy(cntl, cntk, 4 * sizeof(bwtint_t));
		return;
	}
	_k = (k >= bwt->primary)? k-1 : k;
	_l = (l >= bwt->primary)? l-1 : l;
	if (_l/OCC_INTERVAL != _k/OCC_INTERVAL || k == (bwtint_t)(-1) || l == (bwtint_t)(-1)) {
		bwt_occ4(bwt, k, cntk);
		bwt_occ4(bwt, l, cntl);
	} else {
		bwtint_t i, j, x, y;
		uint32_t *p;
		int cl[4];
		if (k >= bwt->primary) --k; // because $ is not in bwt
		if (l >= bwt->primary) --l;
		cl[0] = cl[1] = cl[2] = cl[3] = 0;
		p = bwt_occ_intv(bwt, k);
		memcpy(cntk, p, 4 * sizeof(bwtint_t));
		p += 4;
		// prepare cntk[]
		j = k >> 4 << 4;
		for (i = k / OCC_INTERVAL * OCC_INTERVAL, x = 0; i < j; i += 16, ++p)
			x += __occ_aux4(bwt, *p);
		y = x;
		x += __occ_aux4(bwt, *p & ~((1U<<((~k&15)<<1)) - 1)) - (~k&15);
		// calculate cntl[] and finalize cntk[]
		j = l >> 4 << 4;
		for (; i < j; i += 16, ++p) y += __occ_aux4(bwt, *p);
		y += __occ_aux4(bwt, *p & ~((1U<<((~l&15)<<1)) - 1)) - (~l&15);
		memcpy(cntl, cntk, 16);
		cntk[0] += x&0xff; cntk[1] += x>>8&0xff; cntk[2] += x>>16&0xff; cntk[3] += x>>24;
		cntl[0] += y&0xff; cntl[1] += y>>8&0xff; cntl[2] += y>>16&0xff; cntl[3] += y>>24;
	}
}

int bwt_match_exact(const bwt_t *bwt, int len, const ubyte_t *str, bwtint_t *sa_begin, bwtint_t *sa_end)
{
	bwtint_t k, l, ok, ol;
	int i;
	k = 0; l = bwt->seq_len;
	for (i = len - 1; i >= 0; --i) {
		ubyte_t c = str[i];
		if (c > 3) return 0; // no match
		bwt_2occ(bwt, k - 1, l, c, &ok, &ol);
		k = bwt->L2[c] + ok + 1;
		l = bwt->L2[c] + ol;
		if (k > l) break; // no match
	}
	if (k > l) return 0; // no match
	if (sa_begin) *sa_begin = k;
	if (sa_end)   *sa_end = l;
	return l - k + 1;
}

int bwt_match_exact_alt(const bwt_t *bwt, int len, const ubyte_t *str, bwtint_t *k0, bwtint_t *l0)
{
	//fprintf(stderr,"in match_exact");
	//fprintf(stderr,"length: %d\n", len);
	int i;
	bwtint_t k, l, ok, ol;
	k = *k0; l = *l0;
	for (i = len - 1; i >= 0; --i) {
		ubyte_t c = str[i];
		if (c > 3) return 0; // there is an N here. no match
		bwt_2occ(bwt, k - 1, l, c, &ok, &ol);
		k = bwt->L2[c] + ok + 1;
		l = bwt->L2[c] + ol;
		*k0 = k; *l0 = l;
		if (k > l) return 0; // no match
	}
	*k0 = k; *l0 = l;
	return l - k + 1;
}
