/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: main.c  Main caller function

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

   This program uses the foundation from BWA 0.49

*/

#include <stdio.h>
#include <string.h>
#include "main.h"
#include "barracuda.h"

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "0.6.2 beta"
#endif

void bwa_print_sam_PG()
{
	printf("@PG\tID:barracuda\tPN:barracuda\tVN:%s\n", PACKAGE_VERSION);
}


static int usage()
{
	fprintf(stderr, "Barracuda, Version %s\n", PACKAGE_VERSION);
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage:   barracuda <command> [options]\n\n");
	fprintf(stderr, "Command: index         index sequences in the FASTA format\n");
	fprintf(stderr, "         aln           gapped/ungapped alignment\n");
	fprintf(stderr, "         samse         generate single-end alignments in SAM format\n");
	fprintf(stderr, "         sampe         generate paired-end alignments in SAM format\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "         fa2pac        convert FASTA to PAC format\n");
	fprintf(stderr, "         pac2bwt       generate BWT from PAC\n");
	fprintf(stderr, "         pac2bwtgen    alternative algorithm for generating BWT\n");
	fprintf(stderr, "         bwtupdate     update .bwt to the new format\n");
	fprintf(stderr, "         pac_rev       generate reverse PAC\n");
	fprintf(stderr, "         bwt2sa        generate SA from BWT and Occ\n");
	fprintf(stderr, "         pac2cspac     convert PAC to color-space PAC\n");
	fprintf(stderr, "         stdsw         standard SW/NW alignment\n");
	fprintf(stderr, "\n");
	return 1;
}

int main(int argc, char *argv[])
{
	int result = 0;
	if (argc < 2) return usage();
	if (strcmp(argv[1], "fa2pac") == 0) return bwa_fa2pac(argc-1, argv+1);
	else if (strcmp(argv[1], "pac2bwt") == 0) return bwa_pac2bwt(argc-1, argv+1);
	else if (strcmp(argv[1], "pac2bwtgen") == 0) return bwt_bwtgen_main(argc-1, argv+1);
	else if (strcmp(argv[1], "bwtupdate") == 0) return bwa_bwtupdate(argc-1, argv+1);
	else if (strcmp(argv[1], "pac_rev") == 0) return bwa_pac_rev(argc-1, argv+1);
	else if (strcmp(argv[1], "bwt2sa") == 0) return bwa_bwt2sa(argc-1, argv+1);
	else if (strcmp(argv[1], "index") == 0) return bwa_index(argc-1, argv+1);
	else if (strcmp(argv[1], "aln") == 0) return bwa_aln(argc-1, argv+1);
	else if (strcmp(argv[1], "sw") == 0) return bwa_stdsw(argc-1, argv+1);
	else if (strcmp(argv[1], "samse") == 0) return bwa_sai2sam_se(argc-1, argv+1);
	else if (strcmp(argv[1], "sampe") == 0) return bwa_sai2sam_pe(argc-1, argv+1);
	else if (strcmp(argv[1], "pac2cspac") == 0) return bwa_pac2cspac(argc-1, argv+1);
	else if (strcmp(argv[1], "stdsw") == 0) return bwa_stdsw(argc-1, argv+1);
	else if (strcmp(argv[1], "deviceQuery") == 0) return bwa_deviceQuery();
	else {
		fprintf(stderr, "[main] unrecognized command '%s'\n", argv[1]);
		result =  1;
	}
	return result;
}
