#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

# $Id$
use warnings;
use strict;

my $lowercase = 0;
if ($ARGV[0] eq "-lc") {
  $lowercase = 1;
  shift;
}

my $stem = $ARGV[0];
if (!defined $stem) {
  print STDERR "usage: multi-bleu.pl [-lc] reference < hypothesis\n";
  print STDERR "Reads the references from reference or reference0, reference1, ...\n";
  exit(1);
}

$stem .= ".ref" if !-e $stem && !-e $stem."0" && -e $stem.".ref0";

my @REF;
my $ref=0;
while(-e "$stem$ref") {
    &add_to_ref("$stem$ref",\@REF);
    $ref++;
}
&add_to_ref($stem,\@REF) if -e $stem;
die("ERROR: could not find reference file $stem") unless scalar @REF;

# add additional references explicitly specified on the command line
shift;
foreach my $stem (@ARGV) {
    &add_to_ref($stem,\@REF) if -e $stem;
}

sub add_to_ref {
    my ($file,$REF) = @_;
    my $s=0;
    if ($file =~ /.gz$/) {
	open(REF,"gzip -dc $file|") or die "Can't read $file";
    } else { 
	open(REF,$file) or die "Can't read $file";
    }
    while(<REF>) {
	chop;
	push @{$$REF[$s++]}, $_;
    }
    close(REF);
}

my(@CORRECT,@TOTAL,$length_translation,$length_reference);
my $s=0;
while(<STDIN>) {
    chop;
    $_ = lc if $lowercase;
    my @WORD = split;
    my %REF_NGRAM = ();
    my $length_translation_this_sentence = scalar(@WORD);
    my ($closest_diff,$closest_length) = (9999,9999);
    foreach my $reference (@{$REF[$s]}) {
  $reference = lc($reference) if $lowercase;
	my @WORD = split(' ',$reference);
	my $length = scalar(@WORD);
        my $diff = abs($length_translation_this_sentence-$length);
	if ($diff < $closest_diff) {
	    $closest_diff = $diff;
	    $closest_length = $length;
	} elsif ($diff == $closest_diff) {
            $closest_length = $length if $length < $closest_length;
        }
	for(my $n=1;$n<=4;$n++) {
	    my %REF_NGRAM_N = ();
	    for(my $start=0;$start<=$#WORD-($n-1);$start++) {
		my $ngram = "$n";
		for(my $w=0;$w<$n;$w++) {
		    $ngram .= " ".$WORD[$start+$w];
		}
		$REF_NGRAM_N{$ngram}++;
	    }
	    foreach my $ngram (keys %REF_NGRAM_N) {
		if (!defined($REF_NGRAM{$ngram}) ||
		    $REF_NGRAM{$ngram} < $REF_NGRAM_N{$ngram}) {
		    $REF_NGRAM{$ngram} = $REF_NGRAM_N{$ngram};
		}
	    }
	}
    }
    $length_translation += $length_translation_this_sentence;
    $length_reference += $closest_length;
    for(my $n=1;$n<=4;$n++) {
	my %T_NGRAM = ();
	for(my $start=0;$start<=$#WORD-($n-1);$start++) {
	    my $ngram = "$n";
	    for(my $w=0;$w<$n;$w++) {
		$ngram .= " ".$WORD[$start+$w];
	    }
	    $T_NGRAM{$ngram}++;
	}
	foreach my $ngram (keys %T_NGRAM) {
	    $ngram =~ /^(\d+) /;
	    my $n = $1;
	    $TOTAL[$n] += $T_NGRAM{$ngram};
	    if (defined($REF_NGRAM{$ngram})) {
		$CORRECT[$n] += ($REF_NGRAM{$ngram} >= $T_NGRAM{$ngram}) ? $T_NGRAM{$ngram} : $REF_NGRAM{$ngram};
	    }
	}
    }
    $s++;
}

my $brevity_penalty = 1;
my $bleu = 0;
my @bleu=();

# Ensure CORRECT and TOTAL are initialized
for(my $n=1;$n<=4;$n++) {
  $CORRECT[$n] = 0 unless defined $CORRECT[$n];
  $TOTAL[$n] = 0 unless defined $TOTAL[$n];

  if ($TOTAL[$n] > 0) {
    $bleu[$n] = $CORRECT[$n] / $TOTAL[$n];
  } else {
    $bleu[$n] = 0;
  }
}

if ($length_reference == 0 || $length_translation == 0) {
  print "BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)\n";
  exit(1);
}

if ($length_translation < $length_reference) {
  $brevity_penalty = exp(1 - $length_reference / $length_translation);
} else {
  $brevity_penalty = 1;
}

$bleu = $brevity_penalty * exp(
  (my_log($bleu[1]) +
   my_log($bleu[2]) +
   my_log($bleu[3]) +
   my_log($bleu[4])) / 4
);

printf "%.2f\n", $bleu;

sub my_log {
  return log($_[0]) if $_[0] > 0;
  return 0;
}
