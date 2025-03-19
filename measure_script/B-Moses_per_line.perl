#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

# $Id$
use warnings;
use strict;

# Command-line arguments processing
my $lowercase = 0;
if ($ARGV[0] eq "-lc") {
    $lowercase = 1;
    shift;
}

# The first argument is the reference text passed directly
my $reference_text = $ARGV[0];
if (!defined $reference_text) {
    print STDERR "usage: multi-bleu.pl [-lc] reference < hypothesis\n";
    print STDERR "Reads the reference text and hypothesis from command line arguments.\n";
    exit(1);
}

# The hypothesis is passed via standard input
my $hypothesis_text = <STDIN>;
chomp($hypothesis_text);

# Convert reference and hypothesis to lowercase if needed
$reference_text = lc($reference_text) if $lowercase;
$hypothesis_text = lc($hypothesis_text) if $lowercase;

# Tokenize the reference and hypothesis
my @reference_words = split(' ', $reference_text);
my @hypothesis_words = split(' ', $hypothesis_text);

# Calculate n-grams (for 1-gram to 4-gram) for BLEU score
my (%ref_ngram, %hyp_ngram, %correct_ngram, $length_reference, $length_hypothesis);

# Function to compute n-grams
sub get_ngrams {
    my ($words, $n) = @_;
    my %ngrams;
    for (my $i = 0; $i <= @$words - $n; $i++) {
        my $ngram = join(' ', @$words[$i..$i+$n-1]);
        $ngrams{$ngram}++;
    }
    return %ngrams;
}

# Collect reference n-grams (for 1-gram to 4-gram)
for (my $n = 1; $n <= 4; $n++) {
    my %ref_ngram_n = get_ngrams(\@reference_words, $n);
    foreach my $ngram (keys %ref_ngram_n) {
        $ref_ngram{$ngram} = $ref_ngram_n{$ngram};
    }
}

# Collect hypothesis n-grams (for 1-gram to 4-gram)
my $length_hypothesis = scalar(@hypothesis_words);
for (my $n = 1; $n <= 4; $n++) {
    my %hyp_ngram_n = get_ngrams(\@hypothesis_words, $n);
    foreach my $ngram (keys %hyp_ngram_n) {
        $hyp_ngram{$ngram} = $hyp_ngram_n{$ngram};
    }
}

# Calculate number of correct n-grams
my $correct_ngrams = 0;
foreach my $ngram (keys %hyp_ngram) {
    if (exists $ref_ngram{$ngram}) {
        $correct_ngrams += $hyp_ngram{$ngram};
    }
}

# Calculate BLEU score
my $precision = $correct_ngrams / $length_hypothesis;
my $brevity_penalty = 1;

if ($length_hypothesis < $length_reference) {
    $brevity_penalty = exp(1 - $length_reference / $length_hypothesis);
}

my $bleu_score = $brevity_penalty * $precision;

# Print the BLEU score
printf "%.2f\n", $bleu_score;

