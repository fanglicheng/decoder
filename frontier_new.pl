#! /usr/bin/perl

#How to run :

# frontier_new.pl /p/mt2/users/saurabh/mt-code/data/ISI/ISI_data.parse /p/mt2/users/saurabh/scratch/eng_chi.giza.afile /p/mt2/users/post/data/chi_eng/corpora/ISI_chi_eng_parallel_corpus.chi.retok > rules

# where .parse : parse file
#	.afile : alignment file
#	.retok : chinese sentences

# top node (TOP) in parser output is discarded
# parse tree must have a single root node under TOP node

# alignment file is in format:
#  0-0 1-1 2-2
# with position in both french and english sentence is numbered from zero
# position in english (the parse file) is before the dash, 
# position in french is after the dash

# -reverse command line switch changes french/english direction of alignment
# file

# -nounary command line switch prevents extraction of rules that are
# unary in chinese (rhs has one nonterminal and no terminals in
# chinese).  frontier nodes below unary rules are removed, 
# meaning that any unary rule is attached to the rule below it.

# -depthtwo command line switch enables extraction of composed rules
# consisting of two minimal ghkm rules

# -alignments command line switch prints word-level alignments from 
# input for each rule after " ||| " 

while ($ARGV[0] =~ /^-/) {
    if ($ARGV[0] eq '-reverse') {
	$reverse_opt = shift;
    } elsif ($ARGV[0] eq '-nounary') {
	$no_unary_opt = shift;
    } elsif ($ARGV[0] eq '-depthtwo') {
	$depthtwo_opt = shift;
    } elsif ($ARGV[0] eq '-alignments') {
	$alignments_opt = shift;
    } else {
	shift;
    }
}

open(PARSE, shift) || die;
open(ALIGN, shift) || die;
open(STR, shift) || die;

my $print_progress = 1;
$rule_count = 0;
my $skip_count = 0;
my $treeno = 0;
while ( <PARSE> ) {
    $treeno++;

    if ($print_progress) {
      if (! ($treeno % 10000)) {
        print STDERR "[$treeno," . (100.0 * $skip_count / $treeno) . "\%]";
      } elsif (! ($treeno % 100)) {
        print STDERR ".";
      }
    }
    
    $id2node = {};
    undef %is_aligned;
    undef @s;
    undef @a;
    undef $tree;

    $tree = &read_parse($_,$id2node);

    # get rid of the top node (something saurabh did)
    $tree = @{$tree->{children}}[0];

    $_ = <ALIGN>;
    @a = split;		# word-level alignments
    undef %a;
    for $pair (@a) {
	($e, $c) = split(/-/, $pair);
	$is_aligned{$c} = 1;
	if ($reverse_opt) {
	    $a{$c}{$e}++;
	} else {
	    $a{$e}{$c}++;
	}
    }

    $_ = <STR>;
    $_ =~ s/\"/DOUBLE_QT/g;                      # replace double quotes with DOUBLE_QT in chinese sentence,
                                                 # this is because the binarizer creates problems
    @s = split;		# the chinese sentence
    @s = map("\"C_".$_ ."\"" , @s);

    if (($tree eq 'NOPARSE') || ($#{$tree->{children}} < 0)) {
      $skip_count++;
      if (! $print_progress) {
        print STDERR "* skipping tree $treeno\n";
      }
      next;
    }
    # The second condition above is only to skip sentences which have bad parse trees. The Charniak
    # parser is found to generate these. (for example, with uneven brackets.)
 
    &find_spans($tree);
    &find_complements($tree);

    if ($no_unary_opt) {
	&get_rid_of_unary_rules_in_chinese($tree, $tree->{minsink}, $tree->{maxsink});
    }

    undef @rules;

    # the normal rules are in @rules
    if ($depthtwo_opt) {
	push(@rules, &find_rules_depth_one_and_two($tree));
    } else {
	push(@rules, &find_rules($tree));
    }

    # print print_parse_qtree($tree) . $/;

    # expand and print the rules
    &print_all_rules();
}

print STDERR "skipped $skip_count of $treeno trees (" . (100.0*$skip_count / $treeno) . "\%)\n";

# print english parse tree along with chinese translations 
# of each word in a human readable form
sub print_parse_with_trans {
    my($tree) = @_;
    my $out;
    if (defined $tree->{children}) {
	$out = '('.$tree->{label};

	#$out .= '~'.$tree->{id};
	# print span in this language
	#$out .= '.'.$tree->{min}.'-'.$tree->{max}  ;

	# print sinks
	#for $s ( keys %{$tree->{sinks}} ) {
	#    $out .= "->$s"
	#}

	# print complement span (aka outside span)
	#for $s ( keys %{$tree->{out}} ) {
	#    $out .= "-<$s"
	#}

	if ($tree->{conflict}) {
	    print "conflict\n";
	    $out .= "=".$tree->{conflict};
	}

	$out .= ' ';
	for $c (@{$tree->{children}}) {
	    $out .= &print_parse_with_trans($c);
	}
	$out .= ') ';
    } else {
	# leaf
	$out = $tree->{label}.'.'.$tree->{term};
	for $c (keys %{$tree->{sinks}}) {
	    $out .= '->'.$s[$c].'.'.$c;
	}
	if ($tree->{conflict}) {
	    $out .= '='.$tree->{conflict};
	}
    }
    $out .= ' ';
}

# prints a qtree-style tree
sub print_parse_qtree {
  my($tree) = @_;
  my $label = $tree->{label};
  my $out;
  if (defined $tree->{children}) {

    if (! $tree->{conflict}) {
      $out = "[.\\node[frontier]{$label}; ";
    } else {
      $out = "[.$label ";
    }

    for $c (@{$tree->{children}}) {
      $out .= &print_parse_qtree($c);
    }
    $out .= '] ';
  } else {
    # leaf
    $label =~ s/"E_(.*)"/$1/;
    $out = "\\node[terminal]{$label};";
    # $out = $tree->{label}.'.'.$tree->{term};
    # for $c (keys %{$tree->{sinks}}) {
    #   $out .= '->'.$s[$c].'.'.$c;
    # }
  }
  $out .= ' ';
}

sub print_parse_with_term {
    my($tree) = @_;
    my $out;
    if (defined $tree->{children}) {
	$out = $tree->{label}.'(';

	#$out .= "$tree->{minsink}-$tree->{maxsink}";

	if ($tree->{conflict}) {
	    $out .= "=".$tree->{conflict};
	}

	for $c (@{$tree->{children}}) {
	    $out .= ' ';
	    $out .= &print_parse_with_term($c);
	}
	$out .= ')';
    } else {
	# leaf
#	print "label ", $tree->{label},  " term ", $tree->{term}, "\n";
	if (defined $tree->{term}) {
	    if ($tree->{label} =~ /\"\w*\"/) {
		$out = ($tree->{term}) . ':' . (substr $tree->{label}, 1, -1);
	    } else {
		$out = ($tree->{term}) . ':' . $tree->{label};
	    }
	}
	else {
	    $out = $tree->{label};
	}
#	$out .= ' ';
    }
    $out;
}

# recusively descends english tree,
# caculating for each node,
# the maximum and minimum of chinese positions to which any descendent is aligned.
sub find_spans {
    my($top) = @_;
    if ($top->{children}) {
	for $c (@{$top->{children}}) {
	    &find_spans($c);
	}
	for $c (@{$top->{children}}) {
	    # span in tree (english positions)
	    if ($c->{min} < $top->{min} || !defined $top->{min}) {
		$top->{min} = $c->{min};
	    }
	    if ($c->{max} > $top->{max} || !defined $top->{max}) {
		$top->{max} = $c->{max};
	    }

	    # closure of sinks (chinese positions)
	    for $s ( keys %{$c->{sinks}} ) {
		$top->{sinks}->{$s}++;

		if ($s < $top->{minsink} || !defined $top->{minsink}) {
		    $top->{minsink} = $s;
		}
		if ($s > $top->{maxsink} || !defined $top->{maxsink}) {
		    $top->{maxsink} = $s;
		}
	    }
	}
    } else {
	# this is a leaf
	$top->{max} = $top->{min} = $top->{term};

	for $s ( keys %{$a{$top->{term}}} ) {
	    $top->{sinks}->{$s}++;

	    if ($s < $top->{minsink} || !defined $top->{minsink}) {
		$top->{minsink} = $s;
	    }
	    if ($s > $top->{maxsink} || !defined $top->{maxsink}) {
		$top->{maxsink} = $s;
	    }
	}
    }
}

# the complement of the english node $top is the set of all
# chinese positions that any non-descendents of $top are aligned to.
# if there is any overlap between the complement and the span
# that $top's descendents are aligned to, we cannot extract a rule 
# with $top as its lefthand side.
# the nodes without such an overlap are called frontier nodes,
# and can be the lefthand side of an extracted rule.
sub find_complements {
    my($top) = @_;
    return if ! defined $top->{children} ;
    for $c (@{$top->{children}}) {

	# everything in complement of parent is in complement of child
	%{ $c->{out} } = %{ $top->{out} };

	# next add anything pointed to by a sibling of current child
	for $sib (@{$top->{children}}) {
	    if ($sib eq $c) { next; }
	    for $s ( keys %{$sib->{sinks}} ) {
		$c->{out}->{$s} ||= 1;
		#print "outside of $c->{id}: $s\n";

	    }
	}

	# check if anything in complement falls within closure of span
	for $s ( keys %{$c->{out}} ) { 
	    if ($s >= $c->{minsink} && $s <= $c->{maxsink}) {
		$c->{conflict} = 'CONFLICT';	# not a "frontier node"
	    }
	}

	&find_complements($c);
    }
}

# any frontier node that is under a frontier
# node corresponding to the same span in chinese
# is no longer considered a frontier node.
sub get_rid_of_unary_rules_in_chinese {
    my($top, $minsink, $maxsink) = @_;
    my($c);
    return if ! defined $top->{children} ;
    for $c (@{$top->{children}}) {
	# print "$top->{label}.$c->{label}   $c->{minsink} == $minsink && $c->{maxsink} == $maxsink\n";
	if (! $c->{conflict} &&
	    $c->{minsink} == $minsink && $c->{maxsink} == $maxsink) {
	    $c->{conflict} = 'CONFLICT';	# no longer a frontier node
	}
	
	if ($c->{conflict}) {
	    &get_rid_of_unary_rules_in_chinese($c, $minsink, $maxsink);
	} else {
	    &get_rid_of_unary_rules_in_chinese($c, $c->{minsink}, $c->{maxsink});
	}
    }
}

# once we have identified the frontier nodes,
# traverse the tree looking for tree fragments with frontier
# node at the root and fringe, and nonfrontier nodes inside.
# these are the rules we extract.
sub find_rules {
    my($top) = @_;
    my $out = {};	# a new node in output rule
    $out->{label} = $top->{label};
    $out->{minsink} = $top->{minsink};	# copy chinese span
    $out->{maxsink} = $top->{maxsink};
    $out->{minterm} = $top->{minterm};	# copy english span
    $out->{maxterm} = $top->{maxterm};
    if (!defined $top->{children}) {
	$out->{sinks} = $top->{sinks};
    } else {
	for $c (@{$top->{children}}) {
	    if (($c->{conflict}) || !(defined $c->{minsink} && defined $c->{maxsink})
		|| (($c->{label} =~ /^\"/) && ($c->{label} =~ /\"$/))) {
		# not frontier node.
		# keep copying the subtree into the new rule
		push(@{$out->{children}}, &find_rules($c));
	    } else {
		# a frontier node.
		# make a nonterminal on the righthand side of the output rule
		push(@{$out->{children}}, 
		     {
			 'label' => $c->{label} ,
			 'term' => $c->{minsink} ,
			 'minsink' => $c->{minsink},
			 'maxsink' => $c->{maxsink}
		     } ) ;
		# start building a new rule as we go down the tree from here
		push(@rules, &find_rules($c));
	    }
	}
    }
    $out;
}

# once we have identified the frontier nodes,
# traverse the tree looking for tree fragments with frontier
# node at the root and fringe, and nonfrontier nodes inside.
# these are the rules we extract.
#
# jump_kth_frontier:
# 0 or greater: continue past kth frontier node to build rule of depth two
# -1: have already skipped frontier node
# -2: build rule of depth one, and recurse down tree
# -3: build rule of depth one (as lower part of larger rule) 
#	and do not recurse

sub find_rules_depth_two {
    my($top, $jump_kth_frontier) = @_;
    my $out = {};	# a new node in output rule
    $out->{label} = $top->{label};
    $out->{minsink} = $top->{minsink};	# copy chinese span
    $out->{maxsink} = $top->{maxsink};
    $out->{minterm} = $top->{minterm};	# copy english span
    $out->{maxterm} = $top->{maxterm};
    if (!defined $top->{children}) {
	$out->{sinks} = $top->{sinks};
    } else {
	for $c (@{$top->{children}}) {
	    if (($c->{conflict}) || !(defined $c->{minsink} && defined $c->{maxsink})
		|| (($c->{label} =~ /^\"/) && ($c->{label} =~ /\"$/))) {
		# not frontier node.
		# keep copying the subtree into the new rule
		push(@{$out->{children}}, &find_rules_depth_two($c, $jump_kth_frontier));
	    } elsif ($jump_kth_frontier == 0) {
		$jump_kth_frontier--;
		push(@{$out->{children}}, &find_rules_depth_two($c, -3));
	    } else {
		# a frontier node.
		# make a nonterminal on the righthand side of the output rule
		push(@{$out->{children}}, 
		     {
			 'label' => $c->{label} ,
			 'term' => $c->{minsink} ,
			 'minsink' => $c->{minsink},
			 'maxsink' => $c->{maxsink}
		     } ) ;
		if ($jump_kth_frontier > 0) {
		    $jump_kth_frontier--;
		} elsif ($jump_kth_frontier == -2) {
		    # start building a new rule as we go down the tree from here
		    &find_rules_depth_one_and_two($c);
		}
	    }
	}
    }
    $out;
}

sub find_rules_depth_one_and_two {
    my($top) = @_;
    my $new_rule = &find_rules_depth_two($top, -2);	# first depth one
    push(@rules, $new_rule);
    my $num_rhs_vars = &count_nonterminals($new_rule);
    for ($i = 0; $i < $num_rhs_vars; $i++) {	# then each rule of depth two
	my $new_rule = &find_rules_depth_two($top, $i);
	push(@rules, $new_rule);
    }
}


sub count_nonterminals {
    my($top) = @_;
    my $n = 0;
    $n++ if defined $top->{term};
    if (defined $top->{children}) {
	for $c (@{$top->{children}}) {
	    $n += &count_nonterminals($c);
	}
    }
    $n;
}

# input: the rule $top
# output: the expanded rule $top
# the modification happens in adding a key in the hash table $top, $top->{rhsmore}[]
# $top->{rhsmore}[x] have the right handside of the rule with the unaligned words inserted

# example: suppose we have rule: X -> 2Y4 6Z7 9W10, and unaligned words U0 U1 U2 U5 U8 U11 U12
# the extra rules we get is:
# X -> 2Y4 U5 6Z7 U8 9W10
# X -> U0 2Y4 U5 6Z7 U8 9W10
# X -> U1 2Y4 U5 6Z7 U8 9W10
# X -> 2Y4 U5 6Z7 U8 9W10 U11
# X -> 2Y4 U5 6Z7 U8 9W10 U12
sub tag_rule_with_fspan {
    my($top) = shift;

    # collect all the variables in the right handside of $top, and their beginning and ending position in 
    # the sentence, so that we know which unaligned word could be inserted after which variable
    my ($vars, $beg, $end) = &collect_vars($top);

    $term_index = 0;
    &renumber_terminals($top, $vars, \%vars_index, \$term_index);

    my $i, $first_head, $last_tail;
    my $head, $tail, $p_head=-1, $p_tail=-1, $j, $inserted = 0;

    # go through all the variables and insert all the possible unaligned words into it
    for $k ( sort {$a<=>$b} keys %{$vars} ) {
	++$i;
	$top->{rhs} .= $vars->{$k}. ' ';

	$head = $beg->{$k};
	$tail = $end->{$k};

	if($i == 1){
	    $first_head = $head;
	}
	if($p_head >=0 && $head - $p_tail > 1){
	    for($j = $p_tail+1; $j <= $head-1; $j ++){
		if(!defined $is_aligned{$j} && length($s[$j])>0){
		    $top->{rhsmore }[0] =  $top->{rhsmore}[0]. $s[$j]. ' ';
		    $inserted = 1;
		}
	    }
	}
	$top->{rhsmore}[0] .= $vars->{$k}. ' ';
	
	$vars->{$k} = $i;
	$p_head = $head;
	$p_tail = $tail;
    }
    $last_tail = $tail;
    
    # expand the left by just one unaligned word
    $j = $first_head-1;
    my $index = 0;
    if($inserted){$index = 1;}
    while($j >= 0 && !defined $is_aligned{$j} && length($s[$j])>0){
	if($index == 0){$base_index = 0;}else{$base_index = $index -1;}
	$top->{rhsmore}[$index] = $s[$j] . ' ' . $top->{rhsmore}[$base_index];
	$index ++;
	$j --;
    }

    # expand the right by just one unaligned word
    $j = $last_tail+1;
    $prior_num = $index;
 
    $base_index = $index;
    while($j <= @s-1 && !defined $is_aligned{$j} && length($s[$j])>0){
	for ($k = 0; $k < $base_index; $k++) {
	    $top->{rhsmore}[$index] = $top->{rhsmore}[$index-$base_index] . $s[$j] . ' ' ;
	    $index ++;
	}
	$j ++;
    }

    if($index==0 && !$inserted){undef $top->{rhsmore};}
}


sub collect_and_renumber_terminals {
    my($top) = @_;
    my ($vars, $beg, $end) = &collect_vars($top);
    my $i;
    for $k ( sort {$a<=>$b} keys %{$vars} ) {
	++$i;
	$top->{rhs} .= $vars->{$k}.'.'.$i. ' ';
	$vars->{$k} = $i;
	#print "vars->{$k} = $i;\n";
    }
    &renumber_terminals($top, $vars);
}

# recursively descend tree, renumbering terms
sub renumber_terminals {
    my($top, $vars, $vars_ind, $term_index) = @_;

    if (defined $top->{term}) {

	if ( $vars->{$top->{term}} =~ /^t:/) {
	    $vars->{$top->{term}} = "x" . $$term_index;
	    $top->{term} = "x" . $$term_index;
	    $$term_index ++;
	}
	else {
	    $top->{term} = $vars->{$top->{term}};
	}
    } 

    if ( defined $top->{sinks} ) {
	for $s ( keys %{$top->{sinks}} ) {
	    $top->{newsinks}->{ $vars->{ $s } } ++  ;
	}
	$top->{sinks} = $top->{newsinks};
    } 
    if ( defined $top->{children} ) {
	for $c ( @{$top->{children}} ) {
	    &renumber_terminals($c, $vars, $vars_ind, $term_index);
	}
    }
}

# returns a list of three hashes. vars, begin, end
# first hash maps (out) maps chinese positions to chinese words and
# english positions to english words.
# second maps span begin points, third end points - used for unaligned words.
sub collect_vars {
    my ($top) = @_;
    my $out = {};
    my $begin = {};
    my $end = {};
    if ( defined $top->{term} ) {
	$out->{ $top->{term} } = "t:" . $top->{label} ;
	$begin->{$top->{term}} = $top->{minsink};
	$end->{$top->{term}} = $top->{maxsink};
    } 
    if ( defined $top->{sinks} ) {
	for $s ( keys %{$top->{sinks}} ) {
	    $out->{ $s } = $s[$s] ;
	    $begin->{$s} = $s;
	    $end->{$s} = $s;
	}
    } 
    if ( defined $top->{children} ) {
	# merge each of the three hashes across all subtrees
	for $c (@{$top->{children}}) {
	    my ($h, $b, $e) = &collect_vars( $c );
	    for $t ( keys %{ $h } ) {
		$out->{$t} = $h->{$t};
	    }
	    for $t1 ( keys %{ $b } ) {
		$begin->{$t1} = $b->{$t1};
	    }
	    for $t2 ( keys %{ $e } ) {
		$end->{$t2} = $e->{$t2};
	    }
	}
    }

    ($out, $begin, $end);

}

sub tag_all_rules {
    for $r (@rules) {
	&tag_rule_with_fspan($r);
    }
}

sub print_all_rules {
    for $r (@rules) {
	# expand the rule with the unaligned words
	&tag_rule_with_fspan($r);
	&print_rule($r);
    }
}

sub print_rule {
    my($top) = shift;
    my $lhs = &print_parse_with_term($r);
    my $alignments;
    $alignments = '||| ' . &print_aligned_terms($r) if $alignments_opt;

    if ($lhs =~ /^\"\w*\"$/ ) {
	$lhs = (substr $lhs, 1, -1) . "(" . $lhs . ")";
    }
    if ( $r->{rhs} ne '') {
	print $lhs , ' -> ', $r->{rhs}, $alignments, "\n";

	$no_undef_rules = @{$r->{rhsmore}};

	if (@{$r->{rhsmore}} > 0) {
            # attach all the unaligned Chinese words to this rule, ignore other partial assignments of unaligned words
	    print $lhs, ' -> ', $r->{rhsmore}[@{$r->{rhsmore}}-1], $alignments, "\n";
	}
    }
}

sub print_aligned_terms {
    my($tree) = @_;
    my $out;
    #print "print_aligned_terms $tree->{label}\n";
    if (defined $tree->{children}) {
	#$out .= "$tree->{minsink}-$tree->{maxsink}";

	for $c (@{$tree->{children}}) {
	    $out .= &print_aligned_terms($c);
	}

    } else {
	# leaf
	#print "label ", $tree->{label},  " term ", $tree->{minterm}, "\n";
	if (defined $tree->{minterm}) {
	    my $e = $tree->{minterm};
	    if (defined $a{$e}) {
		for $c (sort {$a <=> $b} keys %{$a{$e}}) {
		    $out .= "$tree->{label}-$s[$c] ";
		}
	    }
	}

    }
    $out;
}

# turns a parenthesized string into a tree data structure.
# each node is a hash with the following fields:
#  label: 	nonterm or term:  NP, the, etc
#  children: 	list of child nodes
#  id: 		numeric id
#  term: 	for leaf nodes, position in string
sub read_parse {
    # takes ref to node index hash
    my($string, $id2node) = @_;
    my($cid, $top);

    return 'NOPARSE' if /^NOPARSE/ || /^null/;

    $max_children = 0;
    $term_count = 0;
    $string =~ s/\)/ \) /g;

    @a = split(' ', $string);
    for $_ (@a) {
	if (s/^\(//) {
	    $c = {};
	    $c->{label} = get_tag($_);
	    $c->{children} = [];
	    $c->{begin} = $term_count;
	    push(@{$c[$#c]->{children}}, $c) unless $#c < 0;

	    push(@c, $c);
	} elsif (/^\)/) {
	    $cid++;
	    $top = pop(@c);
	    $top->{id} = $cid;
	    $top->{rule} = join('_', map {$_->{label}} @{$top->{children}});

	    if ($#{$top->{children}} + 1 > $max_children) {
		$max_children = $#{$top->{children}} + 1;
	    }
	    $top->{minterm} = $top->{children}->[0]->{minterm};
	    $top->{maxterm} = $top->{children}->[-1]->{minterm};
	    $nonterm_count++;
	} else {
	    # terminal
	    $_ = $_ eq "\"" ? "DOUBLE_QT" : $_;  # replace double quotes with DOUBLEQ in chinese sentence,
                                                 # this is because the binarizer creates problems
	    $c = {};
	    $c->{label} = "\"E_". $_ . "\"";
	    $c->{minterm} = $c->{maxterm} = $c->{term} = $term_count;
	    $cid++;
	    $id2node->{$term_count} = $c;
	    $c->{id} = $cid;
	    push(@{$c[$#c]->{children}}, $c) unless $#c < 0;

	    $term_count++;
	}
    }
    $top->{max_children} = $max_children;
    $top->{term_count} = $term_count;
    $top;

}

sub get_tag {
    my $tag = shift;

    my $new_tag = $tag;
    if ($tag eq '#') {
	$new_tag = "PRE_#";
    } elsif ($tag eq '\$') {
	$new_tag = "PRE_\$";
    } elsif ($tag eq '.') {
	$new_tag = "PRE_\.";
    } elsif ($tag eq ',') {
	$new_tag = "PRE_,";
    } elsif ($tag eq ':') {
	$new_tag = "PRE_:";
    } elsif ($tag eq '\`\`') {
	$new_tag = "PRE_\`\`";
    } elsif ($tag eq "\'\'") {
	$new_tag = "PRE_\'\'";
    } elsif ($tag eq '-NONE-') {
	$new_tag = "PRE_-NONE-";
    }
    $new_tag;
}
