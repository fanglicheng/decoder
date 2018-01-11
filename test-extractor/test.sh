#!/bin/bash
exit_on_fail=0

if [[ $1  &&  $1 = freeze ]]; then
    echo
fi

../extractor.py --parallel_corpus=c,e,c-e.a --weightfiles=lex.e2f,lex.f2e --maxabslen 10 --rule_dump dump 2>/dev/null

if test $? -eq 1; then
    echo extractor.py RETURNS 1
    exit 1
fi

if [[ $1  &&  $1 = freeze ]]; then
    cp dump/extract.0001 extract.0001.gold
else
    if diff -bu extract.0001.gold dump/extract.0001; then
        echo extractor.py OK
    else
        echo extractor.py FAIL
        if test $exit_on_fail -eq 1; then exit; fi
    fi
fi

../scorer.py --unfiltered=dump --filter_file=tune --filtered=tune.gr 2>/dev/null

if test $? -eq 1; then
    echo scorer.py RETURNS 1
    exit 1
fi

if [[ $1  &&  $1 = freeze ]]; then
    cp tune.gr tune.gr.gold
else
    if diff -bu tune.gr.gold tune.gr; then
        echo scorer.py OK
    else
        echo scorer.py FAIL
        if test $exit_on_fail -eq 1; then exit; fi
    fi
fi

