#!/bin/bash
passed=0
total=0
exit_on_fail=0

if [[ $1  &&  $1 = freeze ]]; then
    echo
fi

for TESTCASE in d d.3gram smoke; do
    ../decoder.py --flagfile $TESTCASE.flag 2> $TESTCASE.stdout
    if [[ $1  &&  $1 = freeze ]]; then
        cp $TESTCASE.stdout $TESTCASE.stdout.gold
    else
        if diff -bu $TESTCASE.stdout.gold $TESTCASE.stdout; then
	        echo PASSED test $TESTCASE
	        let passed=passed+1
        else
	        echo FAILED test $TESTCASE
            if test $exit_on_fail -eq 1; then exit; fi
        fi
        let total=total+1
    fi
done

#for TESTCASE in `ls ../*_test.py`; do
#    $TESTCASE 2> $TESTCASE.stdout
#    if [[ $1  &&  $1 = freeze ]]; then
#        cp $TESTCASE.stdout $TESTCASE.stdout.gold
#    else
#        if diff -bu $TESTCASE.stdout.gold $TESTCASE.stdout; then
#	        echo PASSED test $TESTCASE
#	        let passed=passed+1
#        else
#	        echo FAILED test $TESTCASE
#            if test $exit_on_fail -eq 1; then exit; fi
#        fi
#        let total=total+1
#    fi
#done

echo PASSED $passed/$total tests

#DOCTESTS=`grep -l doctest ../*.py`
#for TESTCASE in $DOCTESTS; do
#    $TESTCASE
#done
