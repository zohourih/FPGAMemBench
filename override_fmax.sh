#!/bin/bash

skip=0

if [[ -z $1 ]]
then
	echo "Missing kernel file path!"
	exit -1
else
	path=$1
fi

if [[ -z $2 ]]
then
	skip=1
else
	echo "Overriding Fmax!"
	fmax=$2
fi

if [[ $skip -eq 0 ]]
then
	if [[ -z `quartus_map --version | grep "Pro"` ]]
	then
		pro=0
	else
		pro=1
	fi

	if [[ $pro -eq 0 ]]
	then
		orig=`echo "source $::env(ALTERAOCLSDKROOT)/ip/board/bsp/adjust_plls.tcl"`
		orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')

		new=`echo -e "set argv [list -fmax $fmax]\n        set argc 2\n        $orig\n        unset argv\n        unset argc"`
		new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')

		sed -i "s/$orig_fix/$new_fix/" $path/scripts/post_flow.tcl
	else
		orig=`echo "source $::env(ALTERAOCLSDKROOT)/ip/board/bsp/adjust_plls_a10.tcl"`
		orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')

		new=`echo -e "set argv [list -fmax $fmax]\nset argc 2\n$orig\nunset argv\nunset argc"`
		new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')

		sed -i "s/$orig_fix/$new_fix/" $path/scripts/post_flow_pr.tcl
	fi
fi
