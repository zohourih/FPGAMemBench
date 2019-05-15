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
	echo "Missing version indicator!"
	exit -1
else
	legacy=$2
fi

if [[ -z $3 ]]
then
	skip=1
else
	echo -n "Overriding Fmax: "
	fmax=$3
fi

if [[ $skip -eq 0 ]]
then
	if [[ -z `quartus_fit --version | grep "Pro"` ]]
	then
		pro=0
	else
		pro=1
	fi

	if [[ $pro -eq 0 ]]
	then
		file=`echo $path/scripts/post_flow.tcl`
		if [[ $legacy -eq 1 ]]
		then
			orig=`echo "source $::env(ALTERAOCLSDKROOT)/ip/board/bsp/adjust_plls.tcl"`
			orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
			if [[ -z `cat $file | grep "$orig"` ]]
			then
				echo "FAILURE!! (pattern not found)"
				exit -1
			fi

			new=`echo -e "set argv [list -fmax $fmax]\n        set argc 2\n        $orig\n        unset argv\n        unset argc"`
			new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
		else
			orig=`echo "source $::env(INTELFPGAOCLSDKROOT)/ip/board/bsp/adjust_plls.tcl"`
			orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
			if [[ -z `cat $file | grep "$orig"` ]]
			then
				echo "FAILURE!! (pattern not found)"
				exit -1
			fi

			new=`echo -e "set argv [list -fmax $fmax]\n  set argc 2\n  $orig\n  unset argv\n  unset argc"`
			new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
		fi

		sed -i "s/$orig_fix/$new_fix/" $file
		echo "SUCCESS!!"
	else
		file=`echo $path/scripts/post_flow_pr.tcl`
		if [[ $legacy -eq 1 ]]
		then
			orig=`echo "source $::env(ALTERAOCLSDKROOT)/ip/board/bsp/adjust_plls_a10.tcl"`
			orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
			if [[ -z `cat $file | grep "$orig"` ]]
			then
				echo "FAILURE!! (pattern not found)"
				exit -1
			fi

			new=`echo -e "set argv [list -fmax $fmax]\nset argc 2\n$orig\nunset argv\nunset argc"`
			new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
		else
			orig=`echo "source $::env(INTELFPGAOCLSDKROOT)/ip/board/bsp/adjust_plls_a10.tcl"`
			orig_fix=$(printf '%s\n' "$orig" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
			if [[ -z `cat $file | grep "$orig"` ]]
			then
				echo "FAILURE!! (pattern not found)"
				exit -1
			fi

			new=`echo -e "set argv [list -fmax $fmax]\nset argc 2\n$orig\nunset argv\nunset argc"`
			new_fix=$(printf '%s\n' "$new" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
		fi

		sed -i "s/$orig_fix/$new_fix/" $file
		echo "SUCCESS!!"
	fi
fi
