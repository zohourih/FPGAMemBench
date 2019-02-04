#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

iter=5
size=1024
folder=de5net
verify=""
pad_end=1
if [[ "$1" == "--verify" ]] || [[ "$2" == "--verify" ]]
then
	verify="--verify"
fi
if [[ "$1" == "--pad" ]] || [[ "$2" == "--pad" ]]
then
	pad_end=32
fi

echo "Type" | xargs printf "%-8s"
echo "Model" | xargs printf "%-8s"
echo "Cache" | xargs printf "%-8s"
echo "Interleave" | xargs printf "%-12s"
echo "Vector" | xargs printf "%-8s"
echo "Frequency" | xargs printf "%-12s"
echo "Padding" | xargs printf "%-10s"
echo "Copy\ (GiB/s)" | xargs printf "%-15s"
echo "MAC\ (GiB/s)" | xargs printf "%-15s"
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	type=`echo $name | cut -d "-" -f 4 | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]'`
	if [[ `echo $name | cut -d "_" -f 2` == NDR ]]
	then
		ndr=1
		model=NDR
	else
		ndr=0
		model=SWI
	fi
	VEC=`echo $name | cut -d "_" -f 3 | cut -c 4-`
	if [[ -n `echo $name | grep nointer` ]]
	then
		nointer=1
		inter=N
	else
		nointer=0
		inter=Y
	fi
	if [[ -n `echo $name | grep nocache` ]]
	then
		cache=N
	else
		cache=Y
	fi
	freq=`cat $folder/$name/acl_quartus_report.txt | grep Actual | cut -d " " -f 4 | xargs printf %0.2f`

	make clean >/dev/null 2>&1; make INTEL_FPGA=1 HOST_ONLY=1 VEC=$VEC NO_INTER=$nointer NDR=$ndr >/dev/null 2>&1
	rm fpga-stream-kernel.aocx >/dev/null 2>&1
	ln -s "$folder/$i" fpga-stream-kernel.aocx
	aocl program acl0 fpga-stream-kernel.aocx >/dev/null 2>&1

	for ((pad = 0 ; pad < $pad_end ; pad++))
	do
		out=`DEVICE_TYPE=FPGA ./fpga-stream -s $size -n $iter -p $pad $verify 2>&1`
		#echo "$out" >> ast.txt
		copy=`echo "$out" | grep "Copy:" | cut -d " " -f 2`
		mac=`echo "$out" | grep "MAC :" | cut -d " " -f 3`
		copy_ver=`echo "$out" | grep Verify | grep Copy | cut -d " " -f 4 | cut -c 1-1`
		mac_ver=`echo "$out" | grep Verify | grep MAC | cut -d " " -f 4 | cut -c 1-1`

		echo $type | xargs printf "%-8s"
		echo $model | xargs printf "%-8s"
		echo $cache | xargs printf "%-8s"
		echo $inter | xargs printf "%-12s"
		echo $VEC | xargs printf "%-8s"
		echo $freq | xargs printf "%-12s"
		echo $pad | xargs printf "%-10s"
		if [[ "$verify" == "--verify" ]]
		then
			echo "$copy\ ($copy_ver)" | xargs printf "%-15s"
			echo "$mac\ ($mac_ver)" | xargs printf "%-15s"
		else
			echo $copy | xargs printf "%-15s"
			echo $mac | xargs printf "%-15s"
		fi
		echo
	done
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA