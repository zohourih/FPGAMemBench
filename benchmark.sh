#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

iter=5
size=1024
folder=de5net

echo Kernel | xargs printf "%-65s"
echo "Copy\ (GiB/s)" | xargs printf "%-15s"
echo "Mac\ (GiB/s)" | xargs printf "%-15s"
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	echo "$name" | xargs printf "%-65s"
	if [[ `echo $name | cut -d "_" -f 2` == NDR ]]
	then
		ndr=1
	else
		ndr=0
	fi
	VEC=`echo $name | cut -d "_" -f 3 | cut -c 4-`
	if [[ -n `echo $name | grep nointer` ]]
	then
		nointer=1
	else
		nointer=0
	fi
	#freq=`cat $folder/$name/acl_quartus_report.txt | grep Actual | cut -d " " -f 4`

	make clean >/dev/null 2>&1; make INTEL_FPGA=1 HOST_ONLY=1 VEC=$VEC NO_INTER=$nointer NDR=$ndr >/dev/null 2>&1
	rm fpga-stream-kernel.aocx >/dev/null 2>&1
	ln -s "$folder/$i" fpga-stream-kernel.aocx
	aocl program acl0 fpga-stream-kernel.aocx >/dev/null 2>&1

	out=`DEVICE_TYPE=FPGA ./fpga-stream -s $size -n $iter 2>&1`
	#echo "$out" >> ast.txt
	copy=`echo "$out" | grep "Copy:" | cut -d " " -f 2`
	mac=`echo "$out" | grep "MAC :" | cut -d " " -f 3`

	echo $copy | xargs printf "%-15s"
	echo $mac | xargs printf "%-15s"
	echo
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA