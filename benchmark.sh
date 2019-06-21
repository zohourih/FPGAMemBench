#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

iter=10
size=768
dim_x_2d=24576
dim_y_2d=8192
dim_x_3d=1024
dim_y_3d=1024
dim_z_3d=192
size_switch=""
board=`aoc --list-boards | sed -n 2p | tr -d ' ' | cut -d "_" -f 1`
if [[ "$board" == "de5net" ]]
then
	max_bw="25.6"
	mem_freg="1600"
elif [[ "$board" == "p385a" ]]
then
	max_bw="34.128"
	mem_freg="2133"
fi
version=`aoc --version | grep Build | cut -d " " -f 2`
folder=`echo "$board"_"$version"`
verify=""
pad_start=0
pad_end=0
overlap=0
halo=0
overlap_switch=""
if [[ "$1" == "--verify" ]] || [[ "$2" == "--verify" ]]
then
	verify="--verify"
fi
if [[ "$1" == "--pad" ]] || [[ "$2" == "--pad" ]]
then
	pad_end=16
fi

echo "Type" | xargs printf "%-9s"
echo "Model" | xargs printf "%-8s"
echo "Cache" | xargs printf "%-8s"
echo "Inter." | xargs printf "%-9s"
echo "Vector" | xargs printf "%-9s"
echo "Freq." | xargs printf "%-9s"
echo "Pad" | xargs printf "%-6s"
echo "Overlap" | xargs printf "%-10s"
echo "Copy\ (GB/s)" | xargs printf "%-14s"
echo "MAC\ (GB/s)" | xargs printf "%-13s"
echo "Copy\ Eff." | xargs printf "%-12s"
echo "MAC\ Eff." | xargs printf "%-8s"
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	type=`echo $name | cut -d "-" -f 4 | cut -d "_" -f 1`

	if [[ "$type" == "std" ]]
	then
		overlap_switch="-o $overlap"
		size_switch="-s $size"
	elif [[ "$type" == "blk2d" ]] || [[ "$type" == "chblk2d" ]]
	then
		overlap_switch="-hw $halo"
		size_switch="-x $dim_x_2d -y $dim_y_2d"
	elif [[ "$type" == "blk3d" ]] || [[ "$type" == "chblk3d" ]]
	then
		overlap_switch="-hw $halo"
		size_switch="-x $dim_x_3d -y $dim_y_3d -z $dim_z_3d"
	else
		size_switch="-s $size"
	fi

	if [[ "$type" == "blk3d" ]] || [[ "$type" == "chblk3d" ]]
	then
		BSIZE=256
	else
		BSIZE=1024
	fi

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

	make clean >/dev/null 2>&1; make $type INTEL_FPGA=1 HOST_ONLY=1 VEC=$VEC BSIZE=$BSIZE NO_INTER=$nointer NDR=$ndr >/dev/null 2>&1
	rm fpga-stream-kernel.aocx >/dev/null 2>&1
	ln -s "$folder/$i" fpga-stream-kernel.aocx
	aocl program acl0 fpga-stream-kernel.aocx >/dev/null 2>&1

	for ((pad = $pad_start ; pad <= $pad_end ; pad++))
	do
		out=`DEVICE_TYPE=FPGA ./fpga-stream $size_switch -n $iter -pad $pad $overlap_switch $verify 2>&1`
		#echo "$out" >> ast.txt

		copy=`echo "$out" | grep "Copy:" | cut -d " " -f 2`
		mac=`echo "$out" | grep "MAC :" | cut -d " " -f 3`

		bw_copy=`echo "$freq * $VEC * 4 * 2 / 1000" | bc -l | xargs printf %0.3f`
		if [[ `echo "$bw_copy > $max_bw" | bc -l` -eq 1 ]]
		then
			bw_copy=$max_bw
		fi
		bw_mac=`echo "$freq * $VEC * 4 * 3 / 1000" | bc -l | xargs printf %0.3f`
		if [[ `echo "$bw_mac > $max_bw" | bc -l` -eq 1 ]]
		then
			bw_mac=$max_bw
		fi
		copy_eff=`echo "100 * ($copy/$bw_copy)" | bc -l | xargs printf %0.1f`
		mac_eff=`echo "100 * ($mac/$bw_mac)" | bc -l | xargs printf %0.1f`

		copy_ver=`echo "$out" | grep Verify | grep Copy | cut -d " " -f 4 | cut -c 1-1`
		mac_ver=`echo "$out" | grep Verify | grep MAC | cut -d " " -f 4 | cut -c 1-1`

		if [[ -n `echo "$out" | grep Halo` ]]
		then
			halo_overlap=`echo "$out" | grep "Halo" | tr -s " " | cut -d " " -f 3`
		elif [[ -n `echo "$out" | grep Overlap` ]]
		then
			halo_overlap=`echo "$out" | grep "Overlap" | tr -s " " | cut -d " " -f 2`
		else
			halo_overlap="N/A"
		fi

		echo $type  | tr '[:lower:]' '[:upper:]' | xargs printf "%-9s"
		echo $model | xargs printf "%-8s"
		echo $cache | xargs printf "%-8s"
		echo $inter | xargs printf "%-9s"
		echo $VEC | xargs printf "%-9s"
		echo $freq | xargs printf "%-9s"
		echo $pad | xargs printf "%-6s"
		echo $halo_overlap | xargs printf "%-10s"
		if [[ "$verify" == "--verify" ]]
		then
			echo "$copy\ ($copy_ver)" | xargs printf "%-15s"
			echo "$mac\ ($mac_ver)" | xargs printf "%-15s"
		else
			echo $copy | xargs printf "%-14s"
			echo $mac | xargs printf "%-13s"
		fi
		echo "$copy_eff%" | xargs printf "%-12s"
		echo "$mac_eff%" | xargs printf "%-8s"
		echo
	done
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA