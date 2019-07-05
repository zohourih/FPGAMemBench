#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

iter=5
size=768
indexes=$(($size * 256 * 1024))
sqrt=`echo "e(l($indexes)/2)" | bc -l`
cbrt=`echo "e(l($indexes)/3)" | bc -l`
dim_x_2d=`echo "x=l($sqrt)/l(2); scale=0; if (x%1 != 0) x = (x + 1); x = (x / 1); 2 ^ x" | bc -l`
dim_y_2d=$(( $indexes / $dim_x_2d ))
dim_x_3d=`echo "x=l($cbrt)/l(2); scale=0; if (x%1 != 0) x = (x + 1); x = (x / 1); 2 ^ x" | bc -l`
dim_y_3d=$dim_x_3d
dim_z_3d=$(( $indexes / ($dim_x_3d * dim_y_3d) ))
size_switch=""
board=`aoc --list-boards | sed -n 2p | tr -d ' ' | cut -d "_" -f 1`
if [[ "$board" == "de5net" ]]
then
	max_bw="25.6"
elif [[ "$board" == "p385a" ]]
then
	max_bw="34.128"
fi
version=`aoc --version | grep Build | cut -d " " -f 2`
folder=`echo "$board"_"$version"`
verify=""
pad_start=0
pad_end=0
overlap=0
halo_start=0
halo_end=0
halo_step=1
overlap_switch=""
if [[ "$1" == "--verify" ]] || [[ "$2" == "--verify" ]] || [[ "$3" == "--verify" ]]
then
	verify="--verify"
fi
if [[ "$1" == "--pad" ]] || [[ "$2" == "--pad" ]] || [[ "$3" == "--pad" ]]
then
	pad_end=16
fi
if [[ "$1" == "--halo" ]] || [[ "$2" == "--halo" ]] || [[ "$3" == "--halo" ]]
then
	halo_end=16
fi

echo "Type" | xargs printf "%-9s"
echo "Model" | xargs printf "%-8s"
echo "Cache" | xargs printf "%-7s"
echo "Inter." | xargs printf "%-8s"
echo "VEC" | xargs printf "%-6s"
echo "Freq." | xargs printf "%-9s"
echo "Size" | xargs printf "%-15s"
echo "Pad" | xargs printf "%-6s"
echo "O\\\H" | xargs printf "%-6s"
echo "Copy\ (GB/s)" | xargs printf "%-13s"
echo "MAC\ (GB/s)" | xargs printf "%-12s"
echo "Copy\ Eff." | xargs printf "%-11s"
echo "MAC\ Eff." | xargs printf "%-8s"
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	type=`echo $name | cut -d "-" -f 4 | cut -d "_" -f 1`

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
	rm fpga-mem-bench-kernel.aocx >/dev/null 2>&1
	ln -s "$folder/$i" fpga-mem-bench-kernel.aocx
	aocl program acl0 fpga-mem-bench-kernel.aocx >/dev/null 2>&1

	for ((halo = $halo_start ; halo <= $halo_end ; halo += $halo_step))
	do
		compute_bsize_x=$(( $BSIZE - (2 * $halo) ))
		compute_bsize_y=$(( $BSIZE - (2 * $halo) ))
		dim_x_2d_aligned=$(( ($dim_x_2d % $compute_bsize_x) == 0 ? $dim_x_2d : ($dim_x_2d + $compute_bsize_x - $dim_x_2d % $compute_bsize_x) ))
		dim_x_3d_aligned=$(( ($dim_x_3d % $compute_bsize_x) == 0 ? $dim_x_3d : ($dim_x_3d + $compute_bsize_x - $dim_x_3d % $compute_bsize_x) ))
		dim_y_3d_aligned=$(( ($dim_y_3d % $compute_bsize_y) == 0 ? $dim_y_3d : ($dim_y_3d + $compute_bsize_y - $dim_y_3d % $compute_bsize_y) ))

		if [[ "$type" == "std" ]]
		then
			overlap_switch="-o $overlap"
			size_switch="-s $size"
			dim=$size
		elif [[ "$type" == "blk2d" ]] || [[ "$type" == "chblk2d" ]]
		then
			overlap_switch="-hw $halo"
			size_switch="-x $dim_x_2d_aligned -y $dim_y_2d"
			dim=`echo "$dim_x_2d_aligned"x"$dim_y_2d"`
		elif [[ "$type" == "blk3d" ]] || [[ "$type" == "chblk3d" ]]
		then
			overlap_switch="-hw $halo"
			size_switch="-x $dim_x_3d_aligned -y $dim_y_3d_aligned -z $dim_z_3d"
			dim=`echo "$dim_x_3d_aligned"x"$dim_y_3d_aligned"x"$dim_z_3d"`
		else
			size_switch="-s $size"
			dim=$size
		fi

		for ((pad = $pad_start ; pad <= $pad_end ; pad++))
		do
			out=`DEVICE_TYPE=FPGA ./fpga-mem-bench $size_switch -n $iter -pad $pad $overlap_switch $verify 2>&1`
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
			echo $cache | xargs printf "%-7s"
			echo $inter | xargs printf "%-8s"
			echo $VEC | xargs printf "%-6s"
			echo $freq | xargs printf "%-9s"
			echo $dim | xargs printf "%-15s"
			echo $pad | xargs printf "%-6s"
			echo $halo_overlap | xargs printf "%-6s"
			if [[ "$verify" == "--verify" ]]
			then
				echo "$copy\ ($copy_ver)" | xargs printf "%-13s"
				echo "$mac\ ($mac_ver)" | xargs printf "%-12s"
			else
				echo $copy | xargs printf "%-13s"
				echo $mac | xargs printf "%-12s"
			fi
			echo "$copy_eff%" | xargs printf "%-11s"
			echo "$mac_eff%" | xargs printf "%-8s"
			echo
		done
	done
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA