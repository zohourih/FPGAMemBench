#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

iter=5
size=1024
indexes=$(($size * 256 * 1024))
sqrt=`echo "e(l($indexes)/2)" | bc -l`
cbrt=`echo "e(l($indexes)/3)" | bc -l`
dim_x_2d=`echo "x=l($sqrt)/l(2); scale=0; if (x%1 != 0) x = (x + 1); x = (x / 1); 2 ^ x" | bc -l`
dim_y_2d=$(( $indexes / $dim_x_2d ))
dim_x_3d=`echo "x=l($cbrt)/l(2); scale=0; if (x%1 != 0) x = (x + 1); x = (x / 1); 2 ^ x" | bc -l`
dim_y_3d=$dim_x_3d
dim_z_3d=$(( $indexes / ($dim_x_3d * dim_y_3d) ))
size_switch=""
board=`aoc --list-boards | grep Board -A 2 | sed -n 2p | tr -d ' ' | cut -d "_" -f 1`
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
halo_switch=""
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
echo "Model" | xargs printf "%-7s"
echo "Cache" | xargs printf "%-7s"
echo "Inter." | xargs printf "%-8s"
echo "VEC" | xargs printf "%-5s"
echo "Freq." | xargs printf "%-9s"
echo "Size" | xargs printf "%-15s"
echo "Pad" | xargs printf "%-6s"
echo "Halo" | xargs printf "%-6s"
echo "Performance\ (GB/s)" | xargs printf "%-36s"
echo "Efficiency\ (%)" | xargs printf "%-26s"
if [[ "$verify" == "--verify" ]]
then
	echo "Verification" | xargs printf "%-11s"
fi
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	type=`echo $name | cut -d "-" -f 5 | cut -d "_" -f 1`

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

		if [[ "$type" == "std" ]] || [[ "$type" == "chstd" ]]
		then
			halo_switch="-hw $halo"
			size_switch="-s $size"
			dim=$size
		elif [[ "$type" == "blk2d" ]] || [[ "$type" == "chblk2d" ]]
		then
			halo_switch="-hw $halo"
			size_switch="-x $dim_x_2d_aligned -y $dim_y_2d"
			dim=`echo "$dim_x_2d_aligned"x"$dim_y_2d"`
		elif [[ "$type" == "blk3d" ]] || [[ "$type" == "chblk3d" ]]
		then
			halo_switch="-hw $halo"
			size_switch="-x $dim_x_3d_aligned -y $dim_y_3d_aligned -z $dim_z_3d"
			dim=`echo "$dim_x_3d_aligned"x"$dim_y_3d_aligned"x"$dim_z_3d"`
		else
			size_switch="-s $size"
			dim=$size
		fi

		for ((pad = $pad_start ; pad <= $pad_end ; pad++))
		do
			out=`DEVICE_TYPE=FPGA ./fpga-mem-bench $size_switch -n $iter -pad $pad $halo_switch $verify 2>&1`
			#echo "$out" >> ast.txt

			R1W1=`echo "$out" | grep "R1W1:" | cut -d " " -f 2`
			R2W1=`echo "$out" | grep "R2W1:" | cut -d " " -f 2`
			R3W1=`echo "$out" | grep "R3W1:" | cut -d " " -f 2`
			R2W2=`echo "$out" | grep "R2W2:" | cut -d " " -f 2`

			bw_R1W1=`echo "$freq * $VEC * 4 * 2 / 1000" | bc -l | xargs printf %0.3f`
			if [[ `echo "$bw_R1W1 > $max_bw" | bc -l` -eq 1 ]]
			then
				bw_R1W1=$max_bw
			fi
			bw_R2W1=`echo "$freq * $VEC * 4 * 3 / 1000" | bc -l | xargs printf %0.3f`
			if [[ `echo "$bw_R2W1 > $max_bw" | bc -l` -eq 1 ]]
			then
				bw_R2W1=$max_bw
			fi
			bw_R3W1=`echo "$freq * $VEC * 4 * 4 / 1000" | bc -l | xargs printf %0.3f`
			if [[ `echo "$bw_R3W1 > $max_bw" | bc -l` -eq 1 ]]
			then
				bw_R3W1=$max_bw
			fi
			bw_R2W2=`echo "$freq * $VEC * 4 * 4 / 1000" | bc -l | xargs printf %0.3f`
			if [[ `echo "$bw_R2W2 > $max_bw" | bc -l` -eq 1 ]]
			then
				bw_R2W2=$max_bw
			fi

			R1W1_eff=`echo "100 * ($R1W1/$bw_R1W1)" | bc -l | xargs printf %0.1f`
			R2W1_eff=`echo "100 * ($R2W1/$bw_R2W1)" | bc -l | xargs printf %0.1f`
			R3W1_eff=`echo "100 * ($R3W1/$bw_R3W1)" | bc -l | xargs printf %0.1f`
			R2W2_eff=`echo "100 * ($R2W2/$bw_R2W2)" | bc -l | xargs printf %0.1f`

			R1W1_ver=`echo "$out" | grep Verify | grep R1W1 | cut -d " " -f 4 | cut -c 1-1`
			R2W1_ver=`echo "$out" | grep Verify | grep R2W1 | cut -d " " -f 4 | cut -c 1-1`
			R3W1_ver=`echo "$out" | grep Verify | grep R3W1 | cut -d " " -f 4 | cut -c 1-1`
			R2W2_ver=`echo "$out" | grep Verify | grep R2W2 | cut -d " " -f 4 | cut -c 1-1`

			if [[ "$type" == "std" ]] || [[ "$type" == "chstd" ]]
			then
				R1W0=`echo "$out" | grep "R1W0:" | cut -d " " -f 2`

				bw_R1W0=`echo "$freq * $VEC * 4 * 1 / 1000" | bc -l | xargs printf %0.3f`
				if [[ `echo "$bw_R1W0 > $max_bw" | bc -l` -eq 1 ]]
				then
					bw_R1W0=$max_bw
				fi

				R1W0_eff=`echo "100 * ($R1W0/$bw_R1W0)" | bc -l | xargs printf %0.1f`

				R1W0_ver=`echo "N/A"`
			fi

			if [[ "$type" != "sch" ]]
			then
				halo_size=$halo
			else
				halo_size="N/A"
			fi

			echo $type  | tr '[:lower:]' '[:upper:]' | xargs printf "%-9s"
			echo $model | xargs printf "%-7s"
			echo $cache | xargs printf "%-7s"
			echo $inter | xargs printf "%-8s"
			echo $VEC | xargs printf "%-5s"
			echo $freq | xargs printf "%-9s"
			echo $dim | xargs printf "%-15s"
			echo $pad | xargs printf "%-6s"
			echo $halo_size | xargs printf "%-6s"
			if [[ "$type" == "std" ]] || [[ "$type" == "chstd" ]]
			then
				echo "$R1W0|$R1W1|$R2W1|$R3W1|$R2W2" | xargs printf "%-36s"
				echo "$R1W0_eff|$R1W1_eff|$R2W1_eff|$R3W1_eff|$R2W2_eff" | xargs printf "%-26s"
				if [[ "$verify" == "--verify" ]]
				then
					echo "$R1W0_ver|$R1W1_ver|$R2W1_ver|$R3W1_ver|$R2W2_ver" | xargs printf "%-11s"
				fi
			else
				echo "$R1W1|$R2W1|$R3W1|$R2W2" | xargs printf "%-36s"
				echo "$R1W1_eff|$R2W1_eff|$R3W1_eff|$R2W2_eff" | xargs printf "%-26s"
				if [[ "$verify" == "--verify" ]]
				then
					echo "$R1W1_ver|$R2W1_ver|$R3W1_ver|$R2W2_ver" | xargs printf "%-11s"
				fi
			fi
			echo
		done
	done
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA