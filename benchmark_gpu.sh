#!/bin/bash

gpu=V100
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
verify=""
pad_start=0
pad_end=0
pad_array=(0)
overlap=0
halo_start=0
halo_end=0
halo_step=1
halo_switch=""
halo_array=(0)
if [[ "$1" == "--verify" ]] || [[ "$2" == "--verify" ]] || [[ "$3" == "--verify" ]]
then
	verify="--verify"
fi
if [[ "$1" == "--pad" ]] || [[ "$2" == "--pad" ]] || [[ "$3" == "--pad" ]]
then
	pad_end=32
	pad_array=(0 1 2 4 8 16 32)
fi
if [[ "$1" == "--halo" ]] || [[ "$2" == "--halo" ]] || [[ "$3" == "--halo" ]]
then
	halo_end=32
	halo_array=(0 1 2 4 8 16 32)
fi

echo "Type" | xargs printf "%-9s"
echo "Size" | xargs printf "%-15s"
echo "Pad" | xargs printf "%-6s"
echo "Halo" | xargs printf "%-6s"
echo "Redundancy" | xargs printf "%-12s"
echo "Performance\ (GB/s)" | xargs printf "%-41s"
echo "Efficiency\ (%)" | xargs printf "%-31s"
if [[ "$verify" == "--verify" ]]
then
	echo "Verification" | xargs printf "%-11s"
fi
echo

type=std
ndr=1
VEC=1
if [[ $gpu == "V100" ]]
then
	BSIZE=512
	gpu_id=1
	max_bw=897.0
else
	BSIZE=128
	gpu_id=2
	max_bw=249.6
fi

make clean >/dev/null 2>&1; make $type NVIDIA=1 VEC=$VEC BSIZE=$BSIZE NDR=$ndr >/dev/null 2>&1

#for ((halo = $halo_start ; halo <= $halo_end ; halo += $halo_step))
for halo in "${halo_array[@]}"
do
	compute_bsize_x=$(( $BSIZE - (2 * $halo) ))
	compute_bsize_y=$(( $BSIZE - (2 * $halo) ))
	dim_x_2d_aligned=$(( ($dim_x_2d % $compute_bsize_x) == 0 ? $dim_x_2d : ($dim_x_2d + $compute_bsize_x - $dim_x_2d % $compute_bsize_x) ))
	dim_x_3d_aligned=$(( ($dim_x_3d % $compute_bsize_x) == 0 ? $dim_x_3d : ($dim_x_3d + $compute_bsize_x - $dim_x_3d % $compute_bsize_x) ))
	dim_y_3d_aligned=$(( ($dim_y_3d % $compute_bsize_y) == 0 ? $dim_y_3d : ($dim_y_3d + $compute_bsize_y - $dim_y_3d % $compute_bsize_y) ))

	if [[ "$type" == "std" ]]
	then
		halo_switch="-hw $halo"
		size_switch="-s $size"
		dim=$size
	elif [[ "$type" == "blk2d" ]]
	then
		halo_switch="-hw $halo"
		size_switch="-x $dim_x_2d_aligned -y $dim_y_2d"
		dim=`echo "$dim_x_2d_aligned"x"$dim_y_2d"`
	elif [[ "$type" == "blk3d" ]]
	then
		halo_switch="-hw $halo"
		size_switch="-x $dim_x_3d_aligned -y $dim_y_3d_aligned -z $dim_z_3d"
		dim=`echo "$dim_x_3d_aligned"x"$dim_y_3d_aligned"x"$dim_z_3d"`
	else
		size_switch="-s $size"
		dim=$size
	fi

	#for ((pad = $pad_start ; pad <= $pad_end ; pad++))
	for pad in "${pad_array[@]}"
	do
		out=`DEVICE_TYPE=GPU ./fpga-mem-bench $size_switch -n $iter -pad $pad $halo_switch -id $gpu_id $verify 2>&1`
		#echo "$out" >> ast.txt

		redundancy=`echo "$out" | grep "Redundancy:" | cut -d " " -f 2`

		R1W1=`echo "$out" | grep "R1W1:" | cut -d " " -f 2`
		R2W1=`echo "$out" | grep "R2W1:" | cut -d " " -f 2`
		R3W1=`echo "$out" | grep "R3W1:" | cut -d " " -f 2`
		R2W2=`echo "$out" | grep "R2W2:" | cut -d " " -f 2`

		R1W1_eff=`echo "100 * ($R1W1/$max_bw)" | bc -l | xargs printf %0.1f`
		R2W1_eff=`echo "100 * ($R2W1/$max_bw)" | bc -l | xargs printf %0.1f`
		R3W1_eff=`echo "100 * ($R3W1/$max_bw)" | bc -l | xargs printf %0.1f`
		R2W2_eff=`echo "100 * ($R2W2/$max_bw)" | bc -l | xargs printf %0.1f`

		R1W1_ver=`echo "$out" | grep Verify | grep R1W1 | cut -d " " -f 4 | cut -c 1-1`
		R2W1_ver=`echo "$out" | grep Verify | grep R2W1 | cut -d " " -f 4 | cut -c 1-1`
		R3W1_ver=`echo "$out" | grep Verify | grep R3W1 | cut -d " " -f 4 | cut -c 1-1`
		R2W2_ver=`echo "$out" | grep Verify | grep R2W2 | cut -d " " -f 4 | cut -c 1-1`

		if [[ "$type" == "std" ]]
		then
			R1W0=`echo "$out" | grep "R1W0:" | cut -d " " -f 2`

			R1W0_eff=`echo "100 * ($R1W0/$max_bw)" | bc -l | xargs printf %0.1f`

			R1W0_ver=`echo "N/A"`
		fi

		echo $type  | tr '[:lower:]' '[:upper:]' | xargs printf "%-9s"
		echo $dim | xargs printf "%-15s"
		echo $pad | xargs printf "%-6s"
		echo $halo | xargs printf "%-6s"
		echo $redundancy | xargs printf "%-12s"
		if [[ "$type" == "std" ]]
		then
			echo "$R1W0|$R1W1|$R2W1|$R3W1|$R2W2" | xargs printf "%-41s"
			echo "$R1W0_eff|$R1W1_eff|$R2W1_eff|$R3W1_eff|$R2W2_eff" | xargs printf "%-31s"
			if [[ "$verify" == "--verify" ]]
			then
				echo "$R1W0_ver|$R1W1_ver|$R2W1_ver|$R3W1_ver|$R2W2_ver" | xargs printf "%-11s"
			fi
		else
			echo "$R1W1|$R2W1|$R3W1|$R2W2" | xargs printf "%-41s"
			echo "$R1W1_eff|$R2W1_eff|$R3W1_eff|$R2W2_eff" | xargs printf "%-31s"
			if [[ "$verify" == "--verify" ]]
			then
				echo "$R1W1_ver|$R2W1_ver|$R3W1_ver|$R2W2_ver" | xargs printf "%-11s"
			fi
		fi
		echo
	done
done
