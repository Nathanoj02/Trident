#!/bin/bash

datasets=("cage15" "dielFilterV3real" "HV15R" "ldoor" "nlpkkt160")

# grids=("2x2" "4x4")
# gridproc=("4" "16")
grids=("4x4")
gridproc=("16")

#configurations=("--impl get" "--impl main" "--impl main --Acsc" "--impl main --Acsc --spcomm")
#configurations_str=("get_none_none" "main_none_none" "main_Acsc_none" "main_Acsc_spcomm")
configurations=("--impl sendrecv --skip-spgemm" "--impl sendrecv --Acsc --skip-spgemm" "--impl sendrecv --Acsc --spcomm --skip-spgemm")
configurations_str=("sendrecv_none_none_skipspgemm" "sendrecv_Acsc_none_skipspgemm" "sendrecv_Acsc_spcomm_skipspgemm")

for mat in ${datasets[@]}
do
	#matpath=$(./build/name2path.sh build/name2path_list.txt ${mat})
	matpath=$(ls /global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/known_squaring_nnz/*/*/${mat}.bmtx | head -1)
	for i in ${!configurations[@]}
	do
		conf=${configurations[$i]}
                confstr=${configurations_str[$i]}
		for j in ${!grids[@]}
		do
			grid=${grids[$j]}
			nproc=${gridproc[$j]}
			cmd="srun --ntasks=${nproc} --gpus=${nproc} ./build/run_spgemm --matA ${matpath} --matB ${matpath} --2D-pgrid ${grid} ${conf} "
			outfile="hns_${grid}x1_${mat}_${confstr}.out"
			echo "cmd: ${cmd}"
			echo "outfile: ${outfile}"
			${cmd} > ${outfile}
		done
	done
done
