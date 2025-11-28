#!/bin/bash

datasets=("cage15" "dielFilterV3real" "HV15R" "ldoor" "nlpkkt160")

grids=("2x2")
gridproc=("16")
nreps=("4")
#grids=("4x4")
#gridproc=("16")

configurations=("--impl sendrecv --Acsc --spcomm")
configurations_str=("sendrecv_Acsc_spcomm_none")

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
			nrep=${nreps[$j]}
			cmd="srun --ntasks=${nproc} --gpus=${nproc} ./build/run_spgemm --matA ${matpath} --matB ${matpath} --2D-pgrid ${grid} ${conf} "
			outfile="hns_${grid}x${nrep}_${mat}_${confstr}.out"
			echo "cmd: ${cmd}"
			echo "outfile: ${outfile}"
			${cmd} > ${outfile}
		done
	done
done
