import argparse
import os


GPUS_PER_NODE = 4

DATASETS=( "HV15R", "mouse_gene", "nlpkkt160", "cage15", "isolates_subgraph4", "uk-2002")

MAT_DIR = "/global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/known_squaring_nnz/"

GRIDS=("4x4", "2x2", "8x8", "4x4")

GRIDPROCS=("16", "16", "64", "64")

CONFIGURATIONS=("--impl sendrecv --Acsc --mem-efficient", "--impl sendrecv --Acsc --spcomm --mem-efficient", "--impl sendrecv", "--impl sendrecv --Acsc --spcomm")

CONFIGURATIONS_STR=("memeff_nospcomm", "memeff_spcomm", "reuse_nospcomm", "reuse_spcomm")

RESULTS_DIR = "./results_wave2/"



def make_script(nodes):
    header = f"""#!/usr/bin/bash
#SBATCH -N {nodes}
#SBATCH --tasks-per-node {GPUS_PER_NODE}
#SBATCH --gpus-per-node {GPUS_PER_NODE}
#SBATCH -C "gpu&hbm80g"
#SBATCH -G {GPUS_PER_NODE*nodes}
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -A m4646_g

    """

    os.makedirs("./scripts/sbatch_scripts/", exist_ok=True)

    with open(f"./scripts/sbatch_scripts/strong_{nodes}.sh", "w") as file:
        file.write(header + "\n")
        for mat in DATASETS:
            matpath = f"{MAT_DIR}/{mat}.bmtx"
            for i, conf in enumerate(CONFIGURATIONS):
                conf_str = CONFIGURATIONS_STR[i]
                for j, grid in enumerate(GRIDS):
                    gridproc = GRIDPROCS[j]
                    if int(gridproc) != GPUS_PER_NODE*nodes:
                        continue
                    cmd = f"srun ./build/run_spgemm --matA {matpath} --matB {matpath} --2D-pgrid {grid} {conf}"
                    outfile = f"results_wave3/hns_{gridproc}_{grid}_{mat}_{conf_str}.out"
                    file.write(f"{cmd} > {outfile}\n")
                 

def make_scripts(args):
    for nodes in args.nodes:
        make_script(nodes)




if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs='+')
    args = parser.parse_args()

    make_scripts(args)



