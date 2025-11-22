import argparse
import os


GPUS_PER_NODE = 4

DATASETS=( "HV15R", "mouse_gene", "nlpkkt160", "cage15", "isolates_subgraph4", "uk-2002", "archaea")
GROUPS = ( "Fluorem", "Belcastro", "Schenk", "vanHeukelum", "mcl", "LAW", "mcl")

MAT_DIR = "/global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/known_squaring_nnz/"

GRIDS=("4x4", "2x2", "8x8", "4x4")

GRIDPROCS=("16", "16", "64", "64")

CONFIGURATIONS=("--impl sendrecv ", "--impl sendrecv --Acsc --spcomm ")

CONFIGURATIONS_STR=( "kokkos_nospcomm", "kokkos_spcomm")

RESULTS_DIR = "./results_wave4/"



def make_script_hns(nodes):
    header = f"""#!/usr/bin/bash
#SBATCH -N {nodes}
#SBATCH --tasks-per-node {GPUS_PER_NODE}
#SBATCH --gpus-per-node {GPUS_PER_NODE}
#SBATCH -C "gpu&hbm80g"
#SBATCH -G {GPUS_PER_NODE*nodes}
#SBATCH -q regular
#SBATCH -t 0:10:00
#SBATCH -A m4646_g
    """


    os.makedirs("./scripts/sbatch_scripts/", exist_ok=True)

    for k, mat in enumerate(DATASETS):
        with open(f"./scripts/sbatch_scripts/hns_strong_{mat}_{nodes}.sh", "w") as file:
            file.write(header + "\n")
            matpath = f"{MAT_DIR}/{GROUPS[k]}/{mat}/{mat}.bmtx"
            for i, conf in enumerate(CONFIGURATIONS):
                conf_str = CONFIGURATIONS_STR[i]
                for j, grid in enumerate(GRIDS):
                    gridproc = GRIDPROCS[j]
                    if int(gridproc) != GPUS_PER_NODE*nodes:
                        continue
                    cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {RESULTS_DIR}/hns_strong_{mat}_{nodes}.err ./build/run_spgemm --matA {matpath} --matB {matpath} --2D-pgrid {grid} {conf}"
                    outfile = f"{RESULTS_DIR}/hns_{gridproc}_{grid}_{mat}_{conf_str}.out"
                    file.write(f"echo 'HnS {mat}, {conf_str}, {grid}'\n")
                    file.write(f"{cmd} > {outfile}\n")

    with open(f"./scripts/sbatch_scripts/hns_strong_all_{nodes}.sh", "w") as file:
        for mat in DATASETS:
            file.write(f"sbatch ./scripts/sbatch_scripts/hns_strong_{mat}_{nodes}.sh\n")


def make_script_trilinos(nodes):
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

    for k, mat in enumerate(DATASETS):
        with open(f"./scripts/sbatch_scripts/trilinos_strong_{mat}_{nodes}.sh", "w") as file:
            file.write(header + "\n")
            matpath = f"{MAT_DIR}/{GROUPS[k]}/{mat}/{mat}.mtx"
            gridproc = nodes * GPUS_PER_NODE
            cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {RESULTS_DIR}/trilinos_strong_{mat}_{nodes}.err ./build/comparison/trilinos_spgemm --matA={matpath} --matB={matpath}"
            outfile = f"{RESULTS_DIR}/trilinos_{gridproc}_{mat}.out"
            file.write(f"echo 'Trilinos {mat}, {gridproc}'\n")
            file.write(f"{cmd} > {outfile}\n")

    with open(f"./scripts/sbatch_scripts/trilinos_strong_all_{nodes}.sh", "w") as file:
        for mat in DATASETS:
            file.write(f"sbatch ./scripts/sbatch_scripts/trilinos_strong_{mat}_{nodes}.sh\n")
        

                 

def make_scripts(args):
    if args.impl == "hns":
        f = make_script_hns 
    elif args.impl == "trilinos":
        f = make_script_trilinos
    for nodes in args.nodes:
        f(nodes)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs='+')
    parser.add_argument("--impl", type=str)
    args = parser.parse_args()

    make_scripts(args)


