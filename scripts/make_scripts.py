import argparse
import os


GPUS_PER_NODE = 4

DATASETS=( "mouse_gene", "isolates_subgraph4", "kmer_A2a", "archaea", "eukarya", "ldoor", "kmer_V1r")
GROUPS = ( "Belcastro", "mcl", "GenBank", "mcl", "mcl", "GHS_psdef", "GenBank")
#DATASETS=( "HV15R", "nlpkkt160", "uk-2002")
#GROUPS = ( "Fluorem", "Schenk", "LAW")
#DATASETS=( "uk-2002", "nlpkkt240")
#GROUPS = ( "LAW", "Schenk")

SIZES = [10, 10, 10, 10, 10, 10, 10]

MAT_DIR = "/global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/known_squaring_nnz/"

GRIDS=("1x1", "2x2", "4x4", "8x8")
TWODGRIDS=("2x2", "4x4", "8x8", "16x16")
GRIDPROCS=("4", "16", "64", "256")

CONFIGURATIONS=[ "--impl async", "--impl summa"]
CONFIGURATIONS_STR=[ "kokkos_nospcomm_async_nopermute", "kokkos_nospcomm_summa_nopermute"]

RESULTS_DIR = "./results_final/"

GPU_KIND = '\"gpu\"'

def make_script_bcl(nodes, accum_thread=False):
    header = f"""#!/usr/bin/bash
#SBATCH -N {nodes}
#SBATCH --tasks-per-node {GPUS_PER_NODE}
#SBATCH --gpus-per-node {GPUS_PER_NODE}
#SBATCH -C {GPU_KIND}
#SBATCH -G {GPUS_PER_NODE*nodes}
#SBATCH -q regular
#SBATCH -t 0:10:00
#SBATCH -A m4646_g
module load cudatoolkit/12.4
export NVSHMEM_SYMMETRIC_SIZE=12G
    """

    os.makedirs("./scripts/sbatch_scripts/", exist_ok=True)

    for k, mat in enumerate(DATASETS):
        with open(f"./scripts/sbatch_scripts/bcl_strong_{mat}_{nodes}.sh", "w") as file:
            file.write(header + "\n")
            matpath = f"{MAT_DIR}/{GROUPS[k]}/{mat}/{mat}.mtx"
            fname = f"{RESULTS_DIR}/bcl_strong_{mat}_{GPUS_PER_NODE*nodes}"
            outfile = fname + ".out"
            errfile = fname + ".err"
            file.write(f"echo 'BCL {mat}'\n")
            cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {errfile} -o {outfile} ./bcl/examples/experimental/nvshmem/matrix/sparse_test {matpath} "
            file.write(f"{cmd}\n")

    with open(f"./scripts/sbatch_scripts/bcl_strong_all_{nodes}.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sbatch ./scripts/sbatch_scripts/bcl_strong_{mat}_{nodes}.sh\n")

    with open(f"./scripts/sbatch_scripts/bcl_strong_all_{nodes}_run.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sh ./scripts/sbatch_scripts/bcl_strong_{mat}_{nodes}.sh\n")


def make_script_hns(nodes, accum_thread=False):
    header = f"""#!/usr/bin/bash
#SBATCH -N {nodes}
#SBATCH --tasks-per-node {GPUS_PER_NODE}
#SBATCH --gpus-per-node {GPUS_PER_NODE}
#SBATCH -C {GPU_KIND}
#SBATCH -G {GPUS_PER_NODE*nodes}
#SBATCH -q regular
#SBATCH -t 0:10:00
#SBATCH -A m4646_g
    """

    if accum_thread:
        hns_name = "hns_accumthread"
        bin_name = "run_spgemm_accumthread"
    else:
        hns_name = "hns"
        bin_name = "run_spgemm"

    os.makedirs("./scripts/sbatch_scripts/", exist_ok=True)

    for k, mat in enumerate(DATASETS):
        with open(f"./scripts/sbatch_scripts/{hns_name}_strong_{mat}_{nodes}.sh", "w") as file:
            file.write(header + "\n")
            matpath = f"{MAT_DIR}/{GROUPS[k]}/{mat}/{mat}.bmtx"
            for i, conf in enumerate(CONFIGURATIONS):
                conf_str = CONFIGURATIONS_STR[i]

                if "summa" in conf_str:
                    grids = TWODGRIDS
                else:
                    grids = GRIDS

                for j, grid in enumerate(grids):
                    gridproc = GRIDPROCS[j]
                    if int(gridproc) != GPUS_PER_NODE*nodes:
                        continue
                    fname = f"{RESULTS_DIR}/{hns_name}_strong_{gridproc}_{grid}_{mat}_{conf_str}"
                    outfile = fname + ".out"
                    errfile = fname + ".err"
                    file.write(f"echo 'HnS {mat}, {conf_str}, {grid}'\n")
                    cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {errfile} -o {outfile} ./build/{bin_name} --matA {matpath} --matB {matpath} --2D-pgrid {grid} {conf} --c-size {SIZES[k]}"
                    file.write(f"{cmd}\n")

    with open(f"./scripts/sbatch_scripts/{hns_name}_strong_all_{nodes}.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sbatch ./scripts/sbatch_scripts/{hns_name}_strong_{mat}_{nodes}.sh\n")

    with open(f"./scripts/sbatch_scripts/{hns_name}_strong_all_{nodes}_run.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sh ./scripts/sbatch_scripts/{hns_name}_strong_{mat}_{nodes}.sh\n")


def make_script_trilinos(nodes, dummy):
    header = f"""#!/usr/bin/bash
#SBATCH -N {nodes}
#SBATCH --tasks-per-node {GPUS_PER_NODE}
#SBATCH --gpus-per-node {GPUS_PER_NODE}
#SBATCH -C {GPU_KIND}
#SBATCH -G {GPUS_PER_NODE*nodes}
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -A m4646_g
    """

    os.makedirs("./scripts/sbatch_scripts/", exist_ok=True)

    for k, mat in enumerate(DATASETS):
        with open(f"./scripts/sbatch_scripts/trilinos_strong_{mat}_{nodes}.sh", "w") as file:
            file.write(header + "\n")
            matpath = f"{MAT_DIR}/{GROUPS[k]}/{mat}/{mat}.bmtx"

            if mat == "cage15":
                matpath = "/global/cfs/projectdirs/m4646/hns_spgemm_matrices/known_squaring_nnz/vanHeukelum/cage15_/cage15.bmtx"

            gridproc = nodes * GPUS_PER_NODE

            fname = f"{RESULTS_DIR}/trilinos_strong_{mat}_{gridproc}"
            outfile = fname + ".out"
            errfile = fname + ".err"
            cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {errfile} -o {outfile} ./build/comparison/trilinos_spgemm --matA={matpath} --matB={matpath}"
            file.write(f"echo 'Trilinos {mat}, {gridproc}'\n")
            file.write(f"{cmd}\n")

            fname = f"{RESULTS_DIR}/trilinos_strong_{mat}_{gridproc}_permute"
            outfile = fname + ".out"
            errfile = fname + ".err"
            cmd = f"srun --gpus-per-node {GPUS_PER_NODE} -N {nodes} --tasks-per-node {GPUS_PER_NODE} -e {errfile} -o {outfile} ./build/comparison/trilinos_spgemm --matA={matpath} --matB={matpath} --randomize"
            file.write(f"echo 'Trilinos permute {mat}, {gridproc}'\n")
            file.write(f"{cmd}\n")

    with open(f"./scripts/sbatch_scripts/trilinos_strong_all_{nodes}.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sbatch ./scripts/sbatch_scripts/trilinos_strong_{mat}_{nodes}.sh\n")
        
    with open(f"./scripts/sbatch_scripts/trilinos_strong_all_{nodes}_run.sh", "w") as file:
        file.write("#!/usr/bin/bash\n")
        for mat in DATASETS:
            file.write(f"sh ./scripts/sbatch_scripts/trilinos_strong_{mat}_{nodes}.sh\n")

                 

def make_scripts(args):
    if args.impl == "hns":
        f = make_script_hns 
        t = False
    elif args.impl == "trilinos":
        f = make_script_trilinos
        t = False
    elif args.impl == "hns_accumthread":
        f = make_script_hns 
        t = True
    elif args.impl == "bcl":
        f = make_script_bcl
        t = False
    for nodes in args.nodes:
        f(nodes, t)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs='+')
    parser.add_argument("--impl", type=str)
    args = parser.parse_args()

    make_scripts(args)


