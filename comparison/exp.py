import argparse
import pandas
import numpy
import os
import urllib.request
import tarfile
import shutil



NODES = [1, 4, 16]
FLAGS = {"run_spgemm": [["--impl get"],
                       ["--impl main"],
                       ["--Acsc", "--impl main"]],
         "trilinos_spgemm": [' ']
        } 
EXECUTABLES = ["run_spgemm", "trilinos_spgemm"]



def get_matrices(matrix_list_file):
    matrices = []
    with open(matrix_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                matrices.append(line)
    return matrices


def download_matrix(group_matrix, matrices_dir):
    group, matrix_name = group_matrix.split('/')
    matrix_dir = os.path.join(matrices_dir, f"{matrix_name}")

    # Skip if already downloaded
    if os.path.exists(matrix_dir):
        print(f"Matrix {matrix_name} already exists, skipping")
        return

    # Create matrix directory
    #os.makedirs(matrix_dir, exist_ok=True)
    print(matrix_dir)

    # SuiteSparse URL format
    url = f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{matrix_name}.tar.gz"
    tar_path = os.path.join(f"{matrix_name}.tar.gz")

    try:
        print(f"Downloading {group_matrix}...")
        urllib.request.urlretrieve(url, tar_path)

        # Extract the tar.gz file
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(matrices_dir)

        # Remove the tar.gz file to save space
        os.remove(tar_path)
        print(f"Successfully downloaded {group_matrix}")

        print("Converting to binary...")
        os.system(f"../distributed_mmio/build/mtx_to_bmtx {matrices_dir}/{matrix_name}/{matrix_name}.mtx ")

    except Exception as e:
        print(f"Failed to download {group_matrix}: {e}")
        # Clean up failed download directory
        if os.path.exists(matrix_dir):
            shutil.rmtree(matrix_dir)


def download_matrices(args):
    fpath = args.matrix_list
    matrices_dir = "hns_matrices"

    # Ensure matrices directory exists
    os.makedirs(matrices_dir, exist_ok=True)

    matrices = get_matrices(fpath)
    print(matrices)

    for matrix in matrices:
        download_matrix(matrix, matrices_dir)


def make_scripts(matrix):
    for node in NODES:
        gpus = 4 * node
        header = f"""#!/usr/bin/bash
        -N {node}
        -G {gpus}
        -A m4646
        -t 00:20:00
        -C gpu
        -q regular\n

        """
        fname = f"scripts/{matrix}/gpus-{gpus}.sbatch"
        matpath = f"matrices/{matrix}/{matrix}.bmtx"
        with open(fname, 'w') as file:
            file.write(header)
            for ex in EXECUTABLES:
                for flags in FLAGS[ex]:

                    flag_str = ''.join(flags)
                    flag_str = flag_str.replace('-', '')
                    flag_str = flag_str.replace(' ', '')

                    output = f"output/{matrix}/e:{ex}-g:{gpus}-f:{flag_str}"

                    flags_args = ' '.join(flags)

                    cmd = f"srun -G {gpus} -n {gpus} -o {output} {ex} {flags_args} --matA {matpath} --matB {matpath}\n"
                    file.write(cmd)
    


def setup_scripts(args):
    fpath = args.matrix_list
    matrices_dir = "hns_matrices"

    # Ensure matrices directory exists
    os.makedirs(matrices_dir, exist_ok=True)

    matrices = get_matrices(fpath)
    print(matrices)

    for matrix in matrices:
        m = matrix.split("/")[-1]
        os.makedirs(f"output/{m}", exist_ok=True)
        os.makedirs(f"scripts/{m}", exist_ok=True)
        make_scripts(m)
        
    
def submit_scripts(args):
    fpath = args.matrix_list
    matrices_dir = "hns_matrices"

    # Ensure matrices directory exists
    os.makedirs(matrices_dir, exist_ok=True)

    matrices = get_matrices(fpath)
    print(matrices)
    script_dir = "scripts"
    for matrix in matrices:
        path = os.join(script_dir, matrix)
        for script in os.listdir(path):
            path = os.join(path, script)
            os.system(f"sbatch {path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str)
    parser.add_argument("--matrix_list", type=str)
    args = parser.parse_args()

    action = args.action


    if action == "download":
        download_matrices(args)
    elif action == "scripts":
        setup_scripts(args)
    elif action == "submit":
        submit_scripts(args)
    else:
        print(f"Invalid action {action}")
        raise Exception

