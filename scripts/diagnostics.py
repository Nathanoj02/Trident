import os
import argparse
import re

from colors import *



def diagnose_hns(args):
    print("==========HNS Diagnosis==========")
    print("---------")

    files = os.listdir(args.dir)

    files = list(filter(lambda s: "hns" in s and ".out" in s and "mcl" not in s, files))

    bad = []
    for filename in files:
        param_str = filename.split(".out")[0]
        griddims = re.search(r"\dx\d", param_str)[0]
        dim = int(griddims.split("x")[0])
        nprocs = int(re.search(r"_\d+_", param_str)[0].replace("_", ""))
        
        # spcomm + 3D does not work
        if dim**2 != nprocs and "nospcomm" not in param_str:
            continue

        with open(args.dir + "/" + filename, "r") as file:
            content = file.read()
            if content.count("spgemm round: 3")==1 and content.count("A stored in CSC") == 1:
                print("\t" + OKGREEN + param_str + " complete" + ENDC)
            else:
                bad.append(param_str)

    print("---------")
    for b in bad:
        print("\t" + FAIL + b + " incomplete" + ENDC)


def diagnose_trilinos(args):
    print("==========Trilinos Diagnosis==========")
    print("---------")

    files = os.listdir(args.dir)

    files = list(filter(lambda s: "trilinos" in s and ".out" in s, files))

    bad = []
    for filename in files:
        param_str = filename.split(".out")[0]
        with open(args.dir + "/" + filename, "r") as file:
            content = file.read()
            if content.count("Done spgemm") == 1:
                print("\t" + OKGREEN + param_str + " complete" + ENDC)
            else:
                bad.append(param_str)

    print("---------")
    for b in bad:
        print("\t" + FAIL + b + " incomplete" + ENDC)


def diagnose_combblas(args):
    print("==========Combblas Diagnosis==========")
    print("---------")

    files = os.listdir(args.dir)

    files = list(filter(lambda s: "combblas" in s and ".out" in s, files))

    bad = []
    for filename in files:
        param_str = filename.split(".out")[0]
        with open(args.dir + "/" + filename, "r") as file:
            content = file.read()
            if content.count("2D Multiplication done") == 1:
                print("\t" + OKGREEN + param_str + " complete" + ENDC)
            else:
                bad.append(param_str)

    print("---------")
    for b in bad:
        print("\t" + FAIL + b + " incomplete" + ENDC)


def diagnose_mcl(args):
    print("==========MCL Diagnosis==========")
    print("---------")

    files = os.listdir(args.dir)

    files = list(filter(lambda s: "mcl" in s and ".out" in s, files))

    bad = []
    for filename in files:
        param_str = filename.split(".out")[0]
        with open(args.dir + "/" + filename, "r") as file:
            content = file.read()
            if content.count("Done MCL") == 1:
                print("\t" + OKGREEN + param_str + " complete" + ENDC)
            else:
                bad.append(param_str)

    print("---------")
    for b in bad:
        print("\t" + FAIL + b + " incomplete" + ENDC)

def diagnose(args):
    diagnose_hns(args)
    diagnose_trilinos(args)
    diagnose_combblas(args)
    diagnose_mcl(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    diagnose(args)

















