import os
import argparse


HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


def diagnose_hns(args):
    print("==========HNS Diagnosis==========")
    print("---------")

    files = os.listdir(args.dir)

    files = list(filter(lambda s: "hns" in s and ".out" in s, files))

    bad = []
    for filename in files:
        param_str = filename.split(".out")[0]
        with open(args.dir + "/" + filename, "r") as file:
            content = file.read()
            if "spgemm round: 5" in content:
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
            if "Done spgemm" in content:
                print("\t" + OKGREEN + param_str + " complete" + ENDC)
            else:
                bad.append(param_str)

    print("---------")
    for b in bad:
        print("\t" + FAIL + b + " incomplete" + ENDC)


def diagnose(args):
    diagnose_hns(args)
    diagnose_trilinos(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    diagnose(args)

















