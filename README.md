# Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects
Our [ICS 2026](https://dipsa-qub.github.io/ICS2026-webpage/) paper (PDF coming soon) introduces Trident partitioning, a hierarchy-aware hybrid 2D–1D decomposition for distributed SpGEMM that cuts inter-node communication, achieving up to 2.4× speedup and 2× lower communication volume on NERSC’s Perlmutter system.

## Modules

- **Perlmutter**: `cmake/3.30.2 cudatoolkit/12.9 craype-accel-nvidia80 craype-hugepages2G cray-pmi/6.1.15`
- **Leonardo**: `cmake/3.27.9 gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 openblas/0.3.26--gcc--12.2.0`

# Trident

First, install Kokkos:
```bash
scripts/kokkos_install.sh
```

# Baselines

Installation paths are customizable in [`scripts/variables.sh`](scripts/variables.sh).

## Trilinos

To build `Trilinos` as a shared library, run:
```bash
scripts/trilinos_install.sh
```

Notes:
- Currently the script builds for GPU architecture `AMPERE80`

MCL PUT + CSC

## Citation

If you find this repo helpful to your work, please cite our article:

```
@inproceedings{trident,
  author    = {Bellavita, Julian and Pichetti, Lorenzo and Pasquali, Thomas and Vella, Flavio and Guidi, Giulia},
  title     = {{C}ommunication-{A}voiding {SpGEMM} via {T}rident {P}artitioning on {H}ierarchical {GPU} {I}nterconnects},
  booktitle = {Proceedings of the 40th ACM International Conference on Supercomputing},
  series    = {ICS '26},
  year      = {2026},
  month     = {July},
  address   = {Belfast, Northern Ireland, United Kingdom},
  publisher = {ACM},
}
```

## Acknowledgment

This work was a collaboration between the [HiCrest Laboratory at the University of Trento](https://hicrest.unitn.it/) (Italy) and the [Cornell HPC Group at Cornell University](https://giuliaguidi.github.io/) (USA). The [first author](https://hooninator.github.io/website.github.io/) was supported by DOE CSGF. This work was partially developed during the second author's research visit at Cornell University. The authors want to thank Benjamin Brock for his help in running and benchmarking the BCL 2D asynchronous SpGEMM. The authors acknowledge financial support from ICSC – Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing, funded by European Union – NextGenerationEU. This work has received funding from the NETwork for European EXAscale Systems (NET4EXA) EU project under grant agreement No 101175702 and the NationalInstitute of Higher Mathematics Francesco Severi. This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0030076. 
