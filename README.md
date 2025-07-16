# Computational homogenization of mechanical metamaterials
This repository implements a computational homogenization framework for simulating the mechanical response of microstructured materials using the FE² (Finite Element squared) method.

Modeling Assumptions and Limitations:
- **Simulation regime**: Hyperelastic, finite strain, including bifurcation and buckling
- **Plane strain** conditions assumed throughout
- **Material model**: Bertoldi–Boyce hyperelastic law, with parameters:
  - c₁ = 0.55 kPa, c₂ = 0.3 kPa, K = 55 kPa
- **No contact modeling**: in post-processing, trajectories must be truncated once self-contact is detected
- **2×2 RVE** used for all homogenization (see paper for justification)
- **Boundary conditions**: periodic displacement via image-source nodes

The code was used to perform the simulations described in our preprint: https://arxiv.org/abs/2507.11195, which describes a dataset of 1020 microstructures and their mechanical responses. That dataset is available here: https://zenodo.org/records/15849550.
The Python code used to generate the new microstructure geometries is available in a repository on Github here: https://github.com/FHendriks11/wallpaper_microstructures.

### Support
Martin Doskar (MartinDoskar@gmail.com)

### Authors
Martin Doškář, modified by Fleur Hendriks

### License
This repository by Martin Doškář and Fleur Hendriks is licensed under CC BY 4.0. See the license file here: https://creativecommons.org/licenses/by/4.0/ for more details.

### Citation
If you use this code, please cite our preprint [link here] as follows:

```bibtex
@article{hendriks2025wallpaper,
  author       = {Hendriks, Fleur and Menkovski, Vlado and Doškář, Martin and Geers, Marc and Rokoš, Ondřej},
  title        = {Wallpaper Group-Based Mechanical Metamaterials: Dataset Including Mechanical Responses},
  year         = {2025},
  archivePrefix = {arXiv},
  arxivID = {t.b.d.},
  eprint = {t.b.d},
  month = {jul},
  url = {t.b.d}
}
```

## Software
We used Matlab 2021a for the simulations.

## Compilation

Before execution of any of the `RUN_*.m` files, compilation of all `mex` files is required. This can be achieved in two ways:

1. Using CMake to build `mex` files with the standard set of commands. For Linux, type
    ```
    cd mex
    mkdir build
    cd build
    cmake -DBUILD_PARSER=ON -DCMAKE_BUILD_TYPE=Release
    make
    ```
    CMake automatically downloads the latest version of the Eigen library. Alternatively, you can provide a path to pre-installed Eigen on your computer
    ```
    cd mex
    mkdir build
    cd build
    cmake -DBUILD_PARSER=ON -DPROVIDED_EIGEN_PATH="your_path_to_eigen" -DCMAKE_BUILD_TYPE=Release
    make
    ```
    On Windows, we recommend using either Visual Studio 2019 Community or VS Code with CMake Tools extension, which both support CMake projects.

2. The second option is to compile all `mex` files directly in Matlab. To this end, a C/C++ compiler needs to be installed and linked to Matlab, see [Matlab supported compilers](https://www.mathworks.com/support/compilers.html) for available options (note that Matlab's support often lacks behind the latest compilers). The MinGW option works well for Windows; the Xcode has been tested for Mac; the standard gcc works well for Linux. To check that the compiler has been properly linked with Matlab, execute `mex -setup` in the Matlab command prompt.
Once a compiler is linked properly, change `pathEigen = 'your_path_to_eigen';` variable string in `compile_mex.m` to point to Eigen headers on your machine. Finally, perform the compilation executing
    ```
    compile_mex
    ```
    in Matlab from the `mechmat_homogenization` folder.

## Files
This repository contains the following files:

To compile the mex code in mex/src:
* **compile_mex.m**

Running simulations:
* **main.m**: run a simple example.
* **sampling.m**: iterate over all microstructures geometries in a folder. For each, sample 12 different macroscopic deformation gradients F and perform a simulation.
* **sampling_mesh_convergence.m**: very similar to sampling.m, used for a mesh convergence study.

Additional results
* **reference_stiffness.m**: Calculate stiffness in the reference configuration of all microstructures in a given directory.
* **save_special_nodes.m**: Saves the node indices that were used as source and image nodes and the fixed node.

Example inputs:
* **cm_hexagonal1_2024-05-22_14-23-52.891252_00.mat**: an example of a geometry from the new dataset, based on wallpaper group cm with a hexagonal Bravais lattice.
* **RVEdefintionHEX.mat**: RVE of hexagonally arranged circular holes
* **RVEdefinitonSQR.mat**: RVE of square lattice of circular holes
