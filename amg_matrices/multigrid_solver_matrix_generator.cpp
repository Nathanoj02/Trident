#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iomanip>
#include <sys/stat.h>
#include <algorithm>

// Utility function to compute linear index for 1D, 2D, 3D
inline size_t idx1D(size_t i) { return i; }
inline size_t idx2D(size_t i, size_t j, size_t nx) { return i + j*nx; }
inline size_t idx3D(size_t i, size_t j, size_t k, size_t nx, size_t ny) { return i + j*nx + k*nx*ny; }

// Check if index is inside the grid
inline bool in_bounds(int i, int nx) { return i >= 0 && i < nx; }
inline bool in_bounds(int i, int j, int nx, int ny) { return i >= 0 && i < nx && j >= 0 && j < ny; }
inline bool in_bounds(int i, int j, int k, int nx, int ny, int nz) { return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz; }

// Generate 1D 3-point stencil
void generate_1D(size_t nx,
                 std::vector<size_t>& row_ptr,
                 std::vector<size_t>& col_idx,
                 std::vector<double>& values)
{
    row_ptr.resize(nx+1);
    col_idx.reserve(3*nx);
    values.reserve(3*nx);
    size_t counter = 0;
    for (size_t i = 0; i < nx; ++i) {
        row_ptr[i] = counter;
        for (int di=-1; di<=1; ++di) {
            int j = static_cast<int>(i)+di;
            if (in_bounds(j, nx)) {
                col_idx.push_back(j);
                double val = (di==0) ? 2.0 : -1.0;
                values.push_back(val);
                ++counter;
            }
        }
    }
    row_ptr[nx] = counter;
}

// Generate 2D 5-point stencil
void generate_2D(size_t nx, size_t ny,
                 std::vector<size_t>& row_ptr,
                 std::vector<size_t>& col_idx,
                 std::vector<double>& values)
{
    row_ptr.resize(nx*ny+1);
    col_idx.reserve(5*nx*ny);
    values.reserve(5*nx*ny);
    size_t counter = 0;
    for (size_t j=0; j<ny; ++j) {
        for (size_t i=0; i<nx; ++i) {
            row_ptr[idx2D(i,j,nx)] = counter;
            int offsets[5][2] = {{0,0},{-1,0},{1,0},{0,-1},{0,1}};
            for (auto& off : offsets) {
                int ii = static_cast<int>(i) + off[0];
                int jj = static_cast<int>(j) + off[1];
                if (in_bounds(ii,jj,nx,ny)) {
                    col_idx.push_back(idx2D(ii,jj,nx));
                    double val = (off[0]==0 && off[1]==0) ? 4.0 : -1.0;
                    values.push_back(val);
                    ++counter;
                }
            }
        }
    }
    row_ptr[nx*ny] = counter;
}

// Generate 2D 9-point stencil
void generate_2D_9pt(size_t nx, size_t ny,
                     std::vector<size_t>& row_ptr,
                     std::vector<size_t>& col_idx,
                     std::vector<double>& values)
{
    row_ptr.resize(nx*ny+1);
    col_idx.reserve(9*nx*ny);
    values.reserve(9*nx*ny);
    size_t counter = 0;
    for (size_t j=0; j<ny; ++j) {
        for (size_t i=0; i<nx; ++i) {
            row_ptr[idx2D(i,j,nx)] = counter;
            int offsets[9][2] = {{0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
            for (auto& off : offsets) {
                int ii = static_cast<int>(i) + off[0];
                int jj = static_cast<int>(j) + off[1];
                if (in_bounds(ii,jj,nx,ny)) {
                    col_idx.push_back(idx2D(ii,jj,nx));
                    double val;
                    if (off[0]==0 && off[1]==0) val = 8.0;
                    else if (off[0]==0 || off[1]==0) val = -2.0;
                    else val = -1.0;
                    values.push_back(val);
                    ++counter;
                }
            }
        }
    }
    row_ptr[nx*ny] = counter;
}

// Generate 3D 7-point stencil
void generate_3D(size_t nx, size_t ny, size_t nz,
                 std::vector<size_t>& row_ptr,
                 std::vector<size_t>& col_idx,
                 std::vector<double>& values)
{
    row_ptr.resize(nx*ny*nz+1);
    col_idx.reserve(7*nx*ny*nz);
    values.reserve(7*nx*ny*nz);
    size_t counter = 0;
    for (size_t k=0; k<nz; ++k) {
        for (size_t j=0; j<ny; ++j) {
            for (size_t i=0; i<nx; ++i) {
                row_ptr[idx3D(i,j,k,nx,ny)] = counter;
                int offsets[7][3] = {{0,0,0},{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
                for (auto& off : offsets) {
                    int ii = static_cast<int>(i) + off[0];
                    int jj = static_cast<int>(j) + off[1];
                    int kk = static_cast<int>(k) + off[2];
                    if (in_bounds(ii,jj,kk,nx,ny,nz)) {
                        col_idx.push_back(idx3D(ii,jj,kk,nx,ny));
                        double val = (off[0]==0 && off[1]==0 && off[2]==0) ? 6.0 : -1.0;
                        values.push_back(val);
                        ++counter;
                    }
                }
            }
        }
    }
    row_ptr[nx*ny*nz] = counter;
}

// Generate 3D 27-point stencil
void generate_3D_27pt(size_t nx, size_t ny, size_t nz,
                      std::vector<size_t>& row_ptr,
                      std::vector<size_t>& col_idx,
                      std::vector<double>& values)
{
    row_ptr.resize(nx*ny*nz+1);
    col_idx.reserve(27*nx*ny*nz);
    values.reserve(27*nx*ny*nz);
    size_t counter = 0;
    for (size_t k=0; k<nz; ++k) {
        for (size_t j=0; j<ny; ++j) {
            for (size_t i=0; i<nx; ++i) {
                row_ptr[idx3D(i,j,k,nx,ny)] = counter;
                for (int dk=-1; dk<=1; ++dk) {
                    for (int dj=-1; dj<=1; ++dj) {
                        for (int di=-1; di<=1; ++di) {
                            int ii = static_cast<int>(i) + di;
                            int jj = static_cast<int>(j) + dj;
                            int kk = static_cast<int>(k) + dk;
                            if (in_bounds(ii,jj,kk,nx,ny,nz)) {
                                col_idx.push_back(idx3D(ii,jj,kk,nx,ny));
                                if (di==0 && dj==0 && dk==0) {
                                    values.push_back(26.0);
                                }
                                else if ((di==0 && dj==0) || (di==0 && dk==0) || (dj==0 && dk==0)) {
                                    int nonzero_count = (di!=0) + (dj!=0) + (dk!=0);
                                    if (nonzero_count == 1) values.push_back(-4.0);
                                    else values.push_back(-2.0);
                                }
                                else {
                                    values.push_back(-1.0);
                                }
                                ++counter;
                            }
                        }
                    }
                }
            }
        }
    }
    row_ptr[nx*ny*nz] = counter;
}

// Generate 1D Prolongation
void generate_prolongation_1D(size_t nx_coarse,
                              std::vector<size_t>& row_ptr,
                              std::vector<size_t>& col_idx,
                              std::vector<double>& values)
{
    size_t nx_fine = 2 * nx_coarse - 1;
    row_ptr.resize(nx_fine + 1);
    col_idx.reserve(2 * nx_fine);
    values.reserve(2 * nx_fine);
    
    size_t counter = 0;
    for (size_t i = 0; i < nx_fine; ++i) {
        row_ptr[i] = counter;
        if (i % 2 == 0) {
            col_idx.push_back(i / 2);
            values.push_back(1.0);
            ++counter;
        } else {
            size_t left = i / 2;
            size_t right = left + 1;
            col_idx.push_back(left);
            values.push_back(0.5);
            col_idx.push_back(right);
            values.push_back(0.5);
            counter += 2;
        }
    }
    row_ptr[nx_fine] = counter;
}

// Generate 2D Prolongation
void generate_prolongation_2D(size_t nx_coarse, size_t ny_coarse,
                              std::vector<size_t>& row_ptr,
                              std::vector<size_t>& col_idx,
                              std::vector<double>& values)
{
    size_t nx_fine = 2 * nx_coarse - 1;
    size_t ny_fine = 2 * ny_coarse - 1;
    size_t n_fine = nx_fine * ny_fine;
    
    row_ptr.resize(n_fine + 1);
    col_idx.reserve(4 * n_fine);
    values.reserve(4 * n_fine);
    
    size_t counter = 0;
    for (size_t j = 0; j < ny_fine; ++j) {
        for (size_t i = 0; i < nx_fine; ++i) {
            row_ptr[idx2D(i, j, nx_fine)] = counter;
            
            bool i_coarse = (i % 2 == 0);
            bool j_coarse = (j % 2 == 0);
            
            if (i_coarse && j_coarse) {
                col_idx.push_back(idx2D(i/2, j/2, nx_coarse));
                values.push_back(1.0);
                ++counter;
            } else if (i_coarse && !j_coarse) {
                size_t ic = i / 2;
                size_t jc_low = j / 2;
                size_t jc_high = jc_low + 1;
                col_idx.push_back(idx2D(ic, jc_low, nx_coarse));
                values.push_back(0.5);
                col_idx.push_back(idx2D(ic, jc_high, nx_coarse));
                values.push_back(0.5);
                counter += 2;
            } else if (!i_coarse && j_coarse) {
                size_t ic_low = i / 2;
                size_t ic_high = ic_low + 1;
                size_t jc = j / 2;
                col_idx.push_back(idx2D(ic_low, jc, nx_coarse));
                values.push_back(0.5);
                col_idx.push_back(idx2D(ic_high, jc, nx_coarse));
                values.push_back(0.5);
                counter += 2;
            } else {
                size_t ic_low = i / 2;
                size_t ic_high = ic_low + 1;
                size_t jc_low = j / 2;
                size_t jc_high = jc_low + 1;
                col_idx.push_back(idx2D(ic_low, jc_low, nx_coarse));
                values.push_back(0.25);
                col_idx.push_back(idx2D(ic_high, jc_low, nx_coarse));
                values.push_back(0.25);
                col_idx.push_back(idx2D(ic_low, jc_high, nx_coarse));
                values.push_back(0.25);
                col_idx.push_back(idx2D(ic_high, jc_high, nx_coarse));
                values.push_back(0.25);
                counter += 4;
            }
        }
    }
    row_ptr[n_fine] = counter;
}

// Generate 3D Prolongation
void generate_prolongation_3D(size_t nx_coarse, size_t ny_coarse, size_t nz_coarse,
                              std::vector<size_t>& row_ptr,
                              std::vector<size_t>& col_idx,
                              std::vector<double>& values)
{
    size_t nx_fine = 2 * nx_coarse - 1;
    size_t ny_fine = 2 * ny_coarse - 1;
    size_t nz_fine = 2 * nz_coarse - 1;
    size_t n_fine = nx_fine * ny_fine * nz_fine;
    
    row_ptr.resize(n_fine + 1);
    col_idx.reserve(8 * n_fine);
    values.reserve(8 * n_fine);
    
    size_t counter = 0;
    for (size_t k = 0; k < nz_fine; ++k) {
        for (size_t j = 0; j < ny_fine; ++j) {
            for (size_t i = 0; i < nx_fine; ++i) {
                row_ptr[idx3D(i, j, k, nx_fine, ny_fine)] = counter;
                
                bool i_coarse = (i % 2 == 0);
                bool j_coarse = (j % 2 == 0);
                bool k_coarse = (k % 2 == 0);
                
                int n_fine_dirs = (!i_coarse) + (!j_coarse) + (!k_coarse);
                
                if (n_fine_dirs == 0) {
                    col_idx.push_back(idx3D(i/2, j/2, k/2, nx_coarse, ny_coarse));
                    values.push_back(1.0);
                    ++counter;
                } else {
                    double weight = 1.0 / (1 << n_fine_dirs);
                    
                    size_t ic_low = i / 2;
                    size_t ic_high = i_coarse ? ic_low : ic_low + 1;
                    size_t jc_low = j / 2;
                    size_t jc_high = j_coarse ? jc_low : jc_low + 1;
                    size_t kc_low = k / 2;
                    size_t kc_high = k_coarse ? kc_low : kc_low + 1;
                    
                    for (size_t kc = kc_low; kc <= kc_high; ++kc) {
                        for (size_t jc = jc_low; jc <= jc_high; ++jc) {
                            for (size_t ic = ic_low; ic <= ic_high; ++ic) {
                                col_idx.push_back(idx3D(ic, jc, kc, nx_coarse, ny_coarse));
                                values.push_back(weight);
                                ++counter;
                            }
                        }
                    }
                }
            }
        }
    }
    row_ptr[n_fine] = counter;
}

// Generate 1D Restriction
void generate_restriction_1D(size_t nx_coarse,
                             std::vector<size_t>& row_ptr,
                             std::vector<size_t>& col_idx,
                             std::vector<double>& values)
{
    // size_t nx_fine = 2 * nx_coarse - 1;
    row_ptr.resize(nx_coarse + 1);
    col_idx.reserve(3 * nx_coarse);
    values.reserve(3 * nx_coarse);
    
    size_t counter = 0;
    for (size_t i = 0; i < nx_coarse; ++i) {
        row_ptr[i] = counter;
        size_t i_fine = 2 * i;
        
        if (i > 0) {
            col_idx.push_back(i_fine - 1);
            values.push_back(0.25);
            ++counter;
        }
        col_idx.push_back(i_fine);
        values.push_back(0.5);
        ++counter;
        if (i < nx_coarse - 1) {
            col_idx.push_back(i_fine + 1);
            values.push_back(0.25);
            ++counter;
        }
    }
    row_ptr[nx_coarse] = counter;
}

// Generate 2D Restriction
void generate_restriction_2D(size_t nx_coarse, size_t ny_coarse,
                             std::vector<size_t>& row_ptr,
                             std::vector<size_t>& col_idx,
                             std::vector<double>& values)
{
    size_t nx_fine = 2 * nx_coarse - 1;
    size_t ny_fine = 2 * ny_coarse - 1;
    
    row_ptr.resize(nx_coarse * ny_coarse + 1);
    col_idx.reserve(9 * nx_coarse * ny_coarse);
    values.reserve(9 * nx_coarse * ny_coarse);
    
    size_t counter = 0;
    for (size_t j = 0; j < ny_coarse; ++j) {
        for (size_t i = 0; i < nx_coarse; ++i) {
            row_ptr[idx2D(i, j, nx_coarse)] = counter;
            size_t i_fine = 2 * i;
            size_t j_fine = 2 * j;
            
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    int ii = static_cast<int>(i_fine) + di;
                    int jj = static_cast<int>(j_fine) + dj;
                    if (in_bounds(ii, jj, nx_fine, ny_fine)) {
                        col_idx.push_back(idx2D(ii, jj, nx_fine));
                        double weight;
                        if (di == 0 && dj == 0) weight = 0.25;
                        else if (di == 0 || dj == 0) weight = 0.125;
                        else weight = 0.0625;
                        values.push_back(weight);
                        ++counter;
                    }
                }
            }
        }
    }
    row_ptr[nx_coarse * ny_coarse] = counter;
}

// Generate 3D Restriction
void generate_restriction_3D(size_t nx_coarse, size_t ny_coarse, size_t nz_coarse,
                             std::vector<size_t>& row_ptr,
                             std::vector<size_t>& col_idx,
                             std::vector<double>& values)
{
    size_t nx_fine = 2 * nx_coarse - 1;
    size_t ny_fine = 2 * ny_coarse - 1;
    size_t nz_fine = 2 * nz_coarse - 1;
    
    row_ptr.resize(nx_coarse * ny_coarse * nz_coarse + 1);
    col_idx.reserve(27 * nx_coarse * ny_coarse * nz_coarse);
    values.reserve(27 * nx_coarse * ny_coarse * nz_coarse);
    
    size_t counter = 0;
    for (size_t k = 0; k < nz_coarse; ++k) {
        for (size_t j = 0; j < ny_coarse; ++j) {
            for (size_t i = 0; i < nx_coarse; ++i) {
                row_ptr[idx3D(i, j, k, nx_coarse, ny_coarse)] = counter;
                size_t i_fine = 2 * i;
                size_t j_fine = 2 * j;
                size_t k_fine = 2 * k;
                
                for (int dk = -1; dk <= 1; ++dk) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int di = -1; di <= 1; ++di) {
                            int ii = static_cast<int>(i_fine) + di;
                            int jj = static_cast<int>(j_fine) + dj;
                            int kk = static_cast<int>(k_fine) + dk;
                            if (in_bounds(ii, jj, kk, nx_fine, ny_fine, nz_fine)) {
                                col_idx.push_back(idx3D(ii, jj, kk, nx_fine, ny_fine));
                                int n_nonzero = (di != 0) + (dj != 0) + (dk != 0);
                                double weight = 1.0 / (1 << (n_nonzero + 3));
                                values.push_back(weight);
                                ++counter;
                            }
                        }
                    }
                }
            }
        }
    }
    row_ptr[nx_coarse * ny_coarse * nz_coarse] = counter;
}

// Transpose a matrix in CSR format
void transpose_matrix(const std::vector<size_t>& row_ptr,
                     const std::vector<size_t>& col_idx,
                     const std::vector<double>& values,
                     size_t nrows, size_t ncols,
                     std::vector<size_t>& row_ptr_T,
                     std::vector<size_t>& col_idx_T,
                     std::vector<double>& values_T)
{
    size_t nnz = col_idx.size();
    
    // Count entries per column (which becomes row count for transpose)
    std::vector<size_t> col_counts(ncols, 0);
    for (size_t idx : col_idx) {
        col_counts[idx]++;
    }
    
    // Build row_ptr for transpose
    row_ptr_T.resize(ncols + 1);
    row_ptr_T[0] = 0;
    for (size_t i = 0; i < ncols; ++i) {
        row_ptr_T[i + 1] = row_ptr_T[i] + col_counts[i];
    }
    
    // Allocate space for transpose
    col_idx_T.resize(nnz);
    values_T.resize(nnz);
    
    // Track current position for each row of transpose
    std::vector<size_t> current_pos = row_ptr_T;
    
    // Fill transpose
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            size_t col = col_idx[j];
            size_t pos = current_pos[col]++;
            col_idx_T[pos] = i;
            values_T[pos] = values[j];
        }
    }
}

// Generate descriptive filename
std::string generate_filename(int dim, const std::string& type, 
                               size_t nx, size_t ny, size_t nz,
                               size_t nrows, size_t nnz, bool transposed)
{
    std::string name;
    
    if (type == "3pt" || type == "5pt" || type == "9pt" || 
        type == "7pt" || type == "27pt") {
        name = "stencil_" + type;
    } else if (type == "prolongation") {
        name = "prolong";
    } else if (type == "restriction") {
        name = "restrict";
    } else {
        name = type;
    }
    
    name += "_" + std::to_string(dim) + "D";
    
    if (dim == 1) {
        name += "_" + std::to_string(nx);
    } else if (dim == 2) {
        name += "_" + std::to_string(nx) + "x" + std::to_string(ny);
    } else if (dim == 3) {
        name += "_" + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);
    }
    
    name += "_n" + std::to_string(nrows);
    name += "_nnz" + std::to_string(nnz);
    
    if (transposed) {
        name += "_T";
    }
    
    return name;
}

// Check if directory exists
bool directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

// Save matrix in MatrixMarket format
void save_matrix_market(const std::string& filepath,
                       const std::vector<size_t>& row_ptr,
                       const std::vector<size_t>& col_idx,
                       const std::vector<double>& values,
                       size_t ncols = 0)
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << " for writing\n";
        return;
    }
    
    size_t nrows = row_ptr.size() - 1;
    // If ncols not specified, determine from column indices
    if (ncols == 0) {
        if (!col_idx.empty()) {
            ncols = *std::max_element(col_idx.begin(), col_idx.end()) + 1;
        } else {
            ncols = nrows; // Default to square if empty
        }
    }
    size_t nnz = col_idx.size();
    
    // Write MatrixMarket header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << nrows << " " << ncols << " " << nnz << "\n";
    
    // Write matrix entries (1-indexed for MatrixMarket format)
    file << std::scientific << std::setprecision(16);
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            file << (i+1) << " " << (col_idx[j]+1) << " " << values[j] << "\n";
        }
    }
    
    file.close();
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " DIMENSION MATRIX_TYPE [NX NY NZ] [OUTPUT_DIR] [FILENAME] [--transpose]\n";
        std::cerr << "DIMENSION: 1, 2, or 3\n";
        std::cerr << "MATRIX_TYPE: stencil, prolongation, restriction\n";
        std::cerr << "  For stencil: 3pt (1D), 5pt, 9pt (2D), 7pt, 27pt (3D)\n";
        std::cerr << "  For prolongation/restriction: grid sizes are for COARSE grid\n";
        std::cerr << "OUTPUT_DIR: Directory path where to save the .mtx file (default: current directory)\n";
        std::cerr << "FILENAME: Custom filename without extension (default: auto-generated)\n";
        std::cerr << "--transpose: Transpose the matrix before saving\n";
        return 1;
    }
    
    int dim = std::atoi(argv[1]);
    std::string type = argv[2];
    size_t nx = 10, ny = 10, nz = 10;  // defaults
    
    // Parse dimension-specific arguments
    int next_arg = 3;
    if (argc >= 4 && argv[3][0] != '-') {
        nx = std::atoi(argv[3]);
        next_arg = 4;
    }
    if (dim >= 2 && argc >= next_arg + 1 && argv[next_arg][0] != '-') {
        ny = std::atoi(argv[next_arg]);
        next_arg++;
    }
    if (dim >= 3 && argc >= next_arg + 1 && argv[next_arg][0] != '-') {
        nz = std::atoi(argv[next_arg]);
        next_arg++;
    }
    
    // Get output directory (default is current directory)
    std::string output_dir = ".";
    std::string custom_filename = "";
    bool transpose = false;
    
    // Parse remaining optional arguments
    for (int i = next_arg; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--transpose" || arg == "-t") {
            transpose = true;
        } else if (arg.substr(0, 2) != "--" && arg.substr(0, 1) != "-" && output_dir == ".") {
            // First non-flag argument is output directory
            output_dir = arg;
            // Remove trailing slash if present
            if (!output_dir.empty() && (output_dir.back() == '/' || output_dir.back() == '\\')) {
                output_dir.pop_back();
            }
            // Check if directory exists
            if (!directory_exists(output_dir)) {
                std::cerr << "Error: Directory '" << output_dir << "' does not exist\n";
                return 1;
            }
        } else if (arg.substr(0, 2) != "--" && arg.substr(0, 1) != "-" && custom_filename.empty()) {
            // Second non-flag argument is custom filename
            custom_filename = arg;
        }
    }
    
    std::vector<size_t> row_ptr, col_idx;
    std::vector<double> values;
    size_t ncols = 0; // Will be set for non-square matrices
    
    // Generate matrix based on type and dimension
    if (type == "3pt" || type == "5pt" || type == "9pt" || type == "7pt" || type == "27pt") {
        if (dim==1 && type=="3pt") {
            generate_1D(nx, row_ptr, col_idx, values);
        } else if (dim==2 && type=="5pt") {
            generate_2D(nx, ny, row_ptr, col_idx, values);
        } else if (dim==2 && type=="9pt") {
            generate_2D_9pt(nx, ny, row_ptr, col_idx, values);
        } else if (dim==3 && type=="7pt") {
            generate_3D(nx, ny, nz, row_ptr, col_idx, values);
        } else if (dim==3 && type=="27pt") {
            generate_3D_27pt(nx, ny, nz, row_ptr, col_idx, values);
        } else {
            std::cerr << "Unsupported combination of dimension and stencil\n";
            return 1;
        }
    } else if (type == "prolongation") {
        if (dim == 1) {
            generate_prolongation_1D(nx, row_ptr, col_idx, values);
            ncols = nx; // coarse grid size
        } else if (dim == 2) {
            generate_prolongation_2D(nx, ny, row_ptr, col_idx, values);
            ncols = nx * ny; // coarse grid size
        } else if (dim == 3) {
            generate_prolongation_3D(nx, ny, nz, row_ptr, col_idx, values);
            ncols = nx * ny * nz; // coarse grid size
        } else {
            std::cerr << "Invalid dimension for prolongation\n";
            return 1;
        }
    } else if (type == "restriction") {
        if (dim == 1) {
            generate_restriction_1D(nx, row_ptr, col_idx, values);
            ncols = 2 * nx - 1; // fine grid size
        } else if (dim == 2) {
            generate_restriction_2D(nx, ny, row_ptr, col_idx, values);
            ncols = (2 * nx - 1) * (2 * ny - 1); // fine grid size
        } else if (dim == 3) {
            generate_restriction_3D(nx, ny, nz, row_ptr, col_idx, values);
            ncols = (2 * nx - 1) * (2 * ny - 1) * (2 * nz - 1); // fine grid size
        } else {
            std::cerr << "Invalid dimension for restriction\n";
            return 1;
        }
    } else {
        std::cerr << "Unknown matrix type: " << type << "\n";
        std::cerr << "Valid types: 3pt, 5pt, 9pt, 7pt, 27pt, prolongation, restriction\n";
        return 1;
    }
    
    // Determine actual number of rows and columns
    size_t nrows = row_ptr.size() - 1;
    if (ncols == 0) {
        if (!col_idx.empty()) {
            ncols = *std::max_element(col_idx.begin(), col_idx.end()) + 1;
        } else {
            ncols = nrows;
        }
    }
    
    // Transpose if requested
    if (transpose) {
        std::vector<size_t> row_ptr_T, col_idx_T;
        std::vector<double> values_T;
        transpose_matrix(row_ptr, col_idx, values, nrows, ncols, row_ptr_T, col_idx_T, values_T);
        row_ptr = std::move(row_ptr_T);
        col_idx = std::move(col_idx_T);
        values = std::move(values_T);
        std::swap(nrows, ncols); // Swap dimensions
    }
    
    // Generate filename
    std::string filename;
    if (!custom_filename.empty()) {
        // Custom filename provided
        filename = custom_filename;
    } else {
        // Auto-generate filename
        filename = generate_filename(dim, type, nx, ny, nz, 
                                    nrows, col_idx.size(), transpose);
    }
    
    // Construct full filepath
    std::string filepath = output_dir + "/" + filename + ".mtx";
    
    // Save in MatrixMarket format
    save_matrix_market(filepath, row_ptr, col_idx, values, ncols);
    
    std::cout << "Matrix generated: " << nrows << " rows, "
              << ncols << " cols, "
              << col_idx.size() << " nonzeros\n";
    if (transpose) {
        std::cout << "Matrix transposed\n";
    }
    std::cout << "MatrixMarket file saved: " << filepath << "\n";
    
    return 0;
}