CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

multigrid_solver_matrix_generator: multigrid_solver_matrix_generator.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f multigrid_solver_matrix_generator