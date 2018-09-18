/*
 File main.cc
 
 This file contains just an example on how to set-up the matrices for using with
 the solve_quadprog() function.
 
 The test problem is the following:
 
 Given:
 G =  4 -2   g0^T = [6 0]
     -2  4       
 
 Solve:
 min f(x) = 1/2 x G x + g0 x
 s.t.
   x_1 + x_2 = 3
   x_1 >= 0
   x_2 >= 0
   x_1 + x_2 >= 2
 
 The solution is x^T = [1 2] and f(x) = 12
 
 Author: Luca Di Gaspero
 DIEGM - University of Udine, Italy
 l.digaspero@uniud.it
 http://www.diegm.uniud.it/digaspero/
 
 Copyright 2006-2009 Luca Di Gaspero
 
 This software may be modified and distributed under the terms
 of the MIT license.  See the LICENSE file for details.
*/

#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Eigen>
#include <boost/progress.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "quadprogpp/QuadProg++.hh"
#include "quadprog_eigen/quadprog_eigen.hh"
#include "EigenQP/EigenQP.h"

namespace btime = boost::posix_time;

int main (int argc, char *const argv[]) {
  constexpr int n = 2;
  constexpr int m = 1;
  constexpr int p = 3;

  constexpr double GInput[n * n] = {4, -2, -2, 4};

  constexpr double CEInput[n * m] = {1, 1};

  constexpr double CIInput[n * p] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0};

  constexpr double g0Input[n] = {6.0, 0.0};

  constexpr double ce0Input[m] = {-3.0};

  constexpr double ci0Input[p] = {0.0, 0.0, -2.0};

  constexpr double x0Input[n] = {0.0, 0.0};

  // Initialize the matrix to pass to quadprog solver.
  quadprogpp::Matrix<double> G(GInput, n, n);
  quadprogpp::Matrix<double> CE(CEInput, n, m);
  quadprogpp::Matrix<double> CI(CIInput, n, p);

  quadprogpp::Vector<double> g0(g0Input, n);
  quadprogpp::Vector<double> ce0(ce0Input, m);
  quadprogpp::Vector<double> ci0(ci0Input, p);
  quadprogpp::Vector<double> x(x0Input, n);

  int count = 1000000;

  btime::ptime tic = btime::microsec_clock::local_time();
  //boost::timer timer;
  // Does this modify H?
  double objVal;
  for (int i = 0; i < count; ++i)
  {
	objVal = quadprogpp::solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
  }
  btime::time_duration toc = btime::microsec_clock::local_time() - tic;
  std::cout << "Elapsed time of quadprogpp: " << std::setprecision(8) << toc.total_milliseconds() << " ms\n";
  std::cout << "obj: " << objVal << "\nx: " << x << "\n\n";
  

  // Initialize the matrix to pass to quadprog solver.
  Eigen::MatrixXd eG(n, n);
  Eigen::MatrixXd eCE(n, m);
  Eigen::MatrixXd eCI(n, p);

  Eigen::VectorXd eg0(n), ece0(m), eci0(p), ex(n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      eG(i, j) = GInput[i * n + j];
    }
    eg0(i) = g0Input[i];
    ex(i) = x0Input[i];
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      eCE(i, j) = CEInput[i * m + j];
    }
  }

  for (int i = 0; i < m; i++) {
    ece0(i) = ce0Input[i];
  }

  for (int s = 0; s < n; s++) {
    for (int j = 0; j < p; j++) {
      eCI(s, j) = CIInput[s * p + j];
    }
  }

  for (int i = 0; i < p; i++) {
    eci0(i) = ci0Input[i];
  }

  tic = btime::microsec_clock::local_time();
  for (int i = 0; i < count; ++i) {
  	ex.setZero();
    objVal = QP::solve_quadprog(eG, eg0, eCE, ece0, eCI, eci0, ex);
  }
  toc = btime::microsec_clock::local_time() - tic;
  std::cout << "Elapsed time of eigen QP: " << std::setprecision(8) << toc.total_milliseconds() << " ms\n";
  std::cout << "obj: " << objVal << "\nx: " << ex << "\n\n";

  quadprog_eigen::SolverFlag solver_flag;
  tic = btime::microsec_clock::local_time();
  for (int i = 0; i < count; ++i) {
  	ex.setZero();
    solver_flag = quadprog_eigen::solve_quadprog(eG, eg0, eCE, ece0, eCI, eci0, ex, objVal);
  }
  switch(solver_flag) {
    case quadprog_eigen::SolverFlag::kReachMaxIter:
      std::cout << "Reaches Maximum Iteration." << std::endl;
      break;
    case quadprog_eigen::SolverFlag::kSolveSuccess:
      std::cout << "Solve Success." << std::endl;
      break;
    case quadprog_eigen::SolverFlag::kInfeasibleConstr:
      std::cout << "Infeasible Constraints." << std::endl;
      break;
    case quadprog_eigen::SolverFlag::kLinearConstr:
      std::cout << "Infeasible Constraints." << std::endl;
      break;
  }

  toc = btime::microsec_clock::local_time() - tic;
  std::cout << "Elapsed time of quadprog eigen: " << std::setprecision(8) << toc.total_milliseconds() << " ms\n";
  std::cout << "obj: " << objVal << "\nx: " << ex << "\n\n";
}
