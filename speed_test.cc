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
  constexpr int n = 10;
  constexpr int m = 5;
  constexpr int p = 28;

  int count = 1000000;

  btime::ptime tic = btime::microsec_clock::local_time();
  

  // Initialize the matrix to pass to quadprog solver.
  Eigen::MatrixXd eG(n, n);
  Eigen::MatrixXd eCE(m, n);
  Eigen::MatrixXd eCI(p, n);

  Eigen::VectorXd eg0(n), ece0(m), eci0(p), ex(n);

  eG << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 2;

  eg0 << 0, -0.01, -0.02, -0.03, -0.04, -2, -2, -2, -2, -2; 

  eCE <<  1,     0,     0,     0,    0, -0.01,     0,     0,     0,     0,
          1,    -1,     0,     0,    0,     0,  0.01,     0,     0,     0,
          0,     1,    -1,     0,    0,     0,     0,  0.01,     0,     0,
          0,     0,     1,    -1,    0,     0,     0,     0,  0.01,     0,
          0,     0,     0,     1,   -1,     0,     0,     0,     0,  0.01;

  ece0 <<  0, 0, 0, 0, 0;

  eCI <<    0,     0,     0,     0,     0,    -1,     0,     0,     0,     0,
            1,    -1,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     1,    -1,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     1,    -1,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     1,    -1,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     1,     0,     0,     0,     0,
           -1,     1,     0,     0,     0,     0,     0,     0,     0,     0,
            0,    -1,     1,     0,     0,     0,     0,     0,     0,     0,
            0,     0,    -1,     1,     0,     0,     0,     0,     0,     0,
            0,     0,     0,    -1,     1,     0,     0,     0,     0,     0,
         0.02, -0.01,     0,     0,     0,     0,     0,     0,     0,     0,
        -0.01,  0.02, -0.01,     0,     0,     0,     0,     0,     0,     0,
            0, -0.01,  0.02, -0.01,     0,     0,     0,     0,     0,     0,
            0,     0, -0.01,  0.02, -0.01,     0,     0,     0,     0,     0,
        -0.02,  0.01,     0,     0,     0,     0,     0,     0,     0,     0,
         0.01, -0.02,  0.01,     0,     0,     0,     0,     0,     0,     0,
            0,  0.01, -0.02,  0.01,     0,     0,     0,     0,     0,     0,
            0,     0,  0.01, -0.02,  0.01,     0,     0,     0,     0,     0,
           -1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,    -1,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,    -1,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,    -1,     0,     0,     0,     0,     0,
            1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     1,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     1,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     1,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     1,     0,     0,     0,     0,     0;

  eci0 << 2, 0.02, 0.02, 0.02, 0.02, 4.5, 0.045, 0.045, 0.045, 0.045, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 0, 0.01, 0.02, 0.03, 0.04, 0, 0, 0, 0, 0;

  double objVal;
  tic = btime::microsec_clock::local_time();
  for (int i = 0; i < count; ++i) {
    Eigen::MatrixXd eG_test1(n, n);
    Eigen::VectorXd eg0_test1(n);
    eG_test1 = eG;
    eg0_test1 = eg0;
  	ex.setZero();
    objVal = QP::solve_quadprog(eG_test1, eg0_test1, eCE.transpose(), ece0, eCI.transpose(), eci0, ex);
  }
  btime::time_duration toc = btime::microsec_clock::local_time() - tic;
  std::cout << "Elapsed time of eigen QP: " << std::setprecision(8) << toc.total_milliseconds() << " ms\n";
  std::cout << "obj: " << objVal << "\nx: \n" << ex << "\n\n";
  // std::cout << "Equality violation: " << std::endl;
  // std::cout << eCE*ex + ece0 << std::endl;
  // std::cout << "Inequality violation: " << std::endl;
  // std::cout << eCI*ex + eci0 << std::endl;

  // Solving use a different Eigen
  int iter = 0;
  quadprog_eigen::SolverFlag solver_flag;
  tic = btime::microsec_clock::local_time();
  for (int i = 0; i < count; ++i) {
    Eigen::MatrixXd eG_test2(n, n);
    Eigen::VectorXd eg0_test2(n);
    eG_test2 = eG;
    eg0_test2 = eg0;
  	ex.setZero();
    solver_flag = quadprog_eigen::solve_quadprog(eG_test2, eg0_test2, eCE.transpose(), ece0, eCI.transpose(), eci0, ex, objVal, iter);
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
  std::cout << "obj: " << objVal << "\nx: \n" << ex << "\niter:" << iter << "\n\n";
  // std::cout << "Equality violation: " << std::endl;
  // std::cout << eCE*ex + ece0 << std::endl;
  // std::cout << "Inequality violation: " << std::endl;
  // std::cout << eCI*ex + eci0 << std::endl;

  // Solving use precomputed cholosky computation
  tic = btime::microsec_clock::local_time();

  Eigen::MatrixXd eG_test3(n, n);
  eG_test3 = eG;

  Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> chol(eG_test3.cols());

  double c1; 

  /* compute the trace of the original matrix G */
  c1 = eG_test3.trace();

  /* decompose the matrix G in the form LL^T */
  chol.compute(eG_test3);

  for (int i = 0; i < count; ++i) {
    Eigen::VectorXd eg0_test3(n);
    eg0_test3 = eg0;
    ex.setZero();
    solver_flag = quadprog_eigen::solve_quadprog2(chol, c1, eg0_test3, eCE.transpose(), ece0, eCI.transpose(), eci0, ex, objVal, iter);
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
  std::cout << "Elapsed time of quadprog eigen with one computation of cholosky decomposition: " << std::setprecision(8) << toc.total_milliseconds() << " ms\n";
  std::cout << "obj: " << objVal << "\nx: \n" << ex << "\niter:" << iter<< "\n\n";
  // std::cout << "Equality violation: " << std::endl;
  // std::cout << eCE*ex + ece0 << std::endl;
  // std::cout << "Inequality violation: " << std::endl;
  // std::cout << eCI*ex + eci0 << std::endl;
}
