/// NewtonRaphson.cpp : Vector-space Newton-Raphson solver
///inputs: f(x), f'(x), x0-initial guess, err-error limit
///outputs: x0-final answer based on error limit, res-residuals
///error is determined by root-mean squared
//Copyright <2018> <SIMPSONAEROSPACE>
//
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <cmath>
//#include <string>
//#include <vector>
//#include <numeric>
//#include <functional>
//#include <stdlib.h>
//#include <thread>
#include <Eigen/Dense>
#include "stdafx.h"

#ifndef NEWTONRAPHSON_H
#define NEWTONRAPHSON_H

//inputs:
// f(x), f'(x), x0-initial guess, err-error limit
//outputs:
// x0-final answer based on error limit, res-residuals

//NOTES: 
//User will assign inputs based on use case
//Write a class that allows the user to do this

template <class Bs>
class newton_raphson {
	///Define f(x), f'(x), x0, and error limit for NR.
	public:
		//typedef
		typedef Eigen::Matrix<Bs, Eigen::Dynamic, 1> VectorXp;
		typedef Eigen::Matrix<Bs, Eigen::Dynamic, Eigen::Dynamic> MatrixXp;

		//set functions
		void set_function(std::function <VectorXp(VectorXp, VectorXp)> *fun);
		std::function <VectorXp(VectorXp, VectorXp)> fcn;
		void set_deriv(std::function <MatrixXp(VectorXp, VectorXp)> *der);
		std::function <MatrixXp(VectorXp, VectorXp)> drv;
		//declare initial state
		void set_x0(VectorXp state_vector);
		void set_t(VectorXp time);
		//declare max error allowed
		void set_max_error(Bs max_error_allowed);
		//declare number of steps to use
		void set_num_steps(int num_steps);

		//f(x)
		VectorXp func(VectorXp x, VectorXp t);
		//f'(x)
		MatrixXp deriv(VectorXp x, VectorXp t);

		//access
		MatrixXp get_res();
		MatrixXp get_x();
		VectorXp get_x0();
		VectorXp get_rms();

		//iterate
		void iterate();

	private:
		//initial and final state vector
		VectorXp x;
		//time
		VectorXp t;
		//max error allowed
		Bs err;
		//next iteration
		MatrixXp xnew;
		//residuals
		MatrixXp res;
		//number of steps for calculating
		int steps;
		//rms
		VectorXp rms;
		//store all state vector iterations, residuals, and rms
		MatrixXp store_xnew;
		MatrixXp store_res;
		VectorXp store_rms;




};

#endif /*NEWTONRAPHSON_H*/
