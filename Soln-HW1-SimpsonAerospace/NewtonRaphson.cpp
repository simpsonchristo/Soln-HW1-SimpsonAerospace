/// NewtonRaphson.cpp : Vector-space Newton-Raphson solver
///inputs: f(x), f'(x), x-initial guess, err-error limit
///outputs: x0-final answer based on error limit, res-residuals
///error is determined by root-mean squared
//Copyright <2018> <SIMPSONAEROSPACE>
//
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "NewtonRaphson.h"
#include "stdafx.h"
///Setup
//typedef
template <class Bs>
using VectorXp = typename newton_raphson<Bs>::VectorXp;
template <class Bs>
using MatrixXp = typename newton_raphson<Bs>::MatrixXp;

//set
template <class Bs>
void newton_raphson<Bs>::set_function(std::function <VectorXp(VectorXp, VectorXp)> *fun) {
	fcn = *fun;
}
template <class Bs>
void newton_raphson<Bs>::set_function(std::function <MatrixXp(VectorXp, VectorXp)> *der) {
	drv = *der;
}
template <class Bs>
void newton_raphson<Bs>::set_x0(VectorXp state_vector) {
	x = state_vector;
}
template <class Bs>
void newton_raphson<Bs>::set_max_error(Bs max_error_allowed) {
	err = max_error_allowed;
}
template <class Bs>
void newton_raphson<Bs>::set_num_steps(int num_steps) {
	steps = num_steps;
}
template<class Bs>
void newton_raphson<Bs>::set_t(VectorXp time) {
	t = time;
}

//access
template <class Bs>
MatrixXp newton_raphson<Bs>::get_res() {
	return res;
}
template <class Bs>
MatrixXp newton_raphson<Bs>::get_x() {
	return x_new;
}
template <class Bs>
VectorXp newton_raphson<Bs>::get_x0() {
	return x;
}
template <class Bs>
VectorXp newton_raphson<Bs>::get_rms() {
	return rms;
}

//f(x)
template <class Bs>
VectorXp newton_raphson<Bs>::func(VectorXp x, VectorXp t) {
	return fcn(x, t);
}
//f'(x)
template<class Bs>
MatrixXp newton_raphson<Bs>::deriv(VectorXp x, VectorXp t) {
	return drv(x, t);
}

//////////////////////////////////////////////////////////////
///Calculations
Eigen::VectorXd::Index min_index;

//TODO: Make the while loop do something 
template <class Bs>
void newton_raphson<Bs>::iterate() {
	//determine if error conditions have been met
	while (err > rms.minCoeff())
	{
		//iterate for steps
		for (int i = 0; i < steps; i++)
		{
			//calculate next iteration
			xnew[:, i] = x - ((deriv(x,t).inverse())*func(x,t));
			//calculate residuals
			res[:, i] = xnew[:, i] - x;
			//calculate rms
			rms[i] = res[:, i].norm();
			//set x to xnew
			x = xnew[:, i];
		}
		index = rms.minCoeff(&min_index);
		if (err > rms.minCoeff()) {
			x = xnew[:, index];
		}
	}
}
