///Soln-HW1-SimpsonAerospace.cpp : Solution to assigned OD problem.
/// Using Newton-Raphson, the initial state is solved for assuming no error in
/// observations. 
//Copyright <2018> <SIMPSONAEROSPACE>
//
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "stdafx.h"
#include "NewtonRaphson.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>


using namespace std;

ofstream output_file;
std::string state_vector_file = "xnew.csv";
std::string residuals_file = "res.csv";

//kinematics
Eigen::VectorXd state_vector_at(double t, /*given*/ Eigen::VectorXd x0) {
	Eigen::VectorXd state_vector(5);
	state_vector(0) /*x*/     = x0(0) + x0(2) * t;
	state_vector(1) /*y*/     = x0(1) + x0(3) * t - x0(4) * 0.5*pow(t, 2);
	state_vector(2) /*x_dot*/ = x0(2);
	state_vector(3) /*y_dot*/ = x0(3) - x0(4) * t;
	state_vector(4) /*g*/     = x0(4);
	return state_vector;
}
//range
double range_eqn(Eigen::VectorXd x0, double t) {
	Eigen::VectorXd x = state_vector_at(t, x0);
	double rho = sqrt(pow(x(0) - 1.0, 2) + pow(x(1) - 1.0, 2));
	return rho;
}
//f(x)
Eigen::VectorXd residuals_of_x2rho(Eigen::VectorXd x0, Eigen::VectorXd t) {
	Eigen::VectorXd truth_observations(5);
	truth_observations << 7.0,
		8.00390597,
		8.94427191,
		9.801147892,
		10.630145813;

	Eigen::VectorXd rho(5);
	for (int i = 0; i < rho.rows(); i++)
	{
		rho(i) = range_eqn(x0, t(i));
	}

	return truth_observations - rho;
}
//f'(x)
Eigen::MatrixXd deriv_of_x2rho(Eigen::VectorXd x0, Eigen::VectorXd t) {
	Eigen::MatrixXd x_of_t(5,5);
	double rho;
		x_of_t.col(0) = state_vector_at(t(0), x0);
		x_of_t.col(1) = state_vector_at(t(1), x0);
		x_of_t.col(2) = state_vector_at(t(2), x0);
		x_of_t.col(3) = state_vector_at(t(3), x0);
		x_of_t.col(4) = state_vector_at(t(4), x0);
		

	for (int j = 0; j < x_of_t.cols(); j++)
	{
		rho = sqrt(pow(x_of_t(0, j) - 1.0, 2) + pow(x_of_t(1, j) - 1.0, 2));

		x_of_t(2, j) = (t(j)*(x_of_t(0, j) - 1.0)) / rho;

		x_of_t(3, j) = (t(j)*(x_of_t(1, j) - 1.0)) / rho;

		x_of_t(4, j) = (-pow(t(j), 2)*0.5*(x_of_t(1, j) - 1.0)) / rho;

		x_of_t(0, j) = (x_of_t(0, j) - 1.0) / rho;

		x_of_t(1, j) = (x_of_t(1, j) - 1.0) / rho;
	}
	
	return x_of_t;
}


int main()
{
	Eigen::VectorXd x0(5);
	x0(0)/*x*/     = 1.5;
	x0(1)/*y*/     = 10.0;
	x0(2)/*x_dot*/ = 2.2;
	x0(3)/*y_dot*/ = 0.5;
	x0(4)/*g*/     = 0.3;

	Eigen::VectorXd t(5);
	t(0) = 0;
	t(1) = 1;
	t(2) = 2;
	t(3) = 3;
	t(4) = 4;
	
	function <Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> primary_function = bind(&residuals_of_x2rho, placeholders::_1, placeholders::_2);
	function <Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> derivative_function = bind(&deriv_of_x2rho, placeholders::_1, placeholders::_2);
	//initialize problem
	newton_raphson<double> od;
	od.set_function(&primary_function);
	od.set_deriv(&derivative_function);
	od.set_max_error(1.0e-6);
	od.set_num_steps(1.0e+6);
	od.set_t(t);
	od.set_x0(x0);
	////calculate
	od.iterate();
	////get results
	Eigen::MatrixXd residuals = od.get_res();
	Eigen::MatrixXd all_x = od.get_x();
	Eigen::VectorXd x0_final = od.get_x0();
	Eigen::VectorXd all_rms = od.get_rms();

	cout << residuals << std::endl;
	double j;
	cin >> j;

    return 0;
}

