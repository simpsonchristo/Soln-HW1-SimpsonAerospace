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
Eigen::VectorXd state_vector_at(/*given*/ Eigen::VectorXd x0, double t) {
	//x0(0)/*x*/
	//x0(1)/*y*/
	//x0(2)/*x_dot*/
	//x0(3)/*y_dot*/
	//x0(4)/*g*/
	Eigen::VectorXd state_vector(5);
	state_vector(0) /*x*/ = x0(0) + x0(2) * t;
	state_vector(1) /*y*/ = x0(1) + x0(3) * t - x0(4) * 0.5*pow(t, 2);
	state_vector(2) /*x_dot*/ = x0(2);
	state_vector(3) /*y_dot*/ = x0(3) - x0(4) * t;
	state_vector(4) /*g*/ = x0(4);
	return state_vector;
}
//range
double range_eqn(Eigen::VectorXd x0, Eigen::VectorXd xs, double t) {
	Eigen::VectorXd x = state_vector_at(x0, t);
	double rho = sqrt(pow((x(0) - xs(0)), 2) + pow((x(1) - xs(1)), 2));
	return rho;
}
//f(x)
Eigen::VectorXd x2rho(Eigen::VectorXd x0, Eigen::VectorXd xs, Eigen::VectorXd actual_rho, Eigen::VectorXd t) {

	Eigen::VectorXd rho(5);
	for (int i = 0; i < rho.rows(); i++)
	{
		rho(i) = range_eqn(x0, xs, t(i));		
	}

	Eigen::VectorXd rho_res = actual_rho - rho;
	return rho_res;
}
//f'(x)
Eigen::MatrixXd deriv_of_x2rho(Eigen::VectorXd x0, Eigen::VectorXd xs, Eigen::VectorXd t) {
	Eigen::MatrixXd x_of_t(5, 5);
	Eigen::VectorXd rho(5);
	for (int i = 0; i < rho.rows(); i++)
	{
		rho(i) = range_eqn(x0, xs, t(i));
		x_of_t.row(i) = state_vector_at(x0, t(i));
	}


	for (int j = 0; j < x_of_t.cols(); j++)
	{
		x_of_t(j, 0) = (x_of_t(j, 0) - xs(0)) / rho(j);
		x_of_t(j, 1) = (x_of_t(j, 1) - xs(1)) / rho(j);
		x_of_t(j, 2) = t(j)*x_of_t(j,0);
		x_of_t(j, 3) = t(j)*x_of_t(j,1);
		x_of_t(j, 4) = -0.5*pow(t(j),2) * x_of_t(j,1);
	}

	return x_of_t;
}


int main()
{
	//initial state
	Eigen::VectorXd x0(5);
	x0(0)/*x*/     = 1.5;
	x0(1)/*y*/     = 10.0;
	x0(2)/*x_dot*/ = 2.2;
	x0(3)/*y_dot*/ = 0.5;
	x0(4)/*g*/     = 0.3;
	//GS
	Eigen::VectorXd xs(4);
	xs(0)/*x*/ = 1.0;
	xs(1)/*y*/ = 1.0;
	xs(2)/*x_dot*/ = 0.0;
	xs(3)/*y_dot*/ = 0.0;
	//time
	Eigen::VectorXd t(5);
	t(0) = 0;
	t(1) = 1;
	t(2) = 2;
	t(3) = 3;
	t(4) = 4;
	//actual range
	Eigen::VectorXd actual_rho(5);
	actual_rho << 7.0,
                  8.00390597,
		          8.94427191,
		          9.801147892,
		          10.630145813;
	
	
	function <Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)> primary_function = bind(&x2rho, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
	function <Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)> derivative_function = bind(&deriv_of_x2rho, placeholders::_1, placeholders::_2, placeholders::_3);
	//initialize problem
	newton_raphson<double> od;
	od.set_function(&primary_function);
	od.set_deriv(&derivative_function);
	od.set_max_error(1.0e-6);
	od.set_num_steps(1.0e+4);
	od.set_t(t);
	od.set_x0(x0);
	od.set_actual_range(actual_rho);
	od.set_ground_station(xs);
	////calculate
	od.iterate();
	////get results
	Eigen::MatrixXd residuals = od.get_res();
	Eigen::MatrixXd all_x = od.get_x();
	Eigen::VectorXd x0_final = od.get_x0();
	Eigen::VectorXd all_rms = od.get_rms();
	
	////Output
	cout << "Error for each iteration..." << endl;
	cout << all_rms << endl;
	cout << "Each iterations output..." << endl;
	cout << all_x << std::endl;
	//pause/end
	double j;
	cin >> j;

    return 0;
}

