#pragma once
#include <cmath>
#include "vector.h"
#include "rocket.h"
#include "body.h"

// gravitational constant
double G = 6.67428e-11;

class ComputerV1
{
	private:
		
		Rocket rocket;
		Body body;
		
		double dt;
	
	public:
		
		ComputerV1(Rocket& rocket, Body& body, double dt);
		
		double get_prograde(Rocket& rocket, Body& body);
		double get_burn_time(double delta_v);
};

ComputerV1::ComputerV1(Rocket& rocket, Body& body, double dt)
{
	this->rocket = rocket;
	this->body = body;
	this->dt = dt;
}

double ComputerV1::get_prograde(Rocket& rocket, Body& body)
{
	double parallel = atan2(rocket.y-body.y, rocket.x-body.x);
	
	return parallel-M_PI/2;
}

Vector get_delta_v(Vector& targ_vect)
{
	double v_burn = sqrt(pow(targ_vect.get_speed(), 2) + pow(rocket.get_speed(), 2) - 2*targ_vect.get_speed()*rocket.get_speed()*cos(targ_vect.get_angle()-rocket.get_angle()));
	
	double theta_burn = asin(targ_vect.get_speed()/v_burn * sin(targ_vect.get_angle()-theta_burn));
	
	return Vector(v_burn, theta_burn);
}

double get_burn_time(double delta_v)
{
	return rocket.get_mass()/rocket.get_fuel_depr()*(1-exp(-1*delta_v/rocket.get_eject_v()));
}

