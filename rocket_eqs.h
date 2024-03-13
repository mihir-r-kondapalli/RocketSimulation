#pragma once
#include <cmath>
#include "vector.h"
#include "rocket.h"
#include "body.h"

// gravitational constant
double G = 6.67428e-11;

void calc_delta_v_rocket(Rocket& rocket, double dt)
{
	// delta v from rocket acceleration
	double delta_v_r = rocket.get_fuel_depr(dt)*rocket.fuel_v/(rocket.get_mass()-rocket.get_fuel_depr(dt)/2*dt)*dt;
	// dvr = u*vf/(mr - u/2*dt)*dt
	
	// x component
	rocket.x = rocket.x + (rocket.get_speed()*cos(rocket.get_angle()) + delta_v_r*cos(rocket.get_dir()))/2*dt;
	
	// y component
	rocket.y = rocket.y + (rocket.get_speed()*sin(rocket.get_angle()) + delta_v_r*sin(rocket.get_dir()))/2*dt;
	
	rocket.velocity.add_vector(delta_v_r, rocket.get_dir());
}

void calc_delta_v_grav(Rocket& rocket, Body& body, double dt)
{
	// delta v from gravity
	double delta_v_g = G*body.MASS/(pow(rocket.x-body.x, 2)+pow(rocket.y-body.y, 2))*dt;
	// dvg = GMe/(h+Re)^2
	double angle = atan2(body.y-rocket.y, body.x-rocket.x);
	
	// x component
	rocket.x = rocket.x + (rocket.get_speed()*cos(rocket.get_angle()) + delta_v_g*cos(angle))/2*dt;
	
	// y component
	rocket.y = rocket.y + (rocket.get_speed()*sin(rocket.get_angle()) + delta_v_g*sin(angle))/2*dt;
	
	rocket.velocity.add_vector(delta_v_g, angle);
}

double get_prograde(Rocket& rocket, Body& body)
{
	double parallel = atan2(rocket.y-body.y, rocket.x-body.x);
	
	return parallel-M_PI/2;
}
