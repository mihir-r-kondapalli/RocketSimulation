#pragma once
#include "vector.h"

class Rocket
{
	public:
		
		double x; // horizontal position of rocket with respect to gravitational body (m)
		double y; // vertical positionn of rocket with respect to gravitational body (m)
		
		double dir; // direction of rocket and its propulsion (in radians)
		
		Vector velocity; // velocity vector of rocket
		
		double height;   // height of rocket above radius of gravitational body (m)
		
		double mass_rocket; // mass of empty rocket (kg)
		
		double fuel; // amount of fuel (kg)
		double fuel_v; // fuel exit velocity (m/s)
		double fuel_capacity; // maximum amount of fuel (kg)
		double fuel_depr; // amount of fuel that is discharged (kg/s)
		
		double throttle; // fraction of fuel ejected
		
		Rocket(){}
		
		Rocket(double mass_rocket, double fuel, double fuel_depr, double fuel_v, double x, double y)
		{	
			this->mass_rocket = mass_rocket;
			this->fuel = fuel;
			this->fuel_depr = fuel_depr;
			this->fuel_v = fuel_v;
			
			this->x = x;
			this->y = y;
						
			velocity = Vector(0, 90);
			dir = M_PI/2;
		}
		
		double get_fuel_depr(double dt);
		double get_fuel_depr();
		double get_eject_v();
		double get_mass();
		double get_speed();
		double get_angle();
		double get_dir();
		void set_dir(double dir);
		
		// command methods
		void turn(double delta);
		double get_throttle();
		void set_throttle(double throttle);
		void cut();
		void full(); 
};


double Rocket::get_fuel_depr(double dt)
{
	if(fuel_depr*throttle*dt<=fuel){
		return fuel_depr*throttle;
	}else if(fuel>0){
		return fuel/dt;
	}else{
		return 0;
	}
}

double Rocket::get_fuel_depr()
{
	return fuel_depr*throttle;
}

double Rocket::get_eject_v()
{
	return fuel_v;
}

double Rocket::get_mass()
{
	return mass_rocket+fuel;
}

double Rocket::get_speed()
{
	return velocity.get_speed();
}

double Rocket::get_angle()
{
	return velocity.get_angle();
}

double Rocket::get_dir()
{
	return dir;
}

void Rocket::set_dir(double dir)
{
	this->dir = dir;
}

double Rocket::get_throttle()
{
	return throttle;
}

void Rocket::turn(double angle_change)
{
	dir+=angle_change;
}

void Rocket::set_throttle(double throttle)
{
	this->throttle = throttle;
}

void Rocket::cut()
{
	throttle = 0;
}

void Rocket::full()
{
	throttle = 1;
}

