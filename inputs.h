#include <cmath>
#include "rocket.h"

double DEG_TO_RAD = M_PI/180;

class RVars
{
	
	public:
		// Mass of empty rocket
		static const double MASS = 100;
		// Mass of fuel
		static const double FUEL = 1000;
		// Mass of fuel ejected per second
		static const double FUEL_DEPR = 10;
		// Exit velocity of ejected fuel
		static const double FUEL_EXIT_V = 850;
		
		// Torque of rocket
		static const double TORQUE = 3;
			
		static bool is_time(double time, double time_start, double time_end);
		
		// returns angle of rocket direction given height of rocket
		static void control_rocket(Rocket& rocket, double dt, double time, double height, double prograde_angle);
		
};

bool RVars::is_time(double time, double time_start, double time_end)
{
	if(time>=time_start && time<=time_end)
	{
		return true;
	}
	
	return false;
}

void RVars::control_rocket(Rocket& rocket, double dt, double time, double height, double prograde_angle)
{
	// torque =  degrees per second
	double torque = RVars::TORQUE/6;

	if(is_time(time, 0, 30))
	{
		rocket.set_throttle(1);
		rocket.turn(torque*dt*DEG_TO_RAD);		
	}
	else if(is_time(time, 30, 40))
	{
		rocket.set_throttle(0.7);
		rocket.set_dir((prograde_angle*RAD_TO_DEG-5)*DEG_TO_RAD);
	}
	else
	{
		rocket.set_throttle(0.4);
		rocket.set_dir(prograde_angle);
	}
}