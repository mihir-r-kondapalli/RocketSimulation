#include <cmath>
#include <iostream>
#include <fstream>
#include "rocket_eqs.h"
#include "rocket.h"
#include "vector.h"
#include "inputs.h"

using namespace std;


double dist(Rocket r, Body b)
{
	return sqrt(pow(r.x-b.x, 2)+pow(r.y-b.y, 2));
}

class Simulation
{
	public:
		
		int counter = 0;
		
		double max_height = 0;
		double max_time = 0;
		double current_height = 0;
		
		// time limit for simulation
		double limit;
		
		double dt = 0.0001;
		const double G = 6.67430e-11;
		const string OUT_FILE_NAME = "Traj.csv";
		
		ofstream outfile;
		Rocket rocket;
		Body body;
		
		Simulation(Rocket rocket, Body body, double dt, double limit)
		{
			this->rocket = rocket;
			this->body = body;
			this->dt = dt;
			
			this->limit = limit;
			
			outfile.open(OUT_FILE_NAME);
			outfile << to_string(rocket.mass_rocket)+" kgs, "+to_string(rocket.fuel)+" kgs, "+to_string(rocket.fuel_depr)+" kgs/s, "+to_string(rocket.fuel_v)+" m/s\n";
			outfile << "Time (s),X (m),Y (m),A (d)\n";
		}
		
		void update_vars();
		void print_data();
		void runstep();
		void output_to_file();
		bool check_stop();
};


void Simulation::update_vars()
{
	// rocket, time, height, prograde angle
	RVars::control_rocket(rocket, dt, counter*dt, current_height, get_prograde(rocket, body));
	
	calc_delta_v_rocket(rocket, dt);
	calc_delta_v_grav(rocket, body, dt);
	
	rocket.fuel = rocket.fuel - rocket.get_fuel_depr(dt)*dt;
	counter++;
	
	current_height = dist(rocket, body)-body.RADIUS;
	
	if(max_height<current_height)
	{
		max_height = current_height;
		max_time = counter*dt;
	}
}

void Simulation::print_data()
{
		printf("T: %10.5f  |  H: %13.5f  |  X: %13.5f  |  Y: %13.5f  |  V: %10.5f  |  A: %7.4f  |  M: %10.5f  |  F: %10.5f\n", counter*dt, current_height, rocket.x, rocket.y, rocket.get_speed(), rocket.get_angle()*RAD_TO_DEG, rocket.get_mass(), rocket.fuel);
}

void Simulation::output_to_file()
{
	outfile << to_string(counter*dt)+","+to_string(rocket.x)+","+to_string(rocket.y)+","+to_string(rocket.dir*RAD_TO_DEG)+"\n";
}

void Simulation::runstep()
{
	update_vars();
}

bool Simulation::check_stop()
{	
	// stops if rocket crashes, leaves shere of influence, reaches the time limit, or makes a complete orbit
	if(current_height<0 || current_height>6.43e7-body.RADIUS || counter*dt>=limit/* || (abs(rocket.x)<50 && rocket.y>1000+body.RADIUS)*/)
	{
		outfile.close();
		return true;
	}
	
	return false; 
}

void run(Simulation& simul)
{
	cout << "\n\n\n";
	
	int counter = 0;
	
	simul.print_data();
	while(!simul.check_stop())
	{
		counter++;
		simul.runstep();
		
		if(counter%1000==0)
		{
			simul.print_data();
			simul.output_to_file();
		}
		if(counter%100==0)
		{
			simul.output_to_file();
		}
	}
	
	printf("Max Height: %10.5f km  |  Time: %8.5f s\n", simul.max_height/(1e3), simul.max_time);

	cout << "\n\n\n";
}

int main()
{
	Body body(0, 0, 7.34767309e22, 1.7374e6);
	
	// Mass, Fuel, Fuel Depr, Fuel Exit Velocity, X, Y
	Rocket rocket(RVars::MASS, RVars::FUEL, RVars::FUEL_DEPR, RVars::FUEL_EXIT_V, 0, body.RADIUS);
	
	Simulation simul(rocket, body, 0.001, 5e4);

	run(simul);
}



