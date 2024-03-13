import pandas as pd
import pygame
import numpy as np
import math

pi = math.pi

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (65, 65, 65)
P_RADIUS = 3
ROCKET_SIZE = 5
ROCKET_DIR_SIZE = 7

WINDOW_W = 700
WINDOW_H = 700

RADIUS_BODY = 1.7374e6

points = []

sim_speeds = [1, 5, 10, 20, 50, 100, 500, 1000, 2000]

trajectory = pd.read_csv("Traj.csv", skiprows = 1)


# amount of meters per pixel
scale = 1e4


def draw_point(x, y, color, size = P_RADIUS):
	pygame.draw.circle(screen, color, (WINDOW_W/2+x, WINDOW_H/2-y), size)
	
def draw_rocket(x, y, angle, color):
	angle*=pi/180
	fpoint = (int(WINDOW_W/2+x+ROCKET_DIR_SIZE*math.cos(angle)), int(WINDOW_H/2-y-ROCKET_DIR_SIZE*math.sin(angle)))
	tpoint = (int(WINDOW_W/2+x+ROCKET_SIZE*math.cos(angle+2*pi/3)), int(WINDOW_H/2-y-ROCKET_SIZE*math.sin(angle+2*pi/3)))
	bpoint = (int(WINDOW_W/2+x+ROCKET_SIZE*math.cos(angle-2*pi/3)), int(WINDOW_H/2-y-ROCKET_SIZE*math.sin(angle-2*pi/3)))
	pygame.draw.polygon(screen, color, [fpoint, tpoint, bpoint])

def draw_body(x, y, color, size = int(RADIUS_BODY/scale)):
	pygame.draw.circle(screen, color, (WINDOW_W/2+x, WINDOW_H/2-y), size)

def get_height(traj, ind):
	return pow(pow(trajectory["X (m)"][ind], 2)+pow(trajectory["Y (m)"][ind], 2), 1/2) - RADIUS_BODY

pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Rocket")
clock = pygame.time.Clock();

end = False
crashed = False

counter = 0
ind = 0;

traj_length = len(trajectory["X (m)"])

while not end:
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			end = True;
		if event.type == pygame.KEYUP:
			if event.key == pygame.K_RIGHT:
				if ind < len(sim_speeds)-1:
					ind+=1;
			if event.key == pygame.K_LEFT:
				if ind > 0:
					ind-=1;
	
	inc = sim_speeds[ind];
	
	screen.fill((0, 0, 0))
	
	draw_body(0, 0, GRAY)
	
	for i in range(0, counter):
		draw_point(int(float(trajectory["X (m)"][i])/scale), int(float(trajectory["Y (m)"][i])/scale), WHITE, size = P_RADIUS/2)
		
	if not crashed:
		draw_rocket(int(float(trajectory["X (m)"][counter])/scale), int(float(trajectory["Y (m)"][counter])/scale), float(trajectory["A (d)"][counter]), RED)	
	
	pygame.display.update()
	clock.tick(10)
	
	if counter>=traj_length-1:
		crashed = True;
		
	if not crashed:
		
		print("Time: "+round(trajectory["Time (s)"][counter], 2).astype(str)+"  |  Height: "+round(get_height(trajectory, counter), 2).astype(str)+"  |  Warp: "+str(inc)+"x")
		
		counter+=inc
		
		if counter>=traj_length:
			counter = traj_length - 1
	
pygame.quit()
quit()