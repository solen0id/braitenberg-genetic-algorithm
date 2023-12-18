import sys
from pathlib import Path

import numpy as np
from controller import Robot

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(BASE_DIR / "scripts"))
from neural import Fitness, NeuralNetwork

# Get reference to the robot.
robot = Robot()

# Get simulation step length.
timeStep = int(robot.getBasicTimeStep())


left_motor = robot.getDevice("left_wheel_motor")
right_motor = robot.getDevice("right_wheel_motor")

left_wheel_sensor = robot.getDevice("left_wheel_sensor")
right_wheel_sensor = robot.getDevice("right_wheel_sensor")

light_sensor_left = robot.getDevice("ls0")
light_sensor_right = robot.getDevice("ls1")

distance_sensor_left = robot.getDevice("ds0")
distance_sensor_right = robot.getDevice("ds1")

gps_sensor = robot.getDevice("gps")

light_sensor_left.enable(timeStep)
light_sensor_right.enable(timeStep)

distance_sensor_left.enable(timeStep)
distance_sensor_right.enable(timeStep)

left_wheel_sensor.enable(timeStep)
right_wheel_sensor.enable(timeStep)

gps_sensor.enable(timeStep)

# Disable motor PID control mode.
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# initialize velocity with 0
left_motor.setVelocity(0)
right_motor.setVelocity(0)


def evaluate_fitness():
    [_, _, dl, dr, _, _, gps] = get_sensor_readings()

    # compute gps distance since last frame
    x, y, _ = gps
    fitness.distance += np.sqrt(
        (x - fitness.distance_x) ** 2 + (y - fitness.distance_y) ** 2
    )
    fitness.distance_x = x
    fitness.distance_y = y

    fitness.collisions += 1 if dl > 1023 or dr > 1023 else 0


def get_sensor_readings():
    ll = light_sensor_left.getValue()
    lr = light_sensor_right.getValue()

    dl = distance_sensor_left.getValue()
    dr = distance_sensor_right.getValue()

    pl = left_wheel_sensor.getValue()
    pr = right_wheel_sensor.getValue()

    gps = gps_sensor.getValues()

    return [ll, lr, dl, dr, pl, pr, gps]


fitness = Fitness()
nn = NeuralNetwork.from_file(robot.getCustomData())

frame_counter = 0

while robot.step(timeStep) != -1:
    inputs = get_sensor_readings()[:4]
    outputs = nn.think(inputs)

    ml, mr = outputs

    left_motor.setVelocity(ml)
    right_motor.setVelocity(mr)

    frame_counter += 1
    if frame_counter % 10 == 0:
        evaluate_fitness()
        robot.setCustomData(str(fitness.to_val()))
