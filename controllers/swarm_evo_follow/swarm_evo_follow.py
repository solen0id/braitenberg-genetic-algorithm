import sys
from pathlib import Path

import numpy as np
from controller import Robot

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "scripts"))

from neural import FollowLightFitness, NeuralNetwork

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
    [ll, lr, _, _, _, _, _] = get_sensor_readings()

    # track light intensity
    fitness.light_intensity += ll + lr


def get_sensor_readings():
    ll = light_sensor_left.getValue()
    lr = light_sensor_right.getValue()

    dl = distance_sensor_left.getValue()
    dr = distance_sensor_right.getValue()

    pl = left_wheel_sensor.getValue()
    pr = right_wheel_sensor.getValue()

    gps = gps_sensor.getValues()

    return [ll, lr, dl, dr, pl, pr, gps]


def get_nn_sensors_normalized():
    [ll, lr, dl, dr, _, _, _] = get_sensor_readings()

    # normalize sensors
    ll = normalize(ll, 333, 500, -1, 1)  # measured min and max values
    lr = normalize(lr, 333, 500, -1, 1)  # measured min and max values
    dl = normalize(dl, 0, 1024, -1, 1)
    dr = normalize(dr, 0, 1024, -1, 1)

    return [ll, lr, dl, dr]


def normalize(x, xmin, xmax, a, b):
    # normalize x from [xmin, xmax] to [a, b]
    return (b - a) * (x - xmin) / (xmax - xmin) + a


frame_counter = 0
nn = NeuralNetwork.from_file(robot.getCustomData())


def init_robot():
    global fitness
    global frame_counter

    if frame_counter == 1:
        [ll, lr, _, _, _, _, _] = get_sensor_readings()
        fitness = FollowLightFitness(light_intensity=ll + lr)


while robot.step(timeStep) != -1:
    init_robot()

    [ll, lr, _, _] = get_nn_sensors_normalized()
    inputs = [ll, lr]
    ml, mr = nn.think(inputs)
    ml, mr = max(-10, ml), max(-10, mr)
    ml, mr = min(10, ml), min(10, mr)

    left_motor.setVelocity(ml)
    right_motor.setVelocity(mr)

    frame_counter += 1

    if frame_counter % 10 == 0:
        evaluate_fitness()
        robot.setCustomData(str(fitness.to_val()))
