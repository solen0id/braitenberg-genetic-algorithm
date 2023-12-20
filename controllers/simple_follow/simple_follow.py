from controller import Robot

# Get reference to the robot.
robot = Robot()

# Get simulation step length.
timeStep = int(robot.getBasicTimeStep())


left_motor = robot.getDevice("left_wheel_motor")
right_motor = robot.getDevice("right_wheel_motor")

light_sensor_left = robot.getDevice("ls0")
light_sensor_right = robot.getDevice("ls1")

distance_sensor_left = robot.getDevice("ds0")
distance_sensor_right = robot.getDevice("ds1")

light_sensor_left.enable(timeStep)
light_sensor_right.enable(timeStep)

distance_sensor_left.enable(timeStep)
distance_sensor_right.enable(timeStep)


# Disable motor PID control mode.
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# initialize velocity with 0
left_motor.setVelocity(0)
right_motor.setVelocity(0)

LIGHT_THRESHOLD = 700
LIGHT_BOOST = 750
LM_SCALE = 100

while robot.step(timeStep) != -1:
    ll = light_sensor_left.getValue()
    lr = light_sensor_right.getValue()

    dl = distance_sensor_left.getValue()
    dr = distance_sensor_right.getValue()

    # ml = mr = 0

    if ll + lr < LIGHT_THRESHOLD:
        ml = 0
        mr = 0
    else:
        ml = lr / LM_SCALE - ll / LM_SCALE + (LIGHT_BOOST - (lr + ll) / 2) / LM_SCALE
        mr = ll / LM_SCALE - lr / LM_SCALE + (LIGHT_BOOST - (lr + ll) / 2) / LM_SCALE

    print(
        "{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(ll, lr, dl, dr, ml, mr)
    )

    # left_motor.setVelocity(ml)
    # right_motor.setVelocity(mr)
