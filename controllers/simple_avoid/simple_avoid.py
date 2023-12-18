from controller import Robot

# Get reference to the robot.
robot = Robot()

# Get simulation step length.
timeStep = int(robot.getBasicTimeStep())


left_motor = robot.getDevice("left_wheel_motor")
right_motor = robot.getDevice("right_wheel_motor")

light_sensor_left = robot.getDevice("ls0")
light_sensor_right = robot.getDevice("ls1")

light_sensor_left.enable(timeStep)
light_sensor_right.enable(timeStep)

# Disable motor PID control mode.
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# initialize velocity with 0
left_motor.setVelocity(0)
right_motor.setVelocity(0)

LIGHT_THRESHOLD = 700
LIGHT_BOOST = 0
LM_SCALE = 100

while robot.step(timeStep) != -1:
    ll = light_sensor_left.getValue()
    lr = light_sensor_right.getValue()

    if ll + lr < LIGHT_THRESHOLD:
        ml = 0
        mr = 0
    else:
        ml = lr / LM_SCALE - ll / LM_SCALE + (LIGHT_BOOST - (lr + ll) / 2) / LM_SCALE
        mr = ll / LM_SCALE - lr / LM_SCALE + (LIGHT_BOOST - (lr + ll) / 2) / LM_SCALE

    # print(f"{ll:.2f}, {lr:.2f}, {ml:.2f}, {mr:.2f}")

    # cross connections between sensors and motors
    left_motor.setVelocity(mr)
    right_motor.setVelocity(ml)
