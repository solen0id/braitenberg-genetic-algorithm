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
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# initialize velocity with 0
left_motor.setVelocity(0)
right_motor.setVelocity(0)

while robot.step(timeStep) != -1:
    
    # measure values between [0,1000]
    ll = light_sensor_left.getValue()
    lr = light_sensor_right.getValue()
    
    print( "{}, {}".format(ll,lr) )
    
    left_motor.setVelocity ( lr/100 - ll/100 )
    right_motor.setVelocity( ll/100 - lr/100 )
