import numpy as np
import time
import serial
import rospy
from jetbot_follow_person.msg import speed_cmd
import os


ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)

def make_command(cmd,left_speed,right_speed):
	cmd_to_arduino = cmd + chr(int(left_speed)) + chr(int(right_speed))
	return cmd_to_arduino

def nothing(x):
	pass

move = False
cmd = 'S'
left_speed = 0
right_speed = 0
zero_str = 'S' + chr(0) + chr(0)
loop_rate = rospy.Rate(20)
bridge = CvBridge()

def callback(cmd):
	if(cmd.left_speed >= 0):
		left_speed = int(cmd.left_speed * 126 / 100)

	else:
		left_speed = int((-cmd.left_speed/100)*63)


	if(cmd.right_speed >= 0):
		right_speed = int(cmd.right_speed * 126 /100)

	else:
		right_speed = int((-cmd.right_speed/100)*63)

	command = make_command(cmd, left_speed, right_speed)
	print(command)
	rospy.loginfo("Got command from controller!")
	ser.write(command.encode())


def jetbot_move():
	rospy.init_node('jetbot_receiver', anonymous=True)
	command_sub = rospy.Subscriber("speed_command", speed_cmd, callback)

	rospy.spin()

