import numpy as np
import time
import serial
import jetson.inference
import jetson.utils
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os


ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
camera = jetson.utils.videoSource("/dev/video0",['-input-width=1280','-input-height=720'])
# display = jetson.utils.videoOutput("display://0")


def make_command(cmd,left_speed,right_speed):
	cmd_to_arduino = cmd + chr(int(left_speed)) + chr(int(right_speed))
	return cmd_to_arduino

def nothing(x):
	pass

cam = cv2.VideoCapture(0)
move = False
cmd = 'S'
left_speed = 0
right_speed = 0
zero_str = 'S' + chr(0) + chr(0)
rospy.init_node('jetbot', anonymous=True)
loop_rate = rospy.Rate(10)
bridge = CvBridge()

try:
	#while display.IsStreaming():
	img_pub = rospy.Publisher('nano_img',Image,queue_size=10)
	while not rospy.is_shutdown():
		#img = camera.Capture()
		_, img = cam.read()
		frame_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
		cuda_frame = jetson.utils.cudaFromNumpy(frame_rgba)
		detections = net.Detect(cuda_frame)
		for detection in detections:
			print(detection)
			if(detection.ClassID == 1):
				x = int(detection.Left)
				y = int(detection.Bottom)
				w = int(detection.Width)
				h = int(detection.Height)
				cv2.rectangle(img,(x,y),(x+w,y-h),(215,0,0),3)
				x_cord = detection.Center[0]
				print("Object detected at " + str(x_cord))
				move = True

			else:
				x_cord = 320
				move = False

			rospy.loginfo('Publishing image with detection')
			img_pub.publish(bridge.cv2_to_imgmsg(img, encoding="bgr8"))
			loop_rate.sleep()


		if(move):
			if(x_cord < 300):       #let there be 20px deadband each side of center of screen
				#left_speed = (x_cord/320)*62 + 64
				left_speed = 0
				right_speed = 86

			elif(x_cord > 340):
				left_speed = 86
				right_speed = 0
				#right_speed = ((640 - x_cord)/320)*62 + 64

			else:
				left_speed = 120
				right_speed = 120

		else:
			left_speed = 86
			right_speed = 86

		command = make_command(cmd, left_speed, right_speed)
		print(command)
		ser.write(command.encode())


		time.sleep(0.02)
		#end = time.time()

		#print(end-start)
		#time.sleep(0.25)

except KeyboardInterrupt:
	ser.write(zero_str.encode())
