# coding: utf8 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# LIBRARIES

import time
import random
import numpy as np
import cv2 as cv 
from copy import deepcopy
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# START

# input file name 
user_number = raw_input('Please enter user number: ')
file_name = user_number + '.mpg'

# instruct user to select q on keyboard to end recording session 
print 'Select q to end recording session'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# INSTANTIATION

# initialize haar cascade classifier object 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# read and save json pre-formatted model structure
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
# load and save pre-trained model weights in H5 format 
model.load_weights('facial_expression_model_weights.h5')

#print str(model.layers)
#print model.get_config()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# RECORDING 

# initialize video capture object from built in webcam
cap = cv.VideoCapture(0)
# establish video width width as integer
frame_width = int(cap.get(1))
# establish video height height as integer
frame_height = int(cap.get(1))
# initialize video recorder object 
out = cv.VideoWriter(file_name, cv.VideoWriter_fourcc('P','I','M','1'), 10, (frame_width,frame_height))
# create emotion strings 
emotion_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# NORMALIZATION

start_time = time.time()

# create empty lists for emotion recording 
anger_record = []
disgust_record = []
fear_record = []
happy_record = []
sad_record = []
surprise_record = []
neutral_record = []

# create while loop for video recording
while(True):
	# read video capture 
	ret, img = cap.read()
	# convert img from RGB to gray scale 
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# detect faces with gray scale using cascade classifier
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	# instantiate emotion container 
	emotion_list = []
	# create for loop to dynamically box any detected faces 
	for (x,y,w,h) in faces:
		# create rectangle for box shape
		cv.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2) 
		# crop detected faces 
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		# transform deteted faces to gray scale
		detected_face = cv.cvtColor(detected_face, cv.COLOR_BGR2GRAY)
		# resize detected_faces
		detected_face = cv.resize(detected_face, (48, 48), interpolation=cv.INTER_AREA)
		# convert image instance to 3D numpy array
		img_pixels = image.img_to_array(detected_face)
		# expand array shape 
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		# normalize pixel scale 
		img_pixels /= 255
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		
		# PREDICTIONS 

		# predict detected emotions with cascade classifier 
		predictions = model.predict(img_pixels, batch_size=1, verbose=1) 
		# save emotion probabilities
		angry, disgust, fear, happy, sad, surprise, neutral = [round(prob, 2) for lst in predictions for prob in lst]
		

		# store loop time post-prediction
		end_time = time.time()

		# append outputs to empty lists for emotion records 
		anger_record.append(angry)
		disgust_record.append(disgust)
		fear_record.append(fear)
		happy_record.append(happy)
		sad_record.append(sad)
		surprise_record.append(surprise)
		neutral_record.append(neutral)

		#data = np.loadtxt(filename, delimiter=",")

		timing = int(end_time-start_time)

		# save txt file as user number
		filename = user_number + '.txt'

		# open new user text file with append access 
		with open(filename, 'a') as f:
			
			# annotated text version for readability
			#f.write('{}, ANGER: {}, DISGUST: {}, FEAR: {}, HAPPY: {}, SAD: {}, SURPRISE: {}, NEUTRAL: {}\n'.format(int(end_time-start_time),angry,disgust,fear,happy,sad,surprise,neutral))
			# non-annotated text version
			f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(timing, angry, disgust, fear, happy, sad, surprise, neutral))

		# store maximum prediction result  
		max_index = np.argmax(predictions[0])
		# key max result with emotion strings
		top_emotion = emotion_labels[max_index]
		
		# display top emotion as string value 
		cv.putText(img, "EMOTION: {}".format(top_emotion), (int(x+75), int(y+250)), cv.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 1, lineType=cv.LINE_AA)
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	'''
	# RECORDING 
	#full_emotion_record = [int(end_time-start_time), anger_record, disgust_record, fear_record, happy_record, sad_record, surprise_record, neutral_record]
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	def animate(i):
		global filename
		graph_data = open(filename, 'r').read()
		lines = graph_data.split('\n')
		xs = []
		y_angry = []
		y_disgust = []
		y_fear = []
		y_happy = []
		y_sad = []
		y_surprise = []
		y_neutral = []
		for line in lines: 
			if len(line) > 1:
				timing, angry, disgust, fear, happy, sad, surprise, neutral = line.split(',')
				xs.append(timing)
				y_angry.append(angry)
				y_disgust.append(disgut)
				y_fear.append(fear)
				y_happy.append(happy)
				y_sad.append(sad)
				y_surprise.append(surprise)
				y_neutral.append(neutral)
		ax1.clear()
		ax1.plot(xs, y_angry)
		ax1.plot(xs, y_disgust)
		ax1.plot(xs, y_fear)
		ax1.plot(xs, y_happy)
		ax1.plot(xs, y_sad)
		ax1.plot(xs, y_surprise)
		ax1.plot(xs, y_neutral)
		
	ani = animation.FuncAnimation(fig, animate, interval=1000)
	plt.show()	
	plt.pause(0.0001)
	'''

	# if detection True, write recording 
	if ret == True:
				
		#out.write(img)
		# show results
		cv.imshow('imbreakg', img)
		
		# if user hits 'q' end recording 
		if cv.waitKey(1) & 0xFF == ord('q'):
			
			break

	# else break 
	else:

		break

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

'''

 # PLOTTING 

fig = plt.figure()
ax1 = fig.add_subplot(111)

def animate(i):
	graph_data = open(filename, 'r').read()
	lines = graph_data.split('\n')
	xs = []
	y_angry = []
	y_disgust = []
	y_fear = []
	y_happy = []
	y_sad = []
	y_surprise = []
	y_neutral = []
	for line in lines: 
		if len(line) > 0:
			time, angry, disgust, fear, happy, sad, surprise, neutral = line.split(',')
			xs.append(time)
			y_angry.append(angry)
			y_disgust.append(disgust)
			y_fear.append(fear)
			y_happy.append(happy)
			y_sad.append(sad)
			y_surprise.append(surprise)
			y_neutral.append(neutral)
	ax1.clear()
	ax1.plot(xs, y_angry)
	ax1.plot(xs, y_disgust)
	ax1.plot(xs, y_fear)
	ax1.plot(xs, y_happy)
	ax1.plot(xs, y_sad)
	ax1.plot(xs, y_surprise)
	ax1.plot(xs, y_neutral)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# FINISH 

# end video capture
cap.release()
# end video writer 
out.release()
# close session
cv.destroyAllWindows()

print 'Session recording completed for {}\n'. format(user_number)

print "Average Probability:\n"
print 'Anger', round(np.mean(anger_record), 2)
print 'Disgust', round(np.mean(disgust_record), 2)
print 'Fear', round(np.mean(fear_record), 2)
print 'Happiness', round(np.mean(happy_record), 2)
print 'Sadness', round(np.mean(sad_record), 2)
print 'Surprise', round(np.mean(anger_record), 2)
print 'Neutrality', round(np.mean(neutral_record), 2)

print '\n'

print "Standard Deviations:\n"
print 'Anger', round(np.std(anger_record), 2)
print 'Disgust', round(np.std(disgust_record), 2)
print 'Fear', round(np.std(fear_record), 2)
print 'Happiness', round(np.std(happy_record), 2)
print 'Sadness', round(np.std(sad_record), 2)
print 'Surprise', round(np.std(surprise_record), 2)
print 'Neutrality', round(np.std(neutral_record), 2)
