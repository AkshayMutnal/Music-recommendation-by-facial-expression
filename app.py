from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

face_classifier = cv2.CascadeClassifier(r'D:\projects\final\emotion-based-music-recommendation-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\projects\final\emotion-based-music-recommendation-main\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Happy', 'Sad', 'Surprise']
print("+"*50, "loadin gmmodel")
model = load_model('Emotion_little_vgg.h5')
cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/choose_singer', methods = ["POST"])
def choose_singer():
	info['language'] = request.form['language']
	print(info)
	return render_template('choose_singer.html', data = info['language'])


@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
	info['singer'] = request.form['singer']

	found = False

	cap = cv2.VideoCapture(0)

	while True:
		# Grab a single frame of video
		ret, frame = cap.read()
		labels = []
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h,x:x+w]
			roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
		# rect,face,image = face_detector(frame)


			if np.sum([roi_gray])!=0:
				roi = roi_gray.astype('float')/255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi,axis=0)
				cv2.imwrite("static/face.jpg", roi)

			# make a prediction on the ROI, then lookup the class

				preds = classifier.predict(roi)[0]
				label=class_labels[preds.argmax()]
				label_position = (x,y)
				cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
			else:
				cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
		cv2.imshow('Emotion Detector',frame)
		if cv2.waitKey(1) & 0xFF == ord('y'):
			
			break
	prediction = model.predict(roi)

	print(prediction)

	prediction = np.argmax(prediction)
	prediction = label_map[prediction]

	cap.release()


	link  = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction}+{info['language']}+song"
	webbrowser.open(link)

	return render_template("emotion_detect.html", data=prediction, link=link)

if __name__ == "__main__":
	app.run(debug=True)