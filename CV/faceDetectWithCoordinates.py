import numpy as np
import cv2
# Добавляем бибилотеки pip install opencv-python

cap = cv2.VideoCapture(0)
# выбираем номер камеры для работы

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 3)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),255, 2)
		print(faces[0])
		cv2.putText(frame, (str(x)+" "+str(y)), (x,y), cv2.FONT_HERSHEY_SIMPLEX,  1 ,(255,255,255),2)

	cv2.imshow("capture", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("found faces", format(len(faces)))
