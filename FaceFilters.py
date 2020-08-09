import cv2
import numpy as np 
import dlib
import os

directory = "./Filters"
filterlist = os.listdir(directory)
i=0
lines = -1

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\manish\Desktop\Projects\Python\OpenCVCrashCourse\blink-detection\shape_predictor_68_face_landmarks.dat")



while True:
	_,frame = cap.read()
	frame = cv2.flip(frame,1)

	img2 = cv2.imread(f'Filters/{filterlist[i]}')
	# img2 = cv2.resize(img2, (100,100))

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	for face in faces:
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()
		#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
		landmarks = predictor(gray, face)
		if lines >0 :
			for j in range(68):
				if j in [16,21,26,30,35,41,47,59,67] :
					if j in[41,47]:
						cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
						(landmarks.part(j-5).x,landmarks.part(j-5).y), (20,255,20), 1) 
					elif j==59:
						cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
						(landmarks.part(j-11).x,landmarks.part(j-11).y), (20,255,20), 1)
					elif j==30:
						cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
						(landmarks.part(j+5).x,landmarks.part(j+5).y), (20,255,20), 1)
						cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
						(landmarks.part(j+1).x,landmarks.part(j+1).y), (20,255,20), 1) 
					elif j==67:
						cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
						(landmarks.part(j-7).x,landmarks.part(j-7).y), (20,255,20), 1) 
					continue
				cv2.line(frame, (landmarks.part(j).x,landmarks.part(j).y),
				(landmarks.part(j+1).x,landmarks.part(j+1).y), (20,255,20), 1) 	

		
		height = landmarks.part(8).y - landmarks.part(24).y  + 50
		width = landmarks.part(16).x - landmarks.part(0).x
		# print("height:",height)
		# print("width:",width)
		img2 = cv2.resize(img2, (width,height))
		x0 = landmarks.part(0).x
		y0 = landmarks.part(0).y - 50
		rows,cols,channels = img2.shape
		roi = frame[y0:rows+y0, x0:cols+x0]

		# # cv2.imshow("roi",roi)
		# # Now create a mask of logo and create its inverse mask also
		img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)

		# # Now black-out the area of logo in ROI
		try:
			frame_bg = cv2.bitwise_and(roi,roi,mask = mask)
		except:
			continue

		# # Take only region of logo from logo image.
		img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
		
		# # Put logo in ROI and modify the main image
		dst = cv2.add(frame_bg,img2_fg)


		# # frame = cv2.addWeighted(frame, 0.7, img2, 0.3, 0.0)
		frame[y0:rows+y0, x0:cols+x0] = dst

	# cv2.imshow("mask",mask)
	# cv2.imshow("frame_bg",frame_bg)
	cv2.imshow("Frame",frame)
	# cv2.imshow("dst",dst)

	key = cv2.waitKeyEx(1) & 0xFF
	print(key)
	if key==ord('q'):
		break
	if key==ord('p'):
		if i==0:
			i=len(filterlist)-1
		else:
			i-=1
	if key==ord('n'):
		if i==len(filterlist)-1:
			i=0
		else:
			i+=1
	if key==ord('t'):
		lines *= -1
		
	

cap.release()
cv2.destroyAllWindows()