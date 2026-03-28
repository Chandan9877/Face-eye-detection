import cv2
import os

path = 'dataset'
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created!")
    
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

#face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_id = input("Enter the user id and press <return> ==>  ")

print("initializing face capture. Look at the camera...")

count = 0
while True:
    ret,img = cam.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x + w , y + h),(255,0,0),2)
        count += 1
        cv2.imwrite("dataset/user."+str(face_id) + "." + str(count)+".jpg",gray[y:y+h,x:x+w])
        print("image captured"+str(count))
    cv2.imshow("image",img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break
cam.release()
cv2.destroyAllWindows()