import cv2
import mediapipe

counter=0
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(0)
folder = r"C:\Users\lENOVOO\Desktop\Mproject\Data\ASL new"
 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
 
    while (True):
 
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
 
        cv2.imshow('Test hand', frame)
        key=cv2.waitKey(1)
        if key==ord("s"):
           counter+=1
           kk=str(counter)
           frame_new=cv2.resize(frame,(400,400))
           cv2.imwrite(folder+'/Image_'+kk+'.jpg',frame_new)
           print(counter)
 
        elif cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()