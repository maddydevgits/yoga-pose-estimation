import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def calculate_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist

cap = cv2.VideoCapture(0)

counter = 0
stage = None


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.8) as pose:
    
    while not cap.isOpened():
        print("Cannot open camera")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            lknee =  [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            rknee =  [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            # Calculate angle
            langle = calculate_angle(lshoulder, lelbow, lwrist)
            rangle = calculate_angle(rshoulder, relbow, rwrist)
            lsangle=calculate_angle(lhip,lshoulder,lelbow)
            rsangle=calculate_angle(rhip,rshoulder,relbow)
            ankdist=calculate_dist(lankle,rankle)
            rwdist=calculate_dist(rhip,rwrist)
            lwdist=calculate_dist(lhip,lwrist)
            rhangle=calculate_angle(rshoulder,rhip,rknee)
            lhangle=calculate_angle(lshoulder,lhip,lknee)
            rkangle=calculate_angle(rankle,rknee,rhip)
            lkangle=calculate_angle(lankle,lknee,lhip)
            
            print([langle,rangle,lsangle,rsangle],[ankdist,rwdist,lwdist],[rhangle,lhangle,rkangle,lkangle])

            
            # Visualize angle and distances
            cv2.putText(image, str(langle), 
                           tuple(np.multiply(lelbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )                        
            cv2.putText(image, str(rangle), 
                           tuple(np.multiply(relbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA 
                                ) 
            cv2.putText(image, str(ankdist), 
                           tuple(np.multiply(lankle, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(lwdist), 
                           tuple(np.multiply(lwrist, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(rwdist), 
                           tuple(np.multiply(rwrist, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                ) 
            
            
            #Custom logic
            
            if ((lsangle>70 and lsangle<100) or (rsangle>80 and rsangle<110)):
                stage='T pose'
            
            elif((langle>170 and langle<185)and (rangle>170 and rangle<180)and (lsangle>150 and lsangle<180)and(rsangle>10 and rsangle<20)):
                stage="konasana"
            elif((langle>130 and langle<150)and (rangle>130 and rangle<145)and (lsangle>165 and lsangle<180)and(rsangle>160 and rsangle<180)):
                stage="tree pose"
            elif((langle>158 and langle<175)and (rangle>160 and rangle<173)and (lsangle>9 and lsangle<15)and(rsangle>5 and rsangle<10)):
                stage="thunderbolt"
            else:
                stage=''
                       
        except:
            pass
        
      
        #Output overlays
        
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                
        cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
            
        
        # Render pose detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        #Feed output
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    
    #Release OpenCV object from memory
    
    cap.release()
    cv2.destroyAllWindows()