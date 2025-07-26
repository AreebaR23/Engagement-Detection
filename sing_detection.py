import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


def detect_sing():
    stage = None

    MODEL_PATH = "face_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = landmarker.detect(mp_image)

        # Draw face landmarks
        if result.face_landmarks:
            for landmark_list in result.face_landmarks:
                for lm in landmark_list:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        #singning detection
       
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            mouth_open_score = 0.0
            upper_lip = result.face_landmarks[0][13]
            lower_lip = result.face_landmarks[0][14]
            lip_distance = abs(upper_lip.y - lower_lip.y)

            for blendshape in result.face_blendshapes[0]:
                if blendshape.category_name == "jawOpen" :
                    
                    mouth_open_score = blendshape.score
        
        
            #Mouth closing detection
            if mouth_open_score <= 0.1 and stage == "open" and lip_distance < 0.005:
                stage = "closed"
                print("SINGING detected! ", stage)
            
            #Mouth opening detection
            elif mouth_open_score > 0.1 and stage == "closed" and lip_distance > 0.005:
                stage = "open"
                print("SINGING detected! ", stage)
            
            elif stage is None:
                # Initialize state
                stage = "open" if mouth_open_score > 0.1 else "closed"
        else:
            print("no blendshape detected")
           
    
        cv2.imshow("Smile Detection", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    detect_sing()