import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pandas as pd


# Utility function
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def run_combined_detection():
    # --- Drawing Utilities ---
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # --- State Variables ---
    clap_counter = 0
    smile_counter = 0
    sing_counter = 0
    clap_stage = None
    sing_stage = None
    smile_stage = None

    # --- Load Face Blendshape Model ---
    MODEL_PATH = "face_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # --- Capture Webcam ---
    cap = cv2.VideoCapture(0)

    

    with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Clone frame for results
            output_frame = frame.copy()
            h, w, _ = frame.shape

            # --- Detect Pose (for Clapping) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # Draw hands

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # If 2 hands are detected, calculate distance between index fingertips
                fingertips = [4, 8, 12, 16, 20]

                if len(results.multi_hand_landmarks) == 2:
                    lm1 = results.multi_hand_landmarks[0].landmark
                    lm2 = results.multi_hand_landmarks[1].landmark

                    # Find minimum distance between any fingertips
                    min_dist = min(
                        np.linalg.norm(
                            np.array([lm1[t1].x, lm1[t1].y]) -
                            np.array([lm2[t2].x, lm2[t2].y])
                        )
                        for t1 in fingertips for t2 in fingertips
                    )

                    touching = min_dist < 0.08  # threshold; tune this

                    # Count a clap when they transition from not touching to touching
                    global hands_touching_last_frame
                    if touching and not hands_touching_last_frame:
                        clap_counter += 1
                        print("CLAP detected!", clap_counter)

                    hands_touching_last_frame = touching

                                
                cv2.putText(output_frame, f'Claps: {clap_counter}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --- Detect Face Blendshapes (Smiling + Singing) ---
            rgb_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_face)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                for landmark_list in result.face_landmarks:
                    for lm in landmark_list:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(output_frame, (cx, cy), 1, (0, 255, 0), -1)

            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                smile_score = 0.0
                jaw_open_score = 0.0
                for blendshape in result.face_blendshapes[0]:
                    if blendshape.category_name in ["mouthSmileLeft", "mouthSmileRight"]:
                        smile_score = max(smile_score, blendshape.score)
                    if blendshape.category_name == "jawOpen":
                        jaw_open_score = blendshape.score

                # Smile detection
                if smile_score > 0.3:
                    cv2.putText(output_frame, "SMILING!", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if smile_stage is None or smile_stage == "not_smiling":
                        smile_stage = "smiling"
                if smile_score <= 0.3 and smile_stage == "smiling":
                    smile_stage = "not_smiling"
                    smile_counter += 1

                    print("SMILE detected! Total:", smile_counter)

                # Singing detection logic
                upper_lip = result.face_landmarks[0][13]
                lower_lip = result.face_landmarks[0][14]
                lip_distance = abs(upper_lip.y - lower_lip.y)

                if jaw_open_score <= 0.1 and sing_stage == "open" and lip_distance < 0.005:
                    sing_stage = "closed"
                    print("SINGING detected! (closed)")
                    sing_counter += 1
                elif jaw_open_score > 0.1 and sing_stage == "closed" and lip_distance > 0.005:
                    sing_stage = "open"
                    print("SINGING detected! (open)")
                    sing_counter += 1
                elif sing_stage is None:
                    sing_stage = "open" if jaw_open_score > 0.1 else "closed"

                cv2.putText(output_frame, f"Mouth: {sing_stage}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

            # --- Show Result ---
            cv2.imshow("Combined Detection", output_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return clap_counter, smile_counter, (sing_counter/2)



def compute_engagement_score(clap_count, smile_count, sing_count):
    # Simple scoring logic
    return clap_count * 1 + smile_count * 1 + sing_count * 2
    

def record_engagement_data(clap_count=0, smile_count=0, sing_count=0):
    engagement_score = compute_engagement_score(clap_count, smile_count, sing_count)
    print(f"Total Claps: {clap_count}, Smiles: {smile_count}, Singing: {sing_count}")
    data = {
            "claps": [clap_count],
            "smiles": [smile_count],
            "singing": [sing_count],
            "engagement_score": [engagement_score]
        }
    df = pd.DataFrame(data)
    df.to_excel("engagement_data.xlsx", index=False)




if __name__ == "__main__":
    clap_count, smile_count, sing_count = run_combined_detection()

    record_engagement_data(clap_count, smile_count, sing_count)


