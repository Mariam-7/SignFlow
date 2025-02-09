import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '/Users/mariamhafeez/Desktop/Hackie/sign-language-detector-python/data'

data = []
labels = []
no_hands_count = 0  # Track how many images have no hands detected

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    # Only process directories (skip files like .gitignore)
    if os.path.isdir(dir_path):  # Ensures only directories are processed
        for img_path in os.listdir(dir_path):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect 42 values (21 landmarks * 2 coordinates: x, y)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Add x and y values for this hand to data_aux
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalize x value
                        data_aux.append(y - min(y_))  # Normalize y value

                # Only add data if it's complete (42 values for 21 landmarks)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"Skipping image (insufficient landmarks): {img_path}")
            else:
                # Increment counter if no hands were detected
                no_hands_count += 1
                print(f"Skipping image (no hands detected): {img_path}")

# Log how many images had no hands detected
print(f"Number of images with no hands detected: {no_hands_count}")

# Save data as a pickle file
script_dir = os.path.dirname(os.path.abspath(__file__))
pickle_file_path = os.path.join(script_dir, 'data.pickle')

with open(pickle_file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
