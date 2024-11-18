import cv2
import numpy as np
import mediapipe as mp
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from IPython.display import display

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

canvas_width, canvas_height = 640, 480
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None

movement_threshold = 10  
no_movement_time = 0  
movement_timeout = 3  

distance_threshold = 50  

# Increase the stroke width
stroke_width = 10  # Set to your desired thickness

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_x = int(thumb_tip.x * canvas_width)
            thumb_y = int(thumb_tip.y * canvas_height)
            index_x = int(index_tip.x * canvas_width)
            index_y = int(index_tip.y * canvas_height)
            middle_x = int(middle_tip.x * canvas_width)
            middle_y = int(middle_tip.y * canvas_height)

            dist_thumb_index = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            dist_index_middle = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)
            dist_thumb_middle = np.sqrt((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2)

            if dist_thumb_index < distance_threshold and dist_index_middle < distance_threshold and dist_thumb_middle < distance_threshold:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 255, 255), stroke_width)

                prev_x, prev_y = index_x, index_y
                no_movement_time = 0
            else:
                prev_x, prev_y = None, None  
                no_movement_time += 1

    if no_movement_time > movement_timeout * 30:  
        cv2.imwrite('canvas_image.png', canvas)
        print("Canvas image saved as 'canvas_image.png'")
        break  

    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('Drawing', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def show_image(pathStr):
  img = Image.open(pathStr).convert("RGB")
  display(img)
  return img

def ocr_image(src_img):
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return decoded_text.replace(" ", "")  # Removes all spaces

directory = r"C:\Users\ASUS\Downloads\Projects and Research Papers\WRITING\canvas_image.png"
ocr_image(directory)