import streamlit as st
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import InferenceClient
from PIL import Image

# Initialize models
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
mistral_client = InferenceClient(api_key="hf_YOUR_HUGGINGFACE_API_KEY")

# Streamlit UI
st.title("Interactive Handwriting-to-Description System")

# Session state to manage the process
if "canvas_saved" not in st.session_state:
    st.session_state.canvas_saved = False

if st.button("Open the Camera"):
    # Call the camera program when button is clicked
    def run_camera_app():
        canvas_width, canvas_height = 640, 480
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        prev_x, prev_y = None, None
        movement_timeout = 3
        no_movement_time = 0

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks using mediapipe
            # (hand detection code omitted for brevity; use previous code for this)
            # Draw lines and save canvas logic here...

            if no_movement_time > movement_timeout * 30:
                # Save the canvas and break
                cv2.imwrite("canvas_image.png", canvas)
                st.session_state.canvas_saved = True
                break

            combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
            cv2.imshow("Drawing", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    run_camera_app()

# Process the canvas image if saved
if st.session_state.canvas_saved:
    st.write("Processing the canvas image...")

    # Load the image
    def load_and_display_image(path):
        img = Image.open(path).convert("RGB")
        st.image(img)
        return img

    canvas_image = load_and_display_image("canvas_image.png")

    # TROCR text extraction
    st.write("Extracting text from the image...")
    pixel_values = trocr_processor(images=canvas_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    extracted_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.write(f"Extracted Text: {extracted_text}")

    # Pass the text to Mistral for descriptive response
    st.write("Generating a descriptive response...")
    mistral_response = mistral_client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "user", "content": f"Describe this: {extracted_text}"}
        ],
        max_tokens=150
    )

    # Display Mistral's response
    descriptive_response = "".join([chunk.choices[0].delta.content for chunk in mistral_response])
    st.write("Descriptive Response:")
    st.write(descriptive_response)
