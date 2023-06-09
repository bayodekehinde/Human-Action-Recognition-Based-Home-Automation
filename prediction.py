import cv2
import numpy as np
import tensorflow as tf
import time
from collections import Counter

#libraries for Raspberry pi GPIO

#import RPi.GPIO as GPIO

# Define the GPIO pin number for each actuators
led_1=10    #light
led_2=12    #tv
fan=12



#define class labels
class_labels=['clapping', 'lying', 'sitting', 'standing', 'waving']

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("Model_MobileNetv2.h5", compile=False)
    
# Define input size expected by the model
input_width, input_height = 224, 224

# Define the frame rate (30 fps)
frame_rate = 1/30

# Initialize the action counter
action_counter = []

# Set up the video capture
video_capture = cv2.VideoCapture(0)  # Use the appropriate video source (0 for webcam)

# Start capturing frames
while True:
    # Capture a frame
    ret, frame = video_capture.read()

    # Check if the frame was captured successfully
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.) to match the input requirements of your model
    preprocessed_frame = cv2.resize(frame, (input_width, input_height))
        
    # Perform prediction on the preprocessed frame
    input_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension 
    predictions = model.predict(input_frame)

    # Get the predicted action 
    predicted_action = (np.argmax(predictions))

    # Update the action counter
    action_counter.append(predicted_action)
    print("waiting....")
    time.sleep(frame_rate)  #capture 30 frames every second
    print(action_counter)

    # Check if the len of action counter is 300 i.e predict video frames for 5s with approx. 30fps
    if len(action_counter)==50:
        #get the most common action
        most_common=Counter(action_counter).most_common(1)[0]
        #reset action counter
        action_counter.clear()

        action=class_labels[most_common[0]]

        
        if action=="clapping":
            
            GPIO.output(fan, GPIO.LOW)
            GPIO.output(led_1, GPIO.LOW)
            GPIO.output(led_2, GPIO.LOW)

        elif action=="lying":
            GPIO.output(fan, GPIO.HIGH)
            GPIO.output(led_1, GPIO.LOW)
            GPIO.output(led_2, GPIO.HIGH)

        elif action=="sitting":
            GPIO.output(fan, GPIO.HIGH)
            GPIO.output(led_1, GPIO.HIGH)
            GPIO.output(led_2, GPIO.HIGH)
        elif action=="standing":
            
            GPIO.output(fan, GPIO.HIGH)
            GPIO.output(led_1, GPIO.HIGH)
            GPIO.output(led_2, GPIO.LOW)

        elif action=="waving":
            
            GPIO.output(fan, GPIO.LOW)
            GPIO.output(led_1, GPIO.HIGH)
            GPIO.output(led_2, GPIO.HIGH)

        #if you would like to display the video frames to a screen, uncomment the below
        # Display the frame with the predicted action (optional)
        ##cv2.putText(frame, "Predicted Action: {}".format(predicted_action), (10, 30),
                    ##cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ##cv2.imshow("Frame", frame)

        # Check for 'q' key press to exit
        ##if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release the video capture and close any open windows
            ##video_capture.release()
            ##cv2.destroyAllWindows()
            ##break
            
            
           
            
      
