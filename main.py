from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2
import numpy as np
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("signlanguagemodel", compile=False)


# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
sentence = ''
engine.say("Welcome to the ASL Sign Language Interpretor. Do a sign, press p to add it to your sentence, d to delete, and s to speak it. Esc to quit")
engine.runAndWait()
while True:
    org = [25, 25]
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.50
    color = (0, 255, 0)
    thickness = 1
    # Grab the webcamera's image.
    ret, image = camera.read()
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

    if keyboard_input == ord('d'):
        sentence = sentence[:-1]

    if keyboard_input == ord('m'):
        sentence += " "

    if len(sentence) > 0 and keyboard_input == ord('s'):
        engine.say(sentence)
        engine.runAndWait()




    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window


    # Make the image a numpy array and reshape it to the models input shape.
    tensorimage = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    tensorimage = (tensorimage / 127.5) - 1

    # Predicts the model
    prediction = model.predict(tensorimage)

    index = np.argmax(prediction)
    class_line = class_names[index]
    if index >= 10:
        class_name = class_line[3]
    else:
        class_name = class_line[2]
    confidence_score = prediction[0][index]


    # Print prediction and confidence score
    if np.round(confidence_score * 100) > 60:
        print("Class:", class_name)
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        cv2.putText(image, class_name, org, font, fontScale, color, thickness)
        if keyboard_input == ord('p'):
            sentence += class_name
    cv2.putText(image, sentence, [100, 50], font, fontScale, color, thickness)
    cv2.imshow("Webcam Image", image)

        # Listen to the keyboard for presses.

camera.release()
cv2.destroyAllWindows()
