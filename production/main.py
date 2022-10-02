import cv2
import mediapipe as mp
import numpy as np
import torch
import time

# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = "sid"
auth_token = "token"
client = Client(account_sid, auth_token)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


model = torch.jit.load('model_scripted.ptrom')
model.eval()

# For webcam input:
cap = cv2.VideoCapture(0)
rounds = 0
rlen = 5
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    lst = np.zeros((rlen))
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        pose_landmarks = results.pose_landmarks
           
        try:
            data = np.zeros((33, 4))
            for i, data_point in enumerate(pose_landmarks.landmark):
                if data_point.visibility >= 0.5:
                    data[i][0] = data_point.x
                    data[i][1] = data_point.y
                    data[i][2] = data_point.z
                    data[i][3] = data_point.visibility
        except:
            continue

        input = torch.tensor(data).flatten().float()
        out = model.forward(torch.reshape(input, (1, -1)))
        out = torch.argmax(out)
        # print(out)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        #time.sleep(1)

        lst[rounds] = out
        print(out)
        # print(out)
        rounds += 1
        if rounds == len(lst):
            if np.mean(lst) >= 0.8:
                print("Bad posture")
                '''
                message = client.messages \
                    .create(
                        body='You should fix your posture, it is pretty bad!',
                        from_='+17402736640',
                        to='+15147028472'
                    )

                print(message.sid)
'''
            lst = np.zeros((rlen))
            rounds = 0
            

cap.release()
