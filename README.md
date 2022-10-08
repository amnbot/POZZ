# **POZZ**

[Devpost](https://devpost.com/software/pozz?ref_content=my-projects-tab&ref_feature=my_projects)


## Inspiration

As coders, we are aware that our posture is usually bad for our backs when we're coding (especially when some of us already have back problems at 20, yikes). Therefore, we got our insipration from ourselves.

## What it does

It uses the Pose Detection Module from MediaPipe to extract the x, y, z coordinates of multiple body parts (eyes, nose, shoulders, etc) and feeds it to our custom Neural Network to classify the correctness of the posture.

## How we built it

To build it, we first used Google's Teachable Machine to create our training and testing datasets and to train the model we used for our prototype. After testing with our protoype, we decided to create a custom Neural Network with PyTorch to have increased control over our model and scalability. When the model was ready, we exported it and loaded it into our Python app and using Mediapipe to extract the posture in realtime and feed it to our Neural Network.

## Challenges we ran into

We ran into multiple challenges when making our custom Neural Network. For example, we used Cross Entropy Loss even though we had a SoftMax layer (which is basically done in Cross Entropy Loss), which caused our loss to stay constant. We also had the issue that our model seemed to be overfit.

## Accomplishments that we're proud of

Since we do not have much (or no) experience in AI, making and training models, creating our own custom Neural Network was a huge achievement, and something we are extremely proud of.

## What we learned

We learned how to use Teachable Machine to create a dataset and export it to a Tensorflow model and use it in Flask. More importantly, we learned how to create our own Neural Network using PyTorch, export it and load it into a python project to use it.

## What's next for Pozz

We want to make a better training dataset using Data Augmentation to increase the model's accuracy and play around more with parameters and layers. We would also like our model to work with multiple camera angles since it only works when the user sits directly in front of the camera.