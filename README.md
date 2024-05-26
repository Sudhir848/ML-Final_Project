# Google Open Images Mutual Gaze dataset
Detecting Mutual Gaze (facing each other) between individuals in images.

I have chosen the Google Open Images Mutual Gaze dataset for this Machine Learning project which deals with image annotations showing if two people are looking at each other or not. Mutual gaze is a significant social sign mostly showing interaction, attention, and intent.

The program output includes model accuracy, available in the classification report with details on precision, recall as well as F1-scores across classes included in the model; the accuracy achieved was 55.28%. The classification report showed a higher recall for the minority class, indicative of the model's sensitivity to mutual gaze detection but with lower precision, suggesting a number of false positives.

## Analysis of Output
The primary aim here was to find out mutual gaze patterns from a collection of images which have already been annotated. To quantify the spatial dynamics between face pairs, this model was built using the logistic regression method, incorporating engineered characteristics. Such factors involve; geometric distances, and area ratios besides relative positions.

Upon execution, it showed that it had an accuracy rate of 55.28%. This means that the model could predict mutual gaze events greater than what can be expected just by guessing. However, its accuracy was not perfect as far as detecting gaze events is concerned mainly because of different levels of complexity found in the visual data, especially for the recognition of mutual gazes class 1.0 which was a fact though. Instead, the model made so many mistakes in relation to other classes because it did better at finding mutual gazes.

## Output of the Program
<img width="416" alt="image" src="https://github.com/Sudhir848/ML-Final_Project/assets/152313525/72732142-ebd8-4851-bfb4-60401e4ae301">
The program output includes the accuracy of the model and a classification report detailing precision, recall, and F1-scores for each class.
The accuracy achieved: 55.28%.

## Sources
Google AI. Open Images Dataset.
https://github.com/google-research-datasets/Google-Open-Images-Mutual-Gaze-dataset
