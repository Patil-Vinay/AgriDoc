## Weed Detection using Yolov5
We focused on the health of tomato plants and for that , next step is to be able to detect and classify commonly found weeds around tomatoes. Thus to make this work we have used the help of pre-trained model YOLOv5 which gave us an accuracy of almost 98%

## WorkFlow

- Collect the data
- Split the dataset into training set ,test set and validation set
- Label the data using makesense.ai
- Download the YOLOv5 Tutorial Google colab notebook from Github
- Clone the repository and import the necessary dependancies
- In the YOLOv5 folder edit the coco128.yaml file according to your dataset
- Train the model
- Evaluate the model
- Make Predictions

## Model Preview and Test Results

**F1_curve:**

![](https://github.com/TejasARathod/Final-Year-Project-AgriDoc-Agricultural-Robot-/blob/907ac92004d505bde0f043d31fac4b7bd24679f9/Part2/Weed%20Detection%20using%20YOLOv5/F1_curve.png)

**PR_curve:**

![](https://github.com/TejasARathod/Final-Year-Project-AgriDoc-Agricultural-Robot-/blob/907ac92004d505bde0f043d31fac4b7bd24679f9/Part2/Weed%20Detection%20using%20YOLOv5/PR_curve.png)

**P_curve:**

![](https://github.com/TejasARathod/Final-Year-Project-AgriDoc-Agricultural-Robot-/blob/907ac92004d505bde0f043d31fac4b7bd24679f9/Part2/Weed%20Detection%20using%20YOLOv5/P_curve.png)

**Results:**

![](https://github.com/TejasARathod/Final-Year-Project-AgriDoc-Agricultural-Robot-/blob/907ac92004d505bde0f043d31fac4b7bd24679f9/Part2/Weed%20Detection%20using%20YOLOv5/results.png)
