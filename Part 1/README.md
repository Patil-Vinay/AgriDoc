## Real Time Image Classification with NVIDIA Jetson Nano

The objective is to classify diseases and weeds in real time thus to start off with we want to detect simple plants using pre trained models in real time. First part would be to classify downloaded images then move on to real time using camera feed.

For classification I have used jetson inference  and using pre trained model by GoogleNet and ResNet-18. We will use this model for predicting using images as well and also in real time.

- Input Images: 

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/input_image_1.jpg)

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/input_image_2.jpeg)

- Output Images: 

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/output_image_1.jpg)

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/output_image_2.jpg)

- Because the Nano is an embedded device, it is not nearly as powerful as a modern desktop or server built with a powerful graphics card. 
- NVIDIA has a training interface called DIGITS that makes training networks much easier.
- We use “transfer learning” to retrain an existing network. When we do this, we just tweak the model’s parameters to optimize it to our own training data.
- To begin, we first need to set up a swap space on our SD card so that we have more RAM to play with.

For testing purpose we also created our own dataset then trained our model and exported this saved model in Open Neural Network Exchange (ONNX) format. Here are few of the test images.       

Test Images:

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/test_image_1.png)

![](https://github.com/Patil-Vinay/AgriDoc/blob/main/Part%201/test_image_2.png)

