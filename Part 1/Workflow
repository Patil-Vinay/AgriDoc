**Setting up the Jetson**

- Install Packages

        sudo apt install -y git cmake libpython3-dev python3-numpy
        git clone --recursive https://github.com/dusty-nv/jetson-inference

        cd jetson-inference
        mkdir build
        cd build
        cmake ../

- After this we just have to follow up by installing pre trained models like GoogleNet and ResNet-18 and also install pytorch for python 3
- Last step: 

        make
        sudo make install
        sudo ldconfig

**Image Classification for plants on downloaded images**

- Download some images into the downloads folder you want to classify

        cd jetson-inference/build/aarch64/bin
        .detectnet-c
        ./detectnet-console ~/Downloads/*Imagename*.jpg ~/Downloads/out-01.jpg coco-plants
    
**Real Time Image Classification for plants**

- Now we will use the earlier COCO models to do real time predictions , just make sure that a camera is plugged in.

        cd ~/jetson-inference/build/aarch64/bin/

- Because the Nano is an embedded device, it is not nearly as powerful as a modern desktop or server built with a powerful graphics card. 
- NVIDIA has a training interface called DIGITS that makes training networks much easier.
- We use “transfer learning” to retrain an existing network. When we do this, we just tweak the model’s parameters to optimize it to our own training data.
- To begin, we first need to set up a swap space on our SD card so that we have more RAM to play with.

        df -h

        sudo fallocate -l 4G /mnt/4GB.swap
        sudo chmod 0600 /mnt/4GB.swap
        sudo mkswap /mnt/4GB.swap
        sudo swapon /mnt/4GB.swap

- Scroll to the bottom of this file and press ‘o’ to insert a new line and begin editing. Enter the following line:

        /mnt/4GB.swap  none swap sw 0  0

- Next, we need to capture images to create our datasets. I’ll be using 2 different sets of images, as I want my network to identify these categories:
        
        - Background
        - PlantA

        cd ~
        mkdir datasets
        cd ~/datasets
        mkdir utensils
        cd utensils
        touch labels.txt
        echo “background” >> labels.txt
        echo “PlantA” >> labels.txt
        
- Next , we want to capture images for training, validation and testing.

        camera-capture --camera=/dev/video0 --width=640 --height=480

- Now , we train the model

        cd ~/jetson-inference/python/training/classification/
        python train.py --model-dir=plants ~/datasets/plants

- When it’s done, we will need to export the model to the Open Neural Network Exchange (ONNX) format:

        python onnx_export.py --model-dir=plants

- Test it using:

        imagenet-camera --model=plants/resnet18.onnx --labels=/home/sgmustadio/datasets/plants/
        labels.txt --camera=/dev/video0 --width=640 --height=480 --input_blob=input_0 --output_blob=output_0
