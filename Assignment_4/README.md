# Imitation Learning on Turtlebot
The goal of this assignment is to implement an imitation learning algorithm onto the Turtlebot.

### Dataset
We use two types of dataset to feed into our imitation learning model, steering angle from [Udacity](https://github.com/udacity/self-driving-car) and collision probability from ETZ Zurich, [Robotics & Perception Group](http://rpg.ifi.uzh.ch/dronet.html). 

The Udacity dataset provide hours of video recording along with other information such as steering angle, veloctiy, etc,  all stored in rosbag file formats. 

The Robotics & Perception Group dataset provides collision probability of outdoor environment. 

### Pre-processing dataset
With the nature of the dataset, we will need the images along with its labels (steering angle and collision probability). This requires every image frames to be accurately time-stamped to its labels. 

The structure of the dataset was arranged like this below:
```
training/
    HMB_1_3900/*
        images/
        sync_steering.txt
    HMB_2/
    HMB_4/
    HMB_5/
    HMB_6/
validation/
    HMB_1_501/*
testing/
    HMB_3/
```

**Collision dataset**
The dataset was already pre-processed according to the structure shown above.

**Steering angle dataset**
As mentioned earlier, this dataset was presented in rosbag formats. We used
[udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader) to extract all the contents in the rosbag. After extraction, we time-stamped the images with their corresponding steering angles with [time_stamp_matching.py](https://github.com/uzh-rpg/rpg_public_dronet/blob/master/data_preprocessing/time_stamp_matching.py). Once that is completed, arrange the files according to the structure above.  *Enter extraction steps here in the future. 

After both dataset is arranged, merge both datasets together. Eg: Merge `training/` from both steering angle and collision.

### Mini-MobileNet

### Results
After multiple runs, we found that Run 2 gave us the best results.
![loss](images/Result1)

![histogram](images/Result2)

![confusionmatrix](images/Result3)

### Turtlebot Implementation
>This section is still in-progress

Turtlebot

* NVIDIA Tegra K1 - Ubuntu 14.04.2
* RGB-D camera
* 3D distance sensor
* Turtlebot mounting hardware kit

Proposed pipeline:

* Install Keras and Tensorflow on Turtlebot
* Use Turtlebot's live feed camera as input to MobileNet
* Run inference on images using trained model and weights
* Connect predicted steering angles to Turtlebot's steering commands
* Connect predicted collision probablity to Turtlebot's speed commands
* Monitor Turtlebot's performance remotely

### Challenges
What is an assignment without challenges? Every step of this assignment has given us a huge hurdle as high as the Himalayan mountains. Here are the challenges broken down in detail:

* Rosbag extractions
  * Initially, we tried to extract the rosbag without using Udacity's code because we wanted to avoid interfacing with Docker (requirement to run their code). We spent about 4 days just trying other methods, such as: 
  > `rostopic echo -b file.bag -p /topic > data.txt`, 
  > [Exporting image and video data](http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data), 
  > `rosrun image_view extract_images image:=/center_camera/image_color _image_transport:=compressed`
  
   * In the last code above, `rosrun`, worked. But the issue is, everytime we do the extraction it gives us a different number of images extracted. This is probably due to a random capture delays everytime we execute the code (The code requires two terminals open, one to play the bag file, one to capture).
 
 * Version mismatch for Keras and Tensorflow in the cluster. 
   * For MobileNet, it requires the latest version of Keras, 2.1.5 and Tensorflow 1.7.0 for DepthwiseConv. 
   * In the cluster, there is a conflict with PYTHONPATH when upgrading versions. Basically when we update, it doesn't update the path to use the newest update.
   
  * Remotely connect to Turtlebot
  
  
  * Killing Turtlebot
    * The motivation to do this was because we wanted a purely python 3.6 without 2.7 in Ubuntu 14.04.2, due to possible version mismatch when dealing with the latest Keras and Tensorflow for model inference.
    * Our lack of knowledge on Ubuntu led us to do one command `sudo uninstall python`.
    * After a few days of recovery from the incident, we managed to bring back Ubuntu in the Turtlebot and now we're in the process to recover ROS.