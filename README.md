# Human-Action-Recognition-Based-Home-Automation
This project utilizes deep learning to recognize human actions and automate home appliances. By connecting appliances to Raspberry Pi GPIO pins and deploying a deep learning model on the Raspberry Pi, this system enables seamless automation of household tasks.
The deep learning model was trained by using transfer learning with feature extraction on a custom dataset on Google colab.

It aims to design a low-cost, efficient, and smart home automation system that provides ambient assisted living to improve the quality of life in homes. The system utilizes human action recognition through deep learning algorithms to control home appliances based on the occupant's actions.

## **Objectives**

The project focuses on achieving the following objectives:

* Capturing video frames of the occupant using the Raspberry Pi camera.
* Implementing human action recognition using a deep learning algorithm.
* Switching ON/OFF home appliances according to the identified actions of the occupant.

## **Installation Instructions**

To replicate this project, follow these installation instructions:
Install the following libraries using `pip install`:

* TensorFlow
* NumPy
* TensorFlow Hub

Install OpenCV library for reading and preprocessing video frames on the Raspberry Pi(Install the required dependencies for OpenCV and TensorFlow libraries. Detailed step-by-step instructions for installing the dependencies can be found [here](https://www.youtube.com/watch?v=GNRg2P8Vqqs&t=182s))

## **Usage**

This project is a prototype that simulates a TV, a light, and a fan using two LEDs and a 5V motor connected to the GPIO pins of the Raspberry Pi. The system controls these simulated household appliances based on the occupant's actions.

* Download the prediction.py, appliance_control.py, and the trained MobileNetv2.h5 files. Save all the files in the same directory.

* Follow the provided circuit diagram to properly connect the actuators (LEDs and 5V motor) to the GPIO pins of the Raspberry Pi.

![Circuit diagram for the implementation of Home Automation System](https://github.com/bayodekehinde/Human-Action-Recognition-Based-Home-Automation/blob/main/Circuit%20Diagram%20of%20Home%20Automation%20System.JPG)

* Ensure a camera is connected to the Raspberry Pi.

* Run the appliance_control.py file to activate the home automation system.

*Note: This project was developed and tested on the Raspian Bulleye operating system.*

To use the Human Action Recognition model for other applications besides home automation, simply download the Mobilenetv2.h5 model and use for your specifc use case.

To use the Raspberry Pi in this or any other project:

* Install the appropriate operating system (OS) listed on the [Raspberry Pi website](https://www.raspberrypi.com/software/)
* Connect a monitor or keyboard to access the GUI, or establish a headless connection using VNC.

*Note: The specific functionality and control of home appliances can be customized according to individual requirements.*











