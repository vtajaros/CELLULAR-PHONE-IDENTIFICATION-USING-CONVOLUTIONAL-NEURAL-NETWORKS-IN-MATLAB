# CELLULAR PHONE IDENTIFICATION USING CONVOLUTIONAL NEURAL NETWORKS IN MATLAB

## Project Description

This project describes a novel method for identifying cellular phones using **Convolutional Neural Networks (CNNs)** implemented in MATLAB. Accurately identifying mobile phones is crucial for various applications, including strengthening network management, customizing user experiences, and enhancing security protocols. By leveraging the capabilities of CNNs to automatically learn and extract hierarchical features from raw input data, this research attempts to create a reliable and effective method for detecting cell phones based on their unique patterns.

## Key Features and Technical Details

  * **Technology:** Convolutional Neural Networks (CNNs) implemented in MATLAB.
  * **Pre-trained Network:** The software uses the pre-trained CNN "**resnet50**".
  * **Input Image Size:** The ImageInputLayer has an `InputSize` of `[224 224 3]`.
  * **Accuracy:** The software app classifier was able to achieve an accuracy as high as 95%. Training accuracies were generally in the range of 85%-95%, often reaching the high 90's.

## 8 Classification Classes

The model is trained to identify eight different classes of cellular phones:

1.  Blackberry Cingular
2.  Infinix Note 12
3.  iPhone 12
4.  iPhone 15
5.  Samsung A04
6.  Samsung A20
7.  Samsung A50
8.  Samsung Keypad (Keystone)

## User Manual

The Cellular Phone Identifier software has a Graphical User Interface (GUI).

1.  **Load Image Button:** Once clicked, the software prompts you to select an image file (`.jpg`, `.png`, or `.bmp`) to classify. This image should not be part of the training process.
2.  **Classification Results (Label Textbox):** After the image is processed and identified, this textbox displays the classification result and the training accuracy of the dataset.
3.  **Axes (Image Input):** The loaded input image is displayed here.
4.  **To Classify Another Image:** Simply load a new image again with the "Load Image" button.

## Challenges

The project faced challenges, including the creation of the GUI in MATLAB online and the fact that modern smartphones look very similar from the front, which sometimes led to misclassification. It was noted that phone classification on modern phones would more likely need images of the back parts of the phones rather than the front. The model was trained with a limited dataset of 85 images per class, and a larger training dataset would help raise the accuracy.

## App and Dataset Link

You can access the application and dataset here at gitHub.
