[//]: # (Image References)

[image1]: ./images/sign-frequencies.png "Sign Frequencies"
[image2]: ./images/process-images.png "Sign Frequencies"

# German Traffic Sign Recognition
## Data Summary
The code for the data summary is found in the second cell of the Jupyter notebook.
I used the [pandas](http://pandas.pydata.org/) library to calculate summary statistics for the data provided by Udacity:
* The training set featured 34799 examples.
* The validation set featured 4410 examples.
* The test set featured 4410 examples.
* Each traffic sign image was 32x32 pixels.
* There were 43 distinct labels for the data.

## Data Visualization
The code for the data visualization is found in the third cell of the Jupyter notebook.
I created a bar chart illustrating the frequency of each sign in the training data.
![Sign Frequencies][image1]

Please refer to the labels lookup table below for y-axis tick values.

| *Sign ID* |                     *Sign Name*                    | *Counts* |
|:---------:|:--------------------------------------------------:|:--------:|
| 0         | Speed limit (20km/h)                               | 180      |
| 1         | Speed limit (30km/h)                               | 1980     |
| 2         | Speed limit (50km/h)                               | 2010     |
| 3         | Speed limit (60km/h)                               | 1260     |
| 4         | Speed limit (70km/h)                               | 1770     |
| 5         | Speed limit (80km/h)                               | 1650     |
| 6         | End of speed limit (80km/h)                        | 360      |
| 7         | Speed limit (100km/h)                              | 1290     |
| 8         | Speed limit (120km/h)                              | 1260     |
| 9         | No passing                                         | 1320     |
| 10        | No passing for vehicles over 3.5 metric tons       | 1800     |
| 11        | Right-of-way at the next intersection              | 1170     |
| 12        | Priority road                                      | 1890     |
| 13        | Yield                                              | 1920     |
| 14        | Stop                                               | 690      |
| 15        | No vehicles                                        | 540      |
| 16        | Vehicles over 3.5 metric tons prohibited           | 360      |
| 17        | No entry                                           | 990      |
| 18        | General caution                                    | 1080     |
| 19        | Dangerous curve to the left                        | 180      |
| 20        | Dangerous curve to the right                       | 300      |
| 21        | Double curve                                       | 270      |
| 22        | Bumpy road                                         | 330      |
| 23        | Slippery road                                      | 450      |
| 24        | Road narrows on the right                          | 240      |
| 25        | Road work                                          | 1350     |
| 26        | Traffic signals                                    | 540      |
| 27        | Pedestrians                                        | 210      |
| 28        | Children crossing                                  | 480      |
| 29        | Bicycles crossing                                  | 240      |
| 30        | Beware of ice/snow                                 | 390      |
| 31        | Wild animals crossing                              | 690      |
| 32        | End of all speed and passing limits                | 210      |
| 33        | Turn right ahead                                   | 599      |
| 34        | Turn left ahead                                    | 360      |
| 35        | Ahead only                                         | 1080     |
| 36        | Go straight or right                               | 330      |
| 37        | Go straight or left                                | 180      |
| 38        | Keep right                                         | 1860     |
| 39        | Keep left                                          | 270      |
| 40        | Roundabout mandatory                               | 300      |
| 41        | End of no passing                                  | 210      |
| 42        | End of no passing by vehicles over 3.5 metric tons | 210      |

## Model Architecture Design
The code for this step is written in the fifth and sixth cells of the Jupyter notebook.
### Preprocessing
Preprocessing of the images included grayscaling and normalizing all images.
#### Grayscaling
I grayscaled the images due to the variable color distribution of signs with the same label.  For instance, a no entry sign (labeled 17 in our data) is known to the human eye to be red.  However, there are other variables that will influence an image's color data (and thus influencing a convolutional neural network).  Brightness and lumonisty are two examples of variables that may alter an image's color data and make it harder to train with a LeNet model architecture.
#### Normalization
Normalization is done to scale our input values down to be within the same range as our weight matrices and learning rate.  Consider a convolutional neural network that does not use normalized inputs.  The learning rate may overcompensate or undercompensate corrections since the ranges of each feature are vastly different.  This leads to oscillation and a more difficult time in finding the minimum loss.

Source: [Why do we need to normalize the images before we put them into CNN?](http://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn/185857#185857?newreg=7dcec0f56a094d7fb0562f129fd1c673)

An example of an image before and after preprocessing:

![Image][image2]
### Model Design
The LeNet architecture was used for image recognition.  I chose this architecture for its effectiveness, its wide use and Udacity's help in learning the architecture. The code for this architecture is found in the seventh cell of the Jupyer notebook.  The architecture consists of the following layers:

|     *Layer*     |                *Description*                |
|:---------------:|:-------------------------------------------:|
|      Input      |           32x32x1 grayscale image           |
|   Convolution   |  1x1 stride, valid padding, outputs 28x28x6 |
|       RELU      |                                             |
|   Max Pooling   |         2x2 stride, outputs 14x14x6         |
|   Convolution   | 1x1 stride, valid padding, output: 10x10x16 |
|       RELU      |                                             |
|   Max Pooling   |          2x2 stride, output: 5x5x16         |
|     Flatten     |          Input: 5x5x16; output: 400         |
| Fully Connected |                 Output: 120                 |
|       RELU      |                                             |
| Fully Connected |                  Output: 84                 |
|     Dropout     |            Keep probability: 60%            |
| Fully Connected |            Input: 84; output: 43            |
|      Output     |                    Logits                   |

### Model Training
The code to train the model is found in the ninth cell of the Jupyter notebook.

I set the learning rate to 0.0011 found through trial and error.  I used the cross-entropy cost function and the Adam Optimizer to minimize that cost.

### Optimizing the training model
Optimizing the training model was an iterative process full of mistakes, trial and error.

To begin, I set the number of epochs to 10, the standard deviation of the weight initializations to 0.1 and forgot to add dropout to my model.  My first few validation accuracies rose but starkly dropped at the eighth epoch.  I suspected overfitting.

After Googling ways to remedy overfitting, I remembered the dropout module at Udacity.  I looked at existing code of mine where I used dropout and implemented it into my code.  However, this added a new hyperparameter: the keep probabilty.

I began with a keep probability of 80% but quickly lowered that to 50% then brought it up a bit to 60%.  I found 60% worked well but still left me around 85% validation accuracy.

After reading many [Quora](https://www.quora.com) questions regarding CNN accuracy, I realized that weight initialization was important.  I changed my sigma for weight initialization to 0.05 and increased the number of epochs to 30 (to give my CNN more time to find optimal parameters).

The change of sigma was exactly what the training model needed.  My validation accuracy easily hit 93% and my test accuracy hit 92%.
