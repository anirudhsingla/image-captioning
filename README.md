# image-captioning using deep learning

Introduction

Generating a description of an image is called image captioning. Image captioning requires recognizing the important objects, their attributes and their relationships in an image. It also needs to generate syntactically and semantically correct sentences. In this we have to pass the image to the model and the model does some processing and generating captions or descriptions as per its training. This prediction is sometimes not that accurate and generates some meaningless sentences. We need very high computational power and a very huge dataset for better results. Now we will see some information about the dataset and the architecture of the neural network of the Image captions generator.




Importance of Image Captioning

Image captioning is important for many reasons. For example, they can be used for automatic image indexing. Image indexing is important for Content-Based Image Retrieval (CBIR) and therefore, it can be applied to many areas, including biomedicine, commerce, the military, education, digital libraries, and web searching. Social media platforms such as Facebook and Twitter can directly generate descriptions from images. 

Dataset :
In this project, we are using the flickr 8k dataset.
This dataset has 8,092 images with image id and a particular id has multiple captions generated.
One of the image in dataset :- 



The particular captions of this image in dataset :- 



Tech Stack
This project requires good knowledge of Python, working on Jupyter notebooks, Keras library and NumPy.
Make sure you have installed all the following necessary libraries:
Tensorflow
Keras
Pandas
NumPy
nltk ( Natural language toolkit)
Jupyter- IDE



NLTK
NLTK or Natural Language Toolkit is a python package used for analyzing unstructured data and data that contains human readable text. For this project we have used nltk for generating all the stopwords. 
Stopwords are words that fall under the category conjunctions or prepositions or verbs and nouns which are repeated many times. This helps in searching captions which describe a particular image thus reducing the search complexity and makes the algorithm efficient.

Implementation
Convolutional Neural Network
The convolutional neural network (CNN) is a class of deep learning neural networks. They’re most commonly used to analyze visual imagery and in image classification. Here, we are using CNNs rather than a traditional neural network because these networks are fast, efficient, and work well with image data.
A CNN works by extracting features from images.The features are learned while the network trains on a set of images. CNNs learn feature detection through tens or hundreds of hidden layers. 
CNNs have an input layer, an output layer, and hidden layers. The hidden layers usually consist of convolutional layers, ReLU layers, pooling layers, and fully connected layers. 
If the activation function is not applied, the output signal becomes a simple linear function. A neural network without activation function will act as a linear regression with limited learning power. But we also want our neural network to learn non-linear states as we give it complex real-world information.


ReLU Activation Function
ReLU stands for rectified linear activation unit and is considered one of the few milestones in the deep learning revolution. It is simple yet really better than its predecessor activation functions such as sigmoid .
ReLU activation function formula:
f(x)=max(0,x)



One more important property that we consider the advantage of using ReLU activation function is sparsity. 
A matrix in which most entries are 0 is called a sparse matrix and similarly, we desire a property like this in our neural networks where some of the weights are zero.
Sparsity results in concise models that often have better predictive power and less overfitting/noise. 

Scope of Project:

To improve the accuracy of the model, we will further implement Transfer Learning by employing VGG16.
Transfer Learning (VGG16):
Transfer learning, used in machine learning, is the reuse of a pre-trained model on a new problem. VGG16 is a CNN architecture that was the first runner-up in the 2014 ImageNet Challenge. It’s designed by the Visual Graphics Group at Oxford and has 16 layers in total, with 13 convolutional layers themselves. We will load the pre-trained weights of this model so that we can utilize the useful features this model has learned for our task.

After extracting features from images and after getting an acceptable accuracy level, next we will move on to generate captions using LSTM.

LSTM:
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. The success of LSTMs is in their claim to be one of the first implements to overcome the technical problems and deliver on the promise of recurrent neural networks. The two technical problems overcome by LSTMs are vanishing gradients and exploding gradients, both related to how the network is trained. Hence, we will be using LSTM in the near future to achieve the objectives of our project.
Result :- 


---------------------------------------------------------------------------------------------------
 

