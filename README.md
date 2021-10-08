# Human Face Verification with-One-shot Learning Using Siamese Neural Network
A Siamese neural network (sometimes called a twin neural network) is an artificial neural network that uses the same weights while working in tandem on two different input vectors to compute comparable output vectors [[From WIKI]](https://en.wikipedia.org/wiki/Siamese_neural_network). In the modern Deep learning era, Neural networks are almost good at every task, but these neural networks rely on more data to perform well. But, for certain problems like face recognition and signature verification, we can’t always rely on getting more data, to solve this kind of task we have a new type of neural network architecture called Siamese Networks. It uses only a few numbers of images to get better predictions. The ability to learn from very little data made Siamese networks more popular in recent years [[From Towards Data Science]](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942). This repository contains my solution to human face verification with [One-shot learning](https://en.wikipedia.org/wiki/One-shot_learning) using siamese neural network. The dataset used for this problem is the [Celebrities in Frontal-Profile in the Wild (CFPW dataset)](http://www.cfpw.io/).

### Required Package
This project requires **Python** and the following Python packages:
- [NumPy](https://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [os](https://docs.python.org/3/library/os.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/) distribution of Python, which already has most of the above packages. 
