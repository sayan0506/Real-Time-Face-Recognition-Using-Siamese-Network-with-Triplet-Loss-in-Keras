# Real-Time-Face-Recognition-Using-Siamese-Network-with-Triplet-Loss-in-Keras
Real-Time-Face-Recognition-Using-Siamese-Network-with-Triplet-Loss-in-Keras

Here we will build a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). In lecture, we also talked about [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 

Face recognition problems commonly fall into two categories: 

- **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem. 
- **Face Recognition** - "who is this person?". For example, the video lecture showed a face recognition video (https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem. 

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

**In this project, we will:**
- Implement the triplet loss function
- Use a pretrained model to map face images into 128-dimensional encodings
- Use these encodings to perform face verification and face recognition

# Naive Face Verification

In Face Verification, you're given two images and you have to tell if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person! 

Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on. 

We can see that rather than using the raw image, you can learn an encoding $f(img)$ so that element-wise comparisons of this encoding gives more accurate judgements as to whether two pictures are of the same person.

# Encoding face images into a 128-dimensional vector 

## Using an ConvNet  to compute encodings

The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning settings, let's just load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). We have provided an inception network implementation. You can look in the file `inception_blocks.py` to see how it is implemented.

The key things are:

- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$ 
- It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a 128-dimensional vector

By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings the compare two face images as follows:

1.2 - The Triplet Loss
For an image  xx , we denote its encoding  f(x)f(x) , where  ff  is the function computed by the neural network.



Training will use triplets of images  (A,P,N)(A,P,N) :

A is an "Anchor" image--a picture of a person.
P is a "Positive" image--a picture of the same person as the Anchor image.
N is a "Negative" image--a picture of a different person than the Anchor image.
These triplets are picked from our training dataset. We will write  (A(i),P(i),N(i))(A(i),P(i),N(i))  to denote the  ii -th training example.

You'd like to make sure that an image  A(i)A(i)  of an individual is closer to the Positive  P(i)P(i)  than to the Negative image  N(i)N(i) ) by at least a margin  αα :

∣∣f(A(i))−f(P(i))∣∣22+α<∣∣f(A(i))−f(N(i))∣∣22
∣∣f(A(i))−f(P(i))∣∣22+α<∣∣f(A(i))−f(N(i))∣∣22
 
You would thus like to minimize the following "triplet cost":

=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22(1)−∣∣f(A(i))−f(N(i))∣∣22(2)+α]+(3)
(3)J=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22⏟(1)−∣∣f(A(i))−f(N(i))∣∣22⏟(2)+α]+
 
Here, we are using the notation " [z]+[z]+ " to denote  max(z,0)max(z,0) .

Notes:

The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small.
The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large, so it thus makes sense to have a minus sign preceding it.
αα  is called the margin. It is a hyperparameter that you should pick manually. We will use  α=0.2α=0.2 .
Most implementations also normalize the encoding vectors to have norm equal one (i.e.,  ∣∣f(img)∣∣2∣∣f(img)∣∣2 =1); you won't have to worry about that here.

Exercise: Implement the triplet loss as defined by formula (3). Here are the 4 steps:

Compute the distance between the encodings of "anchor" and "positive":  ∣∣f(A(i))−f(P(i))∣∣22∣∣f(A(i))−f(P(i))∣∣22 
Compute the distance between the encodings of "anchor" and "negative":  ∣∣f(A(i))−f(N(i))∣∣22∣∣f(A(i))−f(N(i))∣∣22 
Compute the formula per training example:  ∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α 
Compute the full formula by taking the max with zero and summing over the training examples:
=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α]+(3)
(3)J=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α]+
 
Useful functions: tf.reduce_sum(), tf.square(), tf.subtract(), tf.add(), tf.maximum(). For steps 1 and 2, you will need to sum over the entries of  ∣∣f(A(i))−f(P(i))∣∣22∣∣f(A(i))−f(P(i))∣∣22  and  ∣∣f(A(i))−f(N(i))∣∣22∣∣f(A(i))−f(N(i))∣∣22  while for step 4 you will need to sum over the training examples.
### References:

- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
- The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
