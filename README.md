# Tomato Leaf Disease Classification
Plant diseases put on a heavy toll on the agricultural economy. Early detection and prevention project promising results in cutting losses. However, using expert knowledge is not a scalable solution to this problem in any shape or form. Building autonomous systems based on the state-of-the-art techniques of deep learning is of high importance. Here, we analyze and compare two such models through the lens of four augmented architectures. The best and most stable model architecture yields 86.69% accuracy.  

## 1. Data
The data we are using for this particular task is a publicly open dataset of plant leaves images taken under laboratory conditions (available: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)). The dataset was originally available in three color variations:  
* Color – Images are in RGB color space  
* Grayscale – Images are in grayscale color space  
* Segmented – Leaves are segmented from the background and are in RGB  

Since other papers [2][3] have already worked with this dataset, the results consistently showed that models trained on the color version of the dataset outperformed their equivalent model architectures trained on the other versions, we decided to only work on the color version.  

Contrary to what [4] is suggesting, there is no point in using stylized version of the datasets because diseases mostly manifest in color and texture changes and less in shape changes. By stylizing the dataset, we would lose both texture and color, so it wouldn't be of any help.  

The dataset contains images for 14 plants along with the leaf diseases typical for each plant, summing to a total of 38 distinct [plant, disease] combinations. While other papers have focused on training models to classify these 38 classes, we were interested in testing the robustness of the state-of-the-art models on a particular plant with limited data.  

class | # of samples  
--- | ---  
Bacterial spot | 2127  
Early blight | 1000  
Late blight | 1909  
Leaf mold | 952  
Septoria leaf spot | 1771  
Spider mites	| 1676  
Target spot	| 1404  
Mosaic virus | 373  
Yellow leaf curl virus | 5332  
Healthy | 1591  

For this purpose, we’ve chosen the tomato plant which single-handedly accounts for 10 classes of the dataset, making the task even harder. To push the tests even further, as we can observe from the “number of samples” column in the table above, the dataset is highly imbalanced.  

In order to appropriately train and evaluate the model, we wrote a script that partitions the dataset into train, test and validation sets. The splitting had to be done in the preprocessing segment and not in runtime because of the dataset’s size – there was simply not enough RAM to perform such an operation, so we wrote the results on disk. We split the dataset in 80-20 proportion for the train-test sets. Of course, we need validation set, so we further split the train set in 80-20 proportion for the train-validation sets.   

The training set had 11614 samples, the validation set had 2884 samples and the testing set had 3637 samples. We can totally see how this limited dataset can cause trouble, thus forcing us to be creative with our approach.  

Finally, the data was uploaded to Google Drive, since the training was done on Google Colab – an online platform founded by Google that provides free GPU services up to 12 hours in a single session.  

## 2. Modeling
The modeling segment was implemented in Python using Keras with a Tensorflow backend.  

As we mentioned before, the dataset couldn’t possibly fit into RAM, so in order to train the models we use ImageDataGenerator – a Keras generator that pulls images from disk at the time they are needed to train the models.  
Another neat feature is that the generator can apply a certain set of transformations to the input image with certain probabilities in order to avoid overfitting the models. The set of transformations we use are:  
*	Rescale image intensities to range [0, 1]  
*	Rotate in range [0, 40]  
*	Width shift in range [0, 0.2]  
*	Height shift in range [0, 0.2]  
*	Shear in range [0, 0.2]  
* Zoom in range [0, 0.2]  

For the purpose of this task, we’ve decided to evaluate two well-known architectures that were winners in the ImageNet Large Scale Visual Recognition Challenge at some point in time: **VGG-16** (2014) and **ResNet-50** (2015).  

Since Keras provides the architectures used in this competition, once we instantiate the base models, we drop the last layer which by default is a Dense layer with shape [1x1x1000] that has softmax activation function representing the probabilities for recognizing one of the thousand classes in the ImageNet dataset, and replace it with a Dense layer with shape [1x1x10] that has softmax activation function in order to fulfill our needs.  

For each of the architectures, we’ve trained four variations:  
1.	**Plain** – base model was untrained, augmented model was completely trained on PlantVillage  
2.	**Pretrained** – base model was pretrained on ImageNet, augmented model was fine-tuned on PlantVillage  
3.	**Pretrained (l2)** – introducing Ridge regularization on model 2  
4.	**Pretrained (dropout)** – introducing dropout layer on model 2  

The idea is to find out how important are the learned patterns from the ImageNet competition for leaf disease recognition (model 1 versus model 2). Model 3 and model 4 are used to further fine-tune model 2 and fix any potential overfitting it might have done.  

All model variations are compiled under the **Adam** optimizer and **categorical crossentropy** loss function.

Since a single session in Google Colab lasts up to 12 hours, each VGG-16 variation was trained on 100 epochs and each ResNet-50 variation on 50 epochs due to network complexity.
It’s fair to assume there is a real chance that both models will fail to generalize well on the validation/test sets because, especially the plain variations because:
1.	**Plain ResNet-50** has 24,538,122 trainable parameters
2.	**Plain VGG-16** has 14,965,578 trainable parameters
3.	**Pretrained ResNet-50** has 1,003,530 trainable parameters
4.	**Pretrained VGG-16** has 250,890 trainable parameters

Due to the small size of the dataset, learning that many free parameters will definitely take its toll.

## 3. Results
After training all of the variations to a total of 48 hours, we conclude the results.  

 Model | Plain | Pretrained base | Pretrained base (l2) | Pretrained base (dropout)  
--- | --- | --- | --- | ---  
**VGG-16** | 0.310841 | 0.886615 | 0.772400 | 0.866980  
**ResNet-50** | 0.548119 | 0.019358 | 0.310841 | 0.310841  

From the results in the table above, we can immediately see that all of the ResNet-50 architectures severely underperformed compared to the VGG-16 architectures. This is because even though ResNet is currently the best performing convolutional neural network for classification, we can observe that it is definitely an overkill for this task.  

On the other hand, we can observe that VGG-16’s performance strongly increased when introducing pretrained weights on the ImageNet dataset. Even though the domain of the base model is absolutely different (some of the 1000 classes are: fish, chicken, fruits, broom, car…) it helps in providing a network that has the trained its first layers for efficiently capturing edges and textures.  

![comparison](https://github.com/ZafirStojanovski/tomato-leaf-disease-detection/blob/master/comparison.png)  

Lastly, from the figure above we can observe that the VGG with pretrained base has a slight overfitting, so we try to fix it by introducing Ridge regularization and a dropout layer. However, regularization didn’t seem to help much with the problem, but the dropout definitely did (upper right plot). Even though both the VGG with pretrained base and VGG with pretrained base (dropout) achieved similar accuracies on the test set, we advise to use the second variation because it’s not overfitted and will generalize better on unseen test sets.  

![activations](https://github.com/ZafirStojanovski/tomato-leaf-disease-detection/blob/master/activations.png)  

From the activations of the best performing model (VGG-16 pretrained on ImageNet with a Dropout layer), we see that the convolution layers in the first block mainly focus on extracting texture patterns, edges and the overall contour of the leaf, while the convolution layers in the last block manage to create abstract representations of the key patterns typical for the disease.  

We believe the results could substantially improve in future work that involves working with a balanced dataset that provides much more samples. Also, working with the simpler architectures like AlexNet, winner of ImageNet LSVRC 2012, or even a custom one could potentially increase the overall model accuracy. Finally, there’s room to experiment with other forms of preventing overfitting like Lasso regularization, various values for lambda in both the regularization forms, different keep_probability value for the dropout layer and much more.  


## Conclusion 
Our best stable model achieved accuracy of 86.69% under highly rigid circumstances – unbalanced low-sample 10-class dataset. Convolutional neural networks once again stood the test by being able to recognize intricate patterns in cases where not even human experts could. The agriculture economy is at stake and these networks have the potential of significantly helping by cutting down on losses. Being witnesses to the evergrowing field of deep learning, we expect future techniques to push the boundaries even further than before. 

### References
[1] Agrios GN (1972). Plant Pathology (3rd ed.). Academic Press.  
[2] K.P. Ferentinos, Deep learning models for plant disease detection and diagnosis[J]. Comput. Electron. Agric. 145, 311–318 (2018)  
[3] Wallelign, S., Polceanu, M., Buche, C.: Soybean Plant Disease Identification Using Convolutional Neural Network  
[4] Geirhos, R., Rubisch P.: ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv:1811.12231 (2019)  
[5] Francis, J., Sahaya. A., Anoop, B.K: Identification of leaf diseases in pepper plants using soft computing techniques (2016)  






