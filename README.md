# cnv_neural_network_pokemon

A Neural Network used for Image Classification and Computer Vision. How can we balance all the technical terms, specifications and architectural decisions to make this problem more interesting? With Pokemon!

<p align="center">
  <img width="80%" src="/github_assets/NN.png" alt="NN_structure"/>
</p>

In this code, we will setup a Neural Network that can identify, with a decent level of accuracy, which Pokemon the image contains. While we try to maximize the accuracy, while trying to minimize Dataset size, Training time, and Model size.

Try it online! [https://cnn-pokemon/eval](https://cnn-pokemon-service-wyclairfnq-rj.a.run.app/eval)


## 🧊 Cool features:

### 📐✏️ Convolutional Vs Traditional designs

A Convolutional Neural Network (CNN for short) is a specialized type of neural network model designed for working with two-dimensional image data, like images. 

When passing data to a regular, old-school Neural Network, we break the image into all the individual pixels to fed each one into a neuron from the input layer. Bigger the image, bigger the layer, more weights to be calculated, more time to train it. We also get more depend on individual pixel information. Noise in the data will turn out to be a big thing.

The CNN, on the other hand, tries to look at the big picture (pun intended!). Instead of isolated pixels, it tries to get specific features in the image, to understand what makes this input different from the others. They do this by performing a “convolution”, that is, a linear operation that involves the multiplication of a set of weights (called ”filter” or “kernel”) with the input, much like a traditional neural network. The application of that filter systematically across the entire input image allows the filter an opportunity to discover that feature anywhere in the image.


### 📊📊📊 Data Augmentation

This trick is often performed with image data, where copies of images in the training dataset are created with some image manipulation techniques performed, such as zooms, flips, shifts, and more.
The artificially expanded training dataset can result in a more skillful model, as often the performance of deep learning models continues to scale in concert with the size of the training dataset. In addition, the modified or augmented versions of the images in the training dataset assist the model in extracting and learning features in a way that is invariant to their position, lighting, and more.


## 🗂️ Usage:

Try it online! [https://cnn-pokemon/eval](https://cnn-pokemon-service-wyclairfnq-rj.a.run.app/eval)

Or:

- Download the code:
```
  $ git clone https://github.com/Alessandro1918/cnv_neural_network_pokemon.git
```

- Assemble the dataset:</br>
Download as many images as you can (more images, better training) to a <code>dataset</code> folder, like:
```
cnv_neural_network_pokemon
  ├ dataset
  | ├ Bulbasaur
  | | ├ img1.jpg
  | | ├ ...
  | | └ img20.jpg
  | ├ Charmander
  | ├ Pikachu
  | └ Squirtle
  ├ cnn.py
  └ test.py
```
If you want, there is a link for the images used for this example:
(Github won’t allow directories with 100+ files, so I uploaded them separately.)
```
  https://www.dropbox.com/sh/vkmeeirmi4nb1tr/AACUTojKyBnJ7_FoMzPX1Gp5a?dl=0
```

- Install the required libs:
```
  $ cd cnv_neural_network_pokemon      #change to that directory
  $ pip3 install -r requirements.txt   #download dependencies
```

- Train the network:</br>
  - Split the dataset into Train, Validation, Test;
  - Define the structure of the Convolutional Neural Network;
  - Train the CNN;
  - Show the values of Loss, Accuracy, and Plot the Training Curve;
  - Export the model for future use;
```
  $ python3 cnn.py
```

- Use the  model to classify some images from the Test set:
```
  $ python3 test.py
```


## ▶️ Demo time!

The <code>test.py</code> will plot some images, with the respective classification predicted by the model. The model doesn’t have a 100% accuracy, but can get very close to it, considering how small the Training set was, and how different the images are among the same group!

You can also visualize the Data Augmentation principle here; instead of classify the original images on the Test set, we edit them with the same zooms, flips, shifts and the like we use for the Train test, with pretty good results!

<p align="center">
  <img width="48%" src="/github_assets/plot_test_210.png" alt="plot_test_1"/>
  <img width="48%" src="/github_assets/plot_test_333.png" alt="plot_test_2"/>
</p>
<p align="center">
  <img width="48%" src="/github_assets/plot_test_555.png" alt="plot_test_3"/>
  <img width="48%" src="/github_assets/plot_test_931.png" alt="plot_test_4"/>
</p>

## ⭐ Like, Subscribe, Follow!
Liked the project? Give this repository a Star ⭐!


## 📝 License

This project is under a MIT License. Check out the file [LICENSE](LICENSE.md) for more details.


## 🚫 Disclaimer

This git is unofficial, free fan made, and is NOT affiliated, endorsed or supported by Nintendo, GAME FREAK or The Pokémon company in any way.
Pokémon and Pokémon character names are trademarks of Nintendo.
No copyright infringement intended.
