# authors: Amitesh Mahajan and Vaibhav Sahu
# course: CS6600 Intelligent Systems
# Project: Road sign detection using opencv and tensorflow.


import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    print ('loading datasest')
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each lable, load it's images and add them to the images list.
        # And add the label numer (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
ROOT_PATH = "/traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)


#print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))


def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    print ('Display the first image of each label.')
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

# display_images_and_labels(images, labels)


def display_label_images(images, label):
    """Display images of a specific label."""
    print ('displaying label images')
    limit = 24  # show a max of 16 images
    #plt.title('labled images of a particular type')
    #plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels[start:].index(label+1)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

# display_label_images(images, 32)


def main():
	images, labels = load_data(train_data_dir)
	display_images_and_labels(images, labels)
	display_label_images(images, 32)
	for image in images[:5]:
		print ("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
	images32 = [skimage.transform.resize(image, (32, 32)) for image in images]

	labels_a = np.array(labels)
	images_a = np.array(images32)
	print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
	graph = tf.Graph()

# Create model in the graph.
	with graph.as_default():
	    # Placeholders for inputs and labels.
	    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
	    labels_ph = tf.placeholder(tf.int32, [None])

	    # Flatten input from: [None, height, width, channels]
	    # To: [None, height * width * channels] == [None, 3072]
	    images_flat = tf.contrib.layers.flatten(images_ph)

	    # Fully conected layer. 
	    # Generates logits of size [None, 62]
	    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

	    # Convert logits to one-hot vector. Shape [None, 62], type float.
	    predictions = tf.nn.softmax(logits)

	    # Convert one-hot vector to label index (int). 
	    # Shape [None], which is a 1D vector of length == batch_size.
	    predicted_labels = tf.argmax(predictions, 1)

	    # Define the loss function. 
	    # Cross-entropy is a good choice for classification.
	    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph))

	    # Create training op.
	    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	    # And, finally, an initizliation op to execute before training.
	    init = tf.initialize_all_variables()

	print("images_flat: ", images_flat)
	print("logits: ", logits)
	print("loss: ", loss)
	print("predictions: ", predictions)
	print("predicted_labels: ", predicted_labels)

	print('starting training...')
# Create a session to run the graph we created.
	session = tf.Session(graph=graph)

# First step is always to initalize our variables. 
# We don't care about the return value, though. It's None.
	_ = session.run([init])


	for i in range(201):
	    _, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})
	    if i % 10 == 0:
	    	print("Loss: ", loss_value)
	print('training finished...')
	print(len(images32))
	sample_indexes = random.sample(range(len(images32)), 10)
	print(sample_indexes)
	sample_images = [images32[i] for i in sample_indexes]
	print(len(sample_images))
	sample_labels = [labels[i] for i in sample_indexes]
	print(len(sample_labels))

	# Run the "predicted_labels" op.
	predicted = session.run([predicted_labels], feed_dict={images_ph: sample_images})[0]
	print(sample_labels)
	print(predicted)
	plt.figure(figsize=(15, 15))
	for i in range(len(sample_images)):
	    truth = sample_labels[i]
	    #print('inside for loop which sucks')
	    prediction = predicted[i]
	    print(prediction)
	    plt.subplot(5, 2,1+i)
	    plt.axis('off')
	    #print(truth)
	    color='green' if truth == prediction else 'red'
	    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),fontsize=12, color=color)
	    #print("here!!")
	    plt.imshow(sample_images[i])
	plt.show()
	    #print('imshow shatement executed..')
	#print('I believe we reached to the final!')



if __name__ == "__main__":
	print ('starting main function!')
	main()