```neowise-ML``` is a deep learning transient classification tool for WISE/NEOWISE images. This software allows batch classification of transient sources based on two pre-trained convolutional neural networks: a 'triplet' model that requires a science, reference, and difference image; and an 'echo' model that requires 18 epochs of difference imaging.

Paper forthcoming...

# Usage
Please see the [tutorial notebook](https://github.com/vegajustin26/neowise-ML/blob/main/notebooks/tutorial.ipynb) for an example of scoring datasets.

# GPU support
By default this package requires the CPU version of TensorFlow. To take advantage of GPU acceleration, install the GPU version of TensorFlow using the installation guide [here](https://www.tensorflow.org/guide/gpu) and ensure that the GPU is recognized by TensorFlow before proceeding with scoring.
