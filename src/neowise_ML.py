import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '3' suppresses INFO, WARNING, and ERROR messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import tensorflow as tf
import matplotlib.pyplot as plt

def score(data, model = "triplet", batch_size = 32, raw_logits = False):
    """
    Scores cutout images using the models described in Vega et al. 2026.

    Parameters
    ----------
    data : ndarray
        Input array of shape (n, 61, 61, 3) or (n, 61, 61, 18).
        Triplet cutouts (sci, ref, diff) or difference images (18 cutouts).
    model : {"triplet", "echo"}, optional
        Select which model to use for scoring. Defaults to "triplet".
    batch_size : int
        Number of cutouts evaluated at once. Defaults to 32.
    raw_logits : bool
        If true, returns raw class probabilities. Defaults to False.
    
    Returns
    -------
    y_prob : ndarray
        Raw class probabilities of shape (n, 4). 
        Classes: [reals, highpm, echo, artifact].
    y_pred : ndarray
        Class indices of shape (n,).
        Mapping: {0: 'reals', 1: 'highpm', 2: 'echo', 3: 'artifact'}.
    """
    
    model_dir = "../models/"
    
    if model == "triplet":
        keras_model = tf.keras.models.load_model(model_dir + "triplet_model.keras")
        assert data.shape[1:] == (61, 61, 3), "Please make sure data is shape (n, 61, 61, 3)!"
        
    elif model == "echo":
        keras_model = tf.keras.models.load_model(model_dir + "echo_model.keras")
        assert data.shape[1:] == (61, 61, 18), "Please make sure data is shape (n, 61, 61, 18)!"
    else:
        raise Exception("""Please choose either "triplet" or "echo" as model!""")
    
    # data standardization check
    assert np.min(data) == 0 and np.round(np.max(data)) == 1, "Please make sure each cutout is standardized to within +/- 3 sigma and scaled between 0 and 1."
    
    # data = np.array(data[..., np.newaxis])
    
    if data.shape[0] < batch_size: # if small number of cutouts
        batch_size = 1
    
    test_data = tf.data.Dataset.from_tensor_slices(data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    y_probs = keras_model.predict(test_data, verbose=0)
    
    if raw_logits:
        return(y_probs)
    else:
        y_pred = np.argmax(y_probs, axis = 1)
        return(y_pred)
    
def plot_cutout(data, num_epochs = 3):
    
    # data should only be single triplet or 18 epochs
    
    assert data.shape[0] == 3 or data.shape[0] == 18, "Only one triplet or 18 epochs"
    
    if num_epochs == 3:
        fig, ax = plt.subplots(1, num_epochs)
    elif num_epochs == 18:
        fig, ax = plt.subplots(3, 6, figsize = (12, 5))
    
    ax = ax.flatten() 
    
    for idx, i in enumerate(data):
        ax[idx].imshow(i, cmap = "gray")
        ax[idx].axis("off")
    
    plt.show()
    return(fig)