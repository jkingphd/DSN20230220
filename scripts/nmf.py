from sklearn.decomposition import MiniBatchNMF
from sklearn.preprocessing import normalize
from skimage import io, transform
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import joblib
import time
import sys

STREAM = int(sys.argv[1])
BASE_PATH = '../data/nih/images/'
OUTPUT_SHAPE = (256, 256)
BATCH_SIZE = 1024
EPOCHS = 10
SEED = 101
rng = np.random.RandomState(SEED)

# Load the filenames and generate a train/val split
df = pd.read_csv('../data/nih/sample_labels.csv')
df['val'] = rng.choice([0,1], size=df.shape[0], p=[0.9, 0.1])
idx_train = df['val'] == 0
fnames_train = [BASE_PATH + fname for fname in df.loc[idx_train, 'Image Index'].tolist()]
fnames_val = [BASE_PATH + fname for fname in df.loc[~idx_train, 'Image Index'].tolist()]

def load_image(fname, output_shape=OUTPUT_SHAPE):
    """Function to load image and resize image to target output_shape. 
    Assumes grayscale input and will select first channel if multichannel image is read in."""
    img = io.imread(fname)
    if len(img.shape)>2:
        img = img[:,:,0]
    img = transform.resize(img, output_shape=output_shape)
    return img

def load_image_batch(fnames, n_jobs=4, output_shape=OUTPUT_SHAPE, verbose=False):
    """Helper function for efficiently loading images from disk."""
    if verbose:
        img_batch = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(load_image)(fname) for fname in tqdm(fnames))
    else:
        img_batch = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(load_image)(fname) for fname in fnames)
    img_batch = np.array(img_batch)
    return img_batch

def get_batches(fnames, batch_size=128, n_jobs=-1):
    """Streams images from disk in batches. Uses joblib.Parallel to speed up data loader."""
    splits = np.ceil(len(fnames)/batch_size)
    fname_batches = np.array_split(fnames, splits)
    for fname_batch in fname_batches:
        X_batch = []
        X_batch = load_image_batch(fname_batch, n_jobs=n_jobs)
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1]*X_batch.shape[2])
        X_batch = normalize(X_batch)
        yield X_batch
        
if __name__ == '__main__':
    
    nmf = MiniBatchNMF(
        n_components=20,
        init='nndsvda', 
        batch_size=BATCH_SIZE, # This really dominates noise in the loss as you expect
        beta_loss='frobenius',
        max_no_improvement=20, # Might need to be increased to account for batch_size
        max_iter=EPOCHS, # Maximum number of epochs
        random_state=SEED,
        verbose=True # Sometimes just nice to see that the training is working!
    )
    
    start = time.time()
    if STREAM == 1:
        for i in range(EPOCHS):
            print(f'Working on Epoch {i}')
            batches = get_batches(fnames_train, BATCH_SIZE)
            for X_batch in tqdm(batches, total=np.ceil(len(fnames_train)/BATCH_SIZE)):
                nmf.partial_fit(X_batch)
    else:
        X_train = load_image_batch(fnames_train, n_jobs=-1, verbose=True)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        X_train = normalize(X_train)
        nmf.fit(X_train)
    end = time.time()
    
    print(f'Time to train NMF model: {end-start:0.2f}s')
    joblib.dump(nmf, f'../output/nmf_{STREAM}_{OUTPUT_SHAPE[0]}.joblib')