# gene2gauss: A multi-view gaussian gene embedding learner for analyzing transcriptomic networks
<br/>
<p align="center"><img src="Schematic.png" style="vertical-align:middle" width="800" height="500"></p>

This repository contains the code for the algorithm <i>gene2gauss</i>, part of a manuscript submission for AMIA 2022 Informatics Summit.

## Requirements
 * <a href="https://www.tensorflow.org/install">TensorFlow 2</a>
 * numpy (>=1.18.1), pandas (>=1.0.3), scikit-learn (>=0.22.1)
 * scipy (>=1.4.1) for sparse implementation
 * matplotlib (>=3.1.3) for generating visualizations

&nbsp;&nbsp;All requirements are listed in <b>requirements.txt</b> and can be installed together
```bash
pip install -r requirements.txt
```

## Training
Our model can be trained in either supervised or unsupervised approaches <br/>
```bash
python training_supervised.py (or) training_unsupervised.py 
      --inpath /path to input networks/
      --feat_file /path to matrix(.npy) containing node features/
      --enc type_of_encoder('GCN' or 'GAT' or 'multiGCN' or 'multiGAT')
      --sparse --val_file /path to file containing testing edges/ 
      --hid_dims num_of_filters
      --emb_size final_embedding_size
      --max_iter maximum number of iterations
      --tolerance for_early_stopping
      --outpath /outdir/
```
* <i> `--`inpath</i>: Path to folder one or more input networks in the form of archived numpy (.npz) files.
* <i> `--`feat_file</i>: Path to a numpy array (.npy) containing node features for attributes graphs. Number of rows must match the number of network nodes. If not provides, identity features will be generated and used during the training.
* <i> `--`sparse</i>: A boolean flag to indicate if sparse implementations need to be used during training.
* <i> `--`enc</i>: Type of convolution layer/encoder (one among <i>GCN</i>, <i>GAT</i>, <i>multiGCN</i> or <i>multiGAT</i>)
* <i> `--`val_file</i>: Path to the file containing the edges for link-prediction objective (supervised). In the unsupervised case, training edges will be retrieved from the input networks and these edges will be used for evaluating the final node features.
* <i> `--`hid_dims</i>: Number of convolution filters to be used. Multiple values (separated by a space) are allowed for deep implementations (default = 128).
* <i> `--`emb_size</i>: Dimensionality of the final gaussian embedding mean and variance vectors (default = 64).
* <i> `--`max_iter</i>: Maximum number of training iterations (default = 100)
* <i> `--`tolerance</i>: tolerance parameter value for early stopping criterion.
* <i> `--`outpath</i>: Path to output directory for storing the final node embedding vectors
 