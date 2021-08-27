import argparse

args = argparse.ArgumentParser()
args.add_argument('--inpath',
                  help='Path to folder containing input networks')
args.add_argument('--sparse',
                  action='store_true',
                  help='If sparse adjacency matrices are used as inputs')
args.add_argument('--feat_file',
                  help='File containing node features')
args.add_argument('--val_file',
                  help='File containing validation edges')
args.add_argument('--enc',
                  type=str,
                  default='GCN',
                  choices=['GCN', 'GAT', 'multiGCN', 'multiGAT', 'multihopGAT'])
args.add_argument('--reduce',
                  type=str,
                  choices=['mean', 'sum', 'max'],
                  help='To reduce the concatenated node embeddings')
args.add_argument('--tolerance',
                  type=int,
                  default=5,
                  help='Number of epoch to wait for the score to improve on the validation set before stopping')
args.add_argument('--emb_size',
                  default=64,
                  type=int,
                  help='Embedidng Size L')
args.add_argument('--hid_dims',
                  default=[128],
                  type=int,
                  nargs='+',
                  help='Hidden layer dimensions (multiple values can be used in case of more than one hidden layer)')
args.add_argument('--dropout_hid',
                  default=0.1,
                  type=float,
                  help='Dropout rates in hidden (non-encoder) layers')
args.add_argument('--dropout_enc',
                  default=0.1,
                  type=float,
                  help='Dropout rates in encoder layers')
args.add_argument('--max_hops',
                  default=1,
                  help='Max. number of hops used to sample neighbors')
args.add_argument('--batch_size',
                  default=500,
                  help='Minibatch size (number of nodes)')
args.add_argument('--max_iter',
                  type=int,
                  default=100,
                  help="Maximum number of iterations")
args.add_argument('--outpath',
                  default="/",
                  help="Path to output folder")

args = args.parse_args()
# print(args)