from argparse import ArgumentDefaultsHelpFormatter

DEFAULTS = {
    "chembl22": {
        "uri": "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz",
        "outfile": "data/chembl22.h5"
    },
    "zinc12": {
        "uri": "http://zinc.docking.org/db/bysubset/13/13_prop.xls",
        "outfile": "data/zinc12.h5"
    }
}

MAX_NUM_ROWS = 500000


def func(args, parser):
    import os
    import sys
    import argparse
    import urllib.request
    import tempfile

    import numpy as np
    import pandas
    from sklearn.model_selection import train_test_split
    from progressbar import (ProgressBar, Percentage, Bar,
                             ETA, FileTransferSpeed)

    from rdkit import Chem
    from ..featurizers import OneHotFeaturizer

    uri, outfile, dataset = args.uri, args.outfile, args.dataset

    if dataset and dataset in DEFAULTS.keys():
        uri = DEFAULTS[args.dataset]['uri']
        outfile = outfile or DEFAULTS[dataset]['outfile']
        if not os.path.exists('data'):
            os.makedirs('data')
    elif args.dataset not in DEFAULTS.keys():
        parser.error("Dataset %s unknown. Valid choices are: %s" %
                     (dataset, ", ".join(DEFAULTS.keys())))

    if uri is None:
        parser.error(
            "You must choose either a known --dataset or provide a --uri and --outfile.")
        sys.exit(1)
    if outfile is None:
        parser.error("You must provide an --outfile if using a custom --uri.")
        sys.exit(1)

    fd = tempfile.NamedTemporaryFile()
    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', ETA(),
                                    ' ', FileTransferSpeed()])

    def update(count, blockSize, totalSize):
        if progress.max_value is None:
            progress.max_value = totalSize
            progress.start()
        progress.update(min(count * blockSize, totalSize))

    print('Downloading Dataset...')
    urllib.request.urlretrieve(uri, fd.name, reporthook=update)

    print('Loading Dataset...')
    if dataset == 'zinc12':
        df = pandas.read_csv(fd.name, delimiter='\t')
        df = df.rename(columns={'SMILES': 'structure'})
    elif dataset == 'chembl22':
        df = pandas.read_table(fd.name, compression='gzip')
        df = df.rename(columns={'canonical_smiles': 'structure'})
    else:
        df = pandas.read_csv(fd.name, delimiter='\t')

    keys = df[args.smiles_column].map(len) < 121

    if MAX_NUM_ROWS < len(keys):
        df = df[keys].sample(n=MAX_NUM_ROWS)
    else:
        df = df[keys]

    print('Processing Dataset...')
    smiles = df[args.smiles_column]

    del df

    featurizer = OneHotFeaturizer()
    one_hot = featurizer.featurize(smiles)

    train_idx, test_idx = map(np.array,
                              train_test_split(smiles.index, test_size=0.20))

    h5f = h5py.File(outfile, 'w')
    h5f.create_dataset('charset', data=charset)

    def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                             chunk_size=1000):
        new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                         chunks=tuple([chunk_size] +
                                                      list(dataset_shape[1:]))
                                         )
        for (chunk_ixs, chunk) in chunk_iterator(dataset):
            new_data[chunk_ixs, ...] = chunk

    create_chunk_dataset(h5f, 'data_train', one_hot[train_idx],
                         (len(train_idx), 120, len(charset)))
    create_chunk_dataset(h5f, 'data_test', one_hot[test_idx],
                         (len(test_idx), 120, len(charset)))

    h5f.close()

    print("Done!")


def configure_parser(sub_parsers):
    help = 'Download SMILES datasets'
    p = sub_parsers.add_parser('download', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str,
                   help="%s  ...or specify your own --uri" % ",".join(DEFAULTS.keys()))
    p.add_argument('--uri', type=str,
                   help='URI to download ChEMBL entries from')
    p.add_argument('--outfile', type=str, help='Output file name')
    p.add_argument('--smiles_column', type=str, default='structure',
                   help="Name of the column that contains the SMILES strings.")
    p.set_defaults(func=func)
