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


def func(args, parser):
    import os
    import sys
    import argparse
    import urllib.request
    import pandas
    import tempfile
    from progressbar import (ProgressBar, Percentage, Bar,
                             ETA, FileTransferSpeed)

    uri, outfile, dataset = args.uri, args.outfile, args.dataset

    if dataset and dataset in DEFAULTS.keys():
        uri = DEFAULTS[args.dataset]['uri']
        outfile = outfile or DEFAULTS[dataset]['outfile']
        if not os.path.exists('data'):
            os.makedirs('data')
    elif args.dataset not in DEFAULTS.keys():
        parser.error("Dataset %s unknown. Valid choices are: %s" % (dataset, ", ".join(DEFAULTS.keys())))

    if uri is None:
        parser.error("You must choose either a known --dataset or provide a --uri and --outfile.")
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

    urllib.request.urlretrieve(uri, fd.name, reporthook=update)

    print('Saving to HDF5...')
    if dataset == 'zinc12':
        df = pandas.read_csv(fd.name, delimiter='\t')
        df = df.rename(columns={'SMILES': 'structure'})
        df.to_hdf(outfile, 'table', format='table', data_columns=True)
    elif dataset == 'chembl22':
        df = pandas.read_table(fd.name, compression='gzip')
        df = df.rename(columns={'canonical_smiles': 'structure'})
        df.to_hdf(outfile, 'table', format='table', data_columns=True)
    else:
        df = pandas.read_csv(fd.name, delimiter='\t')
        df.to_hdf(outfile, 'table', format='table', data_columns=True)

    print("Done!")


def configure_parser(sub_parsers):
    help = 'Download SMILES datasets'
    p = sub_parsers.add_parser('download', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(
        description='Download ChEMBL entries and convert them to input for preprocessing')
    p.add_argument('--dataset', type=str,
                   help="%s  ...or specify your own --uri" % ",".join(DEFAULTS.keys()))
    p.add_argument('--uri', type=str,
                   help='URI to download ChEMBL entries from')
    p.add_argument('--outfile', type=str, help='Output file name')
    p.set_defaults(func=func)
