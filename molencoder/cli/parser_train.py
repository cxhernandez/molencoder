from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    import torch.optim as optim

    from ..models import MolEncoder, MolDecoder
    from ..utils import load_dataset, train

    data_train, data_test, charset = load_dataset(args.dataset)


    encoder = MolEncoder(c=len(charset))
    decoder = MolDecoder(c=len(charset))

    for i in range(args.num_epochs):
        train(loader_train, encoder, decoder, optimizer, dtype)



def configure_parser(sub_parsers):
    help = 'Train autoencoder'
    p = sub_parsers.add_parser('train', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.set_defaults(func=func)
