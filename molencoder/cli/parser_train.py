from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    from itertools import chain

    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    from ..models import MolEncoder, MolDecoder
    from ..utils import load_dataset, train_model

    data_train, data_test, charset = load_dataset(args.dataset)

    data_train = torch.FloatTensor(data_train)
    # data_test = torch.FloatTensor(data_test)
    train = TensorDataset(data_train, torch.zeros(data_train.size()[0]))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # test = TensorDataset(data_test, torch.zeros(data_test.size()[0]))
    # test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    dtype = torch.FloatTensor
    encoder = MolEncoder(c=len(charset))
    decoder = MolDecoder(c=len(charset))

    if args.cuda > 0:
        dtype = torch.cuda.FloatTensor
        encoder.cuda()
        decoder.cuda()

    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()),
                           lr=args.learning_rate)
    for i in range(args.num_epochs):
        train_model(train_loader, encoder, decoder, optimizer, dtype)


def configure_parser(sub_parsers):
    help = 'Train autoencoder'
    p = sub_parsers.add_parser('train', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=str, help="Path to HDF5 dataset",
                   required=True)
    p.add_argument('--num-epochs', type=int, help="Number of epochs",
                   default=1)
    p.add_argument('--learning-rate', type=float, help="Learning rate",
                   default=1E-4)
    p.add_argument('--batch-size', type=int, help="Batch size", default=100)
    p.add_argument('--cuda', type=int, help="Use GPU acceleration",
                   default=1)
    p.set_defaults(func=func)
