from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    from itertools import chain

    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    from ..models import MolEncoder, MolDecoder
    from ..utils import (load_dataset, train_model, adjust_learning_rate,
                         save_checkpoint, validate_model)

    data_train, data_val, charset = load_dataset(args.dataset)

    data_train = torch.FloatTensor(data_train)
    data_val = torch.FloatTensor(data_val)

    train = TensorDataset(data_train, torch.zeros(data_train.size()[0]))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    val = TensorDataset(data_val, torch.zeros(data_val.size()[0]))
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True)

    dtype = torch.FloatTensor
    encoder = MolEncoder(c=len(charset))
    decoder = MolDecoder(c=len(charset))

    if args.cuda:
        dtype = torch.cuda.FloatTensor
        encoder.cuda()
        decoder.cuda()

    if args.cont:
        checkpoint = torch.load('checkpoint.pth.tar')
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer = optim.Adam(chain(encoder.parameters(),
                                     decoder.parameters()))
        decoder.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['avg_val_loss']
    else:
        optimizer = optim.Adam(chain(encoder.parameters(),
                                     decoder.parameters()))
        best_loss = 1E6

    for epoch in range(args.num_epochs):
        print('Epoch %s:' % epoch)
        adjust_learning_rate(optimizer, epoch, lr_init=args.learning_rate)
        train_model(train_loader, encoder, decoder, optimizer, dtype)

        avg_val_loss = validate_model(val_loader, encoder, decoder, dtype)
        is_best = avg_val_loss < best_loss
        save_checkpoint({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'avg_val_loss': avg_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)


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
    p.add_argument('--cuda', help="Use GPU acceleration",
                   action='store_true')
    p.add_argument('--cont', help="Continue from saved state",
                   action='store_true')
    p.set_defaults(func=func)
