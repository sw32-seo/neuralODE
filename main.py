import argparse
import train_ode
import train_resnet
import train_cnf


def main(args):
    if args.model == 'odenet':
        train_ode.train_and_evaluate(args.lr, args.n_epoch, args.batch_size, args.tol)
    elif args.model == 'resnet':
        train_resnet.train_and_evaluate(args.lr, args.n_epoch, args.batch_size)
    elif args.model == 'cnf':
        train_cnf.train(0.001, 1000, 512, 2, 32, 64, 0., 10., args.viz, args.sample_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument("--model", type=str, choices=['odenet', 'resnet', 'cnf'], default="odenet",
                        help="Type of model")
    parser.add_argument("--tol", type=float, default=1e-1,
                        help="Error tolerance for ODE solver. This only works with odenet")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epoch", type=int, default=10, help="Total number of epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images in batch")
    parser.add_argument("--sample_dataset", type=str, choices=['circles', 'moons'], default="circles",
                        help="Sample dataset")
    parser.add_argument("--viz", action='store_true')

    args = parser.parse_args()
    main(args)
