import argparse
import train_ode
import train_resnet


def main(args):
    if args.model == 'odenet':
        train_ode.train_and_evaluate(args.lr, args.n_epoch, args.batch_size, args.tol)
    else:
        train_resnet.train_and_evaluate(args.lr, args.n_epoch, args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument("--model", type=str, choices=['odenet', 'resnet'], default="odenet", help="Type of model")
    parser.add_argument("--tol", type=float, default=1e-1,
                        help="Error tolerance for ODE solver. This only works with odenet")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epoch", type=int, default=10, help="Total number of epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images in batch")

    args = parser.parse_args()
    main(args)
