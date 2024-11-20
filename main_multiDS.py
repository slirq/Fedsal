import os
import argparse
import random
import copy

import torch
from pathlib import Path

import setupGC
from training import *
import pickle


def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame,metrics_frame,metrics = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
        m_outfile = os.path.join(outpath, f'M_accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
        m_outfile = os.path.join(outpath, "repeats", f'M_{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    metrics_frame.to_csv(m_outfile)
    frame.to_csv(outfile)
    m_name = outfile[:-4]
    with open(f'{m_name}.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Wrote to file: {outfile},{m_outfile}")


def process_fedsal(args, clients, server):
    print("\nDone setting up fedsal devices.")
    print("Running fedsal ...")
    frame,metrics_frame,metrics = run_fedsal(args, clients, server, args.num_rounds, args.local_epoch, EPS_1_sal, EPS_2_sal)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedsal{suffix}.csv')
        m_outfile = os.path.join(outpath, f'M_accuracy_fedsal{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedsal{suffix}.csv')
        m_outfile = os.path.join(outpath, "repeats", f'M_{args.repeat}_accuracy_fedsal{suffix}.csv')
    metrics_frame.to_csv(m_outfile)
    frame.to_csv(outfile)
    m_name = outfile[:-4]
    with open(f'{m_name}.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Wrote to file: {outfile},{m_outfile}")


def process_fedsalplus(args, clients, server):
    print("\nDone setting up fedsal plus devices.")

    print("Running fedsal plus ...")
    frame,metrics_frame,metrics = run_fedsal(args, clients, server, args.num_rounds, args.local_epoch, EPS_1_sal, EPS_2_sal)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedsalplus{args.type_init}.csv')
        m_outfile = os.path.join(outpath, f'M_accuracy_fedsalplus{args.type_init}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedsalplus{suffix}.csv')
        m_outfile = os.path.join(outpath, "repeats", f'M_{args.repeat}_accuracy_fedsalplus{suffix}.csv')
    metrics_frame.to_csv(m_outfile)
    frame.to_csv(outfile)
    m_name = outfile[:-4]
    with open(f'{m_name}.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Wrote to file: {outfile},{m_outfile}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='mix')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
    
    parser.add_argument('--epssal1', help='the threshold saliency epsilon1 for FedSAL',
                        type=float, default=6)
    parser.add_argument('--epssal2', help='the threshold saliency epsilon2 for FedSAL',
                        type=float, default=8)
    
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Name of algorithms.')
    parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--n_ones', type=int, default=16,
                        help='Size of position encoding (ones).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPS_1_sal = args.epssal1
    EPS_2_sal = args.epssal2

    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')

    if args.overlap:
        outpath = os.path.join(outbase, f"multiDS-overlap")
    else:
        outpath = os.path.join(outbase, f"multiDS-nonOverlap")
    outpath = os.path.join(outpath, args.data_group)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    # preparing data
    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)
    
    
    
    splitedData, df_stats = setupGC.prepareData_multiDS(args,args.datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=100)
    process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_fedsal(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    
    args.alg='salstr'
    args.n_se = args.n_rw + args.n_dg
    splitedData, df_stats = setupGC.prepareData_multiDS(args,args.datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")
    
    process_fedsalplus(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))