import argparse
import pickle
import os


def read_arguments():
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    opt = parser.parse_args()
    return opt


def add_all_arguments(parser):
    #--- general options ---
    parser.add_argument('--window_size', type=int, default=250)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--shift_length', type=int, default=125)
    parser.add_argument('--hidden_size', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--out_name', type=str, default='validation_accuracy', help='result png name')
    parser.add_argument('--continue', default=False, action='store_true', help="continue train from best epoch")
    parser.add_argument('--root_path', default='./',type=str)
    parser.add_argument('--device', default='cpu',type=str)
    parser.add_argument('--bidirectional', default=False,action='store_true')
    parser.add_argument('--dataset',type=str,default='HAPT')
    parser.add_argument('--structure',type=str,default='transformer')
    parser.add_argument('--inputpipeline',type=str,default='s2s')




    return parser