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
    parser.add_argument('--root_path', type=str, default=r"D:\labdata")
    parser.add_argument('--wanted_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--out_name', type=str, default='grad-cam-try', help='result png name')
    parser.add_argument('--regression', action='store_true', help="regressive model or normal")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--EMA', action='store_false', help="regressive model or normal")


    return parser