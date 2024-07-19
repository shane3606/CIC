import os
import argparse
import torch

from CIC import caption

def parse():
    parser = argparse.ArgumentParser(description='Generating captions in test datasets.')
    parser.add_argument('--question_root', type=str, default='Question_prompt.csv', help='root path to the question')
    parser.add_argument('--image_root', type=str, default='sample/', help='root path to the image')

    args = parser.parse_args()
    return args

def main(args):
    caption(args.image_root)

if __name__=='__main__':
    args = parse()
    main(args)      