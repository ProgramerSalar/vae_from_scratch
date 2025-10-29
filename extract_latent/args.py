import argparse


def get_args():

    parser = argparse.ArgumentParser('Pytorch Multi-process Training script', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_dtype', default='bf16', type=str, help='The model dtype: bf16 or df16')
    parser.add_argument('--anno_file', type=str, default='', help="the annoation file")
    parser.add_argument('--width', type=int, default=640, help='The video width')
    parser.add_argument('--height', type=int, default=384, help='The video height')
    parser.add_argument('--num_frames', type=int, default=121, help='The frame number to encode')
    parser.add_argument('--save_memory', action='store_true', help='Open the VAE tilling')
    
    return parser.parse_args()



