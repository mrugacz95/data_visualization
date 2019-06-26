import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

from svd import sklearn_svd_implementation, numpy_svd_implementation, custom_svd_implementation


def compress_image(img, k, svd_implementation=custom_svd_implementation):
    def compress_channel(data, k, implementation):
        u, t, vt = implementation(data, k)
        return u[:, :k] @ t[:k, :k] @ vt[:k, :]

    width, height, channels = img.shape
    data = img.astype(float)
    compressed = np.empty((width, height, channels))
    for channel in range(channels):
        c = data[:, :, channel]
        compressed[:, :, channel] = compress_channel(c, k, svd_implementation)
    return compressed


def main():
    parser = ArgumentParser()
    parser.add_argument('-f', dest="input_file", help="Path to file to compress", required=True)
    parser.add_argument('-out', dest="output_file", default='output.png', help="Path to output file")
    parser.add_argument('-svd', dest="svd_impl", choices=('sklearn', 'custom', 'numpy'), default='custom', help="s")
    parser.add_argument('-k', dest="k", type=int, default='0', help="Number of singular values used. Default all.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f'File {args.input_file} not found', file=sys.stderr)
        return
    img = cv2.imread(args.input_file)

    impl = {
        'sklearn': sklearn_svd_implementation,
        'numpy': numpy_svd_implementation,
        'custom': custom_svd_implementation

    }.get(args.svd_impl, custom_svd_implementation)

    k = args.k
    if k <= 0:
        k = np.iinfo(np.int32).max

    img = compress_image(img, k, impl)

    cv2.imwrite(args.output_file, img)
    print(f'File saved to {args.output_file}')


if __name__ == '__main__':
    main()
