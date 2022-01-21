import argparse
import numpy as np

from pathlib import Path
from numpy.testing import assert_array_equal
from tests.oo import rgb2grey, grey2rgb
from tests.mm import imread

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('rgb_image', type=Path)
    parser.add_argument('converted_greyscale_image', type=Path)
    return parser.parse_args()

def test_greyscale(greyscale, expected):
    assert greyscale.dtype == np.uint8
    assert_array_equal(greyscale, expected)
    print("\nAssertion is true, both arrays are equal and produced no errors.")

def main():

    args = get_args()
    
    print(f'Path to rgb_image: {args.rgb_image}')
    rgb_image = f'{args.rgb_image}'

    print(f'Path to converted_greyscale_image: {args.converted_greyscale_image}')
    converted_greyscale_image = f'{args.converted_greyscale_image}'

    image = imread(str(rgb_image))

    greyscale = rgb2grey(image)
    expected = imread(str(converted_greyscale_image))
    test_greyscale(greyscale, expected)
    
if __name__ == "__main__":
    main()