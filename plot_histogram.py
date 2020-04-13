import argparse
import os

import matplotlib.pyplot as plt

def plot_histogram(filename):
    lines = open(filename, "r").readlines()
    x_data = []
    y_data = []
    cumulative = 0
    for i, l in enumerate(lines):
        if i < 3:
            # Skip the first couple of basic statistics lines
            continue
        val = int(l.split(':')[0])
        count = int(l.split(':')[1])
        for j in range(count):
            x_data.append(cumulative+j)
            y_data.append(val)
        cumulative += count
        """
        x_data.append(cumulative)
        y_data.append(val)
        cumulative += count
        x_data.append(cumulative)
        y_data.append(val)
        """

    plt.scatter(x_data, y_data)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    if args.input_file is not None:
        plot_histogram(args.input_file)
