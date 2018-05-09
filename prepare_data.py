import os
import csv

from glob import glob
from sklearn.model_selection import train_test_split


FLOWERS_IMAGES = './data/jpg/'

FLOWER_NAME = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower', 'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy']


def main():
    data = []

    # get pairs with filename and signal
    for index, image in enumerate(sorted(glob(os.path.join(FLOWERS_IMAGES, '*')))):
        data.append((FLOWER_NAME[(index - 0) // 80], image[2:]))

    # split data for train or test
    train, test = train_test_split(data, random_state=2017)

    # create train dataset
    with open('./data/train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train)

    # create test dataset
    with open('./data/test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test)


if __name__ == '__main__':
    main()
