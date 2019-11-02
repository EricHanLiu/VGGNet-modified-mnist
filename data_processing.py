import pandas as pd


class DataProcessing:
    def __init__(self, train_images_filename, test_images_filename):
        self.train_images = pd.read_pickle(train_images_filename)
        self.test_images = pd.read_pickle(test_images_filename)
