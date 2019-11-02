import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, train_images_filename, test_images_filename):
        self.train_images = pd.read_pickle(train_images_filename)
        self.test_images = pd.read_pickle(test_images_filename)
        self.train_y = None

    def parse_csv(self, filename):
        with open(filename, newline='') as file:
            dataframe = pd.read_csv(file)
            self.train_y = np.array(dataframe)
