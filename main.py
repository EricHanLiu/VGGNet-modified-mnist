from data_processing import DataProcessing
from models import CNN


def main():
    dp = DataProcessing('train_max_x', 'test_max_x')
    print("Data Processed")
    dp.parse_csv('train_max_y.csv')

    print("Building neural net...")
    cnn = CNN(dp.train_images, dp.train_y)
    print("Neural net built!")

    cnn.train_cnn()

    print("Training done.")
    cnn.test_cnn()
    print("Testing done")


main()
