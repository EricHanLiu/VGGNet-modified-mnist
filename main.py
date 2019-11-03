from data_processing import DataProcessing
from models import ConvNet as CNN
import models


def main():
    dp = DataProcessing('train_max_x', 'test_max_x')
    print("Data Processed")
    dp.parse_csv('train_max_y.csv')

    print("dp.train_images")
    print(dp.train_images)
    print("dp.testimages")
    print(dp.test_images)
    print("dp.trainy")
    print(dp.train_y)


    print("Building neural net...")
    cnn = CNN()
    print("Neural net built!")
    cnn.trainCNN(cnn, dp.train_images, dp.train_y)
    print("Training done.")
    cnn.testCNN(cnn, dp.train_images, dp.train_y)



main()
