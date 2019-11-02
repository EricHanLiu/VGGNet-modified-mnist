from data_processing import DataProcessing


def main():
    dp = DataProcessing('train_max_x', 'test_max_x')
    print(dp.train_images)
    print(dp.test_images)


main()
