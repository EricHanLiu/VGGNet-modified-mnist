from data_processing import DataProcessing


def main():
    dp = DataProcessing('train_max_x', 'test_max_x')
    dp.parse_csv('train_max_y.csv')
    print(dp.train_images)
    print(dp.test_images)
    print(dp.train_y)


main()
