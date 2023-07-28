import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    train_test_split()

    # print(train_df.shape)
    # input()

    # test_df = pd.read_csv(filepath_or_buffer='data/test.csv')

    # train_title_max_num = float('-inf')
    # train_title_min_num = float('inf')
    # train_description_max_num = float('-inf')
    # train_description_min_num = float('inf')
    # train_title_over512_data_num = 0
    # train_description_over512_data_num = 0

    # for i in range(train_df.shape[0]):
    #     title = train_df['Title'].iloc[i]
    #     description = train_df['Description'].iloc[i]

    #     title_char_num = len(title.split(' '))
    #     description_char_num = len(description.split(' '))

    #     if title_char_num > 512:
    #         train_title_over512_data_num += 1

    #     if title_char_num > train_title_max_num:
    #         train_title_max_num = title_char_num

    #     if title_char_num < train_title_min_num:
    #         train_title_min_num = title_char_num

    #     if description_char_num > 512:
    #         train_description_over512_data_num += 1

    #     if description_char_num > train_description_max_num:
    #         train_description_max_num = description_char_num

    #     if description_char_num < train_description_min_num:
    #         train_description_min_num = description_char_num

    # print(train_title_max_num, train_title_min_num, train_title_over512_data_num)
    # print(train_description_max_num, train_description_min_num, train_description_over512_data_num)
    # print(train_df.shape[0])



    # print(test_df.shape[0])

    # input()

    # test_description_max_num = float('-inf')
    # test_description_min_num = float('inf')


def train_validation_split():
    train_df = pd.read_csv(
        filepath_or_buffer='data/origin/train.csv'
        , encoding='UTF-8')

    train_df, validation_df = train_test_split(train_df, test_size=0.1)

    train_df.to_csv(
        path_or_buf='data/train.csv'
        , encoding='UTF-8'
        , index=False)
    validation_df.to_csv(
        path_or_buf='data/validation.csv'
        , encoding='UTF-8'
        , index=False)


if __name__ == '__main__':
    main()
