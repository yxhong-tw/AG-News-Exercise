import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    train_df = pd.read_csv(filepath_or_buffer='data/origin/train.csv')
    test_df = pd.read_csv(filepath_or_buffer='data/origin/test.csv')

    analyze_dataset(df=train_df, df_name='train')
    analyze_dataset(df=test_df, df_name='test')

    # train_validation_split(df=train_df)


def analyze_dataset(df, df_name):
    def simple_check(char_num, over_512_data_num, max_num, min_num):
        if char_num > 512:
            over_512_data_num += 1

        if char_num > max_num:
            max_num = char_num

        if char_num < min_num:
            min_num = char_num

        return over_512_data_num, max_num, min_num

    title_over_512_data_num = 0
    title_max_num = float('-inf')
    title_min_num = float('inf')
    desc_over_512_data_num = 0
    desc_max_num = float('-inf')
    desc_min_num = float('inf')

    for i in range(df.shape[0]):
        title = df['Title'].iloc[i]
        desc = df['Description'].iloc[i]

        title_char_num = len(title.split(' '))
        desc_char_num = len(desc.split(' '))

        title_over_512_data_num, title_max_num, title_min_num = simple_check(
            char_num=title_char_num
            , over_512_data_num=title_over_512_data_num
            , max_num=title_max_num
            , min_num=title_min_num
        )

        desc_over_512_data_num, desc_max_num, desc_min_num = simple_check(
            char_num=desc_char_num
            , over_512_data_num=desc_over_512_data_num
            , max_num=desc_max_num
            , min_num=desc_min_num
        )

    print(f'Shape of {df_name}_df: {df.shape}')
    print(f'title_over_512_data_num of {df_name}_df: {title_over_512_data_num}')
    print(f'title_max_num of {df_name}_df: {title_max_num}')
    print(f'title_min_num of {df_name}_df: {title_min_num}')
    print(f'desc_over_512_data_num of {df_name}_df: {desc_over_512_data_num}')
    print(f'desc_max_num of {df_name}_df: {desc_max_num}')
    print(f'desc_min_num of {df_name}_df: {desc_min_num}')
    print()


def train_validation_split(df):
    train_df, validation_df = train_test_split(df, test_size=0.1)

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
