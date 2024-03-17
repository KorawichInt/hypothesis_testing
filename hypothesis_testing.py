import pandas as pd


def sampling_data(read_path, write_path):
    data = pd.read_csv(read_path, delimiter=',')
    data.to_csv(write_path[0], index=False)
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    group_sizes = [80, 38, 32]
    train = data_shuffled[:group_sizes[0]]
    validate = data_shuffled[group_sizes[0]:group_sizes[0]+group_sizes[1]]
    test = data_shuffled[group_sizes[0]+group_sizes[1]:]
    train.to_csv(write_path[1][0], index=False, header=False)
    validate.to_csv(write_path[1][1], index=False, header=False)
    test.to_csv(write_path[1][2], index=False, header=False)


if __name__ == "__main__":
    data_path = r'assets\iris.data'
    data_path_csv = r'csv\iris.csv'
    sampling_path = [r'csv\train.csv', r'csv\validate.csv', r'csv\test.csv']

    sampling_data(data_path, [data_path_csv, sampling_path])
