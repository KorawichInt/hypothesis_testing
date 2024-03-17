import pandas as pd
import os
import math


def sampling_data(read_path, write_path):
    data = pd.read_csv(read_path, delimiter=',', header=None)
    data.to_csv(write_path[0], index=False, header=None)
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    group_sizes = [80, 38, 32]
    train = data_shuffled[:group_sizes[0]]
    validate = data_shuffled[group_sizes[0]:group_sizes[0]+group_sizes[1]]
    test = data_shuffled[group_sizes[0]+group_sizes[1]:]
    train.to_csv(write_path[1][0], index=False, header=False)
    validate.to_csv(write_path[1][1], index=False, header=False)
    test.to_csv(write_path[1][2], index=False, header=False)


def create_dataframe(path):
    train_df = pd.read_csv(path[0], names=column_names)
    validate_df = pd.read_csv(path[1], names=column_names)
    test_df = pd.read_csv(path[2], names=column_names)
    return train_df, validate_df, test_df


def class_proportion(dataframe):
    class_counts = dataframe['Class'].value_counts()
    count_dict = class_counts.to_dict()
    count_dict = {
        class_name: count_dict[class_name]/len(dataframe) for class_name in classes}
    return count_dict


def cal_z_stat(p_sample, n):
    p0 = 1/3
    z_stat = (p_sample-p0)/math.sqrt((p0*(1-p0))/n)
    print(f"z stat = {z_stat:.4f}")
    return z_stat


def cal_z_stat_train(location):
    n = len(train_df)
    z_train_setosa = cal_z_stat(location[0], n)
    z_train_versicolor = cal_z_stat(location[1], n)
    z_train_virginica = cal_z_stat(location[2], n)
    return z_train_setosa, z_train_versicolor, z_train_virginica


def cal_z_stat_validate(location):
    n = len(validate_df)
    z_validate_setosa = cal_z_stat(location[0], n)
    z_validate_versicolor = cal_z_stat(location[1], n)
    z_validate_virginica = cal_z_stat(location[2], n)
    return z_validate_setosa, z_validate_versicolor, z_validate_virginica


def cal_z_stat_test(location):
    n = len(test_df)
    z_test_setosa = cal_z_stat(location[0], n)
    z_test_versicolor = cal_z_stat(location[1], n)
    z_test_virginica = cal_z_stat(location[2], n)
    return z_test_setosa, z_test_versicolor, z_test_virginica


if __name__ == "__main__":
    data_path = r'assets\iris.data'
    data_path_csv = r'csv\iris.csv'
    sampling_path = [r'csv\train.csv', r'csv\validate.csv', r'csv\test.csv']
    column_names = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Class']
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    group_names = ['Training_set', 'Validation_set', 'Testing_set']

    # read, sampling, write train validate test csv
    if not (os.path.exists(sampling_path[0])):
        sampling_data(data_path, [data_path_csv, sampling_path])

    # create train, validate, test dataframe
    train_df, validate_df, test_df = create_dataframe(sampling_path)
    # print(
    #     f"# Number of record\nTraining Set = {len(train_df)}\nValidation Set = {len(validate_df)}\nTesting Set = {len(test_df)}")

    # calculate proportion of each group
    print()
    train_proportion = class_proportion(train_df)
    validate_proportion = class_proportion(validate_df)
    test_proportion = class_proportion(test_df)
    proportion_df = pd.DataFrame(
        [train_proportion, validate_proportion, test_proportion]).T
    proportion_df.columns = group_names
    print(proportion_df)

    # calculate train z statistic
    alpha = 0.05
    z_critical = 1.96
    print("\n# Train z")
    train_location = [proportion_df.iloc[0, 0],
                      proportion_df.iloc[1, 0],
                      proportion_df.iloc[2, 0]]
    z_train_setosa, z_train_versicolor, z_train_virginica = cal_z_stat_train(
        train_location)
    # calculate validate z statistic
    print("\n# Validate z")
    validate_location = [proportion_df.iloc[0, 1],
                         proportion_df.iloc[1, 1],
                         proportion_df.iloc[2, 1]]
    z_validate_setosa, z_validate_versicolor, z_validate_virginica = cal_z_stat_validate(
        validate_location)
    # calculate test z statistic
    print("\n# Test z")
    test_location = [proportion_df.iloc[0, 2],
                     proportion_df.iloc[1, 2],
                     proportion_df.iloc[2, 2]]
    z_test_setosa, z_test_versicolor, z_test_virginica = cal_z_stat_test(
        test_location)

    # decision and conclusion
