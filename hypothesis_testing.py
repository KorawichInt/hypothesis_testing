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


def cal_z_stat_p(p_sample, n):
    p0 = 1/3
    z_stat = (p_sample-p0)/math.sqrt((p0*(1-p0))/n)
    return z_stat


def cal_z_stat_train_p(location):
    n = len(train_df)
    z_train_setosa = cal_z_stat_p(location[0], n)
    z_train_versicolor = cal_z_stat_p(location[1], n)
    z_train_virginica = cal_z_stat_p(location[2], n)
    return z_train_setosa, z_train_versicolor, z_train_virginica


def cal_z_stat_validate_p(location):
    n = len(validate_df)
    z_validate_setosa = cal_z_stat_p(location[0], n)
    z_validate_versicolor = cal_z_stat_p(location[1], n)
    z_validate_virginica = cal_z_stat_p(location[2], n)
    return z_validate_setosa, z_validate_versicolor, z_validate_virginica


def cal_z_stat_test_p(location):
    n = len(test_df)
    z_test_setosa = cal_z_stat_p(location[0], n)
    z_test_versicolor = cal_z_stat_p(location[1], n)
    z_test_virginica = cal_z_stat_p(location[2], n)
    return z_test_setosa, z_test_versicolor, z_test_virginica


def make_decision(z_stat, z_critical):
    if abs(z_stat[0]) < z_critical and abs(z_stat[1]) < z_critical and abs(z_stat[2]) < z_critical:
        return f"Fail to reject null hypothesis (H0)"
    else:
        return f"Reject null hypothesis (H0)"


def make_conclusion(decisions):
    if (decisions[0][:4] == "Fail") and (decisions[1][:4] == "Fail") and (decisions[2][:4] == "Fail"):
        return f"Since, Training Set Decision and Validation Set Decision and Testing Set Decision are Fail to reject null hypothesis (H0)\
              \nSo, there is sufficient evidence to conclude that the proportions of 3 types of iris in all 3 dataset are the same at the significance level of 0.05"
    else:
        return f"Since, Training Set Decision and Validation Set Decision and Testing Set Decision are not Fail to reject null hypothesis (H0)\
              \nSo, there is insufficient evidence to conclude that the proportions of 3 types of iris in all 3 dataset are the same at the significance level of 0.05"


def find_mean(dataframes):
    feature_means_list = []
    for df in dataframes:
        df_mean = df[feature_names].mean()
        mean_dict = df_mean.to_dict()
        feature_means_list.append(mean_dict)
    return feature_means_list


def cal_z_stat_m(sample_mean, pop_mean, pop_sd, n):
    z_stat = (sample_mean-pop_mean)/(pop_sd/math.sqrt(n))
    print(f"z stat = {z_stat:.4f}")
    return z_stat


def cal_z_stat_train_m(location):
    n = len(train_df)
    z_train_sepal_length = cal_z_stat_m(
        location[0], feature_mean_list[0], feature_sd_list[0], n)
    z_train_sepal_width = cal_z_stat_m(
        location[1], feature_mean_list[1], feature_sd_list[1], n)
    z_train_petal_length = cal_z_stat_m(
        location[2], feature_mean_list[2], feature_sd_list[2], n)
    z_train_petal_width = cal_z_stat_m(
        location[3], feature_mean_list[3], feature_sd_list[3], n)
    return z_train_sepal_length, z_train_sepal_width, z_train_petal_length, z_train_petal_width


def cal_z_stat_validate_m(location):
    n = len(validate_df)
    z_validate_sepal_length = cal_z_stat_m(
        location[0], feature_mean_list[0], feature_sd_list[0], n)
    z_validate_sepal_width = cal_z_stat_m(
        location[1], feature_mean_list[1], feature_sd_list[1], n)
    z_validate_petal_length = cal_z_stat_m(
        location[2], feature_mean_list[2], feature_sd_list[2], n)
    z_validate_petal_width = cal_z_stat_m(
        location[3], feature_mean_list[3], feature_sd_list[3], n)
    return z_validate_sepal_length, z_validate_sepal_width, z_validate_petal_length, z_validate_petal_width


def cal_z_stat_test_m(location):
    n = len(test_df)
    z_test_sepal_length = cal_z_stat_m(
        location[0], feature_mean_list[0], feature_sd_list[0], n)
    z_test_sepal_width = cal_z_stat_m(
        location[1], feature_mean_list[1], feature_sd_list[1], n)
    z_test_petal_length = cal_z_stat_m(
        location[2], feature_mean_list[2], feature_sd_list[2], n)
    z_test_petal_width = cal_z_stat_m(
        location[3], feature_mean_list[3], feature_sd_list[3], n)
    return z_test_sepal_length, z_test_sepal_width, z_test_petal_length, z_test_petal_width


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
    print(
        f"# Number of record\nTraining Set = {len(train_df)}\nValidation Set = {len(validate_df)}\nTesting Set = {len(test_df)}")

    """Assignment 1: Hypothesis is the proportions of 3 types of iris in all 3 dataset are the same at the significance level of 0.05"""
    # calculate proportion of each group
    print()
    train_proportion = class_proportion(train_df)
    validate_proportion = class_proportion(validate_df)
    test_proportion = class_proportion(test_df)
    proportion_df = pd.DataFrame(
        [train_proportion, validate_proportion, test_proportion]).T
    proportion_df.columns = group_names
    # print(proportion_df)

    # calculate train z statistic
    alpha = 0.05
    z_critical = 1.96
    train_location = [proportion_df.iloc[0, 0],
                      proportion_df.iloc[1, 0],
                      proportion_df.iloc[2, 0]]
    print()
    z_train_setosa, z_train_versicolor, z_train_virginica = cal_z_stat_train_p(
        train_location)
    # calculate validate z statistic
    validate_location = [proportion_df.iloc[0, 1],
                         proportion_df.iloc[1, 1],
                         proportion_df.iloc[2, 1]]
    print()
    z_validate_setosa, z_validate_versicolor, z_validate_virginica = cal_z_stat_validate_p(
        validate_location)
    # calculate test z statistic
    test_location = [proportion_df.iloc[0, 2],
                     proportion_df.iloc[1, 2],
                     proportion_df.iloc[2, 2]]
    print()
    z_test_setosa, z_test_versicolor, z_test_virginica = cal_z_stat_test_p(
        test_location)

    # make decision
    train_decision = make_decision(
        [z_train_setosa, z_train_versicolor, z_train_virginica], z_critical)
    validate_decision = make_decision(
        [z_validate_setosa, z_validate_versicolor, z_validate_virginica], z_critical)
    test_decision = make_decision(
        [z_test_setosa, z_test_versicolor, z_test_virginica], z_critical)

    # make conclusion
    # print("\n# Assignment 1 Conclusion")
    asm1_conclusion = make_conclusion(
        [train_decision, validate_decision, test_decision])
    # print(asm1_conclusion)
    #
    # ***********************************************************
    #
    """Assignment 2: Hypothesis is the means of 4 features of iris in all 3 dataset are the same at the significance level of 0.05"""
    feature_mean = {'SepalLength': 5.84, 'SepalWidth': 3.05,
                    'PetalLength': 3.76, 'PetalWidth': 1.20}
    feature_mean_list = [val for val in feature_mean.values()]
    feature_sd = {'SepalLength': 0.83, 'SepalWidth': 0.43,
                  'PetalLength': 1.76, 'PetalWidth': 0.76}
    feature_sd_list = [val for val in feature_sd.values()]
    # calculate mean of each group
    print()
    feature_names = column_names[:-1]
    train_mean, validate_mean, test_mean = find_mean(
        [train_df, validate_df, test_df])
    mean_df = pd.DataFrame(
        [train_mean, validate_mean, test_mean]).T
    mean_df.columns = group_names
    print(mean_df)

    # calculate train z statistic
    train_location_2 = [mean_df.iloc[0, 0],
                        mean_df.iloc[1, 0],
                        mean_df.iloc[2, 0],
                        mean_df.iloc[3, 0]]
    print()
    z_train_sepal_length, z_train_sepal_width, z_train_petal_length, z_train_petal_width = cal_z_stat_train_m(
        train_location_2)
    # calculate validate z statistic
    validate_location_2 = [mean_df.iloc[0, 1],
                           mean_df.iloc[1, 1],
                           mean_df.iloc[2, 1],
                           mean_df.iloc[3, 1]]
    print()
    z_validate_sepal_length, z_validate_sepal_width, z_validate_petal_length, z_validate_petal_width = cal_z_stat_validate_m(
        validate_location_2)
    # calculate test z statistic
    test_location_2 = [mean_df.iloc[0, 2],
                       mean_df.iloc[1, 2],
                       mean_df.iloc[2, 2],
                       mean_df.iloc[3, 2]]
    print()
    z_test_sepal_length, z_test_sepal_width, z_test_petal_length, z_test_petal_width = cal_z_stat_test_m(
        test_location_2)
