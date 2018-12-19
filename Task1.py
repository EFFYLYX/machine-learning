import pandas as pd
import matplotlib.pyplot as plt

csv_data = pd.read_csv('irismissing.csv')

[row_size, column_size] = csv_data.shape

row_nan = []
colum_nan = []

# traverse the data frame
for i in range(row_size):

    for j in csv_data.keys():
        # if the value is NA (missing values), put the row number into a list
        if pd.isna(csv_data[j][i]):
            row_nan.append(i)
            # if the the column index does not appear in the list before, add it
            if j not in colum_nan:
                colum_nan.append(j)

print("The column index of instances that have missing values: ", colum_nan)
print("The row numbers of instances that have missing values: ", row_nan)

data_drop = csv_data.dropna()

# find the mean, median, mode for each column that may have missing values
for key in colum_nan:
    print("---------------------")
    print("Handle column: ", key)

    mean = csv_data[key].mean()
    print('mean ', mean)

    data_mean = csv_data.fillna(mean)

    median = csv_data[key].median()
    print('median ', median)
    data_median = csv_data.fillna(median)

    mode = csv_data[key].mode()
    print('mode ', mode.iloc[0])
    data_mode = csv_data.fillna(mode.iloc[0])



    fig = plt.figure(figsize=(8, 6))

    # put the result of each method into a dict
    data_dict = {'mean': data_mean[key], 'median': data_median[key], 'mode': data_mode[key], 'drop': data_drop[key]}

    data = [data_mean[key], data_median[key], data_mode[key], data_drop[key]]
    print("--------")

    print(pd.DataFrame.from_dict(data_dict).describe())
    label = 'measurement ' + str(key)

    # visualize the results of each method
    plt.boxplot(data,
                notch=False,
                sym='rs',
                vert=True,
                showmeans=True)

    plt.xticks([y + 1 for y in range(len(data))], ['mean', 'median', 'mode', 'drop missing values'])
    plt.xlabel(label)
    t = plt.title('Box plot')

    plt.show()
