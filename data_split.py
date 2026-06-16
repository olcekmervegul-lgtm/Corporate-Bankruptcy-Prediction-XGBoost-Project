import pandas as pd


file_name = "data.xls"
df = pd.read_excel(file_name)

print("Data successfully loaded!")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")


demo_df = df.tail(300)


train_test_df = df.head(len(df) - 300)


demo_df.to_excel("demo_data_300.xlsx", index=False)
train_test_df.to_excel("train_test_data.xlsx", index=False)

print("Data successfully partitioned and saved as 'demo_data_300.xlsx' and 'train_test_data.xlsx'!")
