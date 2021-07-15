import os
import pandas as pd
import torch

data_file = os.path.join('..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换成张量模式

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)
