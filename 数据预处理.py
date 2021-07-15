import os
import pandas as pd
import torch

# 创建csv文件
data_file = os.path.join('..','data','house_tiny.csv')
os.makedirs(os.path.join('..','data'),exist_ok=True)

with open(data_file,'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取csv
data = pd.read_csv(data_file)
print(data)

# 转换成张量模式
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
X, y = torch.tensor(inputs.values), torch.tensor(outputs.value)
X, y