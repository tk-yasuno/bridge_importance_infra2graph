import torch
import numpy as np

# Load HeteroData
data = torch.load('output/bridge_importance/heterogeneous_graph_heterodata.pt')
X = data['bridge'].x.numpy()

print('Feature ranges (20 features):')
feature_names = ['健全度Ⅰ', '健全度Ⅱ', '健全度Ⅲ', '健全度Ⅳ', '橋齢', '橋長', '幅員', 
                'log_dist_river', 'log_dist_coast', 'num_buildings', 'num_public_facilities', 
                'num_hospitals', 'num_schools', '離島架橋', '長大橋', '特殊橋', '重要物流道路', 
                '緊急輸送道路', '跨線橋', '跨道橋']

for i in range(X.shape[1]):
    print(f'{i:2d}. {feature_names[i]:20s}: [{X[:,i].min():8.3f}, {X[:,i].max():8.3f}] mean={X[:,i].mean():8.3f} std={X[:,i].std():8.3f}')

# Check for features with zero variance
zero_var = []
for i in range(X.shape[1]):
    if X[:,i].std() < 1e-6:
        zero_var.append((i, feature_names[i]))

if zero_var:
    print(f'\nFeatures with zero variance: {len(zero_var)}')
    for idx, name in zero_var:
        print(f'  {idx}. {name}')
else:
    print('\nAll features have non-zero variance')
