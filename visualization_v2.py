import pandas as pd

import pickle
with open('./result/result_trainsE_v2.pickle', 'rb') as f:
    result = pickle.load(f)

# 確認客戶ID及消費類別的mapping表
ent2idx_table = result['ent2idx_table']

# 確認relation的mapping表
rel2idx_table = result['rel2idx_table']


# 所有的向量
# entity_vector
entity_vector = result['entity_vector']
# rel_vector
rel_vector = result['rel_vector']

# 取消費類別的向量
# Consumption_types = ['tag_2', 'tag_10', 'tag_15', 'tag_36', 'tag_37']
# Consumption_types = ['10161917', '10260106', '10180464','10272691']
Consumption_types = ['tag_2', 'tag_10', 'tag_15', 'tag_36', 'tag_37'] + ['10161917', '10260106', '10180464','10272691']
Consumption_types_idx  = [ent2idx_table[i] for i in Consumption_types]

Consumption_vector = entity_vector[Consumption_types_idx,:]


# 以PCA降維度，並視覺化
import pandas as pd 
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


input_data = pd.DataFrame(Consumption_vector)

# PCA降維
PCA_model = PCA( n_components = 2)
PCA_model_s = PCA_model.fit(input_data).explained_variance_ratio_
output = PCA_model.fit_transform(input_data)


output_data = pd.DataFrame(output)
output_data.columns = ['PCA_'+str(i) for i in output_data.columns.tolist()]
output_data['label'] = Consumption_types

fig = px.scatter(output_data, x="PCA_0", y="PCA_1", text="label")
fig.update_traces(textposition='top center')