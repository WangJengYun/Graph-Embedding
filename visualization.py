import mlflow
from mlflow.tracking import MlflowClient
#client = MlflowClient()
mlflow.set_tracking_uri('http://61.70.194.215:5000/')

# 取部署的模型
# kg_model = mlflow.pytorch.load_model(model_uri=f"models:/KG_Model/Production")
import torch
kg_model = torch.load('./model_files/TransE/model.pth',map_location={'cuda:0':"cpu"})
kg_model = mlflow.pytorch.load_model(model_uri='./model_files/TransE/model.pth')
kg_model = kg_model.cpu()

# 確認客戶ID及消費類別的mapping表
ent2idx_table = kg_model.ent2idx

# 確認relation的mapping表
rel2idx_table = kg_model.rel2idx


# 所有的向量
# entity_vector
entity_vector = kg_model.ent_emb.weight.data.numpy()
# rel_vector
rel_vector = kg_model.rel_emb.weight.data.numpy()

# 取消費類別的向量
Consumption_types = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '2']
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
output = PCA_model.fit_transform(input_data)


output_data = pd.DataFrame(output)
output_data.columns = ['PCA_'+str(i) for i in output_data.columns.tolist()]
output_data['label'] = Consumption_types

fig = px.scatter(output_data, x="PCA_0", y="PCA_1", text="label")
fig.update_traces(textposition='top center')


