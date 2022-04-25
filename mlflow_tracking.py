import mlflow

mlflow.set_tracking_uri('http://192.168.0.21:5213')
mlflow.set_experiment('tesst')
mlflow.start_run()

for i in range(100):
   mlflow.log_metrics(metrics={'A':i+10}, step=i)

features = "rooms, zipcode, median_price, school_rating, transport"
with open("features.txt", 'w') as f:
    f.write(features)

mlflow.log_artifact("features.txt")
mlflow.log_artifact("./model_files/model_test.pt")

import mlflow.pytorch
import torch 
# Log PyTorch model
model = torch.load('./model_files/model_2022.pt')
mlflow.pytorch.log_model(model, "model")

env = mlflow.pytorch.get_default_conda_env()
print("conda env: {}".format(env))

# run_id = mlflow.active_run().info.run_id

mlflow.end_run()


import mlflow.pyfunc
mlflow.set_tracking_uri('http://192.168.0.21:5000')

model_name = "modelA"
stage = 'Staging'

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

dir(model)

import mlflow
modelA = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}")

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.delete_registered_model(name="modelA")


result = mlflow.register_model(
    "runs:/5850531634344803bd094d7a1d2340c1/model",
    "sk-learn-random-forest-reg"
)