import pandas as pd
from utils import utils

for model_key in utils.MODELS:
    if not utils.MODELS[model_key]["excluded"]:
        data = pd.read_pickle(utils.DATA_DIR/"Experiment2"/f"{model_key}_heatmap_data.pkl")

        df_c = pd.DataFrame(data["heat_c"])
        df_c.index = pd.Index([f"Layer {str(i)}" for i in df_c.index])
        df_c.columns = pd.Index([f"Head {str(c)}" for c in df_c.columns])
        df_c = df_c.round(4)
        df_c.to_csv(utils.DATA_DIR/"Experiment2"/f"{model_key}_heat_c.csv")

        df_p = pd.DataFrame(data["heat_c"])
        df_p.index = pd.Index([f"Layer {str(i)}" for i in df_p.index])
        df_p.columns = pd.Index([f"Head {str(c)}" for c in df_p.columns])
        df_p = df_p.round(4)
        df_p.to_csv(utils.DATA_DIR/"Experiment2"/f"{model_key}_heat_p.csv")