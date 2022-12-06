{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-12-06T01:03:32.461742Z\",\"iopub.execute_input\":\"2022-12-06T01:03:32.462566Z\",\"iopub.status.idle\":\"2022-12-06T01:03:32.467857Z\",\"shell.execute_reply.started\":\"2022-12-06T01:03:32.462509Z\",\"shell.execute_reply\":\"2022-12-06T01:03:32.466367Z\"}}\nimport pandas as pd\nimport numpy as np\nfrom matplotlib import pyplot as plt\nimport seaborn as sns\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-12-06T01:03:35.218048Z\",\"iopub.execute_input\":\"2022-12-06T01:03:35.218441Z\",\"iopub.status.idle\":\"2022-12-06T01:03:35.284133Z\",\"shell.execute_reply.started\":\"2022-12-06T01:03:35.218411Z\",\"shell.execute_reply\":\"2022-12-06T01:03:35.282830Z\"}}\ndf = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')\ndf_drop = df.drop(labels=['Unnamed: 0','Date','type','region'], axis=1)\nmean = np.mean(df_drop,axis=0)\nmRd = df_drop - mean\nmRd.head()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-12-06T01:04:05.280871Z\",\"iopub.execute_input\":\"2022-12-06T01:04:05.281268Z\",\"iopub.status.idle\":\"2022-12-06T01:04:05.293771Z\",\"shell.execute_reply.started\":\"2022-12-06T01:04:05.281234Z\",\"shell.execute_reply\":\"2022-12-06T01:04:05.292509Z\"}}\ncov_matrix = np.cov(mRd.T)\ncov_matrix\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-12-06T01:04:07.929299Z\",\"iopub.execute_input\":\"2022-12-06T01:04:07.929718Z\",\"iopub.status.idle\":\"2022-12-06T01:04:07.937103Z\",\"shell.execute_reply.started\":\"2022-12-06T01:04:07.929684Z\",\"shell.execute_reply\":\"2022-12-06T01:04:07.935852Z\"}}\nautval,autvet = np.linalg.eig(cov_matrix)\nprint(autval)\nautval = np.array(np.sort(autval[::-1]))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-12-06T01:04:48.606683Z\",\"iopub.execute_input\":\"2022-12-06T01:04:48.607095Z\",\"iopub.status.idle\":\"2022-12-06T01:04:53.919423Z\",\"shell.execute_reply.started\":\"2022-12-06T01:04:48.607060Z\",\"shell.execute_reply\":\"2022-12-06T01:04:53.918111Z\"}}\nsns.lineplot(x=df_drop.columns[0],y=df_drop.columns[1],data=df_drop)\nplt.show()","metadata":{"_uuid":"f41f8be3-e2ef-4c9b-8879-94adf7dd3f3c","_cell_guid":"68097755-8a5a-4d6f-87bf-0da71905b7f9","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}