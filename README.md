# 准点有数 · F4 v6
用于本队参赛复现的私有代码仓库。运行环境为 Windows、Python 3.11。原始赛事文件和生成的预测均保存在本地。

## 生成预测

从主办方获取授权数据，把两份原始 CSV 按原名放入本地 `data` 文件夹。在仓库根目录运行以下命令。

```powershell
python -m pip install -r requirements.lock.txt
python tools/v6_expected_loss.py candidate --raw-csv --rebuild-base
```

结果写入本地 `output/v6/submission.csv`。仓库提供生成 v6 预测的代码及固定配置；历史实验记录保存在参赛交付包。
