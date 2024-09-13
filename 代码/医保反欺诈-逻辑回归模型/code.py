import secretflow as sf
# 初始化 SecretFlow 框架，指定参与方为 'hospital' 和 'bank'，并设置地址为 'local'
sf.init(['hospital', 'bank'], address='local')
# 定义参与方 'hospital' 和 'bank'
hospital, bank = sf.PYU('hospital'), sf.PYU('bank')
# 定义 SPU（Secure Processing Unit）
spu = sf.SPU(sf.utils.testing.cluster_def(['hospital', 'bank']))
import pandas as pd
# 读取数据
df = pd.read_csv('data.csv')
# 为数据集添加索引列 'uid'
df["uid"] = df.index + 1
import numpy as np
# 随机采样 95% 的医院数据和银行数据
df_hospital = df.iloc[:, np.r_[0:8, -1]].sample(frac=0.95)
df_bank = df.iloc[:, 8:].sample(frac=0.95)
import tempfile
import os
# 创建临时目录
temp_dir = tempfile.mkdtemp()
# 定义医院和银行数据的文件路径
hospital_path = os.path.join(temp_dir, 'hospital.csv')
bank_path = os.path.join(temp_dir, 'bank.csv')
# 重置索引并将采样后的数据保存为 CSV 文件
df_hospital.reset_index(drop=True).to_csv(hospital_path, index=False)
df_bank.reset_index(drop=True).to_csv(bank_path, index=False)
from secretflow.data.vertical import read_csv as v_read_csv
# 隐私求交，使用 ECDH_PSI_2PC 协议
vdf = v_read_csv(
    {hospital: hospital_path, bank: bank_path},
    spu=spu,
    keys="uid",
    drop_keys="uid",
    psi_protocl="ECDH_PSI_2PC",
)
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
# 数据处理
# 初始化标签编码器
encoder = LabelEncoder()
# 对类别特征进行编码
vdf['member_name'] = encoder.fit_transform(vdf['member_name'])
vdf['gender'] = encoder.fit_transform(vdf['gender'])
vdf['location'] = encoder.fit_transform(vdf['location'])
vdf['relationship'] = encoder.fit_transform(vdf['relationship'])
vdf['patient_name'] = encoder.fit_transform(vdf['patient_name'])
vdf['patient_dob'] = encoder.fit_transform(vdf['patient_dob'])
vdf['cause'] = encoder.fit_transform(vdf['cause'])
vdf['employer'] = encoder.fit_transform(vdf['employer'])
# 初始化最小最大缩放器
scaler = MinMaxScaler()
# 对数据进行缩放
data = scaler.fit_transform(vdf)
from secretflow.data.split import train_test_split
random_state = 1234
# 将数据集分割为训练集和测试集
train_vdf, test_vdf = train_test_split(data, train_size=0.8, random_state=random_state)
# 定义训练数据和标签
train = train_vdf.drop(columns=['label'])
train_label = train_vdf['label']
# 定义测试数据和标签
test = test_vdf.drop(columns=['label'])
test_label = test_vdf['label']
from secretflow.ml.linear.ss_sgd import SSRegression
# 训练逻辑回归模型
lr_model = SSRegression(spu)
lr_model.fit(
    x=train,
    y=train_label,
    epochs=3,
    learning_rate=0.1,
    batch_size=1024,
    sig_type='t1',
    reg_type='logistic',
    penalty='l2',
    l2_norm=0.5,
)
# 使用模型进行预测
lr_y_hat = lr_model.predict(x=test, batch_size=1024, to_pyu=bank)
from secretflow.stats.biclassification_eval import BiClassificationEval
# 评估模型性能
biclassification_evaluator = BiClassificationEval(
    y_true=test_label, y_score=lr_y_hat, bucket_size=20
)
# 获取评估报告
lr_report = sf.reveal(biclassification_evaluator.get_all_reports())
# 打印评估指标
print(f'positive_samples: {lr_report.summary_report.positive_samples}')
print(f'negative_samples: {lr_report.summary_report.negative_samples}')
print(f'total_samples: {lr_report.summary_report.total_samples}')
print(f'auc: {lr_report.summary_report.auc}')
print(f'ks: {lr_report.summary_report.ks}')
print(f'f1_score: {lr_report.summary_report.f1_score}')
