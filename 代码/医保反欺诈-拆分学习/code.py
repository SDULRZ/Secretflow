import secretflow as sf
# 初始化 SecretFlow 框架，指定参与方为 'hospital' 和 'bank'，并设置地址为 'local'
sf.init(['hospital', 'bank'], address='local')
# 定义参与方 'hospital' 和 'bank'
hospital, bank = sf.PYU('hospital'), sf.PYU('bank')

# 数据分割
import pandas as pd
# 读取数据
df = pd.read_csv('data.csv')
# 根据数据特征分割数据为医院数据和银行数据
hospital_data = df[["member_name","gender","location","relationship","patient_name","patient_id","patient_dob","cause","label"]]
bank_data = df[["employer","Fee Charged","membership_period","number_of_claims","number_of_dependants"]]
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
# 定义 SPU（Secure Processing Unit）
spu = sf.SPU(sf.utils.testing.cluster_def(['hospital', 'bank']))
import tempfile
import os
# 创建临时目录
temp_dir = tempfile.mkdtemp()
# 定义文件路径
v_hospital_path = os.path.join(temp_dir, 'v_hospital.csv')
v_bank_path = os.path.join(temp_dir, 'v_bank.csv')
v_label_path = os.path.join(temp_dir, 'v_label.csv')
# 将数据分割为医院数据、银行数据和标签
v_hospital, v_bank, v_label = df.iloc[:, :8], df.iloc[:, 8:13], df.iloc[:, 13:14]
# 将分割后的数据保存为 CSV 文件
v_hospital.to_csv(v_hospital_path, index=False)
v_bank.to_csv(v_bank_path, index=False)
v_label.to_csv(v_label_path,index=False)
from secretflow.data.vertical import read_csv as v_read_csv
# 读取 CSV 文件作为数据集
data = v_read_csv({hospital: v_hospital_path, bank: v_bank_path})
label = v_read_csv({hospital: v_label_path})
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder

# 数据处理
# 初始化标签编码器
encoder = LabelEncoder()
# 对类别特征进行编码
data['member_name'] = encoder.fit_transform(data['member_name'])
data['gender'] = encoder.fit_transform(data['gender'])
data['location'] = encoder.fit_transform(data['location'])
data['relationship'] = encoder.fit_transform(data['relationship'])
data['patient_name'] = encoder.fit_transform(data['patient_name'])
data['patient_dob'] = encoder.fit_transform(data['patient_dob'])
data['cause'] = encoder.fit_transform(data['cause'])
data['employer'] = encoder.fit_transform(data['employer'])
# 初始化最小最大缩放器
scaler = MinMaxScaler()
# 对数据进行缩放
data = scaler.fit_transform(data)
from secretflow.data.split import train_test_split
random_state = 1234
# 将数据集分割为训练集和测试集
train_data, test_data = train_test_split(
    data, train_size=0.8, random_state=random_state
)
train_label, test_label = train_test_split(
    label, train_size=0.8, random_state=random_state
)

# 创建拆分学习训练模型
def create_base_model(input_dim, output_dim, name='base_model'):
    # 定义基础模型
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model
    return create_model
hidden_size = 64
# 创建医院的基础模型
model_base_hospital = create_base_model(8, hidden_size)
# 创建银行的基础模型
model_base_bank = create_base_model(5, hidden_size)
def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    # 定义融合模型
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )
        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='sigmoid')(fuse_layer)
        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model
    return create_model
# 创建融合模型
model_fuse = create_fuse_model(input_dim=hidden_size, party_nums=2, output_dim=1)
base_model_dict = {hospital: model_base_hospital, bank: model_base_bank}
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
train_batch_size = 128
# 初始化高斯嵌入差分隐私
gaussian_embedding_dp = GaussianEmbeddingDP(
    noise_multiplier=0.5,
    l2_norm_clip=1.0,
    batch_size=train_batch_size,
    num_samples=train_data.values.partition_shape()[hospital][0],
    is_secure_generator=False,
)
# 初始化标签差分隐私
label_dp = LabelDP(eps=64.0)
# 为医院设置差分隐私策略
dp_strategy_hospital = DPStrategy(label_dp=label_dp)
# 为银行设置差分隐私策略
dp_strategy_bank = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_dict = {hospital: dp_strategy_hospital, bank: dp_strategy_bank}
dp_spent_step_freq = 10
# 创建安全学习模型
sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=hospital,
    model_fuse=model_fuse,
    dp_strategy_dict=dp_strategy_dict,
)

# 训练模型
history = sl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=10,
    batch_size=train_batch_size,
    shuffle=True,
    verbose=1,
    validation_freq=1,
    dp_spent_step_freq=dp_spent_step_freq,
)
# 评估模型
global_metric = sl_model.evaluate(test_data, test_label, batch_size=128)
print(global_metric)