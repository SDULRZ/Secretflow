import secretflow as sf
sf.init(['hospital', 'bank'], address='local')
hospital, bank = sf.PYU('hospital'), sf.PYU('bank')
import pandas as pd

#数据分割
df = pd.read_csv('data.csv')
hospital_data = df[["Age","Gender","Blood Type","Medical Condition","Doctor","Hospital","Medication","Room Number"]]
bank_data = df[["Admission Type","Billing Amount","Insurance Provider"]]
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
spu = sf.SPU(sf.utils.testing.cluster_def(['hospital', 'bank']))
import tempfile
import os
temp_dir = tempfile.mkdtemp()
v_hospital_path = os.path.join(temp_dir, 'v_hospital.csv')
v_bank_path = os.path.join(temp_dir, 'v_bank.csv')
v_label_path = os.path.join(temp_dir, 'v_label.csv')
v_hospital, v_bank, v_label = df.iloc[:, :8], df.iloc[:, 8:10], df.iloc[:, 10:11]
v_hospital.to_csv(v_hospital_path, index=False)
v_bank.to_csv(v_bank_path, index=False)
v_label.to_csv(v_label_path,index=False)
from secretflow.data.vertical import read_csv as v_read_csv
data = v_read_csv({hospital: v_hospital_path, bank: v_bank_path})
label = v_read_csv({bank: v_label_path})
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder

#数据处理
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
data['Blood Type'] = encoder.fit_transform(data['Blood Type'])
data['Medical Condition'] = encoder.fit_transform(data['Medical Condition'])
data['Doctor'] = encoder.fit_transform(data['Doctor'])
data['Hospital'] = encoder.fit_transform(data['Hospital'])
data['Medication'] = encoder.fit_transform(data['Medication'])
data['Admission Type'] = encoder.fit_transform(data['Admission Type'])
label = encoder.fit_transform(label)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
from secretflow.data.split import train_test_split
random_state = 1234
train_data, test_data = train_test_split(
    data, train_size=0.8, random_state=random_state
)
train_label, test_label = train_test_split(
    label, train_size=0.8, random_state=random_state
)

#建立多分类训练模型
def create_base_model(input_dim, output_dim, name='base_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="softmax"),
            ]
        )
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model
    return create_model
hidden_size = 64
model_base_hospital = create_base_model(8, hidden_size)
model_base_bank = create_base_model(2, hidden_size)
def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
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
        output = layers.Dense(output_dim, activation='softmax')(fuse_layer)
        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model
    return create_model
model_fuse = create_fuse_model(input_dim=hidden_size, party_nums=2, output_dim=1)
base_model_dict = {hospital: model_base_hospital, bank: model_base_bank}
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
train_batch_size = 128
gaussian_embedding_dp = GaussianEmbeddingDP(
    noise_multiplier=0.5,
    l2_norm_clip=1.0,
    batch_size=train_batch_size,
    num_samples=train_data.values.partition_shape()[bank][0],
    is_secure_generator=False,
)
label_dp = LabelDP(eps=64.0)
dp_strategy_hospital = DPStrategy(label_dp=label_dp)
dp_strategy_bank = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_dict = {hospital: dp_strategy_hospital, bank: dp_strategy_bank}
dp_spent_step_freq = 10
sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=bank,
    model_fuse=model_fuse,
    dp_strategy_dict=dp_strategy_dict,
)

#训练模型
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
global_metric = sl_model.evaluate(test_data, test_label, batch_size=128)
print(sl_model.predict(test_data))
print(global_metric)
