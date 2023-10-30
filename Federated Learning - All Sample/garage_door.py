# %%
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import resample
import os
import flwr as fl
from sklearn.model_selection import train_test_split
import tensorflow as tf

# %%
df_fridge = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Fridge.csv")
df_gps_tracker = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_GPS_Tracker.csv")
df_garage_door = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Garage_Door.csv", low_memory=False)
df_modbus = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Modbus.csv")
df_motion_light = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Motion_Light.csv")
df_thermostat = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Thermostat.csv")
df_weather = pd.read_csv("/Users/oluwafisayomi/Downloads/Analysis/data/IoT_Weather.csv")

# %%
all_df = [df_fridge, df_gps_tracker, df_garage_door, df_modbus, df_motion_light, df_thermostat, df_weather]
# %%
# # displaying huge class imbalance in the types of attacks in the dataset
# for i in all_df:
#   sns.catplot(data=i, x='type', kind="count", orient='v')

# plt.show
# %% [markdown]
# # Preprocessing
# %%
for device in all_df:
  # Combine 'date' and 'time' and convert to datetime object
  device['datetime'] = pd.to_datetime(device['date'] + ' ' + device['time'])

  # Drop the original 'date' and 'time' columns
  device.drop(['date', 'time'], axis=1, inplace=True)

# %%
# dropping the label since we want to do a multi-class classification
for device in all_df:
    device.drop(['label'], axis=1, inplace=True)

# %%
# Clean the 'temp_condition' column
df_fridge['temp_condition'] = df_fridge['temp_condition'].str.strip()
df_fridge['temp_condition'] = df_fridge['temp_condition'].str.replace(' +', ' ')

# %%
# dropping null values in respective dataframes
df_garage_door.dropna(inplace=True)
df_thermostat.dropna(inplace=True)

# %%
# Convert all values to strings
df_garage_door['sphone_signal'] = df_garage_door['sphone_signal'].astype(str)

# Strip leading and trailing spaces
df_garage_door['sphone_signal'] = df_garage_door['sphone_signal'].str.strip()

# Replace 'false' and 'true' with 0 and 1 respectively
df_garage_door['sphone_signal'] = df_garage_door['sphone_signal'].replace({'false': 0, 'true': 1, '0.0': 0, '1.0': 1, '0': 0, '1': 1})

# %%
# Strip leading and trailing spaces
df_motion_light['light_status'] = df_motion_light['light_status'].str.strip()

# %%
# label encoding each class in all dataframes

# Global Label Encoder
all_labels = np.concatenate([df['type'] for df in all_df])  # concatenate all labels
global_encoder = LabelEncoder().fit(all_labels)  # fit the encoder

# We do this so that all labels across each dataframe has the encoding
for df in all_df:
    df['type'] = global_encoder.transform(df['type'])  # transform the labels

# %%
df_garage_door = pd.get_dummies(df_garage_door, columns=['door_state'])
df_fridge = pd.get_dummies(df_fridge, columns=['temp_condition'])
df_motion_light = pd.get_dummies(df_motion_light, columns = ['light_status'])

for n, df in enumerate(all_df):
    if 'door_state' in df.columns:
        all_df[n] = df_garage_door
    elif 'temp_condition' in df.columns:
        all_df[n] = df_fridge
    elif 'light_status' in df.columns:
      all_df[n] = df_motion_light

# %%
def process_datetime(df):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(int)
    df.drop(columns=['datetime'], axis=1, inplace=True)
    return df

df_final = [process_datetime(df.copy()) for df in [df_fridge, df_gps_tracker, df_garage_door, df_modbus, df_motion_light, df_thermostat, df_weather]]

# %%
for df in df_final:
    df = df.astype(float)
# %%
# Define a function to standardize numerical columns
def standardize_data(df, numerical_cols):
    scaler = StandardScaler()
    for col in numerical_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    return df

numerical_cols = ['fridge_temperature', 'latitude', 'longitude', 'FC1_Read_Input_Register',
                  'FC2_Read_Discrete_Value', 'FC3_Read_Holding_Register', 'FC4_Read_Coil',
                  'current_temperature', 'temperature', 'pressure', 'humidity', 'hour']

# Create a new list to hold the standardized DataFrames
df_final_standardized = []

# Iterate over each DataFrame in df_federated
for df in df_final:
    df_standardized = standardize_data(df, numerical_cols)
    df_final_standardized.append(df_standardized)

# %%
from sklearn.decomposition import PCA

# Assuming df_final_standardized is loaded and contains the dataframes in the order you mentioned.
dataframes = {
    "df_fridge": df_final_standardized[0],
    "df_gps_tracker": df_final_standardized[1],
    "df_garage_door": df_final_standardized[2],
    "df_modbus": df_final_standardized[3],
    "df_motion_light": df_final_standardized[4],
    "df_thermostat": df_final_standardized[5],
    "df_weather": df_final_standardized[6]
}

reduced_data = {}

for name, df in dataframes.items():
    pca = PCA(n_components=5)
    X_reduced = pca.fit_transform(df.drop(columns=["type"]))
    reduced_df = pd.DataFrame(X_reduced)
    reduced_df["type"] = df["type"].values
    reduced_data[name] = reduced_df

all_unique_classes = set()
for key, df in reduced_data.items():
    all_unique_classes = all_unique_classes.union(set(df['type'].unique()))


one_hot = OneHotEncoder(categories=[list(all_unique_classes)], sparse_output=False)

for key, df in reduced_data.items():
    # One-hot encode
    encoded_data = one_hot.fit_transform(df[['type']])
    
    # Create a dataframe from the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot.get_feature_names_out(['type']))
    
    # Concatenate original data (minus 'type' column) with the encoded dataframe
    reduced_data[key] = pd.concat([df.drop('type', axis=1), encoded_df], axis=1)

# %% [markdown]
# ## All Sample CDL (Feedforward Neural Network)

# %%
# Separate features and target for "df_garage_door"
X_all = reduced_data["df_garage_door"].drop(list(one_hot.get_feature_names_out(['type'])), axis=1)
y_all = reduced_data["df_garage_door"][list(one_hot.get_feature_names_out(['type']))]

# Split the data into training, validation, and test sets
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model
model_fed = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_all_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')  # number of classes
])

model_fed.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model_fed.get_weights()

    def fit(self, parameters, config):
        model_fed.set_weights(parameters)
        model_fed.fit(X_all_train, y_all_train, epochs=4, batch_size=32)
        return model_fed.get_weights(), len(X_all_train), {}

    def evaluate(self, parameters, config):
        model_fed.set_weights(parameters)
        results = model_fed.evaluate(X_all_test, y_all_test, verbose=0)
        return results[0], len(X_all_test), {"Test Recall": results[2], "Test Precision": results[1]}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())