# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import resample
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

# %%
# Merge all dataframes
merged_df = pd.concat(reduced_data.values(), ignore_index=True)

# %% [markdown]
# # Centralized Deep Learning

# %% [markdown]
# ## All Sample CDL (Feedforward Neural Network)

# %%
# Separate features and target
X_all = merged_df.drop('type', axis=1)
y_all = merged_df['type']

# One-hot encode the labels
one_hot = OneHotEncoder(sparse_output=False)
y_all = one_hot.fit_transform(y_all.values.reshape(-1, 1))

# Split the data into training, validation, and test sets
X_all_train, X_all_temp, y_all_train, y_all_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
X_all_val, X_all_test, y_all_val, y_all_test = train_test_split(X_all_temp, y_all_temp, test_size=0.5, random_state=42)


# %%
# Create the model
model_all = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_all_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_all.shape[1], activation='softmax')  # number of classes
])

# Compile the model
model_all.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model_all.fit(
    X_all_train, y_all_train,
    validation_data=(X_all_val, y_all_val),
    epochs=10, batch_size=32, verbose=2
)

# Evaluate the model
results = model_all.evaluate(X_all_test, y_all_test, verbose=0)
print("===========================")
print("For All Sample in CDL")
print(f"Test Loss: {results[0]}")
print(f"Test Precision: {results[1]}")
print(f"Test Recall: {results[2]}")
print(f"Test F1 Score: {2 * (results[1] * results[2]) / (results[1] + results[2])}")


# %% [markdown]
# ## Undersampled CDL (Feedforward Neural Network)

# %%
# Function to undersample the majority class in a dataframe
def undersample(df, target_column):
    # Split the dataframe by class
    df_majority = df[df[target_column] == 3]
    df_minority = df[df[target_column] != 3]

    # Downsample the majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)

    # Combine the downsampled majority class dataframe with the original minority class dataframe
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled

# Apply the undersample function to each dataframe
df_undersampled = undersample(merged_df, 'type')

# %%
# Separate features and target
X_under = df_undersampled.drop('type', axis=1)
y_under = df_undersampled['type']

# One-hot encode the labels
one_hot = OneHotEncoder(sparse_output=False)
y_under = one_hot.fit_transform(y_under.values.reshape(-1, 1))

# Split the data into training, validation, and test sets
X_under_train, X_under_temp, y_under_train, y_under_temp = train_test_split(X_under, y_under, test_size=0.3, random_state=42)
X_under_val, X_under_test, y_under_val, y_under_test = train_test_split(X_under_temp, y_under_temp, test_size=0.5, random_state=42)


# %%
# Create the model
model_under = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_under_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_under.shape[1], activation='softmax')  # number of classes
])

# Compile the model
model_under.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model_under.fit(
    X_under_train, y_under_train,
    validation_data=(X_under_val, y_under_val),
    epochs=10, batch_size=32, verbose=2
)

# Evaluate the model
results = model_under.evaluate(X_under_test, y_under_test, verbose=0)

print("===========================")
print("For Undersampled in CDL")
print(f"Test Loss: {results[0]}")
print(f"Test Precision: {results[1]}")
print(f"Test Recall: {results[2]}")
print(f"Test F1 Score: {2 * (results[1] * results[2]) / (results[1] + results[2])}")


# %% [markdown]
# ## Oversample & Undersample (Feedforward Neural Network)

# %%
print('Original dataset shape %s' % Counter(merged_df['type']))

# %%
# Define oversampling and undersampling methods

# For class 3, it should represent 50% of the total data.
# Since the total number of samples is 3552244, we want approximately 1776122 samples in class 3 after resampling.
# For the other classes, we want them to sum up to 50% of the data.
# To keep things simple, let's distribute this equally among the 7 classes. So each of these classes will have about 253128 samples.

over = SMOTE(sampling_strategy={0: 253128, 4: 253128, 1: 253128, 2: 253128, 5: 253128, 7: 253128, 6: 253128})
under = RandomUnderSampler(sampling_strategy={3: 1776122})

# Define pipeline
pipeline = Pipeline(steps=[('o', over), ('u', under)])

# Separate features and target
X = merged_df.drop('type', axis=1)
y = merged_df['type']

# Apply the resampling
X_resampled, y_resampled = pipeline.fit_resample(X, y)


# One-hot encode the labels
one_hot = OneHotEncoder(sparse_output=False)
y_resampled = one_hot.fit_transform(y_resampled.values.reshape(-1, 1))

# %%
# Split the data into training, validation, and test sets
X_resampled_train, X_resampled_temp, y_resampled_train, y_resampled_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_resampled_val, X_resampled_test, y_resampled_val, y_resampled_test = train_test_split(X_resampled_temp, y_resampled_temp, test_size=0.5, random_state=42)

# %%
# Create the model
model_resampled = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_resampled_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_resampled.shape[1], activation='softmax')  # number of classes
])

# Compile the model
model_resampled.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model_resampled.fit(
    X_resampled_train, y_resampled_train,
    validation_data=(X_resampled_val, y_resampled_val),
    epochs=10, batch_size=32, verbose=2
)

# Evaluate the model
results = model_resampled.evaluate(X_resampled_test, y_resampled_test, verbose=0)

print("===========================")
print("For Resampled - (Over & Under) in CDL")
print(f"Test Loss: {results[0]}")
print(f"Test Precision: {results[1]}")
print(f"Test Recall: {results[2]}")
print(f"Test F1 Score: {2 * (results[1] * results[2]) / (results[1] + results[2])}")


