import streamlit as st 
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

with open("models/xgbc_model.pkl", "rb") as f:
    xgbc_model = pickle.load(f)

# Dictionaries to encode categorical features
d_land_surface = {'n':0, 'o':1, 't':2}
d_foundation = {'h':0, 'i':1, 'r':2, 'u':3, 'w':4}
d_roof = {'n':0, 'q':1, 'x':2}
d_ground_floor = {'f':0, 'm':1, 'v':2, 'x':3, 'z':4}
d_other_floor = {'j':0, 'q':1, 's':2, 'x':3}
d_position = {'j':0, 'o':1, 's':2, 't':3}
d_plan_config = {'a':0, 'c':1, 'd':2, 'f':3, 'm':4, 'n':5, 'o':6, 'q':7, 's':8, 'u':9}
d_ownership = {'a':0, 'r':1, 'v':2, 'w':3}


st.title("🏠 Earthquake Damage Prediction")

st.write("Enter building details to predict damage grade:")

# Numeric Inputs
geo_level_1_id = st.number_input("Geo Level 1 ID (0-30)", 0, 30, 0)
geo_level_2_id = st.number_input("Geo Level 2 ID (0-1427)", 0, 1427, 0)
geo_level_3_id = st.number_input("Geo Level 3 ID (0-12567)", 0, 12567, 0)
count_floors_pre_eq = st.number_input("Number of floors before EQ", 0, 20, 1)
age = st.number_input("Building age (years)", 0, 200, 10)
area_percentage = st.number_input("Area percentage", 0, 100, 50)
height_percentage = st.number_input("Height percentage", 0, 100, 50)
count_families = st.number_input("Number of families", 0, 20, 1)

# Categorical Inputs
land_surface_condition = st.selectbox("Land Surface Condition", list(d_land_surface.keys()))
foundation_type = st.selectbox("Foundation Type", list(d_foundation.keys()))
roof_type = st.selectbox("Roof Type", list(d_roof.keys()))
ground_floor_type = st.selectbox("Ground Floor Type", list(d_ground_floor.keys()))
other_floor_type = st.selectbox("Other Floor Type", list(d_other_floor.keys()))
position = st.selectbox("Building Position", list(d_position.keys()))
plan_configuration = st.selectbox("Plan Configuration", list(d_plan_config.keys()))
legal_ownership_status = st.selectbox("Legal Ownership Status", list(d_ownership.keys()))

# Binary Features
binary_features = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
    "has_secondary_use",
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other"
]

binary_inputs = []
for feature in binary_features:
    val = st.selectbox(f"{feature} (0 = No, 1 = Yes)", [0,1])
    binary_inputs.append(val)

# Transform categorical inputs
land_surface_condition = d_land_surface[land_surface_condition]
foundation_type = d_foundation[foundation_type]
roof_type = d_roof[roof_type]
ground_floor_type = d_ground_floor[ground_floor_type]
other_floor_type = d_other_floor[other_floor_type]
position = d_position[position]
plan_configuration = d_plan_config[plan_configuration]
legal_ownership_status = d_ownership[legal_ownership_status]

# Final feature vector
features = [
    geo_level_1_id, geo_level_2_id, geo_level_3_id,
    count_floors_pre_eq, age, area_percentage, height_percentage,
    land_surface_condition, foundation_type, roof_type,
    ground_floor_type, other_floor_type, position, plan_configuration,
    legal_ownership_status, count_families
] + binary_inputs

features = np.array(features).reshape(1, -1)

# Prediction
if st.button("Predict Damage Level"):
    prediction = xgbc_model.predict(features)
    st.success(f"Predicted Damage Level: {prediction[0]}")



