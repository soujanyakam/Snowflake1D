import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
import geopandas as gpd
from shapely.geometry import shape
import json
import streamlit as st
import plotly.express as px
import json
import os
from dotenv import load_dotenv

st.title(f"Navigating Healthcare Deserts: Snowflake 1D")
st.write(
    """
    **Team Members:** Anushmita, Michael, Paula, Sara, Soujanya, & Vivian
    
    **Challenge Advisors:** Joe Warbington & Tess Dicker
    
    **BTTAI Coach:** Rajshri Jain
    """
)
st.subheader("Current Challenge")
st.write(
  """
     Over 80% of U.S. counties are considered healthcare deserts, and around 120 million people lack adequate access to medical services. 
     Natural disasters, healthcare facility availability, and transportation systems impact indivudals' abilities to quickly and efficiently seek medical resources. 
     Currently, it can be challenging to effectively identify and allocate resources to counties across the United States. 
  """
)
st.subheader("Our Project")
st.write(
    """
    The goal of our project is to develop an interactive geospatial tool that predicts healthcare deserts in disaster-prone regions across the U.S. By combining demographic, transportation, healthcare facility, and natural disaster data that we find, we use supervised learning to generate regional risk scores highlighting areas where residents face limited healthcare access and high disaster vulnerability. The app helps health planners and policymakers visualize disparities and identify where new facilities or mobile clinics would have the greatest impact, with GenAI summarizing insights and suggesting targeted interventions to support equitable and resilient healthcare systems.

    Our project primarily focuses on the Southern region of the U.S., as this region is home to numerous healthcare deserts and is highly prone to natural disasters, making it a critical region to analyze where limited access and environmental risks intersect.
    """
)

load_dotenv()
connection_parameters = {
    "user": os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "account": os.getenv("ACCOUNT"),
    "warehouse": os.getenv("WAREHOUSE"),
    "database": os.getenv("DATABASE"),
    "schema": os.getenv("SCHEMA")
}
session = Session.builder.configs(connection_parameters).create()

st.subheader("Dataset")

st.write(
  """
  Below is our dataset that we used throughout our project. 
  To geographically identify different regions, we used FIPS codes, which are standardized numeric codes used by the U.S. government to identify geographic areas like states, counties, cities, and other locations. 
  We also used other features such as an numerical score used to determine if a geographic area is a Health Professional Shortage Area (HPSA), with higher scores indicating a greater shortage.
  Another metric we used is a Social Vulnerability Index (SVI), which measures a community's vulnerability to external stresses. Higher scores indicate that a community is more vulnerable compared to a community with a lower SVI score.
  """
)
st.write(
  """
  In total, we used four datasets and combined them into one dataset. Below are the datasets that we used for our project:
  * NOAA Storm Data 
    * Insights on Natural Disasters (Rain, Fire, Lightning, Tornadoes, Floods) in 2025 
    * Includes state, latitude/longitude, disaster type/description
  * CDC SVI and Places
    * Demographics
    * Community resilience
  * Healthcare Shortage HPSA 
    * Includes areas with shortages and shortage types
  * Transportation Congestion Causes
    * Insights on traffic congestion potential impact on healthcare access

  Some columns from our dataset can be found below:
  """
)
st.dataframe(session.table("FINAL_MASTER_DATASET").limit(10).collect())

st.write(
  """
  Additionally, to understand the relationship between our data, we conducted exploratory data analysis and created a correlation matrix. 
  Higher values indicating a stronger correlation between two variables. 
  """
)

st.image("images/correlation_matrix.png", caption="Correlation Matrix for Final Dataset", use_container_width =True)



st.subheader("Storm Prediction Model")

st.write(
  """
  We utilized NOAA storm data, which contains insights on different types of natural disasters (i.e. rain, fire, lightning) and 
  state, latitude/longitude, and disaster type/description to predict different types of natural disasters. 

  We trained a decision tree model to classify what type of disasters are likely to hit different regions of the United States. 
  We started with 33,904 individual NOAA storm events, used 5 digit FIPS codes to identify counties in the U.S., and grouped 48+ storm types into 13 categories (Tornado, Flood, Wind, etc.).
  The overall accuracy of our model is 91%, and we then used the prediction probabilities as severity scores (85% confident = 0.85 severity).
  Below is a small snippet from the Jupyter Notebook where we cleaned the data, trained the model, evaluated the model performance, and created storm predictions by county. 
  The full Jupyter Notebook (NOAA_STORM_EVENTS_MODEL.ipynb) can be found on our GitHub. 
  """
)

st.code("""
# encode and train model 
df_model = pd.get_dummies(df_pandas, columns=['STATE', 'MONTH_NAME'], drop_first=True)
X = df_model.drop(columns=['EVENT_TYPE', 'EVENT_GROUP', 'STATE_FIPS', 'CZ_FIPS', 'FIPS_CODE'])
y = df_model['EVENT_GROUP']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred)) """
, language='python')


st.write(
  """
  Evaluation metrics for our model and sample classification can be seen below:
  """
)

st.image("images/classification_report.png", caption="Storm Decision Tree Classification Report", use_container_width =True)
st.image("images/confusion_matrix.png", caption="Storm Decision Tree Confusion Matrix", use_container_width =True)
st.image("images/storm_predictions.png", caption="Sample Storm Predictions", use_container_width =True)


st.write(
  """
  We saved the results from this model to a table after aggregating storms to a county level. 
  We grouped 33,904 events by FIPS code and created 8 storm features per county:
  * Average severity score
  * Max severity score
  * Severity variability
  * Total injuries
  * Total deaths
  * Total property damage
  * Total crop damage
  * Event count

  This resulted in 4,760 counties in the Southeast region of the U.S. with storm risk profiles.
  """
)

st.subheader("Linear Regression Risk Score Model")

st.write(
  """
  We created different categories of risk based on our initial merged dataset and the storm predictions dataset from the storm decision tree model.

* Medical Risk = HPSA × SVI
  * Multiplied healthcare shortage by social vulnerability
  * Scaled to 0-100
* Disaster Risk = Weighted storm features
  * 30% avg severity, 20% max severity, 20% property damage, 15% event count, 10% injuries, 5% deaths
  * Uses storm model predictions, not raw data
  * Scaled to 0-100
* Infrastructure Risk = Traffic
  * State-level traffic congestion normalized to 0-100
  * Overall Risk = Weighted combination
  * 40% Medical, 35% Disaster, 25% Infrastructure (change change if we want)
  * Scaled to 0-100
* Built regression model
  * Features: 24 numeric columns (including 8 storm features)
  * Target: OVERALL_RISK score
  * Split: 632 training, 158 testing
  * Model: Gradient Boosting (100 trees)
* Results
  * Test accuracy: 80.4% (R²)
  * Average error: ±3.4 points
  * 790 counties with:
  * MEDICAL_RISK, DISASTER_RISK, INFRASTRUCTURE_RISK, OVERALL_RISK, predicted risk, and all 8 storm features
""")

st.image("images/risk_scores.png", caption="Risk Score Breakdown", use_container_width =True)


#TODO: add in code snippets + potentially evaluation metrics from the linear regression model
# st.write(
#   """
#   Below is a code snippet from our Jupyter Notebook where we created all of the risk categories and trained the linear regression model. 
#   View the full Jupyter Notebook on our GitHub at 
#   """
# )


st.subheader("Geospatial Maps")

# load tables
final_df = session.table("FINAL_MASTER_DATASET") #columns: FIPS_CODE, STATE_ABBR, MAX_HPSA_SCORE, AVG_HPSA_SCORE, NUM_DESIGNATION_TYPES, HPSA_POPULATION, POVERTY_RATE, RURAL_STATUS, SVI_SCORE, SOCIOECONOMIC_SVI, HOUSEHOLD_SVI, MINORITY_SVI, HOUSING_TRANSPORT_SVI, TOTAL_POPULATION, TOTAL_DISASTER_EVENTS, TOTAL_DAMAGE_COST, NUM_DISASTER_TYPES, STATE_TRAFFIC_CONGESTION, STATE_WEATHER_TRAFFIC, STATE_INCIDENT_TRAFFIC
risk_predictions = session.table("COUNTY_RISK_PREDICTIONS_WITH_STORM") #columns: FIPS_CODE, STATE_ABBR, MEDICAL_RISK, DISASTER_RISK, INFRASTRUCTURE_RISK, OVERALL_RISK, PREDICTED_OVERALL_RISK, STORM_SEVERITY_MEAN, STORM_SEVERITY_MAX, STORM_SEVERITY_STD, STORM_INJURIES_TOTAL, STORM_DEATHS_TOTAL, STORM_DAMAGE_PROPERTY_TOTAL, STORM_DAMAGE_CROPS_TOTAL, STORM_EVENT_COUNT
geojson = session.table("GEOJSON").select(
    col("FEATURE").alias("FEATURE"),
    col("FEATURE")["properties"]["GEOID"].cast("STRING").alias("GEOID")
) #contains one column (FEATURE), and each cell is a json consisting of geographical information ({"geometry": {"coordinates": [[[-98.620692,) ... }})

# extract county geoid from each entry in the geojson column, as county geoid = fips code
# join FINAL_MASTER_DATASET with the GEOJSON table using FIPS_CODE = GEOID
final_df = final_df.withColumnRenamed("FIPS_CODE", "FIPS_CODE")
risk_predictions = risk_predictions.withColumnRenamed("FIPS_CODE", "FIPS_CODE")
geojson = geojson.withColumnRenamed("GEOID", "FIPS_CODE")


# CREATE POPULATION DENSITY BASED PLOTTING FUNCTION 
df = final_df.join(geojson, "FIPS_CODE")
df = df.join(risk_predictions, "FIPS_CODE")

pdf = df.to_pandas()

# convert geojson into shapely geometry types
pdf['geometry'] = pdf['FEATURE'].apply(
    lambda x: shape(json.loads(str(x))['geometry'])
)

# create geodataframe (contain one column with geometry information that can be used for geographical visualizations)
gdf = gpd.GeoDataFrame(pdf, geometry='geometry')


# use EPSG:4326, which is the geographic coordinate system that plotly.express uses
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

gdf_projected = gdf.to_crs(epsg=3857) 
gdf["AREA_SQKM"] = gdf_projected.geometry.area / 1e6
gdf["POP_DENSITY"] = gdf["TOTAL_POPULATION"] / gdf["AREA_SQKM"]

#create centroids
gdf_points = gdf.copy()
gdf_points["geometry"] = gdf_points.geometry.centroid
gdf_points = gdf_points.to_crs(epsg=4326)
gdf_points["lon"] = gdf_points.geometry.x
gdf_points["lat"] = gdf_points.geometry.y

#define function to plot a population density map
def plot_population_density_map(color_type):
  fig = px.scatter_mapbox(
      gdf_points,
      lat="lat",
      lon="lon",
      color=color_type,
      size="POP_DENSITY",  
      hover_name="FIPS_CODE",
      hover_data=[
          "FIPS_CODE",
          "TOTAL_POPULATION",
          "AREA_SQKM",
          "POP_DENSITY",
          "POVERTY_RATE",
          "RURAL_STATUS",
          "SVI_SCORE",
          "TOTAL_DISASTER_EVENTS",
          "TOTAL_DAMAGE_COST",
          "NUM_DISASTER_TYPES",
          "MEDICAL_RISK", 
          "DISASTER_RISK", 
          "INFRASTRUCTURE_RISK", 
          "OVERALL_RISK",
      ],
      color_continuous_scale="viridis",
      size_max=20,  # maximum circle size in pixels
      zoom=4,
      center={"lat": 32, "lon": -86},
      height=800,
      mapbox_style="carto-positron"
  )

  st.plotly_chart(fig)


#PLOT POLYGON BASED MAPS FOR DIFFERENT RISK SCORES
df2 = risk_predictions.join(geojson, "FIPS_CODE")

pdf2 = df2.to_pandas()

# convert geojson into shapely geometry types
pdf2['geometry'] = pdf2['FEATURE'].apply(
    lambda x: shape(json.loads(str(x))['geometry'])
)

# create geodataframe (contain one column with geometry information that can be used for geographical visualizations)
gdf2 = gpd.GeoDataFrame(pdf2, geometry='geometry')

def plot_polygon_map(color_type):
    map = px.choropleth_mapbox(
        gdf2,
        geojson=gdf2.geometry.__geo_interface__,  
        locations=gdf2.index,
        color=color_type,
        hover_name='FIPS_CODE',
        hover_data=["STATE_ABBR", "MEDICAL_RISK", "DISASTER_RISK", "INFRASTRUCTURE_RISK", "OVERALL_RISK", "PREDICTED_OVERALL_RISK", "STORM_SEVERITY_MEAN", "STORM_SEVERITY_MAX", "STORM_SEVERITY_STD", "STORM_INJURIES_TOTAL", "STORM_DEATHS_TOTAL", "STORM_DAMAGE_PROPERTY_TOTAL", "STORM_DAMAGE_CROPS_TOTAL", "STORM_EVENT_COUNT"],
        mapbox_style="carto-positron",
        zoom=4,
        center={"lat": 32, "lon": -86},
        opacity=0.6
    )

    st.plotly_chart(map)

#use functions defined above to plot
plot_population_density_map('OVERALL_RISK')
plot_polygon_map('OVERALL_RISK')

plot_population_density_map('DISASTER_RISK')
plot_polygon_map('DISASTER_RISK')

plot_population_density_map('MEDICAL_RISK')
plot_polygon_map('MEDICAL_RISK')



