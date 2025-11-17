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
    
    **Challenge Advisors:** 
    
    **BTTAI Coach:**
    """
)
st.subheader("Current Challenge")
st.write(
  """
     talk about what healthcare deserts are + why they are importnat
     add a statistic
  """
)
st.subheader("Our Project")
st.write(
  """
     (still a wip, feel free to edit the grammar / change as much as you want)
  """
)
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
  Below is our dataset that we used throughout our project. To geographically identify different regions, we used FIPS codes, which are standardized numeric codes used by the U.S. government to identify geographic areas like states, counties, cities, and other locations. 

  """
)
st.write(
  """
  Discuss more of the columns in depth, how we found all of the data (add links), how we clenaed the data (sara / michael can potentially add more for that), + trends in the data (exploratory data analysis)

  """
)

st.dataframe(session.table("FINAL_MASTER_DATASET").limit(10).collect())

st.subheader("Storm Prediction Model")
st.subheader("Linear Regression Risk Score Model")
st.subheader("Geospatial Map")

# load tables
final_df = session.table("FINAL_MASTER_DATASET") #columns: FIPS_CODE, STATE_ABBR, MAX_HPSA_SCORE, AVG_HPSA_SCORE, NUM_DESIGNATION_TYPES, HPSA_POPULATION, POVERTY_RATE, RURAL_STATUS, SVI_SCORE, SOCIOECONOMIC_SVI, HOUSEHOLD_SVI, MINORITY_SVI, HOUSING_TRANSPORT_SVI, TOTAL_POPULATION, TOTAL_DISASTER_EVENTS, TOTAL_DAMAGE_COST, NUM_DISASTER_TYPES, STATE_TRAFFIC_CONGESTION, STATE_WEATHER_TRAFFIC, STATE_INCIDENT_TRAFFIC
geojson = session.table("GEOJSON").select(
    col("FEATURE").alias("FEATURE"),
    col("FEATURE")["properties"]["GEOID"].cast("STRING").alias("GEOID")
) #contains one column (FEATURE), and each cell is a json consisting of geographical information ({"geometry": {"coordinates": [[[-98.620692,) ... }})

# extract county geoid from each entry in the geojson column, as county geoid = fips code
# join FINAL_MASTER_DATASET with the GEOJSON table using FIPS_CODE = GEOID
df = final_df.join(
    geojson,
    final_df["FIPS_CODE"] == geojson["GEOID"]
)

pdf = df.to_pandas()

# convert geojson into shapely geometry types
pdf['geometry'] = pdf['FEATURE'].apply(
    lambda x: shape(json.loads(str(x))['geometry'])
)

# create geodataframe (contain one column with geometry information that can be used for geographical visualizations)
gdf = gpd.GeoDataFrame(pdf, geometry='geometry')

# polygon-based map
fig = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry.__geo_interface__,  
    locations=gdf.index,
    color='AVG_HPSA_SCORE',
    hover_name='FIPS_CODE',
    hover_data=['TOTAL_POPULATION', 'HPSA_POPULATION'],
    mapbox_style="carto-positron",
    zoom=4,
    center={"lat": 32, "lon": -86},
    opacity=0.6
)

st.plotly_chart(fig)


