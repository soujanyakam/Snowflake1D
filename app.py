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

st.info("The full Jupyter Notebook (FINALDATASETCREATION.ipynb) containing our data processing can be found on GitHub")

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
  """
)

st.info("The full Jupyter Notebook (NOAA_STORM_EVENTS_MODEL.ipynb) can be found on our GitHub")

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

## INTERACTIVE COMPONENT 
st.subheader("Interactive Risk Analysis")

st.write(
    """
    Use the filters below to explore counties based on storm severity, frequency, and damage levels. 
    These filters help identify counties with the most significant disaster impacts.
    """
)

# Create filter section
col1, col2 = st.columns(2)

with col1:
    st.write("**Storm Impact Filters**")
    
    severity_threshold = st.select_slider(
        "Minimum Storm Severity",
        options=["Any Severity", "Mild (>0.2)", "Moderate (>0.4)", "Severe (>0.6)", "Extreme (>0.8)"],
        value="Any Severity",
        help="Filter counties by their average storm severity (0 = mild, 1 = extreme)"
    )
    
    min_events = st.slider(
        "Minimum Storm Events",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Show counties with at least this many storm events"
    )

with col2:
    st.write("**Damage and Display Options**")
    
    damage_filter = st.select_slider(
        "Property Damage Level",
        options=["Any Damage", "Low (<$500K)", "Medium ($500K-$5M)", "High ($5M-$50M)", "Extreme (>$50M)"],
        value="Any Damage",
        help="Filter by total property damage from storms"
    )
    
    show_risk_type = st.radio(
        "Map Display",
        options=["Overall Risk", "Disaster Risk Only", "Medical Risk Only", "All Three Risk Types"],
        index=0,
        help="Choose which risk scores to display on maps"
    )

# Apply filters button
apply_filters = st.button("🔍 Apply Filters", type="primary", use_container_width=True)

if apply_filters or 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = True

if st.session_state.get('filters_applied', False):
    
    st.success("✅ Filters applied! Showing counties matching your criteria.")
    
    # Load base table
    filtered_risk = session.table("PROJECT_DB_FINAL.PROCESSED_DATA.COUNTY_RISK_PREDICTIONS_WITH_STORM")
    
    # Apply severity threshold
    severity_map = {
        "Any Severity": 0.0,
        "Mild (>0.2)": 0.2,
        "Moderate (>0.4)": 0.4,
        "Severe (>0.6)": 0.6,
        "Extreme (>0.8)": 0.8
    }
    min_severity = severity_map[severity_threshold]
    
    from snowflake.snowpark.functions import lit
    
    if min_severity > 0:
        filtered_risk = filtered_risk.filter(col("STORM_SEVERITY_MEAN") >= lit(min_severity))
    
    # Apply event count filter
    if min_events > 0:
        filtered_risk = filtered_risk.filter(col("STORM_EVENT_COUNT") >= lit(min_events))
    
    # Apply damage filter
    damage_map = {
        "Any Damage": 0,
        "Low (<$500K)": (0, 500000),
        "Medium ($500K-$5M)": (500000, 5000000),
        "High ($5M-$50M)": (5000000, 50000000),
        "Extreme (>$50M)": (50000000, 1e12)
    }
    
    if damage_filter != "Any Damage":
        min_damage, max_damage = damage_map[damage_filter]
        filtered_risk = filtered_risk.filter(
            (col("STORM_DAMAGE_PROPERTY_TOTAL") >= lit(min_damage)) &
            (col("STORM_DAMAGE_PROPERTY_TOTAL") < lit(max_damage))
        )
    
    # Join with other tables
    filtered_final = session.table("PROJECT_DB_FINAL.PROCESSED_DATA.FINAL_MASTER_DATASET")
    filtered_geojson = session.table("PROJECT_DB_FINAL.PROCESSED_DATA.GEOJSON").select(
        col("FEATURE").alias("FEATURE"),
        col("FEATURE")["properties"]["GEOID"].cast("STRING").alias("GEOID")
    )
    
    filtered_geojson = filtered_geojson.withColumnRenamed("GEOID", "FIPS_CODE")
    filtered_risk = filtered_risk.withColumnRenamed("FIPS_CODE", "FIPS_CODE")
    filtered_final = filtered_final.withColumnRenamed("FIPS_CODE", "FIPS_CODE")
    
    # Join data
    filtered_df = filtered_final.join(filtered_geojson, "FIPS_CODE")
    filtered_df = filtered_df.join(filtered_risk, "FIPS_CODE")
    
    # Convert to pandas
    filtered_pdf = filtered_df.to_pandas()
    
    # Display statistics
    st.write("---")
    
    if len(filtered_pdf) == 0:
        st.warning("⚠️ No counties match your filter criteria. Try adjusting the filters.")
    else:
        st.write(f"### Filtered Results: {len(filtered_pdf)} counties")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            avg_severity = filtered_pdf['STORM_SEVERITY_MEAN'].mean()
            st.metric(
                "Avg Storm Severity",
                f"{avg_severity:.2f}",
                help="Mean storm severity (0-1 scale)"
            )
        
        with col_stat2:
            total_events = filtered_pdf['STORM_EVENT_COUNT'].sum()
            st.metric(
                "Total Events",
                f"{int(total_events):,}",
                help="Sum of all storm events"
            )
        
        with col_stat3:
            avg_disaster_risk = filtered_pdf['DISASTER_RISK'].mean()
            st.metric(
                "Avg Disaster Risk",
                f"{avg_disaster_risk:.1f}/100",
                help="Mean disaster risk score"
            )
        
        with col_stat4:
            total_damage = filtered_pdf['STORM_DAMAGE_PROPERTY_TOTAL'].sum()
            st.metric(
                "Total Damage",
                f"${total_damage/1e6:.1f}M",
                help="Property damage sum"
            )
        
        # Top counties table
        st.write("---")
        st.write("### 🔴 Top 10 Highest Disaster Risk Counties")
        
        # Find columns
        def find_column(base_name, columns):
            matches = [c for c in columns if base_name in c]
            return matches[0] if matches else None
        
        available_cols = filtered_pdf.columns.tolist()
        
        fips_col = find_column('FIPS_CODE', available_cols)
        state_col = find_column('STATE_ABBR', available_cols) 
        disaster_col = find_column('DISASTER_RISK', available_cols)
        overall_col = find_column('OVERALL_RISK', available_cols)
        severity_col = find_column('STORM_SEVERITY_MEAN', available_cols)
        events_col = find_column('STORM_EVENT_COUNT', available_cols)
        damage_col = find_column('STORM_DAMAGE_PROPERTY_TOTAL', available_cols)
        
        display_cols = [c for c in [fips_col, state_col, disaster_col, overall_col, severity_col, events_col, damage_col] if c]
        
        if disaster_col:
            top_10 = filtered_pdf.nlargest(10, disaster_col)[display_cols].copy()
            
            # Rename for display
            rename_map = {}
            if fips_col:
                rename_map[fips_col] = 'FIPS'
            if state_col:
                rename_map[state_col] = 'State'
            if disaster_col:
                rename_map[disaster_col] = 'Disaster Risk'
            if overall_col:
                rename_map[overall_col] = 'Overall Risk'
            if severity_col:
                rename_map[severity_col] = 'Avg Severity'
            if events_col:
                rename_map[events_col] = 'Events'
            if damage_col:
                rename_map[damage_col] = 'Damage ($)'
            
            top_10 = top_10.rename(columns=rename_map)
            
            # Format damage column
            if 'Damage ($)' in top_10.columns:
                top_10['Damage ($)'] = top_10['Damage ($)'].apply(lambda x: f"${x/1e6:.1f}M" if x > 0 else "$0")
            
            # Round numeric columns
            numeric_cols = ['Disaster Risk', 'Overall Risk', 'Avg Severity']
            for col_name in numeric_cols:
                if col_name in top_10.columns:
                    top_10[col_name] = top_10[col_name].round(2)
            
            st.dataframe(top_10, use_container_width=True, hide_index=True)
        
        # Create geodataframe for maps
        filtered_pdf['geometry'] = filtered_pdf['FEATURE'].apply(
            lambda x: shape(json.loads(str(x))['geometry'])
        )
        
        filtered_gdf = gpd.GeoDataFrame(filtered_pdf, geometry='geometry')
        
        if filtered_gdf.crs is None:
            filtered_gdf = filtered_gdf.set_crs(epsg=4326)
        
        filtered_gdf_projected = filtered_gdf.to_crs(epsg=3857)
        filtered_gdf["AREA_SQKM"] = filtered_gdf_projected.geometry.area / 1e6
        filtered_gdf["POP_DENSITY"] = filtered_gdf["TOTAL_POPULATION"] / filtered_gdf["AREA_SQKM"]
        
        # Centroids for bubble map
        filtered_gdf_points = filtered_gdf.copy()
        filtered_gdf_points["geometry"] = filtered_gdf_points.geometry.centroid
        filtered_gdf_points = filtered_gdf_points.to_crs(epsg=4326)
        filtered_gdf_points["lon"] = filtered_gdf_points.geometry.x
        filtered_gdf_points["lat"] = filtered_gdf_points.geometry.y
        
        # Display maps
        st.write("---")
        st.write("### 🗺️ Interactive Maps - Filtered Counties")
        
        # Determine which risk types to show
        risk_types_to_show = []
        if show_risk_type == "Overall Risk":
            risk_types_to_show = [("Overall Risk", "OVERALL_RISK")]
        elif show_risk_type == "Disaster Risk Only":
            risk_types_to_show = [("Disaster Risk", "DISASTER_RISK")]
        elif show_risk_type == "Medical Risk Only":
            risk_types_to_show = [("Medical Risk", "MEDICAL_RISK")]
        else:  # All Three
            risk_types_to_show = [
                ("Disaster Risk", "DISASTER_RISK"),
                ("Medical Risk", "MEDICAL_RISK"),
                ("Infrastructure Risk", "INFRASTRUCTURE_RISK")
            ]
        
        for risk_name, risk_col in risk_types_to_show:
            st.write(f"#### {risk_name}")
            
            # Find actual column name
            actual_risk_col = find_column(risk_col, filtered_gdf_points.columns.tolist())
            
            if actual_risk_col:
                # Find STATE_ABBR column (handles Snowflake prefixes)
                state_abbr_col = find_column('STATE_ABBR', filtered_gdf_points.columns.tolist())
                
                # Build hover data dict
                hover_data_dict = {
                    "FIPS_CODE": True,
                    "POP_DENSITY": ":.0f",
                    "STORM_SEVERITY_MEAN": ":.2f",
                    "STORM_EVENT_COUNT": ":.0f",
                    actual_risk_col: ":.1f",
                    "lat": False,
                    "lon": False
                }
                
                # Add STATE_ABBR if it exists
                if state_abbr_col:
                    hover_data_dict[state_abbr_col] = True
                
                fig = px.scatter_mapbox(
                    filtered_gdf_points,
                    lat="lat",
                    lon="lon",
                    color=actual_risk_col,
                    size="POP_DENSITY",
                    hover_name="FIPS_CODE",
                    hover_data=hover_data_dict,
                    color_continuous_scale="viridis",
                    size_max=25,
                    zoom=4,
                    center={"lat": 32, "lon": -86},
                    height=600,
                    mapbox_style="carto-positron",
                    title=f"{risk_name} - Filtered Counties"
                )
                
                fig.update_layout(
                    margin={"r":0,"t":40,"l":0,"b":0},
                    coloraxis_colorbar=dict(title=risk_name)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not find {risk_col} column in data")
        
        # Comparison section
        st.write("---")
        st.write("### 📊 Compare Filtered vs. All Counties")
        
        col_compare1, col_compare2 = st.columns(2)
        
        with col_compare1:
            st.write("**Filtered Counties**")
            st.write(f"Count: {len(filtered_pdf)}")
            st.write(f"Avg Severity: {filtered_pdf['STORM_SEVERITY_MEAN'].mean():.3f}")
            st.write(f"Avg Overall Risk: {filtered_pdf['OVERALL_RISK'].mean():.1f}")
        
        with col_compare2:
            st.write("**All Counties (Baseline)**")
            st.write(f"Count: 790")
            
            # Load full dataset for comparison
            full_risk = session.table("PROJECT_DB_FINAL.PROCESSED_DATA.COUNTY_RISK_PREDICTIONS_WITH_STORM").to_pandas()
            st.write(f"Avg Severity: {full_risk['STORM_SEVERITY_MEAN'].mean():.3f}")
            st.write(f"Avg Overall Risk: {full_risk['OVERALL_RISK'].mean():.1f}")

st.write("---")
## INTERACTIVE COMPONENT END


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



