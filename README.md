# Navigating Healthcare Desserts Through Natural Disasters

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Sara Alaidroos | @Sara5722 | Data preprocessing, dataset documentation, exploratory data analysis (EDA)|
| Anushmita Dey   | @anushmita06  | Model creation, Model integration, Model evaluation, Streamlit development |
| Vivian Gao    | @vi4849  | Geospatial visualizations, Snowflake Cortex LLM integration, Streamlit development                |
| Soujanya Kamalapur | | Model creation & development |
| Paula Agyeman | | Risk score development |
| Michael Montanez | | Data preprocessing | 

## 🎯 **Project Highlights**

- Developed a comprehensive risk assessment system using **multiple regression models** to predict community vulnerability across Southern US counties
- Integrated **four diverse datasets** (medical access, social vulnerability, natural disasters, infrastructure) to create a holistic risk score
- Achieved **county-level risk mapping** enabling targeted emergency preparedness and resource allocation
- Implemented **data integration techniques** to handle mixed geographic resolutions while preserving analytical integrity
- Built an **interactive Streamlit dashboard** for visualizing risk factors and exploring vulnerable communities
- Integrated **Claude Sonnet 4.5** to provide data insights and resource allocation recommendations

## 👩🏽‍💻 **Setup and Installation**

Set up .env file (not on GitHub for security reasons)

<b> Create Virtual Environment</b>
<br>
(powershell) <br>
``python -m venv .\venv `` <br>

``.\venv\Scripts\activate`` <br>
<br>
<b> Install Requirements</b>

``pip install -r requirements.txt`` <br>
<br>
<b> Run the App Locally</b>

``python -m streamlit run app.py`` <br>

Use two-factor authentication to establish a connection with Snowflake (ex. Duo Mobile push)

NOTE: It may take a minute or two for all of the graphs and visualizations to be loaded


## 🏗️ **Project Overview**

Over 80% of U.S. counties are considered healthcare deserts, and around 120 million people lack adequate access to medical services. Natural disasters, healthcare facility availability, and transportation systems impact individuals' abilities to quickly and efficiently seek medical resources. Currently, it can be challenging to effectively identify and allocate resources to counties across the United States.

As a part of Break Through Tech’s AI Studio program, we collaborated with Snowflake to create a Streamlit app that would allow organizations to understand and assess healthcare deserts. We limited our project scope to the Southeast region of the United States, as this region is home to numerous healthcare deserts and is highly prone to natural disasters, making it a critical region to analyze where limited access and environmental risks intersect. The goal of our project is to develop an interactive geospatial tool that predicts healthcare deserts in disaster-prone regions across the U.S. By combining demographic, transportation, healthcare facility, and natural disaster data that we find, we use supervised learning to generate regional risk scores highlighting areas where residents face limited healthcare access and high disaster vulnerability.

We trained a random forest model, decision tree model, and a linear regression model to predict storms for different counties in the Southeast region of the United States. We also created polygon and population density geospatial visualizations, and integrated Claude Sonnet 4.5 to provide recommendations for resources allocation. 

## 📊 **Data Exploration**

**Datasets Integrated**
- HPSA Data: Health Professional Shortage Areas scoring medical underservice
- CDC SVI: Social Vulnerability Index measuring community resilience
- NOAA Storm Events: Historical natural disaster frequency and damage
- Traffic Congestion: Infrastructure stress indicators <br> 
- To geographically identify different regions, we used FIPS codes, which are standardized numeric codes used by the U.S. government to identify geographic areas like states, counties, cities, and other locations.


**Key Insights from Exploratory Data Analysis:**
- Strong correlation between medical underservice (HPSA scores) and social vulnerability (SVI)
- Southern states show significant variation in disaster risk within states
- Poverty rates strongly correlate with both medical and social vulnerability metrics
- Traffic congestion patterns vary significantly by state rather than county

<img src="https://github.com/soujanyakam/Snowflake1D/blob/main/images/correlation_matrix.png" width="400" height="400" alt="Correlation Matrix" />


**Data Challenges:**
- Mixed geographic resolutions required careful data fusion strategies
- FIPS code inconsistencies across datasets necessitated standardization
- Missing data handling for sparse county-level disaster records

## 🧠 **Model Development**

**Two-Stage ML Architecture**
- Natural Disaster Model (Random Forest)
  - Transforms individual storm events into severity assessments
  - Uses ML to distinguish between destructive vs. mild storms
  - Relates insights to county level for broader risk context
- Risk Score Model (Gradient Boosting Regression)
  - Combines disaster history with healthcare access, social vulnerability, and infrastructure data
  - Learns complex relationships between multiple risk factors
  - Identifies counties where limited resources meet high disaster exposure
- Why This Design:
  - Two specialized models handle different aspects of the problem
  - Decision Tree outputs (probabilities) go well into Gradient Boosting input

**Natural Disaster Prediction Model**
- Challenge: **33,904** raw NOAA storm events lacked severity context - a county with 100 mild storms looks identical to one with 100 devastating tornadoes
- Approach: Built classification model using damage, location, and impact patterns to score storm severity on a continuous 0-1 scale 
- Discovery: Random Forest outperformed Decision Tree (**82% vs 70% accuracy**) with perfect precision on Wind and Hail events - the most common high-impact disasters
- Impact: Transformed storm events into **8 county-level risk metrics for 4,760 counties**; these ML-derived features later proved more predictive than raw event counts in final risk model

<img width="364" height="124" alt="1" src="https://github.com/user-attachments/assets/7b66f8d9-9fe4-4cc8-8ad8-2cae8dfe4bc5" />
<img width="445" height="104" alt="2" src="https://github.com/user-attachments/assets/677a2fa5-28c8-4f73-8929-6ebf9b71b50a" />
<br><br>

**Risk Score Model**
- Challenge: Raw outputs from the Natural Disaster Prediction Model, combined with HPSA, SVI, and traffic data, lacked a unified way to quantify how well counties withstand and recover from disasters.
- Approach: Trained a **Gradient Boosting Regression model** using disaster frequency, healthcare shortage, vulnerability, and mobility indicators to estimate community resilience across Southern U.S. counties.
- Discovery: The optimized model achieved an **R² of ~0.80**, effectively capturing resilience disparities linked to infrastructure access and social vulnerability.
- Impact: Produced resilience scores for each county, enabling targeted policy insights and resource allocation to communities most at risk.
<img width="237" height="170" alt="reg" src="https://github.com/user-attachments/assets/39d60d39-9dca-4ff7-b951-f0bc17aad753" />

## 📈 **Model Evaluation**

### **Natural Disaster Prediction Model** 
<img width="212" height="202" alt="storm_classification" src="https://github.com/user-attachments/assets/621e350f-d5c1-4d39-a623-f6eafce6e73a" />
<br> The classification report shows that the model performs strongly on common storm types like Wind, Hail, and Flood, but struggles with rarer or visually similar categories such as Lightning and Dust Storm.
<br> <br>
<br><img width="241" height="206" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5f19e4af-cabc-4df2-ac90-243cab69632f" />
<br>The confusion matrix reveals that most predictions are correct for dominant classes, though frequent mix-ups occur between related categories like Cold and Snow/Ice.
<br> <br>
<br><img width="308" height="231" alt="important_features" src="https://github.com/user-attachments/assets/d16257ce-407b-4a6d-bb99-0f9e8a7655c6" />
<br>Geographic and Magnitude features are the most significant and repeated, and damage costs are a key indicator for storm type
<br> 

### **Risk Score Model**
<img width="358" height="214" alt="actual" src="https://github.com/user-attachments/assets/07578d66-53ea-46ca-a114-7313e357cbab" />
<br><img width="345" height="193" alt="data" src="https://github.com/user-attachments/assets/8b3509ef-230b-44ae-8d9c-b0d7c4091898" />
<br><img width="314" height="209" alt="most_important" src="https://github.com/user-attachments/assets/aca946fc-4cdf-414f-a8c4-488c421d4cb7" />
<br>*NOTE: Red = storm features <br> <br>

- Infrastructure (traffic congestion) and storms are very significant when predicting risk areas
- Medical Access is crucial to predicting community vulnerability
- Social factors compiled alongside disaster and medical access risk increase an area’s chances of being at risk



## 🌐 **Geospatial Visualizations**

- To create geospatial visualizations, we created a geojson table to store the geometries for all of the FIPS code regions
- We downloaded a Shapefile that contained all of all the counties in the United States from the U.S. Census Bureau, converted coordinates to the WGS84 system (EPSG:4326) to ensure compatibility with the Snowflake GEOGRAPHY type, uploaded data into a ``geojson`` table, and joined this table with other tables to create visualizations

### **Interactive Storm Impact Map**



https://github.com/user-attachments/assets/7aedf390-604c-4dbf-8ee9-dd8ed4698561



- Users can filter based on number of storm events, storm severity, property damage level, and the type of risk (overall risk, disaster risk, & medical risk)
- Results are displayed in a tabular and geospatial format

###  **Geospatial Risk Maps**
- We created maps based on three different risk scores: overall risk, medical risk, and disaster risk
- Both a polygon and population density map were created to visualize the different types of risks, and higher risk scores indicate that a county is more at risk
- Claude Sonnet 4.5 was integrated to provide data insights and resource allocation recommendations 


<img width="400" height="265" alt="population_density_overall_risk" src="https://github.com/user-attachments/assets/0d790952-65c5-48dc-a570-946bf77e8d7b" /> <img width="500" height="235" alt="polygon_overall_risk" src="https://github.com/user-attachments/assets/b9bee2c3-f9df-47a6-ae32-c2d8af61c1fe" />



https://github.com/user-attachments/assets/34663307-1224-4ce2-ba90-60f2916d1111




## 🚀 **Next Steps**
- Expand the scope of the project to include more regions of the United States, as our current project focuses on the Southeast region specifically
- Decrease latency for geospatial visualizations, as sometimes the visualizations can take 1-2 minutes to load 
- Modify the Streamlit app based on feedback from end users

## 🙏 **Acknowledgements**
Thank you to our Challenge Advisors, Joe Warbington and Tess Dicker, from Snowflake, our coach Rajshri Jain, and the Breakthrough Tech team for their guidance and support throughout this project.




