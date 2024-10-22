import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import pydeck as pdk
import json
from keplergl import KeplerGl
from shapely.geometry import Point
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
from streamlit_keplergl import keplergl_static

# Set up the Streamlit page with custom title and layout
st.set_page_config(page_title="Water Consumption Visualization", layout="wide")

# Main Title with description
st.title("🌊 Water Consumption and Building Visualization")
st.markdown("This app visualizes water consumption and building information, with breakdowns by zone and user type. Use the sidebar to provide average consumption details and view interactive maps, graphs, and tables.")
st.markdown("Please upload a .csv file with the specific columns' names X, Y, Block_Number, Zone, DMA, and Status")


# File upload section with icon
st.markdown("### 📂 Upload Your Data File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Define the expected columns
    expected_columns = ["X", "Y", "Zone", "Block_Number", "DMA", "Status"]

    # Select valid columns and fill missing ones with default values
    df = df[[col for col in df.columns if col in expected_columns]]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    # Step 1: Categorize Status into "legal", "illegal", and "non-user"
    df['User Type'] = df['Status'].apply(lambda x: 'Legal' if x == 'water meter' else ('Illegal' if x == 'illegal connection' else ('Non-user' if x == 'non-user' else 'No Data')))

    # Filter out rows with "No Data" in User Type for percentage calculations
    filtered_df = df[df['User Type'] != 'No Data']

    # Sidebar inputs section with sliders only for the average litres per person
    st.sidebar.header("🔧 Average Inputs")
    avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.0)
    avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.0)
    avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=500, step=10, value=150)
    # Choose which heatmap to display
    st.sidebar.header("🔍 Heatmap Options")
    heatmap_type = st.sidebar.selectbox(
        "Choose a heatmap to display:",
        ["All Buildings", "Illegal Connections", "Legal Connections", "Non-Users"]
    )

    # Calculate total population using the full DataFrame (df), aggregated by Zone and DMA
    df['Population'] = avg_floors * avg_people_per_family
    total_population_by_zone = df.groupby('Zone')['Population'].sum() if 'Zone' in df.columns else None
    total_population_by_dma = df.groupby('DMA')['Population'].sum() if 'DMA' in df.columns else None

    # Calculate percentages for legal, illegal, and non-users per Zone
    if 'Zone' in filtered_df.columns:
        filtered_df['Population'] = avg_floors * avg_people_per_family
        user_summary_zone = filtered_df.pivot_table(
            values='Population',
            index='Zone',
            columns='User Type',
            aggfunc='sum',
            fill_value=0
        )
        user_summary_zone['Total Population'] = total_population_by_zone

        for user_type in ['Legal', 'Illegal', 'Non-user']:
            user_summary_zone[f'{user_type} %'] = (user_summary_zone[user_type] / user_summary_zone['Total Population']) * 100

        user_summary_zone = user_summary_zone.round(1)
        overall_summary_zone = user_summary_zone[['Total Population', 'Legal %', 'Illegal %', 'Non-user %']].copy()

    # Add a final row with the sum of all Zones
        total_population_all_zones = user_summary_zone['Total Population'].sum()
        legal_sum_zone = user_summary_zone['Legal %'].mean()
        illegal_sum_zone = user_summary_zone['Illegal %'].mean()
        non_user_sum_zone = user_summary_zone['Non-user %'].mean()
        overall_summary_zone.loc['Total'] = [total_population_all_zones, legal_sum_zone, illegal_sum_zone, non_user_sum_zone]

    # Calculate percentages for legal, illegal, and non-users per DMA (if 'DMA' column exists)
    if 'DMA' in filtered_df.columns:
        user_summary_dma = filtered_df.pivot_table(
            values='Population',
            index='DMA',
            columns='User Type',
            aggfunc='sum',
            fill_value=0
        )
        user_summary_dma['Total Population'] = total_population_by_dma

        for user_type in ['Legal', 'Illegal', 'Non-user']:
            user_summary_dma[f'{user_type} %'] = (user_summary_dma[user_type] / user_summary_dma['Total Population']) * 100

        user_summary_dma = user_summary_dma.round(1)
        overall_summary_dma = user_summary_dma[['Total Population', 'Legal %', 'Illegal %', 'Non-user %']].copy()

    # Combine "overall" input with the sum of everything for both Zone and DMA
    overall_population_zone = total_population_by_zone.sum() if total_population_by_zone is not None else 0
    overall_population_dma = total_population_by_dma.sum() if total_population_by_dma is not None else 0

     # Add a final row with the sum of all DMAs
    total_population_all_dmas = user_summary_dma['Total Population'].sum()
    legal_sum_dma = user_summary_dma['Legal %'].mean()
    illegal_sum_dma = user_summary_dma['Illegal %'].mean()
    non_user_sum_dma = user_summary_dma['Non-user %'].mean()
    overall_summary_dma.loc['Total'] = [total_population_all_dmas, legal_sum_dma, illegal_sum_dma, non_user_sum_dma]


    # Streamlit tabs for organized visualization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Network Users Summary", "📅 Seazonal Water Distribution", "💧 Water Demand Model", "🗺️ Data Visualization"])

    with tab1:
        st.markdown("### 📊 User Type Summary by Zone")
        st.dataframe(overall_summary_zone)

        if 'DMA' in filtered_df.columns:
            st.markdown("### 📊 User Type Summary by DMA")
            st.dataframe(overall_summary_dma)

        st.markdown("### 📈 Population by User Type")
        fig, ax = plt.subplots(figsize=(10, 4))

        user_summary_zone[['Total Population', 'Legal', 'Illegal', 'Non-user']].plot(kind='bar', ax=ax)
        ax.set_ylabel('Population')
        ax.set_title('Population Distribution by Zone and User Type')
        st.pyplot(fig)

    with tab2:
        st.markdown("### 📅 Monthly Water Consumption Calculation")
        
        #Seazonality factors
        month_factors = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dez'],
            'Factor': [0.07, 0.07, 0.07, 0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.08, 0.07, 0.07]
        }
        # Create a DataFrame for the month factors
        df_factors = pd.DataFrame(month_factors)        
        
        st.markdown("#### Seasonal Variation Factor")        
        # Slider to adjust the factors' variation (0 = all equal, 1 = current, 2 = amplified)
        variation_factor = st.slider("Adjust Variation of Factors (0 = No Variation, 1 = Proposed, 2 = Amplified)", min_value=0.0, max_value=2.0, step=0.1, value=1.0)




        # Calculate monthly water consumption based on factors
        df_factors['Factor - Updated'] = (1 - variation_factor) * np.mean(df_factors["Factor"]) + variation_factor * df_factors["Factor"]
        df_factors['Monthly Daily Consumption - l/p/d'] = round(df_factors['Factor - Updated'] * avg_litres_per_person * 12)
        df_factors["Total Monthly Consumption - m3"] = round(df_factors['Monthly Daily Consumption - l/p/d'] * sum(df["Population"]) / 1000, -2)
        
        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
             st.markdown("### Monthly Water Consumption")
                # Display the table with calculated values
             st.dataframe(df_factors)

        with col2:
            # Plot a graph of monthly water consumption
            st.markdown("### Monthly Water Consumption Distribution (l/p/d)")

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(df_factors['Month'], df_factors['Monthly Daily Consumption - l/p/d'], marker='o', color='blue', linewidth=1.0)
            ax.set_ylabel('Monthly Water Consumption (l/p/d)')
            ax.set_title('Monthly Water Consumption Distribution')
            ax.grid(True, linestyle ='-', axis = 'y')

            # Apply the if condition for y-axis limits
            if avg_litres_per_person < 200:
                ax.set_ylim(50, 300)  # Set y-axis limits for avg_litres_per_person < 160
            elif avg_litres_per_person < 280:
                ax.set_ylim(130, 380)  # Set y-axis limits for avg_litres_per_person < 260
            else:
                ax.set_ylim(190, 600)  # Set y-axis limits for avg_litres_per_person >= 260

            # Display the plot
            st.pyplot(fig)

    with tab3:
        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)
        
           
        # Calculate water consumption per zone and overall consumption (for monthly values)
        filtered_df['Cubic Metres'] = filtered_df['Population'] * avg_litres_per_person / 1000 * 30

        # Group water consumption data per zone
        water_per_zone = filtered_df.groupby('Zone').agg({
            'Cubic Metres': 'sum',
            'Population': 'sum'
        }).reset_index()

        # Add a row for total values across all zones
        total_row = pd.DataFrame([['Total', water_per_zone['Cubic Metres'].sum(), water_per_zone['Population'].sum()]],
                                 columns=water_per_zone.columns)        
        water_per_zone = pd.concat([water_per_zone, total_row], ignore_index=True)  

        with col1:
            st.markdown("### 💧 Water Consumption per Zone (Monthly)")
            st.dataframe(water_per_zone)

        with col2:
            st.markdown("### 📉 Monthly Water Consumption Variation by Zone")
            fig, ax = plt.subplots(figsize=(10, 4))
            water_per_zone.plot(x='Zone', y='Cubic Metres', kind='bar', ax=ax, color='#87CEEB')
            ax.set_ylabel('Cubic Metres')
            ax.set_title('Monthly Water Consumption by Zone')
            st.pyplot(fig)

    with tab4:
        st.markdown("### 🗺️ Interactive Maps with Google Satellite Basemap")
        
        # Create a selectbox above the map
        selected_attribute = st.selectbox("Color points by:", options=["Zone", "DMA"], index=0)


        # Create a GeoDataFrame for processing
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
        gdf = gdf.set_crs(epsg=4326)

        # Calculate the center of the uploaded data
        center_lat, center_lon = gdf["Y"].mean(), gdf["X"].mean()

        # Determine a reasonable zoom level based on data spread
        lat_range = gdf["Y"].max() - gdf["Y"].min()
        lon_range = gdf["X"].max() - gdf["X"].min()
        zoom = 12 if max(lat_range, lon_range) < 1 else 10
        
        # Create a dynamic configuration for KeplerGL
        # Dynamically recreate the configuration for KeplerGL based on the user selection
        def create_kepler_config(attribute):
            return {
                'version': 'v1',
                'config': {
                    'mapState': {
                        'latitude': center_lat,
                        'longitude': center_lon,
                        'zoom': 14
                    },
                    "mapStyle": {
                        "styleType": "satellite"
                    },
                    "visState": {
                        "layers": [
                            {
                                "id": "building_layer",
                                "type": "point",
                                "config": {
                                    "dataId": "Water Consumption Data",
                                    "label": "Building Locations",
                                    "columns": {
                                        "lat": "lat",
                                        "lng": "lng"
                                    },
                                    "visConfig": {
                                        "radius": 5,
                                        "opacity": 0.8,
                                        "colorField": {
                                            "name": attribute,  # Dynamically set color field based on user selection
                                            "type": "integer"  # Set the type to integer (since Zone and DMA are integers)
                                        },
                                        "colorRange": {
                                            "colors": ["#FF5733", "#33FF57", "#3357FF", "#F5B041", "#8E44AD"]
                                        }
                                    },
                                    "isVisible": True
                                }
                            }
                        ]
                    }
                }
            }

        # Generate the dynamic KeplerGL config based on user selection
        config_1 = create_kepler_config(selected_attribute)

        # Rename for easier recognition in Kepler
        df = df.rename(columns={"X": "longitude", "Y": "latitude"})

        kepler_map = KeplerGl(height=800, config=config_1)
        kepler_map.add_data(data=df, name="Water Consumption Data")
        keplergl_static(kepler_map)

        # Set up the Folium map with Google Satellite layer
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            width='100%'
        )
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)
        with col1:

            # Create heatmaps based on selection
            if heatmap_type == "All Buildings":
                st.markdown("#### 🔥 Heatmap of All Building Locations")
                heat_data = [[row['Y'], row['X']] for idx, row in gdf.iterrows()]
                HeatMap(heat_data, radius=15).add_to(m)

            elif heatmap_type == "Illegal Connections":
                st.markdown("#### 🔥 Heatmap of Illegal Connections")
                heat_data_illegal = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Illegal'].iterrows()]
                HeatMap(heat_data_illegal, radius=15).add_to(m)

            elif heatmap_type == "Legal Connections":
                st.markdown("#### 🔥 Heatmap of Legal Connections")
                heat_data_legal = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Legal'].iterrows()]
                HeatMap(heat_data_legal, radius=15).add_to(m)
                
            elif heatmap_type == "Non-Users":
                st.markdown("#### 🔥 Heatmap of Non-Users")
                heat_data_non_users = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Non-user'].iterrows()]
                HeatMap(heat_data_non_users, radius=15).add_to(m)

            # Add a layer control panel
            folium.LayerControl().add_to(m)

            # Display the Folium map in Streamlit
            folium_static(m, width=None, height=900)
        
        with col2:
            test = 100 / 1000
            config_grid = {
            'version': 'v1',
            'config': {
                'mapState': {
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'zoom': 14
                },
                "mapStyle": {
                    "styleType": "satellite"
                },
                "visState": {
                    "layers": [
                        {
                            "id": "building_layer",
                            "type": "grid",  
                            "config": {
                                "dataId": "Water Consumption Data",
                                "label": "Building Locations",
                                "color": [30, 144, 255],  # Color of points
                                "columns": {
                                    "lat": "Y",
                                    "lng": "X"
                                },
                                "visConfig": {
                                    "radius": test,
                                    "opacity": 0.8,
                                },
                                "isVisible" : True
                            }
                        }]}}}

            # Create heatmaps based on selection
            if heatmap_type == "All Buildings":
                st.markdown("#### 🔥 Heatmap of All Building Locations")
                kepler_map = KeplerGl(height=900, config=config_1)
                kepler_map.add_data(data=gdf, name="Water Consumption Data")
                keplergl_static(kepler_map)

            elif heatmap_type == "Illegal Connections":
                st.markdown("#### 🔥 Heatmap of Illegal Connections")
                gdf_illegal = gdf[gdf['User Type'] == 'Illegal'] 
                kepler_map = KeplerGl(height=900, config=config_1)
                kepler_map.add_data(data=gdf_illegal, name="Water Consumption Data")
                keplergl_static(kepler_map)

            elif heatmap_type == "Legal Connections":
                st.markdown("#### 🔥 Heatmap of Legal Connections")
                gdf_legal = gdf[gdf['User Type'] == 'Legal']
                kepler_map = KeplerGl(height=900, config=config_1)
                kepler_map.add_data(data=gdf_legal, name="Water Consumption Data")
                keplergl_static(kepler_map)
                
            elif heatmap_type == "Non-Users":
                st.markdown("#### 🔥 Heatmap of Non-Users")
                gdf_non_user = gdf[gdf['User Type'] == 'Non-user']
                kepler_map = KeplerGl(height=900, config=config_1)
                kepler_map.add_data(data=gdf_non_user, name="Water Consumption Data")
                keplergl_static(kepler_map)
   

else:
    st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
