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
st.title("💧 Water Demand Estimation and Visualization 💧")
st.markdown("This app calculates water consumption based on buildings information, with breakdowns by zone and user type. Use the sidebar to provide average consumption details and view interactive maps, graphs, and tables.")
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

    # Define the expected values for the Status column
    expected_values = ['water meter', 'illegal connection', 'non-user', '', " "]
    
    # Replace NaN values with empty strings in the 'Status' column
    df['Status'] = df['Status'].fillna('')

    # Step 1: Validate the 'Status' column for any unexpected values
    unexpected_values = df[~df['Status'].isin(expected_values)]

    # If any unexpected values are found, raise an error or show a message
    if not unexpected_values.empty:
        unexpected_unique = unexpected_values['Status'].unique()
        unexpected_rows = unexpected_values.index.tolist()  # Get the index (line numbers) of unexpected values

        st.warning(f"Warning: Found unexpected values in the 'Status' column: {unexpected_values['Status'].unique()}")
        st.write(f"Expected values for the 'Status' column are: {expected_values}.")
        st.write(f"These unexpected values were found in the following rows: {unexpected_rows}")
        st.write("You can either proceed without these records or adjust your file to include only the expected values.")
        st.write("These records will not be processed if you choose to proceed.")

        df = df[df['Status'].isin(expected_values)]  # Optional: Filter out rows with unexpected values

    # Step 2: Categorize Status into "legal", "illegal", and "non-user"
    df['User Type'] = df['Status'].apply(
        lambda x: 'Legal' if x == 'water meter' else (
            'Illegal' if x == 'illegal connection' else (
                'Non-user' if x == 'non-user' else 'No Data'
            )
        )
    )

    # Filter out rows with "No Data" in User Type for percentage calculations
    filtered_df = df[df['User Type'] != 'No Data']

    # Sidebar inputs section with sliders only for the average litres per person
    st.sidebar.header("🔧 Assumptions")
    avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.0)
    avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.0)
    avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=500, step=10, value=150)
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
        
        # Calculate total population for each zone, including all inputs (with or without User Type)
        zone_counts = df.groupby('Zone').size()  # Count the number of inputs per Zone
        total_population_by_zone = zone_counts * avg_floors * avg_people_per_family  # Calculate total population

        # Count the number of Legal, Illegal, and Non-user inputs per zone
        legal_count = filtered_df[filtered_df['User Type'] == 'Legal'].groupby('Zone').size()
        illegal_count = filtered_df[filtered_df['User Type'] == 'Illegal'].groupby('Zone').size()
        non_user_count = filtered_df[filtered_df['User Type'] == 'Non-user'].groupby('Zone').size()

        # Ensure all zones are represented (fill missing values with 0)
        legal_count = legal_count.reindex(total_population_by_zone.index, fill_value=0)
        illegal_count = illegal_count.reindex(total_population_by_zone.index, fill_value=0)
        non_user_count = non_user_count.reindex(total_population_by_zone.index, fill_value=0)

        # Calculate the percentages for each user type (based on counts of Legal, Illegal, and Non-user)
        total_known_users = legal_count + illegal_count + non_user_count
        legal_percentage = (legal_count / total_known_users) * 100
        illegal_percentage = (illegal_count / total_known_users) * 100
        non_user_percentage = (non_user_count / total_known_users) * 100

        # Create a DataFrame to store the results
        user_summary_zone = pd.DataFrame({
            'Total Population': total_population_by_zone,
            'Legal %': legal_percentage,
            'Illegal %': illegal_percentage,
            'Non-user %': non_user_percentage
        })

        # Handle cases where no known users exist to avoid division by zero
        user_summary_zone[['Legal %', 'Illegal %', 'Non-user %']] = user_summary_zone[['Legal %', 'Illegal %', 'Non-user %']].fillna(0)

        # Add a final row with the sum of all Zones (weighted average for percentages)
        total_population_all_zones = total_population_by_zone.sum()

        legal_sum_zone = (legal_percentage * total_population_by_zone).sum() / total_population_all_zones
        illegal_sum_zone = (illegal_percentage * total_population_by_zone).sum() / total_population_all_zones
        non_user_sum_zone = (non_user_percentage * total_population_by_zone).sum() / total_population_all_zones
        
        user_summary_zone.loc['Total'] = [total_population_all_zones, legal_sum_zone, illegal_sum_zone, non_user_sum_zone]

   # Calculate total population and percentages per DMA (if 'DMA' column exists)
    if 'DMA' in filtered_df.columns:
        
        # Calculate total population for each DMA, including all inputs (with or without User Type)
        dma_counts = df.groupby('DMA').size()  # Count the number of inputs per DMA
        total_population_by_dma = dma_counts * avg_floors * avg_people_per_family  # Calculate total population

        # Count the number of Legal, Illegal, and Non-user inputs per DMA
        legal_count_dma = filtered_df[filtered_df['User Type'] == 'Legal'].groupby('DMA').size()
        illegal_count_dma = filtered_df[filtered_df['User Type'] == 'Illegal'].groupby('DMA').size()
        non_user_count_dma = filtered_df[filtered_df['User Type'] == 'Non-user'].groupby('DMA').size()

        # Ensure all DMAs are represented (fill missing values with 0)
        legal_count_dma = legal_count_dma.reindex(total_population_by_dma.index, fill_value=0)
        illegal_count_dma = illegal_count_dma.reindex(total_population_by_dma.index, fill_value=0)
        non_user_count_dma = non_user_count_dma.reindex(total_population_by_dma.index, fill_value=0)

        # Calculate the percentages for each user type (based on counts of Legal, Illegal, and Non-user)
        total_known_users_dma = legal_count_dma + illegal_count_dma + non_user_count_dma
        legal_percentage_dma = (legal_count_dma / total_known_users_dma) * 100
        illegal_percentage_dma = (illegal_count_dma / total_known_users_dma) * 100
        non_user_percentage_dma = (non_user_count_dma / total_known_users_dma) * 100

        # Create a DataFrame to store the results for DMAs
        user_summary_dma = pd.DataFrame({
            'Total Population': total_population_by_dma,
            'Legal %': legal_percentage_dma,
            'Illegal %': illegal_percentage_dma,
            'Non-user %': non_user_percentage_dma
        })

        # Handle cases where no known users exist to avoid division by zero
        user_summary_dma[['Legal %', 'Illegal %', 'Non-user %']] = user_summary_dma[['Legal %', 'Illegal %', 'Non-user %']].fillna(0)

        # Add a final row with the sum of all DMAs (weighted average for percentages)
        total_population_all_dmas = total_population_by_dma.sum()

        legal_sum_dma = (legal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
        illegal_sum_dma = (illegal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
        non_user_sum_dma = (non_user_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
        
        user_summary_dma.loc['Total'] = [total_population_all_dmas, legal_sum_dma, illegal_sum_dma, non_user_sum_dma]



    # Streamlit tabs for organized visualization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Network Users Summary", "📅 Seasonal Water Demand Distribution", "💧 Water Demand Model", "🗺️ Data Visualization"])

    with tab1:

        # Title for Summary of the Network Users
        st.markdown("## Summary of the Network Users")

        # Round the population and percentages for user_summary_zone
        user_summary_zone['Total Population'] = user_summary_zone['Total Population'].round(-2)  # Round population to nearest hundreds
        user_summary_zone['Legal %'] = user_summary_zone['Legal %'].round(1)  # Round percentages to 1 decimal place
        user_summary_zone['Illegal %'] = user_summary_zone['Illegal %'].round(1)
        user_summary_zone['Non-user %'] = user_summary_zone['Non-user %'].round(1)

        # Round the population and percentages for user_summary_dma
        user_summary_dma['Total Population'] = user_summary_dma['Total Population'].round(-2)  # Round population to nearest hundreds
        user_summary_dma['Legal %'] = user_summary_dma['Legal %'].round(1)  # Round percentages to 1 decimal place
        user_summary_dma['Illegal %'] = user_summary_dma['Illegal %'].round(1)
        user_summary_dma['Non-user %'] = user_summary_dma['Non-user %'].round(1)


        # Create columns for side-by-side layout for the tables
        col1, col2 = st.columns(2)

        # Place the two tables in the side-by-side layout
        with col1:
            if 'Zone' in filtered_df.columns:
                st.markdown("#### 📊 Water Network Summary - Zone")
                st.dataframe(user_summary_zone)

        with col2:
            if 'DMA' in filtered_df.columns:
                st.markdown("#### 📊 Water Network Summary - DMA")
                st.dataframe(user_summary_dma)

        # Title for Population by User Type
        st.markdown("### 📈 Population by User Type")

        # Create columns for side-by-side layout for the graphs
        col1, col2 = st.columns(2)
    
        # Calculate the number of people for each user type by multiplying Total Population with percentages
        user_summary_zone['Legal'] = (user_summary_zone['Total Population'] * user_summary_zone['Legal %'] / 100).astype(int)
        user_summary_zone['Illegal'] = (user_summary_zone['Total Population'] * user_summary_zone['Illegal %'] / 100).astype(int)
        user_summary_zone['Non-user'] = (user_summary_zone['Total Population'] * user_summary_zone['Non-user %'] / 100).astype(int)

        # Calculate the number of people for each user type by multiplying Total Population with percentages
        user_summary_dma['Legal'] = (user_summary_dma['Total Population'] * user_summary_dma['Legal %'] / 100).astype(int)
        user_summary_dma['Illegal'] = (user_summary_dma['Total Population'] * user_summary_dma['Illegal %'] / 100).astype(int)
        user_summary_dma['Non-user'] = (user_summary_dma['Total Population'] * user_summary_dma['Non-user %'] / 100).astype(int)

        # Exclude the 'Total' row for graph plotting
        user_summary_zone_plot = user_summary_zone[user_summary_zone.index != 'Total']
        user_summary_dma_plot = user_summary_dma[user_summary_dma.index != 'Total']

        # Place the two graphs in the side-by-side layout
        # First graph for Zone
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))  # Adjust figure size
            user_summary_zone_plot[['Non-user', 'Illegal', 'Legal', 'Total Population']].plot(
                kind='bar', 
                stacked=False, 
                color=['#A9A9A9', '#FFA500', '#90EE90', '#87CEEB'],  
                ax=ax, 
                edgecolor='black'
            )
            ax.set_title('Network Users Summary - Zones', fontsize=12)
            ax.set_xlabel('Zones')
            ax.set_ylabel('Number of Users')
            ax.legend(['Non users', 'Illegal Users', 'Legal Users', 'Total Population'])
            ax.set_xticklabels(user_summary_zone_plot.index, rotation=0)
            y_max = user_summary_zone_plot[['Non-user', 'Illegal', 'Legal', 'Total Population']].values.max()
            ax.set_yticks(range(0, int(y_max) + 5000, 5000))
            st.pyplot(fig)


       # Second graph for DMA
        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))  # Adjust figure size
            user_summary_dma_plot[['Non-user', 'Illegal', 'Legal', 'Total Population']].plot(
                kind='bar', 
                stacked=False, 
                color=['#A9A9A9', '#FFA500', '#90EE90', '#87CEEB'],  
                ax=ax, 
                edgecolor='black'
            )
            ax.set_title('Network Users Summary - DMAs', fontsize=12)
            ax.set_xlabel('DMAs')
            ax.set_ylabel('Number of Users')
            ax.legend(['Non users', 'Illegal Users', 'Legal Users', 'Total Population'])
            y_max = user_summary_dma_plot[['Non-user', 'Illegal', 'Legal', 'Total Population']].values.max()
            ax.set_yticks(range(0, int(y_max) + 5000, 5000))
            ax.set_xticklabels(user_summary_dma_plot.index, rotation=0)
            st.pyplot(fig)


    with tab2:
        st.markdown("### 📅 Monthly Water Consumption Calculation")
        
        #Seazonality factors
        month_factors = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dez'],
            'Factor': [0.07, 0.07, 0.07, 0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.08, 0.07, 0.07]
        }
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Number of days in each month
        # Create a DataFrame for the month factors
        df_factors = pd.DataFrame(month_factors)        
        
        st.markdown("#### Seasonal Variation Factor")        
        # Slider to adjust the factors' variation (0 = all equal, 1 = current, 2 = amplified)
        variation_factor = st.slider("Adjust Variation of Factors (0 = No Variation, 1 = Proposed, 2 = Amplified)", min_value=0.0, max_value=2.0, step=0.1, value=1.0)




        # Calculate monthly water consumption based on factors
        for i in range(len(days_in_month)):
            df_factors['Factor - Updated'] = (1 - variation_factor) * np.mean(df_factors["Factor"]) + variation_factor * df_factors["Factor"]
            df_factors['Monthly Daily Consumption - l/p/d'] = round(df_factors['Factor - Updated'] * avg_litres_per_person * 12)
            df_factors["Total Monthly Consumption - m3"] = round(df_factors['Monthly Daily Consumption - l/p/d'] * sum(df["Population"]) * days_in_month[i] / 1000, -2)
            
        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
             st.markdown("### Monthly Water Consumption")
                # Display the table with calculated values
             st.dataframe(df_factors, height=500)

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
        
        # Create two columns, with empty space in the left and right columns for centering
        col1, col2, col3 = st.columns([1, 4, 1])  # Make the center column wider for the plot
        
        with col2:
            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df_factors['Month'], df_factors['Total Monthly Consumption - m3'], color='skyblue')
            ax.set_ylabel('Monthly Water Consumption (m³)')
            ax.set_title('Monthly Water Consumption Distribution')
            ax.grid(True, linestyle='-', axis='y')
            st.pyplot(fig2)

    with tab3:
        
        # Prepare population and non-user percentages for DMAs
        population_dma = user_summary_dma['Total Population']
        non_users_dma = user_summary_dma['Non-user %'] / 100  # Convert percentage to a proportion

        # Monthly Daily Consumption from Seasonal Distribution
        monthly_consumption = df_factors['Monthly Daily Consumption - l/p/d']
        

        # Create an empty DataFrame to store the results
        water_demand_dma = pd.DataFrame(columns=['DMA'] + df_factors['Month'].tolist())

        # Calculate the water consumption for each DMA and month
        for dma in population_dma.index:
            # For each DMA, calculate monthly consumption
            dma_consumption = []
            for i, month in enumerate(df_factors['Month']):
                consumption_m3 = population_dma[dma] * (1 - non_users_dma[dma]) * monthly_consumption[i] * days_in_month[i] / 1000
                dma_consumption.append(round(consumption_m3, -2))  # Round to the nearest 100

            # Add the DMA and its monthly consumption to the DataFrame
            water_demand_dma.loc[len(water_demand_dma)] = [dma] + dma_consumption
            
        water_demand_dma.set_index('DMA', inplace=True)        
        water_demand_dma = water_demand_dma.transpose()
        
        
        
        
        
        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)
         

        with col1:
            st.markdown("### 💧 Water Consumption per Zone (Monthly)")
            st.dataframe(water_demand_dma, height=500)

        # with col2:
            # st.markdown("### 📉 Monthly Water Consumption Variation by Zone")
            # fig, ax = plt.subplots(figsize=(10, 4))
            # water_demand_dma.plot(x='DMA', y='Total', kind='bar', ax=ax, color='#87CEEB')
            # ax.set_ylabel('Cubic Metres')
            # ax.set_title('Monthly Water Consumption by Zone')
            # st.pyplot(fig)

    with tab4:
        st.markdown("### 🗺️ Interactive Maps with Google Satellite Basemap")
        
        # Dynamically generate a list of columns for the user to select from, excluding X (latitude) and Y (longitude)
        selectable_columns = [col for col in df.columns if col not in ['X', 'Y']]

        # Create a selectbox above the map
        selected_attribute = st.selectbox("Color points by:", options=["Zone", "DMA"], index=0)
        
        # Reorder the DataFrame so the selected attribute comes after lat/lon, but keep all columns
        cols = ['X', 'Y', selected_attribute] + [col for col in df.columns if col not in ['X', 'Y', selected_attribute]]
        df = df[cols]  # Dynamically reorder columns


        # Create a GeoDataFrame for processing
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
        gdf = gdf.set_crs(epsg=4326)

        # Calculate the center of the uploaded data
        center_lat, center_lon = gdf["Y"].mean(), gdf["X"].mean()

        # Determine a reasonable zoom level based on data spread
        lat_range = gdf["Y"].max() - gdf["Y"].min()
        lon_range = gdf["X"].max() - gdf["X"].min()
        zoom = 12 if max(lat_range, lon_range) < 1 else 10
        
        # Create dynamic KeplerGL configuration
        config_1 = {
            'version': 'v1',
            'config': {
                'mapState': {
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'zoom': 15
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
                                    "radius": 4,
                                    "opacity": 1,
                                },
                                "isVisible": True
                            }
                        }
                    ]
                }
            }
        }

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
                    'zoom': 16
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
