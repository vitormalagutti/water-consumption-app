import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from branca.element import Template, MacroElement



# Set up the Streamlit page with custom title and layout
st.set_page_config(page_title="Water Consumption Visualization", layout="wide")

# Main Title with description
st.title("🌊 Water Consumption and Building Visualization")
st.markdown("This app visualizes water consumption and building information, with breakdowns by zone and user type. Use the sidebar to provide average consumption details and view interactive maps, graphs, and tables.")

# File upload section with icon
st.markdown("### 📂 Upload Your Data File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Define the expected columns
    expected_columns = ["X", "Y", "Zone", "Status"]

    # Identify the columns that are in the CSV but also in the expected list
    valid_columns = [col for col in df.columns if col in expected_columns]

    # Select only the valid columns
    df = df[valid_columns]

    # Ensure all expected columns are present, even if some are missing in the input file
    missing_columns = [col for col in expected_columns if col not in df.columns]
    for col in missing_columns:
        df[col] = None  # Assign a default value or handle as required

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

    # Calculate population based on the number of buildings, average floors, and people per family
    total_population = len(filtered_df) * avg_floors * avg_people_per_family
    filtered_df['Population'] = avg_floors * avg_people_per_family

    # Calculate percentages of legal, illegal, and non-users per zone
    user_summary = filtered_df.pivot_table(values='Population', index='Zone', columns='User Type', aggfunc='sum', fill_value=0)
    user_summary['Total Population'] = user_summary.sum(axis=1)

    for user_type in ['Legal', 'Illegal', 'Non-user']:
        user_summary[f'{user_type} %'] = (user_summary[user_type] / user_summary['Total Population']) * 100

    user_summary = user_summary.round(1)
    overall_summary = user_summary[['Total Population', 'Legal %', 'Illegal %', 'Non-user %']].copy()

    # Streamlit tabs for organized visualization
    tab1, tab2, tab3 = st.tabs(["📊 Network Users Summary", "💧 Water Demand Model", "🗺️ Data Visualization"])

    with tab1:
        st.markdown("### 📊 User Type Summary with Estimated Population")
        st.dataframe(overall_summary)

        st.markdown("### 📈 Population by User Type")
        fig, ax = plt.subplots(figsize=(10, 4))
        user_summary[['Total Population', 'Legal', 'Illegal', 'Non-user']].plot(kind='bar', ax=ax)
        ax.set_ylabel('Population')
        ax.set_title('Population Distribution by Zone and User Type')
        st.pyplot(fig)

    with tab2:
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

        st.markdown("### 💧 Water Consumption per Zone (Monthly)")
        st.dataframe(water_per_zone)

        st.markdown("### 📉 Monthly Water Consumption Variation by Zone")
        fig, ax = plt.subplots(figsize=(10, 4))
        water_per_zone.plot(x='Zone', y='Cubic Metres', kind='bar', ax=ax, color='#87CEEB')
        ax.set_ylabel('Cubic Metres')
        ax.set_title('Monthly Water Consumption by Zone')
        st.pyplot(fig)

    with tab3:
        st.markdown("### 🗺️ Interactive Maps with Google Satellite Basemap")

        # Create a GeoDataFrame from the DataFrame (if needed for processing)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
        gdf = gdf.set_crs(epsg=4326)

        # Convert GeoDataFrame to DataFrame for Plotly
        df_plotly = pd.DataFrame(gdf.drop(columns="geometry"))

        # Plotting a Scatter Map using Plotly
        st.markdown("#### 🗺️ Map of Building Locations with Plotly")
        fig_scatter = px.scatter_mapbox(
            df_plotly,
            lat="Y",
            lon="X",
            color="User Type",
            zoom=10,
            mapbox_style="carto-positron",
            hover_name="Zone",
            title="Building Locations by User Type"
        )

        # Update the layout to adjust the map size
        fig_scatter.update_layout(
            width=700,  # Set the desired width in pixels
            height=1000  # Set the desired height in pixels
        )

        # Display the scatter map in Streamlit
        st.plotly_chart(fig_scatter)

        # Convert GeoDataFrame to DataFrame for Folium
        df_plotly = pd.DataFrame(gdf.drop(columns="geometry"))

        # Set up the Folium map with Google Satellite layer
        m = folium.Map(
            location=[gdf['Y'].mean(), gdf['X'].mean()], 
            zoom_start=14,  # Set your desired zoom level (higher numbers zoom in)
            width='100%',  # Set map width as a percentage or pixels
            height='1000px'  # Set map height as a percentage or pixels
        )
        # Add Google Satellite Tiles
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        # Create heatmaps based on selection
        if heatmap_type == "All Buildings":
            st.markdown("#### 🔥 Heatmap of All Building Locations")
            heat_data = [[row['Y'], row['X']] for idx, row in gdf.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)

        elif heatmap_type == "Illegal Connections":
            st.markdown("#### 🔥 Heatmap of Illegal Connections")
            heat_data_illegal = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Illegal'].iterrows()]
            HeatMap(heat_data_illegal, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

        elif heatmap_type == "Legal Connections":
            st.markdown("#### 🔥 Heatmap of Legal Connections")
            heat_data_legal = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Legal'].iterrows()]
            HeatMap(heat_data_legal, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

        elif heatmap_type == "Non-Users":
            st.markdown("#### 🔥 Heatmap of Non-Users")
            heat_data_non_users = [[row['Y'], row['X']] for idx, row in gdf[gdf['User Type'] == 'Non-user'].iterrows()]
            HeatMap(heat_data_non_users, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
 
        # Overlay specific building type with higher transparency
        for idx, row in gdf.iterrows():
            if heatmap_type == "All Buildings" or row['User Type'].lower() in heatmap_type.lower():
                folium.CircleMarker(
                    location=[row['Y'], row['X']],
                    radius=3,
                    color='black',
                    fill=True,
                    fill_opacity=0.6  # Higher transparency for markers
                ).add_to(m)

        # Add a layer control panel
        folium.LayerControl().add_to(m)

        # Display the Folium map in Streamlit
        folium_static(m)

else:
    st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")




