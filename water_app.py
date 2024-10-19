import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx  # Import contextily for basemap support

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

    # Check for required columns
    if 'X' in df.columns and 'Y' in df.columns and 'Zone' in df.columns and 'Status' in df.columns:
        
        # Step 1: Categorize Status into "legal", "illegal", and "non-user"
        df['User Type'] = df['Status'].apply(lambda x: 'Legal' if x == 'water meter' else ('Illegal' if x == 'illegal connection' else ('Non-user' if x == 'non user' else 'No Data')))

        # Filter out rows with "No Data" in User Type for percentage calculations
        filtered_df = df[df['User Type'] != 'No Data']

        # Sidebar inputs section with sliders only for the average litres per person
        st.sidebar.header("🔧 Average Inputs")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.0)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.0)
        avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=1000, step=10, value=150)

        # Calculate estimated population using the original calculation logic
        filtered_df['Population'] = avg_floors * avg_people_per_family

        # Group and sum up the population by Zone and User Type
        user_summary = filtered_df.pivot_table(values='Population', index='Zone', columns='User Type', aggfunc='sum', fill_value=0)
        user_summary['Total Population'] = user_summary.sum(axis=1)

        # Round population values to the nearest hundreds
        user_summary = user_summary.round(-2)

        # Calculate water consumption per zone and overall consumption (for monthly values)
        if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
            total_buildings = len(filtered_df[filtered_df['User Type'].isin(['Legal', 'Illegal'])])
            total_people = total_buildings * avg_floors * avg_people_per_family
            total_cumecs_needed = total_people * avg_litres_per_person / 1000 * 30  # Monthly consumption

            filtered_df['People'] = avg_floors * avg_people_per_family
            filtered_df['Cubic Metres'] = filtered_df['People'] * avg_litres_per_person / 1000 * 30

            water_per_zone = filtered_df[filtered_df['User Type'].isin(['Legal', 'Illegal'])].groupby('Zone').agg({
                'Cubic Metres': 'sum',
                'People': 'sum'
            }).rename(columns={'People': 'Estimated Population'}).reset_index()
            total_row = pd.DataFrame([['Total', water_per_zone['Cubic Metres'].sum(), water_per_zone['Estimated Population'].sum()]],
                                     columns=water_per_zone.columns)
            water_per_zone = pd.concat([water_per_zone, total_row], ignore_index=True)

        # Streamlit tabs for organized visualization
        tab1, tab2, tab3 = st.tabs(["📊 Network Users Summary", "💧 Water Demand Model", "🗺️ Data Visualization"])

        with tab1:
            st.markdown("### 📊 User Type Summary with Estimated Population")
            st.dataframe(user_summary)

            st.markdown("### 📈 Population by User Type")
            fig, ax = plt.subplots(figsize=(10, 4))
            user_summary.plot(kind='bar', y=['Total Population', 'Legal', 'Illegal', 'Non-user'], ax=ax)
            ax.set_ylabel('Population')
            ax.set_title('Population Distribution by Zone and User Type')
            st.pyplot(fig)

        with tab2:
            st.markdown("### 💧 Water Consumption per Zone (Monthly)")
            st.dataframe(water_per_zone)

            st.markdown("### 📉 Monthly Water Consumption Variation by Zone")
            fig, ax = plt.subplots(figsize=(10, 4))
            water_per_zone.plot(x='Zone', y='Cubic Metres', kind='bar', ax=ax, color='#87CEEB')
            ax.set_ylabel('Cubic Metres')
            ax.set_title('Monthly Water Consumption by Zone')
            st.pyplot(fig)

        with tab3:
            st.markdown("### 🗺️ Map and Heatmaps of Building Locations")

            # Create GeoDataFrame from the DataFrame
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
            gdf = gdf.set_crs(epsg=4326)

            # Display the GeoDataFrame on a map with a satellite basemap
            st.markdown("#### 🗺️ Map of Building Locations with Satellite Basemap")
            fig, ax = plt.subplots(figsize=(10, 6))
            gdf.plot(ax=ax, color='blue', markersize=5, alpha=0.6, legend=True)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)  # Satellite basemap
            ax.set_title("Building Locations on Satellite Map")
            st.pyplot(fig)

            # Create a heatmap for total buildings
            st.markdown("#### 🔥 Heatmap of Total Buildings")
            fig, ax = plt.subplots(figsize=(10, 6))
            gdf.plot(ax=ax, color='red', alpha=0.5, markersize=20)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
            ax.set_title("Heatmap of Total Buildings")
            st.pyplot(fig)

            # Create a heatmap for illegal connections
            st.markdown("#### 🔥 Heatmap of Illegal Connections")
            illegal_gdf = gdf[gdf['User Type'] == 'Illegal']
            fig, ax = plt.subplots(figsize=(10, 6))
            illegal_gdf.plot(ax=ax, color='orange', alpha=0.5, markersize=20)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
            ax.set_title("Heatmap of Illegal Connections")
            st.pyplot(fig)

    else:
        st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
