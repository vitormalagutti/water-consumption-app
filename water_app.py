import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Set up the Streamlit page
st.title("Water Consumption and Building Map Visualization")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Check for required columns
    if 'X' in df.columns and 'Y' in df.columns and 'Zone' in df.columns and 'Status' in df.columns:
        # Input section for averages
        st.sidebar.header("Average Inputs")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=0.0, step=0.1)
        avg_litres_per_person = st.sidebar.number_input("Average Litres per Person per Day", min_value=0.0, step=0.1)

        # Let the user choose which column to use for categorization
        category = st.sidebar.selectbox("Choose a characteristic to display on the map", options=['Zone', 'Status'])

        # Get unique values for the chosen category
        unique_values = df[category].unique()

        # Display total litres needed if averages are provided
        if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
            total_buildings = len(df)
            total_people = total_buildings * avg_floors * avg_people_per_family
            total_litres_needed = total_people * avg_litres_per_person
            st.write(f"### Total litres needed per day: {total_litres_needed:.2f}")

        # Generate multiple maps based on selected characteristic
        for value in unique_values:
            st.write(f"### Map for {category}: {value}")

            # Filter data for the current category value
            filtered_df = df[df[category] == value]

            # Create a folium map centered around the average coordinates of the filtered data
            map_center = [filtered_df['Y'].mean(), filtered_df['X'].mean()]
            my_map = folium.Map(location=map_center, zoom_start=12)
            
            # Add markers for each point in the filtered data
            marker_cluster = MarkerCluster().add_to(my_map)
            for _, row in filtered_df.iterrows():
                folium.Marker(
                    location=[row['Y'], row['X']],
                    popup=f"ID: {row['ID']}, Zone: {row['Zone']}, Status: {row['Status']}"
                ).add_to(marker_cluster)

            # Display the map
            st_data = st_folium(my_map, width=700, height=500)

    else:
        st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
