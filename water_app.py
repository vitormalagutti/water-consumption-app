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

        # Assign a color for each unique value in the chosen category
        unique_values = df[category].unique()
        color_palette = [
            'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
            'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
            'gray', 'black', 'lightgray'
        ]
        color_dict = {value: color_palette[i % len(color_palette)] for i, value in enumerate(unique_values)}

        # Display total litres needed if averages are provided
        if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
            total_buildings = len(df)
            total_people = total_buildings * avg_floors * avg_people_per_family
            total_litres_needed = total_people * avg_litres_per_person
            st.write(f"### Total litres needed per day: {total_litres_needed:.2f}")

        # Create a folium map centered around the average coordinates of the data with satellite basemap
        map_center = [df['Y'].mean(), df['X'].mean()]
        my_map = folium.Map(
            location=map_center, 
            zoom_start=12,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='ESRI World Imagery'
        )

        # Add markers for each point with a different color based on the selected category
        for _, row in df.iterrows():
            marker_color = color_dict.get(row[category], 'gray')  # Use 'gray' as default if no match
            folium.Marker(
                location=[row['Y'], row['X']],
                popup=f"ID: {row['ID']}, Zone: {row['Zone']}, Status: {row['Status']}",
                icon=folium.Icon(color=marker_color)
            ).add_to(my_map)

        # Add a legend to the map for the selected category
        legend_html = f'''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: {25 * len(unique_values)}px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white; padding: 10px;">
            <h4 style="margin-bottom:10px;">{category} Legend</h4>
            '''
        for value, color in color_dict.items():
            legend_html += f'<i style="background:{color}; width:15px; height:15px; float:left; margin-right:5px;"></i>{value}<br>'
        legend_html += '</div>'
        my_map.get_root().html.add_child(folium.Element(legend_html))

        # Display the map
        st.write("### Map of Building Locations with Satellite View")
        st_data = st_folium(my_map, width=700, height=500)

    else:
        st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
