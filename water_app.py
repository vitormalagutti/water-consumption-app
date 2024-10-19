import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Set up the Streamlit page
st.title("Water Consumption and Building Visualization")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Check for required columns
    if 'X' in df.columns and 'Y' in df.columns and 'Zone' in df.columns and 'Status' in df.columns:
        
        # Step 1: Categorize Status into "legal", "illegal", and "non-user"
        df['User Type'] = df['Status'].apply(lambda x: 'Legal' if x == 'water meter' else ('Illegal' if x == 'illegal connection' else ('Non-user' if x == 'non user' else 'No Data')))

        # Calculate overall percentages of each user type
        user_counts = df['User Type'].value_counts(normalize=True) * 100
        st.write("### Overall User Type Percentages")
        st.dataframe(user_counts)

        # Calculate percentages per zone
        user_per_zone = df.groupby('Zone')['User Type'].value_counts(normalize=True).unstack().fillna(0) * 100
        st.write("### User Type Percentages per Zone")
        st.dataframe(user_per_zone)

        # Input section for averages
        st.sidebar.header("Average Inputs")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=0.0, step=0.1)
        avg_litres_per_person = st.sidebar.number_input("Average Litres per Person per Day", min_value=0.0, step=0.1)

        # Display total cubic meters needed if averages are provided
        if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
            # Calculate total buildings with legal or illegal connections
            total_buildings = len(df[df['User Type'].isin(['Legal', 'Illegal'])])
            total_people = total_buildings * avg_floors * avg_people_per_family
            total_cumecs_needed = total_people * avg_litres_per_person / 1000

            # Calculate water consumption per zone
            df['People'] = avg_floors * avg_people_per_family
            df['Cubic Metres'] = df['People'] * avg_litres_per_person / 1000
            water_per_zone = df[df['User Type'].isin(['Legal', 'Illegal'])].groupby('Zone')['Cubic Metres'].sum().reset_index()
            water_per_zone['Percentage'] = (water_per_zone['Cubic Metres'] / total_cumecs_needed) * 100

            st.write(f"### Total cubic metres needed per day: {total_cumecs_needed:.2f}")
            st.write("### Water Consumption per Zone")
            st.dataframe(water_per_zone)

        # Let the user choose which column to use for categorization
        category = st.sidebar.selectbox("Choose a characteristic to display on the map", options=['Zone', 'Status', 'User Type'])

        # Assign a color for each unique value in the chosen category
        unique_values = df[category].unique()
        color_palette = [
            'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
            'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
            'gray', 'black', 'lightgray'
        ]
        color_dict = {value: color_palette[i % len(color_palette)] for i, value in enumerate(unique_values)}

        # Create a folium map centered around the average coordinates of the data with satellite basemap
        map_center = [df['Y'].mean(), df['X'].mean()]
        my_map = folium.Map(
            location=map_center, 
            zoom_start=9,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='ESRI World Imagery'
        )

        # Add markers for each point with a CircleMarker and smaller size
        for _, row in df.iterrows():
            circle_color = color_dict.get(row[category], 'gray')  # Use 'gray' as default if no match
            folium.CircleMarker(
                location=[row['Y'], row['X']],
                radius=4,  # This defines the size of the dot; you can adjust this as needed
                color=circle_color,
                fill=True,
                fill_color=circle_color,
                fill_opacity=0.7,
                popup=f"ID: {row['ID']}, Zone: {row['Zone']}, Status: {row['Status']}"
            ).add_to(my_map)

        # Add a legend to the map for the selected category
        legend_html = f'''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: {30 * len(unique_values)}px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white; padding: 10px;">
            <h4 style="margin-bottom:10px;">{category} Legend</h4>
            '''
        for value, color in color_dict.items():
            legend_html += f'<i style="background:{color}; width:20px; height:20px; float:left; margin-right:5px; border-radius:50%;"></i>{value}<br>'
        legend_html += '</div>'
        my_map.get_root().html.add_child(folium.Element(legend_html))

        # Display the map
        st.write("### Map of Building Locations with Satellite View and Legend")
        st_data = st_folium(my_map, width=700, height=500)

    else:
        st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
