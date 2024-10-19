import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set up the Streamlit page with custom title and layout and white background
st.set_page_config(page_title="Water Consumption Visualization", layout="wide")

# Apply custom CSS to set the background color to white
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

        # Calculate overall percentages of each user type (excluding "No Data")
        user_counts = filtered_df['User Type'].value_counts(normalize=True) * 100

        # Calculate percentages per zone (excluding "No Data")
        user_per_zone = filtered_df.groupby('Zone')['User Type'].value_counts(normalize=True).unstack().fillna(0) * 100

        # Combine the overall and per-zone percentages into a single table (raw for plotting)
        combined_table_raw = user_per_zone.copy()
        combined_table_raw.loc['Overall'] = user_counts

        # Create a formatted version of the combined table for display
        combined_table_formatted = combined_table_raw.copy().applymap(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")

        # Sidebar inputs section with sliders only for the average litres per person
        st.sidebar.header("🔧 Average Inputs")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.0)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.0)
        avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=500, step=10, value=150)

        # Display total cubic meters needed if averages are provided
        if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
            total_buildings = len(filtered_df[filtered_df['User Type'].isin(['Legal', 'Illegal'])])
            total_people = total_buildings * avg_floors * avg_people_per_family
            total_cumecs_needed = total_people * avg_litres_per_person / 1000

            filtered_df['People'] = avg_floors * avg_people_per_family
            filtered_df['Cubic Metres'] = filtered_df['People'] * avg_litres_per_person / 1000

            water_per_zone = filtered_df[filtered_df['User Type'].isin(['Legal', 'Illegal'])].groupby('Zone')['Cubic Metres'].sum().reset_index()
            water_per_zone['Percentage'] = (water_per_zone['Cubic Metres'] / total_cumecs_needed) * 100
            water_per_zone['Percentage'] = water_per_zone['Percentage'].apply(lambda x: f"{x:.1f}%")

        # Streamlit tabs for organized visualization
        tab1, tab2, tab3 = st.tabs(["📊 Tables", "📈 Graph", "🗺️ Map"])

        with tab1:
            st.markdown("### 📊 User Type Percentages (Overall and Per Zone)")
            styled_combined_table = combined_table_formatted.style.set_properties(**{
                'background-color': '#f5f5f5',
                'color': 'black',
                'border-color': '#cccccc'
            })
            st.dataframe(styled_combined_table)

            if avg_floors > 0 and avg_people_per_family > 0 and avg_litres_per_person > 0:
                st.markdown(f"### 💧 Total Cubic Metres Needed per Day: **{total_cumecs_needed:.2f}**")
                st.markdown("### 🏢 Water Consumption per Zone")
                styled_water_per_zone = water_per_zone.style.set_properties(**{
                    'background-color': '#f0f0f0',
                    'color': 'black',
                    'border-color': '#cccccc'
                })
                st.dataframe(styled_water_per_zone)

        with tab2:
            st.markdown("### 📈 User Type Percentages Overview")
            fig, ax = plt.subplots(figsize=(10, 6))
            combined_table_raw.drop('Overall').plot(kind='bar', stacked=True, ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
            ax.set_ylabel('Percentage')
            ax.set_title('User Type Percentages by Zone')
            st.pyplot(fig)

            st.markdown("### 💧 Water Consumption Variation per Zone")
            if 'Cubic Metres' in water_per_zone.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                water_per_zone['Cubic Metres'] = pd.to_numeric(water_per_zone['Cubic Metres'], errors='coerce')
                water_per_zone.plot(x='Zone', y='Cubic Metres', kind='bar', ax=ax, color='#87CEEB')
                ax.set_ylabel('Cubic Metres')
                ax.set_title('Water Consumption by Zone')
                st.pyplot(fig)

        with tab3:
            st.markdown("### 🗺️ Map of Building Locations with Satellite View")
            category = st.sidebar.selectbox("Choose a characteristic to display on the map", options=['Zone', 'Status', 'User Type'], index=0)

            # Assume that the coordinates are in WGS84 format
            st.markdown("**Note**: Coordinates are assumed to be in WGS84 (latitude/longitude).")

            # Create a Plotly map with scatter_mapbox
            fig = px.scatter_mapbox(
                df,
                lat='Y',
                lon='X',
                color=category,
                hover_name='ID',
                hover_data={'Zone': True, 'Status': True},
                color_discrete_sequence=px.colors.qualitative.Safe,
                zoom=9,
                height=500
            )

            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 0, "l": 0, "b": 0}
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("The uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', or 'Status'.")
