import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import pydeck as pdk
import json
import matplotlib.ticker as ticker
import io
import re
import seaborn as sns
from dateutil import parser
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

# Streamlit tabs for organized visualization
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📂 Input Files Upload", "📊 Network Users Summary", "📅 Seasonal Water Demand Distribution", "💧 Water Demand Model", "💰 Billed Water Analysis", "🗺️ Data Visualization"])

def convert_to_csv(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        try:
            if file_extension == 'xlsx':
                return pd.read_excel(uploaded_file, sheet_name=0)  # Ensure you're reading the correct sheet
            elif file_extension == 'csv':
                return pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

def process_volume_or_value_file(uploaded_file):
    """
    This function processes volume or value files by ensuring that a 'Subscription Number' column
    is present, identifying date columns in various formats, and converting them to a standard format.
    """
    if uploaded_file is not None:
        df = convert_to_csv(uploaded_file)

        # Ensure that we have a 'Subscription Number' column
        if 'Subscription Number' not in df.columns:
            st.error("The file does not contain a 'Subscription Number' column.")
            return None

        # Function to identify and standardize date columns
        def standardize_date(col):
            col_str = str(col)  # Ensure the column is a string
            try:
                # If the column contains a time part, split it and use only the date part
                if ' ' in col_str:
                    col_str = col_str.split(' ')[0]

                # Try parsing the column as a date
                parsed_date = parser.parse(col_str, fuzzy=True, dayfirst=False)

                # Convert the date to 'mm/yy' format
                return parsed_date.strftime('%m/%y')

            except (ValueError, TypeError):
                return None  # Return None if the column is not a date

        # Separate out date and non-date columns
        standardized_date_columns = []
        non_date_columns = []

        for col in df.columns:
            standardized_date = standardize_date(col)
            if standardized_date:
                standardized_date_columns.append(standardized_date)
            elif col == 'Subscription Number':
                non_date_columns.append(col)

        # Keep only 'Subscription Number' and valid date columns
        df = df[non_date_columns + [col for col in df.columns if standardize_date(col)]]

        # Rename columns with standardized date formats
        if len(standardized_date_columns) == len(df.columns) - 1:
            df.columns = ['Subscription Number'] + standardized_date_columns
        else:
            st.warning("Some columns were not recognized as dates. Extra columns have been removed.")

        # Check for non-numeric values in the date columns
        for col in standardized_date_columns:
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()]
            if not non_numeric.empty:
                st.warning(f"Non-numeric values found in column '{col}' at rows: {non_numeric.index.tolist()}")

        # Replace blanks (NaN) with 0
        df[standardized_date_columns] = df[standardized_date_columns].fillna(0)

        return df
    else:
        return None

def process_block_subscription_file(uploaded_file):
    """
    This function processes the uploaded Block Number - Subscription Number file, ensuring that the expected 
    columns ['Block Number', 'Subscription Number'] are present, and that the values in both columns are numeric. 
    If non-numeric values are found, a warning is raised. Extra columns will be dropped.
    """
    if uploaded_file is not None:
        df = convert_to_csv(uploaded_file)

        # Ensure that we have the required columns
        required_columns = ['Block Number', 'Subscription Number']
        if not all(col in df.columns for col in required_columns):
            st.error(f"The file must contain the columns: {required_columns}.")
            return None

        # Drop any columns not in required_columns
        df = df[required_columns]

        # Check for non-numeric values in 'Block Number' and 'Subscription Number'
        for col in required_columns:
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()]
            if not non_numeric.empty:
                st.warning(f"Non-numeric values found in column '{col}' at rows: {non_numeric.index.tolist()}")

        # Convert blanks or NaNs to 0 (if needed)
        df[required_columns] = df[required_columns].fillna(0)

        # Optionally convert the columns to integers after replacing NaNs
        df[required_columns] = df[required_columns].astype(int)

        return df
    else:
        return None

def align_and_calculate_percentage_separately(water_demand_df, billed_df):
    # Convert the columns in billed_df to the comparable month and year format (e.g., 'mm/yy')
    billed_df.columns = pd.to_datetime(billed_df.columns, errors='coerce').strftime('%m/%y')

    # Extract only the month part for water demand data (assuming same year for water demand)
    water_demand_df.columns = pd.to_datetime(water_demand_df.columns + " 2023", format='%b %Y', errors='coerce').strftime('%m')

    # Prepare a result dictionary to store percentages for each year-month separately
    percentage_results = {}

    # Loop over the years in the billed data
    for year in pd.to_datetime(billed_df.columns, format='%m/%y', errors='coerce').year.unique():
        # Filter billed_df for the specific year
        billed_for_year = billed_df[[col for col in billed_df.columns if pd.to_datetime(col, errors='coerce').year == year]]
        
        # Extract the months (without the year) to align with water demand
        billed_df_months = pd.to_datetime(billed_for_year.columns, errors='coerce').strftime('%m')

        # Align with water demand by filtering only common months
        common_months = billed_df_months.intersection(water_demand_df.columns)

        # Slice data to include only common months
        water_demand_for_months = water_demand_df[common_months]
        billed_for_months = billed_for_year[[col for col, month in zip(billed_for_year.columns, billed_df_months) if month in common_months]]

        # Calculate percentage billed
        percentage_billed = (billed_for_months / water_demand_for_months) * 100

        # Fill any NaN or infinite values with 0, and ensure numeric type
        percentage_billed = percentage_billed.fillna(0).replace([float('inf'), -float('inf')], 0).astype(float)
        
        # Store the result for the specific year
        percentage_results[f'Year {year}'] = percentage_billed

    return percentage_results

# Functions to merge water demand and billed based on month
def add_month_column_from_index(billed_df):
    # Temporarily reset index
    billed_df_temp = billed_df.reset_index()
    # Add a Month column based on the current index, without permanently changing the index
    billed_df_temp['Month'] = pd.to_datetime(billed_df_temp['index'], format='%m/%y', errors='coerce').dt.strftime('%b')
    # Set the index back to its original column, if desired
    billed_df_temp.set_index('index', inplace=True)
    # Rename the index to the original name if needed
    billed_df_temp.index.name = billed_df.index.name
    return billed_df_temp

def join_billed_with_demand(billed_df, demand_df):
    # Temporarily reset index on billed_df and demand_df to expose 'Month' column for merging
    billed_df_temp = billed_df.reset_index()
    demand_df = demand_df.reset_index()  # 'Month' becomes a column here in demand_df

    # Ensure the first column in demand_df is now 'Month'
    if 'index' in demand_df.columns:
        demand_df.rename(columns={'index': 'Month'}, inplace=True)

    # Perform the merge on 'Month' column
    merged_df = pd.merge(billed_df_temp, demand_df, on='Month', how='left', suffixes=('', '_demand'))

    # Restore the original index on billed_df (after merge)
    merged_df.set_index("index", inplace=True)
    merged_df = merged_df.drop(columns="Month")
    return merged_df

def calculate_percentage_billed(merged_df, n):
    # Store the original DMA/Zone numbers for accurate naming in the % columns
    original_names = [merged_df.columns[i] for i in range(n)]

    # Rename the first n columns to "Volume Billed - DMA/Zone number"
    for i in range(n):
        merged_df.rename(columns={merged_df.columns[i]: f"Volume Billed - {original_names[i]}"}, inplace=True)

    # Rename the next n columns to "Water Demand - DMA/Zone number"
    for i in range(n, 2 * n):
        merged_df.rename(columns={merged_df.columns[i]: f"Water Demand - {original_names[i - n]}"}, inplace=True)

    # Calculate the percentage of billed volumes compared to demand using renamed columns
    for i in range(n):
        billed_column = merged_df.columns[i]  # Volume Billed column
        demand_column = merged_df.columns[i + n]  # Corresponding Water Demand column
        merged_df[f'% Billed - {original_names[i]}'] = round((merged_df[billed_column] / merged_df[demand_column]) * 100, 1)

    # Replace NaN or infinite values with zeroes
    merged_df.replace([float('inf'), -float('inf')], 0, inplace=True)
    merged_df.fillna(0, inplace=True)

    return merged_df

def plot_multiple_demand_billed(df, n, selected_dmas_zones, title="Water Demand vs Billed Volumes"):
    # Ensure `selected_dmas_zones` is a list of strings to match column suffixes
    selected_dmas_zones = [str(zone) for zone in selected_dmas_zones]
    
    # Identify demand and billed columns, filtering based on selected DMAs/Zones
    demand_columns = [col for col in df.columns if "Water Demand" in col and col.split(" - ")[-1] in selected_dmas_zones]
    billed_columns = [col for col in df.columns if "Volume Billed" in col and col.split(" - ")[-1] in selected_dmas_zones]

    # Use the DataFrame index as the x-axis labels (assuming it's the dates)
    x_labels = df.index.astype(str)  # Convert index to strings for Plotly compatibility
    positions = np.arange(len(x_labels))  # Positions should match the number of index entries (rows)

    # Create a plotly figure
    fig = go.Figure()

    # Define a color palette for consistency between bars and lines
    colors = px.colors.qualitative.Set1  # A good default color palette with distinct colors

    # Plot Demand Bars for each demand column
    for i, demand_column in enumerate(demand_columns):
        # Extract the DMA/Zone number from the column name
        dma_zone_number = demand_column.split(" - ")[-1]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=df[demand_column],
                name=f"Demand - {dma_zone_number}",
                opacity=0.6,
                marker_color=colors[i % len(colors)],
                )
        )

    # Plot Billed Percentages as lines
    for i, billed_column in enumerate(billed_columns):
        # Extract the DMA/Zone number from the column name
        dma_zone_number = billed_column.split(" - ")[-1]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df[billed_column],
                mode='lines+markers',
                name=f"Billed - {dma_zone_number}",
                line=dict(color=colors[i % len(colors)], dash='dash'),
                marker=dict(size=8)
            )
        )

    # Set labels and title
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volume - m3",
        barmode='group',  # Bars are overlayed on top of each other, similar to side-by-side
        legend=dict(title="Legend", orientation="v"),
    )

    # Add formatting to y-axis
    fig.update_yaxes(tickformat=",.0f")

    # Show the plot in Streamlit
    st.plotly_chart(fig)



def calculate_expected_egp_and_percentage(merged_df, avg_price, n):
    # Rename the first n columns to "Total Billed EGP - (DMA number)"
    for i in range(n):
        dma_number = merged_df.columns[i]  # Use current column name as DMA number
        merged_df.rename(columns={dma_number: f'Total Billed EGP - {dma_number}'}, inplace=True)

    # Step 1: Calculate Expected EGP Value for each demand column
    for column in merged_df.columns:
        if column.endswith('_demand'):  # Ensure we are working with demand columns
            base_column = column.replace('_demand', '')  # Get the original column name
            dma_number = base_column  # Assuming the DMA identifier is the same as the column name
            # Calculate expected EGP value
            merged_df[f'Expected EGP Value - {dma_number}'] = merged_df[column] * avg_price

    # Step 2: Calculate the percentage billed in EGP for each DMA
    for i in range(n):
        billed_column = merged_df.columns[i]  # Get the renamed billed column
        dma_number = billed_column.replace('Total Billed EGP - ', '')  # Extract DMA number
        expected_column = f'Expected EGP Value - {dma_number}'

        if expected_column in merged_df.columns:
            merged_df[f'% Billed in EGP - {dma_number}'] = round((merged_df[billed_column] / merged_df[expected_column]) * 100, 1)

    # Replace NaN or infinite values with zeroes
    merged_df = merged_df.replace([float('inf'), -float('inf')], 0).fillna(0)

    # Step 3: Drop the '_demand' columns
    demand_columns = [col for col in merged_df.columns if col.endswith('_demand')]
    merged_df.drop(columns=demand_columns, inplace=True)

    return merged_df


def plot_billed_vs_expected(df, n, selected_dmas_zones, start_date_dt, end_date_dt, title="Total Billed vs Expected EGP Values"):

    selected_dmas_zones = [str(zone) for zone in selected_dmas_zones]
    # Filter columns based on selected DMAs/Zones
    billed_columns = [col for col in df.columns[:n] if col.split(' ')[-1] in selected_dmas_zones]
    expected_columns = [col for col in df.columns[n:2*n] if col.split(' ')[-1] in selected_dmas_zones]

    # Filter DataFrame to only include selected columns
    filtered_df = df[billed_columns + expected_columns]
    filtered_df.index = pd.to_datetime(filtered_df.index, format='%m/%y')
    # Filter based on the selected date range
    filtered_df = filtered_df[(filtered_df.index >= start_date_dt) & (filtered_df.index <= end_date_dt)]

    start_date_dt
    df.index
    # Use the DataFrame index as the x-axis labels (assuming it's the dates)
    x_labels = df.index

    # Create a Plotly figure
    fig = go.Figure()

    # Define a color palette for consistency between bars and lines
    colors = px.colors.qualitative.Set1  # A good default color palette with distinct colors

    # Plot Expected as bars
    for i, expected_column in enumerate(expected_columns):
        dma_zone = expected_column.split(' ')[-1]  # Extract DMA/Zone number for label
        fig.add_trace(
            go.Bar(
                x=x_labels, 
                y=df[expected_column], 
                name=f"Expected Billing - {dma_zone}", 
                marker_color=colors[i % len(colors)],
                opacity=0.6
            )
        )

    # Plot Total Billed as lines, reusing the colors from bars for consistency
    for i, billed_column in enumerate(billed_columns):
        dma_zone = billed_column.split(' ')[-1]  # Extract DMA/Zone number for label
        fig.add_trace(
            go.Scatter(
                x=x_labels, 
                y=df[billed_column], 
                mode='lines+markers',
                name=f"Total Billed - {dma_zone}", 
                line=dict(color=colors[i % len(colors)], dash='dash'),
                marker=dict(size=8)
            )
        )

    # Update layout for readability and interactivity
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", tickmode='array', tickvals=x_labels),
        yaxis=dict(title="EGP £", tickformat=","),
        barmode='group',  # Adjust for side-by-side bars
        template="plotly_white",  # Use a clean white theme
        legend=dict(title="Series", x=1.05, y=1),  # Place legend outside the plot
        margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins for better layout
    )
    
    # Render the plot using Plotly in Streamlit
    st.plotly_chart(fig)



with tab1:

    # File upload section with icon
    st.markdown("### 📂 Upload Your Buildings File")
    st.markdown("Please upload a .csv file with the specific columns' names X, Y, Block Number, Zone, DMA, and Status")
    st.markdown("Values accepted for the Status column are water meter, illegal connection, non user, and blank cells")
    buildings_file = st.file_uploader("Choose a CSV file", type=["csv", "xlsx"])
    
    st.markdown("### 📂 Upload Block Number - Subscription Number File")
    st.markdown("It must include the following column names [Block Number, Subscription Number]")
    correlation_file = st.file_uploader("Choose a CSV file for Block Number - Subscription Number", type=["csv", "xlsx"])

    st.markdown("### 📂 Upload Your Billed Value File")
    st.markdown("It must include the following column names [Subscription Number, mm/yy, mm/yy, ...]. Check carefully the date format!")
    value_file = st.file_uploader("Choose a CSV file for Billed Value", type=["csv", "xlsx"])

    st.markdown("### 📂 Upload Your Billed Volume File")
    st.markdown("It must include the following column names [Subscription Number, mm/yy, mm/yy, ...]. Check carefully the date format!")
    volume_file = st.file_uploader("Choose a CSV file for Billed Volume", type=["csv", "xlsx"])

    if buildings_file:
        # Read the CSV file
        df = convert_to_csv(buildings_file)

        # Define the expected columns
        expected_columns = ["X", "Y", "Zone", "Block Number", "DMA", "Status"]

        # Select valid columns and fill missing ones with default values
        df = df[[col for col in df.columns if col in expected_columns]]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        # Replace NaN values with empty strings in the 'Status' column
        df['Status'] = df['Status'].fillna('')

        # Define the expected variations for each user type
        expected_legal = ['water meter', 'water_meter', 'water-meter', 'meter', 'water metre']
        expected_illegal = ['illegal connection', 'illegal_connection', 'illegal-connection']
        expected_non_user = ['non-user', 'non_user', 'non user']
        expected_values = expected_legal + expected_illegal + expected_non_user + ['', ' ']

        # Step 2: Categorize Status into "legal", "illegal", "non-user", and "No Data" only for blanks
        df['User Type'] = df['Status'].apply(
            lambda x: 'Legal' if x.strip().lower() in expected_legal else (
                'Illegal' if x.strip().lower() in expected_illegal else (
                    'Non-user' if x.strip().lower() in expected_non_user else (
                        'No Data' if x.strip() == '' else 'Unexpected'
                    )
                )
            )
        )

        # Identify rows where 'Status' does not match the expected values and are categorized as 'Unexpected'
        unexpected_values = df[df['User Type'] == 'Unexpected']

        # If there are unexpected values, raise a warning
        if not unexpected_values.empty:
            unexpected_unique = unexpected_values['Status'].unique()  # Unique unexpected values
            unexpected_rows = unexpected_values.index.tolist()  # Get the index (line numbers) of unexpected values

            st.warning(f"Warning: Found unexpected values in the 'Status' column: {unexpected_unique}")
            st.write(f"Expected values for the 'Status' column are: {expected_values}.")
            st.write(f"These unexpected values were found in the following rows: {unexpected_rows}")
            st.write("You can either proceed without these records or adjust your file to include only the expected values.")
            st.write("These records will not be processed if you choose to proceed.")

            # Optionally filter out rows with unexpected values
            df = df[df['User Type'] != 'Unexpected']

        # Filter out rows with "No Data" in User Type for percentage calculations
        filtered_df = df[df['User Type'] != 'No Data']
        buildings_df = df

        available_options = []
        if 'Zone' in df.columns:
            available_options.append("Zone")
        if 'DMA' in df.columns:
            available_options.append("DMA")

        # Sidebar inputs section with sliders only for the average litres per person
        st.sidebar.header("🔧 Assumptions")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.63)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.65)
        avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=300, step=5, value=150)
        
        # Sidebar menu for choosing DMA or Zone
        st.sidebar.header("📊 Data Visualization Options")
        if available_options:
            visualization_type = st.sidebar.selectbox("Choose visualization type:", available_options)
        else:
            st.sidebar.write("No data available for Zone or DMA visualization.")
        
        # Sidebar multiselect for filtering DMAs/Zones
        if volume_file and value_file and correlation_file and buildings_file:
            # Extract unique DMA values from the filtered DataFrame and convert to integers
            available_dmas_zones = [int(value) for value in filtered_df[visualization_type].unique() if not pd.isna(value)]
            selected_dmas_zones = st.sidebar.multiselect("Select DMAs/Zones to Display", available_dmas_zones, default=available_dmas_zones)

            # Sidebar input to select analysis type
            st.sidebar.header("💰 Billing Analysis")
            billing_type = st.sidebar.selectbox("Choose Billing Analysis Type", ["Volume (m3) Analysis", "Value (EGP £) Analysis"])

        # Sidebar menu for HeatMap Options
        st.sidebar.header("🔍 Heatmap Options")
        heatmap_type = st.sidebar.selectbox(
            "Choose a heatmap to display:",
            ["All Buildings", "Illegal Connections", "Legal Connections", "Non-Users"]
        )

        # Calculate total population using the full DataFrame (df), aggregated by Zone and DMA
        df['Population'] = avg_floors * avg_people_per_family
        total_population_by_zone = df.groupby('Zone')['Population'].sum() if 'Zone' in df.columns else None
        total_population_by_dma = df.groupby('DMA')['Population'].sum() if 'DMA' in df.columns else None


        if 'visualization_type' in locals():
            if visualization_type == "Zone" and "Zone" in available_options:

                # Extract unique DMA and Zone values
                unique_zones = [int(zone) for zone in filtered_df['Zone'].unique() if pd.notnull(zone)]
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
                legal_percentage_zone = (legal_count / total_known_users) * 100
                illegal_percentage_zone = (illegal_count / total_known_users) * 100
                non_user_percentage_zone = (non_user_count / total_known_users) * 100

                # Create a DataFrame to store the results
                user_summary_zone = pd.DataFrame({
                    'Total Population': total_population_by_zone,
                    'Legal %': legal_percentage_zone,
                    'Illegal %': illegal_percentage_zone,
                    'Non-user %': non_user_percentage_zone
                })

                # Handle cases where no known users exist to avoid division by zero
                user_summary_zone[['Legal %', 'Illegal %', 'Non-user %']] = user_summary_zone[['Legal %', 'Illegal %', 'Non-user %']].fillna(0)

                # # Add a final row with the sum of all Zones (weighted average for percentages)
                total_population_all_zone = total_population_by_zone.sum()

                # Calculate the weighted average for legal, illegal, and non-user percentages
                legal_sum_zone = (legal_percentage_zone * total_population_by_zone).sum() / total_population_all_zone
                illegal_sum_zone = (illegal_percentage_zone * total_population_by_zone).sum() / total_population_all_zone
                non_user_sum_zone = (non_user_percentage_zone * total_population_by_zone).sum() / total_population_all_zone

            elif visualization_type == "DMA" and "DMA" in available_options:
            # Calculate total population and percentages per DMA (if 'DMA' column exists)

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

                # Extract unique DMA and Zone values
                unique_dmas = [int(dma) for dma in filtered_df['DMA'].unique() if pd.notnull(dma)]
                
                # # Add a final row with the sum of all DMAs (weighted average for percentages)
                total_population_all_dmas = total_population_by_dma.sum()

                # Calculate the weighted average for legal, illegal, and non-user percentages
                legal_sum_dma = (legal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
                illegal_sum_dma = (illegal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
                non_user_sum_dma = (non_user_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
                    
        with tab2:

            # Title for Summary of the Network Users
            st.markdown("## Summary of the Network Users")

            # Create columns for side-by-side layout for the tables
            col1, col2, col3 = st.columns([4,1, 1])

            # Place the two tables in the side-by-side layout
            
            if 'visualization_type' in locals():
                if visualization_type == "Zone" and "Zone" in available_options:
                    # Round the population and percentages for user_summary_zone
                    user_summary_zone['Total Population'] = user_summary_zone['Total Population'].round(-2)  # Round population to nearest hundreds
                    user_summary_zone['Legal %'] = user_summary_zone['Legal %'].round(1)  # Round percentages to 1 decimal place
                    user_summary_zone['Illegal %'] = user_summary_zone['Illegal %'].round(1)
                    user_summary_zone['Non-user %'] = user_summary_zone['Non-user %'].round(1)
                    
                    
                    
                    # Calculate the number of people for each user type by multiplying Total Population with percentages
                    user_summary_zone['Legal'] = (user_summary_zone['Total Population'] * user_summary_zone['Legal %'] / 100).astype(int)
                    user_summary_zone['Illegal'] = (user_summary_zone['Total Population'] * user_summary_zone['Illegal %'] / 100).astype(int)
                    user_summary_zone['Non-user'] = (user_summary_zone['Total Population'] * user_summary_zone['Non-user %'] / 100).astype(int)
                    user_summary_zone_plot = user_summary_zone[user_summary_zone.index != 'Total']

                    with col1:
                        st.markdown("#### 📊 Water Network Summary - Zone")
                        st.dataframe(user_summary_zone)
                        
                        # Title for Population by User Type
                        st.markdown("### 📈 Population by User Type")
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
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Format the y-axis labels with a thousand separator
                        st.pyplot(fig)

                    with col2:
                        # Display the calculated total values
                        st.write(f"Total Population: {round(total_population_all_zone,-2)}")
                        st.write(f"Legal %: {legal_sum_zone:.1f}%")
                        st.write(f"Illegal %: {illegal_sum_zone:.1f}%")
                        st.write(f"Non-user %: {non_user_sum_zone:.1f}%")
                        
                elif visualization_type == "DMA" and "DMA" in available_options:
                    # Round the population and percentages for user_summary_dma
                    user_summary_dma['Total Population'] = user_summary_dma['Total Population'].round(-2)  # Round population to nearest hundreds
                    user_summary_dma['Legal %'] = user_summary_dma['Legal %'].round(1)  # Round percentages to 1 decimal place
                    user_summary_dma['Illegal %'] = user_summary_dma['Illegal %'].round(1)
                    user_summary_dma['Non-user %'] = user_summary_dma['Non-user %'].round(1)
                    
                    # Calculate the number of people for each user type by multiplying Total Population with percentages
                    user_summary_dma['Legal'] = (user_summary_dma['Total Population'] * user_summary_dma['Legal %'] / 100).astype(int)
                    user_summary_dma['Illegal'] = (user_summary_dma['Total Population'] * user_summary_dma['Illegal %'] / 100).astype(int)
                    user_summary_dma['Non-user'] = (user_summary_dma['Total Population'] * user_summary_dma['Non-user %'] / 100).astype(int)
                    user_summary_dma_plot = user_summary_dma[user_summary_dma.index != 'Total']

                    with col1:
                        st.markdown("#### 📊 Water Network Summary - DMA")
                        st.dataframe(user_summary_dma)

                        # Title for Population by User Type
                        st.markdown("### 📈 Population by User Type")
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
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Format the y-axis labels with a thousand separator
                        st.pyplot(fig)

                    with col2:
                        
                        # Display the calculated total values
                        st.write(f"Total Population: {round(total_population_all_dmas,-2)}")
                        st.write(f"Legal %: {legal_sum_dma:.1f}%")
                        st.write(f"Illegal %: {illegal_sum_dma:.1f}%")
                        st.write(f"Non-user %: {non_user_sum_dma:.1f}%")

        with tab3:
            st.markdown("### 📅 Monthly Water Consumption Calculation")
            
            #Seazonality factors
            month_factors = {
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
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
                df_factors['Average Daily Consumption - l/p/d'] = round(df_factors['Factor - Updated'] * avg_litres_per_person * 12)
                df_factors["Total Monthly Consumption - m3"] = round(df_factors['Average Daily Consumption - l/p/d'] * sum(df["Population"]) * days_in_month[i] / 1000, -2)
                
            # Create columns for side-by-side layout
            col1, col2 = st.columns(2)
            
            with col1:
                    st.markdown("### Monthly Water Consumption")
                    # Display the table with calculated values
                    st.dataframe(df_factors, height=500)

            with col2:
                # Plot a graph of monthly water consumption
                st.markdown("#### Average Water Consumption Distribution (l/p/d)")

                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(df_factors['Month'], df_factors['Average Daily Consumption - l/p/d'], marker='o', color='skyblue', linewidth=1.0)
                ax.set_ylabel('Average Water Consumption (l/p/d)')
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
            
            # Create a column, with empty space in the left and right for centering
            col1, col2, col3 = st.columns([1, 3, 1])  # Make the center column wider for the plot
            
            with col2:
                fig2, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df_factors['Month'], df_factors['Total Monthly Consumption - m3'], color='deepskyblue')
                ax.set_ylabel('Monthly Water Consumption (m³)')
                ax.set_title('Monthly Water Consumption Distribution')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Format the y-axis labels with a thousand separator
                ax.grid(True, linestyle='-', axis='y')
                st.pyplot(fig2)

        with tab4:
            
            # Monthly Daily Consumption from Seasonal Distribution
            monthly_consumption = df_factors['Average Daily Consumption - l/p/d'] 

            
            # Create columns for side-by-side layout
            col1, col2 = st.columns(2)
            
            if 'visualization_type' in locals():
                if visualization_type == "DMA" and "DMA" in available_options:
                    col1, col2 = st.columns([1,2])
                    with col1:
                        # Prepare population and non-user percentages for DMAs
                        population_dma = user_summary_dma['Total Population']
                        non_users_dma = user_summary_dma['Non-user %'] / 100  
                        
                        # Specify 'DMA' as float or integer when initializing
                        water_demand_dma = pd.DataFrame(columns=['DMA'] + df_factors['Month'].tolist())
                        water_demand_dma['DMA'] = pd.Series(dtype='int')  # Specify the desired numeric type

                        # Calculate the water consumption for each DMA and month
                        for dma in population_dma.index:
                            # For each DMA, calculate monthly consumption
                            dma_consumption = []
                            for i, month in enumerate(df_factors['Month']):
                                consumption_m3 = population_dma[dma] * (1 - non_users_dma[dma]) * monthly_consumption[i] * days_in_month[i] / 1000
                                dma_consumption.append(round(consumption_m3, -2))  # Round to the nearest 100

                            # Add the DMA and its monthly consumption to the DataFrame
                            water_demand_dma.loc[len(water_demand_dma)] = [dma] + dma_consumption  # Fixing float DMA to integer

                        # Set 'DMA' as the index first
                        water_demand_dma.set_index('DMA', inplace=True)
                        # Transpose the DataFrame
                        water_demand_dma = water_demand_dma.transpose()
                        water_demand_dma.columns = pd.to_numeric(water_demand_dma.columns, errors='coerce').astype('Int64')
                        st.markdown("### 💧 Monthly Water Demand per DMA - m3")
                        st.dataframe(water_demand_dma, height=500)

                    with col2:
                    
                        # Plot the stacked bars
                        fig, ax = plt.subplots(figsize=(10, 6))
                        columns_to_plot = water_demand_dma.columns
                        
                        # Use a light blue color palette
                        pastel_colors = sns.color_palette("pastel", len(columns_to_plot))
            
                        water_demand_dma[columns_to_plot].plot(kind='bar', stacked=True, color=pastel_colors, ax=ax)
                        ax.set_title('Monthly Water Demand by DMA - m3', fontsize=15)
                        ax.set_xlabel('Month', fontsize=12)
                        ax.set_ylabel('Water Demand (m3)', fontsize=13)
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                        ax.legend(columns_to_plot, loc='upper left')

                        # Show the grid for y-axis
                        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

                        # Rotate x-axis labels if needed
                        ax.set_xticklabels(water_demand_dma.index, rotation=0)

                        # Show the plot
                        plt.tight_layout()
                        st.pyplot(fig)

                elif visualization_type == "Zone" and "Zone" in available_options:
                    col1, col2 = st.columns([1,2])
                    with col1:
                        # Prepare population and non-user percentages for Zones
                        population_zone = user_summary_zone['Total Population']
                        non_users_zone = user_summary_zone['Non-user %'] / 100  
                        
                        # Create an empty DataFrame to store the results
                        water_demand_zone = pd.DataFrame(columns=['Zone'] + df_factors['Month'].tolist())

                        # Calculate the water consumption for each DMA and month
                        for zone in population_zone.index:
                            # For each Zone, calculate monthly consumption
                            zone_consumption = []
                            for i, month in enumerate(df_factors['Month']):
                                consumption_m3 = population_zone[zone] * (1 - non_users_zone[zone]) * monthly_consumption[i] * days_in_month[i] / 1000
                                zone_consumption.append(round(consumption_m3, -2))  # Round to the nearest 100

                            # Add the zone and its monthly consumption to the DataFrame
                            water_demand_zone.loc[len(water_demand_zone)] = [zone] + zone_consumption
                            
                        water_demand_zone.set_index('Zone', inplace=True)        
                        water_demand_zone = water_demand_zone.transpose()
                        water_demand_zone.columns = pd.to_numeric(water_demand_zone.columns, errors='coerce').astype('Int64')                    
                        st.markdown("### 💧 Monthly Water Demand per Zone - m3")
                        st.dataframe(water_demand_zone, height=500)

                    with col2:
                
                        # Plot the stacked bars
                        fig, ax = plt.subplots(figsize=(10, 6))
                        columns_to_plot = water_demand_zone.columns
                        # Use a light blue color palette
                        pastel_colors = sns.color_palette("pastel", len(columns_to_plot))

                        water_demand_zone[columns_to_plot].plot(kind='bar', stacked=True, color=pastel_colors, ax=ax)
                        ax.set_title('Monthly Water Demand by Zone - m3', fontsize=15)
                        ax.set_xlabel('Month', fontsize=12)
                        ax.set_ylabel('Water Demand (m3)', fontsize=13)
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                        ax.legend(columns_to_plot, loc='upper left')

                        # Show the grid for y-axis
                        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

                        # Rotate x-axis labels if needed
                        ax.set_xticklabels(water_demand_zone.index, rotation=0)

                        # Show the plot
                        plt.tight_layout()
                        st.pyplot(fig)

        with tab5:

            if volume_file and value_file and correlation_file and buildings_file:

                # Process each file
                volume_df = process_volume_or_value_file(volume_file)
                value_df = process_volume_or_value_file(value_file)
                correlation_df = process_block_subscription_file(correlation_file)

                # Step 2: Join the correlation file with the original buildings file on 'Block Number'
                merged_df = pd.merge(buildings_df, correlation_df, on='Block Number', how='left')

                # Step 3: Merge volume_df and value_df on 'Subscription Number'
                volume_df = pd.merge(correlation_df, volume_df, on='Subscription Number', how='left')
                value_df = pd.merge(correlation_df, value_df, on='Subscription Number', how='left')

                # Step 4: Group by 'Block Number' and sum Volume and Value
                volume_summed = volume_df.groupby('Block Number').sum(numeric_only=True).reset_index()
                value_summed = value_df.groupby('Block Number').sum(numeric_only=True).reset_index()

                # Step 5: Merge the summed volumes and values back to the original df
                billed_df = pd.merge(merged_df, volume_summed, on='Block Number', how='left')
                billed_df = pd.merge(billed_df, value_summed, on='Block Number', how='left', suffixes=('_volume', '_value'))

                # Drop unnecessary columns
                columns_to_drop = ['Population', 'Status', 'Subscription Number_x', 'Subscription Number_y', "Subscription Number"]
                billed_df = billed_df.drop(columns=columns_to_drop, errors='ignore')

                if 'visualization_type' in locals():
                    
                    if visualization_type == "DMA" and "DMA" in available_options:
                        n = len(unique_dmas)

                        if billing_type == "Volume (m3) Analysis":
                        
                            # Group by DMA for Volume
                            dma_volume_df = pd.merge(merged_df[['Block Number', 'DMA']], volume_summed, on='Block Number', how='left')
                            dma_volume_df = dma_volume_df.groupby('DMA').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                            dma_volume_df = dma_volume_df.round(0).astype(int)
                            dma_volume_df.set_index('DMA', inplace=True)        
                            dma_volume_df = dma_volume_df.transpose()

                            dma_volume_df = add_month_column_from_index(dma_volume_df)
                            dma_merged_df = join_billed_with_demand(dma_volume_df, water_demand_dma)
                            dma_merged_df = calculate_percentage_billed(dma_merged_df,n)

                            st.markdown("### Percentage of Billed Volume per DMA")
                            st.dataframe(dma_merged_df.iloc[:,-n:])


                            dma_merged_df.index = pd.to_datetime(dma_merged_df.index, format='%m/%y')
                            unique_dates = dma_merged_df.index.sort_values().strftime('%m/%y').tolist()  # Get unique sorted dates as month-year strings

                            # Date range selection
                            start_date, end_date = st.select_slider(
                                "Select Date Range",
                                options=unique_dates,
                                value=(unique_dates[0], unique_dates[-1])  # Default to full range
                            )

                            # Convert selected dates back to datetime format to filter
                            start_date_dt = pd.to_datetime(start_date, format='%m/%y')
                            end_date_dt = pd.to_datetime(end_date, format='%m/%y')       

                            
                            plot_multiple_demand_billed(dma_merged_df, n, selected_dmas_zones, title="Water Demand vs Billed Volumes per DMA")

                        elif billing_type == "Value (EGP £) Analysis" :

                            avg_price_per_m3 = st.number_input("Average Price per m³ in EGP£", min_value=0.0, value=2.0)  # Default value is 5 EGP£ for example
                            
                            dma_value_df = pd.merge(merged_df[['Block Number', 'DMA']], value_summed, on='Block Number', how='left')
                            dma_value_df = dma_value_df.groupby('DMA').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                            dma_value_df = dma_value_df.round(0).astype(int)
                            dma_value_df.set_index('DMA', inplace=True)        
                            dma_value_df = dma_value_df.transpose()

                            dma_value_df = add_month_column_from_index(dma_value_df)
                            dma_value_merged_df = join_billed_with_demand(dma_value_df, water_demand_dma)
                            result_df = calculate_expected_egp_and_percentage(dma_value_merged_df, avg_price_per_m3, n)

                            st.markdown("### Billing Analysis by EGP£ per DMA")
                            st.dataframe(result_df)
                            
                            dma_value_df.index = pd.to_datetime( dma_value_df.index, format='%m/%y')
                            unique_dates =  dma_value_df.index.sort_values().strftime('%m/%y').tolist()  # Get unique sorted dates as month-year strings

                            # Date range selection
                            start_date, end_date = st.select_slider(
                                "Select Date Range",
                                options=unique_dates,
                                value=(unique_dates[0], unique_dates[-1])  # Default to full range
                            )

                            # Convert selected dates back to datetime format to filter
                            start_date_dt = pd.to_datetime(start_date, format='%m/%y')
                            end_date_dt = pd.to_datetime(end_date, format='%m/%y')

                            plot_billed_vs_expected(result_df, n, selected_dmas_zones, title="Total Billed vs Expected EGP £")



                    elif visualization_type == "Zone" and "Zone" in available_options:
                        n = len(unique_zones)
                        if billing_type == "Volume (m3) Analysis":
                            # Group by Zone for Volume
                            zone_volume_df = pd.merge(merged_df[['Block Number', 'Zone']], volume_summed, on='Block Number', how='left')
                            zone_volume_df = zone_volume_df.groupby('Zone').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                            zone_volume_df = zone_volume_df.round(0).astype(int)
                            zone_volume_df.set_index('Zone', inplace=True)        
                            zone_volume_df = zone_volume_df.transpose()

                            zone_volume_df = add_month_column_from_index(zone_volume_df)
                            zone_merged_df = join_billed_with_demand(zone_volume_df, water_demand_zone)
                            zone_merged_df = calculate_percentage_billed(zone_merged_df,n)

                            st.markdown("### Percentage of Billed Volume per Zone")
                            st.dataframe(zone_merged_df.iloc[:,-n:])

                            zone_merged_df.index = pd.to_datetime(zone_merged_df.index, format='%m/%y')
                            unique_dates = zone_merged_df.index.sort_values().strftime('%m/%y').tolist()  # Get unique sorted dates as month-year strings

                            # Date range selection
                            start_date, end_date = st.select_slider(
                                "Select Date Range",
                                options=unique_dates,
                                value=(unique_dates[0], unique_dates[-1])  # Default to full range
                            )

                            # Convert selected dates back to datetime format to filter
                            start_date_dt = pd.to_datetime(start_date, format='%m/%y')
                            end_date_dt = pd.to_datetime(end_date, format='%m/%y')

                            plot_multiple_demand_billed(zone_merged_df, n, selected_dmas_zones, title="Water Demand vs Billed Volumes per Zone")

                        elif billing_type == "Value (EGP £) Analysis" :

                            avg_price_per_m3 = st.number_input("Average Price per m³ in EGP£", min_value=0.0, value=2.0)  # Default value is 5 EGP£ for example
                            
                            zone_value_df = pd.merge(merged_df[['Block Number', 'Zone']], value_summed, on='Block Number', how='left')
                            zone_value_df = zone_value_df.groupby('Zone').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                            zone_value_df = zone_value_df.round(0).astype(int)
                            zone_value_df.set_index('Zone', inplace=True)        
                            zone_value_df = zone_value_df.transpose()
                            
                            zone_value_df = add_month_column_from_index(zone_value_df)
                            zone_value_merged_df = join_billed_with_demand(zone_value_df, water_demand_zone)
                            result_df = calculate_expected_egp_and_percentage(zone_value_merged_df, avg_price_per_m3, n)                            

                            st.markdown("### Billing Analysis by EGP£ per Zone")
                            st.dataframe(result_df)

                            zone_value_df.index = pd.to_datetime(zone_value_df.index, format='%m/%y')
                            unique_dates =  zone_value_df.index.sort_values().strftime('%m/%y').tolist()  # Get unique sorted dates as month-year strings

                            # Date range selection
                            start_date, end_date = st.select_slider(
                                "Select Date Range",
                                options=unique_dates,
                                value=(unique_dates[0], unique_dates[-1])  # Default to full range
                            )

                            # Convert selected dates back to datetime format to filter
                            start_date_dt = pd.to_datetime(start_date, format='%m/%y')
                            end_date_dt = pd.to_datetime(end_date, format='%m/%y')

                            result_df
                            plot_billed_vs_expected(result_df, n, selected_dmas_zones, start_date_dt, end_date_dt, title="Total Billed vs Expected EGP £")
                
            else:
                st.markdown("Input the (1) Billed Volumes, (2) Billed Values, and the (3) Building Blocks / Subscription Number key data to proceed with Billing Analysis")

        with tab6:  
            
            # Create a GeoDataFrame for processing
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
            gdf = gdf.set_crs(epsg=4326)
            gdf = gdf.drop(columns="Population")

            # Calculate the center of the uploaded data
            center_lat, center_lon = gdf["Y"].mean(), gdf["X"].mean()

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
                            # First layer for the original GeoDataFrame
                            {
                                "id": "building_layer",
                                "type": "point",
                                "config": {
                                    "dataId": "Water Consumption Data",  # Link to first GeoDataFrame
                                    "label": "Building Locations",
                                    "color": [150, 200, 255],  # Default color for first layer
                                    "columns": {
                                        "lat": "Y",
                                        "lng": "X"
                                    },
                                    "visConfig": {
                                        "radius": 4,
                                        "opacity": 1,
                                    },
                                    "isVisible": True
                                },
                                "visualChannels": {
                                    "colorField": {
                                        "name": visualization_type,
                                        "type": "string"  # Use 'string' for categorical coloring
                                    }
                                }
                            }
                        ]
                    }
                }
            }

 
            config_2 = {
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
                                    "color": [150, 20, 255],  # Color of points
                                    "columns": {
                                        "lat": "Y",
                                        "lng": "X"
                                    },
                                    "visConfig": {
                                        "radius": 4,
                                        "opacity": 0.2,
                                    },
                                    "isVisible": True
                                }
                            },
                            # Second layer for the gdf_config GeoDataFrame
                            {
                                "id": "dynamic_layer",
                                "type": "hexagon",
                                "config": {
                                    "dataId": "Dynamic Data",  # Use the appropriate data source ID
                                    "label": "Dynamic Data",
                                    "columns": {
                                        "lat": "Y",  # Replace with the latitude column name
                                        "lng": "X"   # Replace with the longitude column name
                                    },
                                    "visConfig": {
                                        "opacity": 0.6,
                                        "worldUnitSize": 0.05,  # Adjust to control the hex size (higher = smaller hexes)
                                        "colorRange": {
                                            "colors": [
                                                "#edf8fb", "#b2e2e2", "#66c2a4", "#2ca25f", "#006d2c"
                                            ]  # Color gradient for hex density
                                        },
                                        "coverage": 1,
                                        "sizeRange": [0, 500],  # Adjust based on data density
                                        "percentile": [0, 100]
                                    },
                                    "isVisible": True
                                },
                                "visualChannels": {
                                    "colorField": {
                                        "name": "density",  # KeplerGL automatically calculates density for hex layers
                                        "type": "real"      # Use real or integer based on your needs
                                    },
                                    "sizeField": {
                                        "name": "density",
                                        "type": "real"
                                    }
                                }
                            }
                        ]
                    }
                }
            }


            # Determine a reasonable zoom level based on data spread
            lat_range = gdf["Y"].max() - gdf["Y"].min()
            lon_range = gdf["X"].max() - gdf["X"].min()

            # Rename for easier recognition in Kepler
            df = df.rename(columns={"X": "longitude", "Y": "latitude"})

            kepler_map = KeplerGl(height=800, config=config_1)   
            st.markdown("### 🗺️ Interactive Map all points")
            # Create heatmaps based on selection
           
            if heatmap_type == "All Buildings":
                kepler_map.add_data(data=gdf, name="Dynamic Data")

            elif heatmap_type == "Illegal Connections":
                gdf_illegal = gdf[gdf['User Type'] == 'Illegal']
                kepler_map.add_data(data=gdf_illegal, name="Dynamic Data")

            elif heatmap_type == "Legal Connections":
                gdf_legal = gdf[gdf['User Type'] == 'Legal']
                kepler_map.add_data(data=gdf_legal, name="Dynamic Data")
 
            elif heatmap_type == "Non-Users":
                gdf_non_user = gdf[gdf['User Type'] == 'Non-user']
                kepler_map.add_data(data=gdf_non_user, name="Dynamic Data")
            
              
            kepler_map.add_data(data=gdf, name="Water Consumption Data")
            keplergl_static(kepler_map)

            kepler_map = KeplerGl(height=800, config=config_2)
            if heatmap_type == "All Buildings":
                st.markdown("#### 🔥 Heatmap and Location of All Building Locations")
                kepler_map.add_data(data=gdf, name="Water Consumption Data")
                kepler_map.add_data(data=gdf, name="Dynamic Data")
                keplergl_static(kepler_map)

            elif heatmap_type == "Illegal Connections":
                st.markdown("#### 🔥 Heatmap and Location of Illegal Connections")
                gdf_illegal = gdf[gdf['User Type'] == 'Illegal']
                kepler_map.add_data(data=gdf_illegal, name="Water Consumption Data")
                kepler_map.add_data(data=gdf_illegal, name="Dynamic Data")
                keplergl_static(kepler_map)

            elif heatmap_type == "Legal Connections":
                st.markdown("#### 🔥 Heatmap and Location of Legal Connections")
                gdf_legal = gdf[gdf['User Type'] == 'Legal']
                kepler_map.add_data(data=gdf_legal, name="Water Consumption Data")
                kepler_map.add_data(data=gdf_legal, name="Dynamic Data")
                keplergl_static(kepler_map)
                
            elif heatmap_type == "Non-Users":
                st.markdown("#### 🔥 Heatmap and Location of Non-Users")
                gdf_non_user = gdf[gdf['User Type'] == 'Non-user']
                kepler_map.add_data(data=gdf_non_user, name="Water Consumption Data")
                kepler_map.add_data(data=gdf_non_user, name="Dynamic Data")
                keplergl_static(kepler_map)
            
           

        
    else:
        st.error("You havent yet uploaded a file or the uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', 'Block Number', or 'Status'. If information is not available, create the column and leave it blank")
