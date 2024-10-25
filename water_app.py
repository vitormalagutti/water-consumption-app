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
import matplotlib.ticker as ticker
import io
import re
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

    return merged_df

def calculate_percentage_billed(merged_df):
    # Calculate the percentage of billed volumes compared to demand directly on the existing merged DataFrame
    for column in merged_df.columns:
        if column.endswith('_demand'):  # Ensure we are working with demand columns
            base_column = column.replace('_demand', '')  # Get the original column name
            if base_column in merged_df.columns:
                merged_df[f'{base_column} % Billed'] = (merged_df[base_column] / merged_df[column]) * 100

    # Replace NaN or infinite values with zeroes
    merged_df = merged_df.replace([float('inf'), -float('inf')], 0).fillna(0)
    
    return merged_df


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

        # Sidebar inputs section with sliders only for the average litres per person
        st.sidebar.header("🔧 Assumptions")
        avg_floors = st.sidebar.number_input("Average Floors per Building", min_value=0.0, step=0.1, value=1.63)
        avg_people_per_family = st.sidebar.number_input("Average People per Family", min_value=1.0, step=1.0, value=5.65)
        avg_litres_per_person = st.sidebar.slider("Average Litres per Person per Day", min_value=50, max_value=300, step=5, value=150)
        st.sidebar.header("🔍 Heatmap Options")
        heatmap_type = st.sidebar.selectbox(
            "Choose a heatmap to display:",
            ["All Buildings", "Illegal Connections", "Legal Connections", "Non-Users"]
        )

        # Calculate total population using the full DataFrame (df), aggregated by Zone and DMA
        df['Population'] = avg_floors * avg_people_per_family
        total_population_by_zone = df.groupby('Zone')['Population'].sum() if 'Zone' in df.columns else None
        total_population_by_dma = df.groupby('DMA')['Population'].sum() if 'DMA' in df.columns else None

        # Calculate total population and percentages per Zone (if 'Zone' column exists)
        if 'Zone' in filtered_df.columns:
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
        else:
            st.markdown("Your file does not have 'Zone' column")

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

            # Extract unique DMA and Zone values
            unique_dmas = [int(dma) for dma in filtered_df['DMA'].unique() if pd.notnull(dma)]
            
            # # Add a final row with the sum of all DMAs (weighted average for percentages)
            total_population_all_dmas = total_population_by_dma.sum()

            # Calculate the weighted average for legal, illegal, and non-user percentages
            legal_sum_dma = (legal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
            illegal_sum_dma = (illegal_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas
            non_user_sum_dma = (non_user_percentage_dma * total_population_by_dma).sum() / total_population_all_dmas

            
        else:
            st.markdown("Your file does not have 'DMA' column")

        with tab2:

            # Title for Summary of the Network Users
            st.markdown("## Summary of the Network Users")
            if 'Zone' in filtered_df.columns and 'user_summary_zone' in locals():
                # Round the population and percentages for user_summary_zone
                user_summary_zone['Total Population'] = user_summary_zone['Total Population'].round(-2)  # Round population to nearest hundreds
                user_summary_zone['Legal %'] = user_summary_zone['Legal %'].round(1)  # Round percentages to 1 decimal place
                user_summary_zone['Illegal %'] = user_summary_zone['Illegal %'].round(1)
                user_summary_zone['Non-user %'] = user_summary_zone['Non-user %'].round(1)

            else:
                st.markdown("Your file does not have 'Zone' column")

            if 'DMA' in filtered_df.columns and 'user_summary_dma' in locals():
                # Round the population and percentages for user_summary_dma
                user_summary_dma['Total Population'] = user_summary_dma['Total Population'].round(-2)  # Round population to nearest hundreds
                user_summary_dma['Legal %'] = user_summary_dma['Legal %'].round(1)  # Round percentages to 1 decimal place
                user_summary_dma['Illegal %'] = user_summary_dma['Illegal %'].round(1)
                user_summary_dma['Non-user %'] = user_summary_dma['Non-user %'].round(1)
                
            else:
                st.markdown("Your file does not have 'DMA' column")

            # Create columns for side-by-side layout for the tables
            col1, col2 = st.columns(2)

            # Place the two tables in the side-by-side layout
            with col1:
                if 'Zone' in filtered_df.columns:
                    st.markdown("#### 📊 Water Network Summary - Zone")
                    st.dataframe(user_summary_zone)

                else:
                    st.markdown("Your file does not have 'Zone' column")

            with col2:
                if 'DMA' in filtered_df.columns:
                    st.markdown("#### 📊 Water Network Summary - DMA")
                    st.dataframe(user_summary_dma)

                else:
                    st.markdown("Your file does not have 'DMA' column")

            # Display the calculated total values
            st.write(f"Total Population: {round(total_population_all_dmas,-2)}")
            st.write(f"Legal %: {legal_sum_dma:.1f}%")
            st.write(f"Illegal %: {illegal_sum_dma:.1f}%")
            st.write(f"Non-user %: {non_user_sum_dma:.1f}%")

            # Title for Population by User Type
            st.markdown("### 📈 Population by User Type")

            # Create columns for side-by-side layout for the graphs
            col1, col2 = st.columns(2)
            if 'Zone' in filtered_df.columns:
                # Calculate the number of people for each user type by multiplying Total Population with percentages
                user_summary_zone['Legal'] = (user_summary_zone['Total Population'] * user_summary_zone['Legal %'] / 100).astype(int)
                user_summary_zone['Illegal'] = (user_summary_zone['Total Population'] * user_summary_zone['Illegal %'] / 100).astype(int)
                user_summary_zone['Non-user'] = (user_summary_zone['Total Population'] * user_summary_zone['Non-user %'] / 100).astype(int)
                user_summary_zone_plot = user_summary_zone[user_summary_zone.index != 'Total']       
            else:
                st.markdown("Your file does not have 'Zone' column")

            if 'DMA' in filtered_df.columns:
                # Calculate the number of people for each user type by multiplying Total Population with percentages
                user_summary_dma['Legal'] = (user_summary_dma['Total Population'] * user_summary_dma['Legal %'] / 100).astype(int)
                user_summary_dma['Illegal'] = (user_summary_dma['Total Population'] * user_summary_dma['Illegal %'] / 100).astype(int)
                user_summary_dma['Non-user'] = (user_summary_dma['Total Population'] * user_summary_dma['Non-user %'] / 100).astype(int)
                user_summary_dma_plot = user_summary_dma[user_summary_dma.index != 'Total']
            else:
                st.markdown("Your file does not have 'DMA' column")

            # Place the two graphs in the side-by-side layout
            if 'Zone' in filtered_df.columns:     
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
            else:
                st.markdown("Your file does not have 'Zone' column")

            if 'DMA' in filtered_df.columns: 
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
            else:
                st.markdown("Your file does not have 'DMA' column")

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

            if 'DMA' in filtered_df.columns:
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

            if 'Zone' in filtered_df.columns:
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
            
            
            # Create columns for side-by-side layout
            col1, col2 = st.columns(2)

            if 'DMA' in filtered_df.columns:
                with col1:
                    st.markdown("### 💧 Monthly Water Consumption per DMA")
                    st.dataframe(water_demand_dma, height=500)
                    # Plot the stacked bars
                    fig, ax = plt.subplots(figsize=(10, 6))
                    columns_to_plot = water_demand_dma.columns
                    water_demand_dma[columns_to_plot].plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title('Monthly Water Demand by DMA', fontsize=15)
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

            if 'Zone' in filtered_df.columns:            
                with col2:
                    st.markdown("### 💧 Monthly Water Consumption per Zone")
                    st.dataframe(water_demand_zone, height=500)

                    # Plot the stacked bars
                    fig, ax = plt.subplots(figsize=(10, 6))
                    columns_to_plot = water_demand_zone.columns
                    water_demand_zone[columns_to_plot].plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title('Monthly Water Demand by Zone', fontsize=15)
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

                # Display the final merged dataframe
                # st.markdown("### Final Merged DataFrame")
                # st.dataframe(billed_df)

                # Group by Zone for Volume
                if 'Zone' in merged_df.columns:
                    zone_volume_df = pd.merge(merged_df[['Block Number', 'Zone']], volume_summed, on='Block Number', how='left')
                    zone_volume_df = zone_volume_df.groupby('Zone').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                    zone_volume_df = zone_volume_df.round(0).astype(int)
                    zone_volume_df.set_index('Zone', inplace=True)        
                    zone_volume_df = zone_volume_df.transpose()
                    # st.markdown("### Zone Billed Data Summed by Volume")
                    # st.dataframe(zone_volume_df)

                # Group by DMA for Volume
                if 'DMA' in merged_df.columns:
                    dma_volume_df = pd.merge(merged_df[['Block Number', 'DMA']], volume_summed, on='Block Number', how='left')
                    dma_volume_df = dma_volume_df.groupby('DMA').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                    dma_volume_df = dma_volume_df.round(0).astype(int)
                    dma_volume_df.set_index('DMA', inplace=True)        
                    dma_volume_df = dma_volume_df.transpose()
                    # st.markdown("### DMA Billed Data Summed by Volume")
                    # st.dataframe(dma_volume_df)
                
                # Group by Zone for Value
                if 'Zone' in merged_df.columns:
                    zone_value_df = pd.merge(merged_df[['Block Number', 'Zone']], value_summed, on='Block Number', how='left')
                    zone_value_df = zone_value_df.groupby('Zone').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                    zone_value_df = zone_value_df.round(0).astype(int)
                    zone_value_df.set_index('Zone', inplace=True)        
                    zone_value_df = zone_value_df.transpose()
                    # st.markdown("### Zone Billed Data Summed by Value")
                    # st.dataframe(zone_value_df)
                    zone_volume_df = add_month_column_from_index(zone_volume_df)
                    zone_merged_df = join_billed_with_demand(zone_volume_df, water_demand_zone)

                # Group by DMA for Value
                if 'DMA' in merged_df.columns:
                    dma_value_df = pd.merge(merged_df[['Block Number', 'DMA']], value_summed, on='Block Number', how='left')
                    dma_value_df = dma_value_df.groupby('DMA').sum(numeric_only=True).reset_index().drop(columns=["Block Number", "Subscription Number"])
                    dma_value_df = dma_value_df.round(0).astype(int)
                    dma_value_df.set_index('DMA', inplace=True)        
                    dma_value_df = dma_value_df.transpose()
                    # st.markdown("### DMA Billed Data Summed by Value")
                    # st.dataframe(dma_value_df)
                    dma_volume_df = add_month_column_from_index(dma_volume_df)
                    dma_merged_df = join_billed_with_demand(dma_volume_df, water_demand_dma)
              
                

                # Calculate the percentage for the already-merged zone and DMA dataframes
                zone_merged_df = calculate_percentage_billed(zone_merged_df)
                dma_merged_df = calculate_percentage_billed(dma_merged_df)
                zone_merged_df = zone_merged_df.drop(columns="Month")
                dma_merged_df = dma_merged_df.drop(columns="Month")

                # Display the merged dataframes with calculated percentages
                st.markdown("### Merged DataFrame for Zone with Percentage Billed")
                st.dataframe(zone_merged_df)

                st.markdown("### Merged DataFrame for DMA with Percentage Billed")
                st.dataframe(dma_merged_df)

                def plot_combined_demand_billed(df, unique_labels, title="Water Demand vs Billed Volumes with % Billed"):
                    # Separate columns into demand, billed, and percentage columns
                    demand_columns = [col for col in df.columns if col.endswith('_demand')]
                    billed_columns = [col for col in df.columns if col not in demand_columns and not col.endswith('% Billed')]
                    percent_columns = [col for col in df.columns if col.endswith('% Billed')]
                    demand_columns
                    fig, ax1 = plt.subplots(figsize=(12, 6))

                    # Widths and positions for the bars
                    bar_width = 0.3
                    demand_positions = np.arange(len(unique_labels))

                    # Check column lengths to prevent broadcasting errors
                    for column in demand_columns + billed_columns + percent_columns:
                        if len(df[column]) != len(demand_positions):
                            df = df[df[column].notna()]  # Remove NaNs to ensure matching lengths

                    # Plot Demand Bars
                    for i, column in enumerate(demand_columns):
                        positions = demand_positions + i * bar_width  # Create positions based on demand_positions
                        positions 
                        if len(df[column].values) == len(positions):
                            ax1.bar(
                                positions,
                                df[column].values,
                                width=bar_width,
                                label=f"{column} Demand",
                                color='blue',
                                alpha=0.6
                            )


                    # Plot Billed Bars
                    for i, column in enumerate(billed_columns):
                        ax1.bar(
                            [p + (i + len(demand_columns)) * bar_width for p in demand_positions],
                            df[column].values,
                            width=bar_width,
                            label=f"{column} Billed",
                            color='green',
                            alpha=0.6
                        )

                    # Plot Percentage Billed as Lines on Secondary Y-axis
                    ax2 = ax1.twinx()
                    for i, column in enumerate(percent_columns):
                        ax2.plot(
                            demand_positions,
                            df[column].values,
                            label=f"{column} % Billed",
                            marker='o',
                            linestyle='--'
                        )

                    # Set labels and title
                    ax1.set_xlabel("Zone/DMA")
                    ax1.set_ylabel("Volume")
                    ax2.set_ylabel("Percentage Billed (%)")
                    ax1.set_title(title)
                    
                    # Set x-ticks and labels
                    ax1.set_xticks(demand_positions + bar_width * (len(demand_columns) - 1) / 2)
                    ax1.set_xticklabels(unique_labels, rotation=45)
                    
                    # Legends
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    
                    plt.show()

                # Call the function for both zone and DMA merged dataframes
                plot_combined_demand_billed(zone_merged_df,unique_zones, title="Zone Demand vs Billed Volumes with % Billed")
                plot_combined_demand_billed(dma_merged_df,unique_dmas, title="DMA Demand vs Billed Volumes with % Billed")







            else:
                st.error("Please upload all the necessary files (volume, value, correlation, and buildings files).")



        with tab6:
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
                    st.markdown("#### 📍 Location of All Building Locations")
                    kepler_map = KeplerGl(height=900, config=config_1)
                    kepler_map.add_data(data=gdf, name="Water Consumption Data")
                    keplergl_static(kepler_map)

                elif heatmap_type == "Illegal Connections":
                    st.markdown("#### 📍 Location of Illegal Connections")
                    gdf_illegal = gdf[gdf['User Type'] == 'Illegal'] 
                    kepler_map = KeplerGl(height=900, config=config_1)
                    kepler_map.add_data(data=gdf_illegal, name="Water Consumption Data")
                    keplergl_static(kepler_map)

                elif heatmap_type == "Legal Connections":
                    st.markdown("#### 📍 Location of Legal Connections")
                    gdf_legal = gdf[gdf['User Type'] == 'Legal']
                    kepler_map = KeplerGl(height=900, config=config_1)
                    kepler_map.add_data(data=gdf_legal, name="Water Consumption Data")
                    keplergl_static(kepler_map)
                    
                elif heatmap_type == "Non-Users":
                    st.markdown("#### 📍 Location of Non-Users")
                    gdf_non_user = gdf[gdf['User Type'] == 'Non-user']
                    kepler_map = KeplerGl(height=900, config=config_1)
                    kepler_map.add_data(data=gdf_non_user, name="Water Consumption Data")
                    keplergl_static(kepler_map)
        
    else:
        st.error("You havent yet uploaded a file or the uploaded CSV file does not contain the required columns 'X', 'Y', 'Zone', 'Block Number', or 'Status'. If information is not available, create the column and leave it blank")
