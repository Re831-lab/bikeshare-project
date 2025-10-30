"""
US Bikeshare Data Analysis Application
Analyzes bikeshare usage patterns across Chicago, New York City, and Washington.
"""

import time
import pandas as pd
import numpy as np
import os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

CITY_DATA = {
    'chicago': 'chicago.csv',
    'new york city': 'new_york_city.csv',
    'washington': 'washington.csv'
}

# Constants for validation
VALID_CITIES = ['chicago', 'new york city', 'washington']
VALID_MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'all']
VALID_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'all']


def get_user_input(prompt, valid_options, error_msg):
    """
    Generic function to get validated user input.
    
    Args:
        prompt (str): Message to display to user
        valid_options (list): List of acceptable inputs
        error_msg (str): Error message for invalid input
        
    Returns:
        str: Validated user input
    """
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_options:
            return user_input
        print(error_msg)


def get_filters():
    """
    Ask user to specify a city, month, and day to analyze.
    Uses input validation to ensure data quality.

    Returns:
        tuple: (city, month, day)
            city (str): name of the city to analyze
            month (str): name of the month to filter by, or "all" for no filter
            day (str): name of the day of week to filter by, or "all" for no filter
    """
    print("=" * 50)
    print("Welcome to US Bikeshare Data Analysis!")
    print("=" * 50)

    # Get city selection with validation (keeps trying until valid input)
    city = get_user_input(
        "\nEnter city (Chicago, New York City, Washington): ",
        VALID_CITIES,
        "Invalid city. Please choose from: Chicago, New York City, or Washington."
    )

    # Get month selection with validation (keeps trying until valid input)
    month = get_user_input(
        "\nEnter month (January to June) or 'all' for no filter: ",
        VALID_MONTHS,
        "Invalid month. Please enter a month from January to June, or 'all'."
    )

    # Get day selection with validation (keeps trying until valid input)
    day = get_user_input(
        "\nEnter day of week (Monday-Sunday) or 'all' for no filter: ",
        VALID_DAYS,
        "Invalid day. Please enter a day of the week, or 'all'."
    )

    print('-' * 50)
    return city, month, day


def remove_outliers(df, column, method='iqr'):
    """
    Remove outliers from a specific column using IQR or Z-score method.
    
    Args:
        df (DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        method (str): 'iqr' for IQR method or 'zscore' for Z-score method
        
    Returns:
        DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"  Warning: Column '{column}' not found in data.")
        return df
    
    # Check for NaN values
    if df[column].isna().all():
        print(f"  Warning: Column '{column}' contains only NaN values.")
        return df
    
    initial_count = len(df)
    
    if method == 'iqr':
        # IQR method: Remove values outside 1.5 * IQR range
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        # Z-score method: Remove values with |z-score| > 3
        mean = df[column].mean()
        std = df[column].std()
        
        # Check for zero standard deviation
        if std == 0:
            print(f"  Warning: Standard deviation for '{column}' is 0, skipping Z-score method")
            return df
        
        z_scores = np.abs((df[column] - mean) / std)
        df = df[z_scores < 3]
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"  Removed {removed_count} outliers from {column}")
    
    return df


def load_data(city, month, day, remove_outliers_flag=False):
    """
    Load data for the specified city and apply month and day filters.
    
    This function reads the CSV file for the selected city, converts the 
    Start Time to datetime, and filters based on month and day preferences.
    Optionally removes outliers from trip duration data.

    Args:
        city (str): name of the city to analyze
        month (str): month name to filter by, or "all" for no month filter
        day (str): day of week to filter by, or "all" for no day filter
        remove_outliers_flag (bool): whether to remove outliers from trip duration

    Returns:
        DataFrame: Pandas DataFrame containing filtered city data, or None if error occurs
    """
    try:
        # Load data file for the selected city
        print(f"\nLoading data for {city.title()}...")
        
        # Check if file exists
        if city not in CITY_DATA:
            print(f"Error: City '{city}' not found in data configuration.")
            return None
        
        csv_file = CITY_DATA[city]
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            return None
        
        df = pd.read_csv(csv_file)
        initial_records = len(df)
        
        # Check if data is empty
        if df.empty:
            print(f"Error: {csv_file} is empty.")
            return None

        # Convert Start Time column to datetime for easier manipulation
        if 'Start Time' not in df.columns:
            print("Error: 'Start Time' column not found in data.")
            return None
        
        df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')
        
        # Remove rows with invalid Start Time
        invalid_times = df['Start Time'].isna().sum()
        if invalid_times > 0:
            print(f"  Warning: Found {invalid_times} invalid datetime entries, removing them.")
            df = df.dropna(subset=['Start Time'])
        
        # Extract month and day of week from Start Time
        df['month'] = df['Start Time'].dt.month
        df['day_of_week'] = df['Start Time'].dt.day_name().str.lower()

        # Apply month filter if specified
        if month != 'all':
            # Convert month name to index (1-6 for Jan-June)
            month_index = VALID_MONTHS.index(month) + 1
            df = df[df['month'] == month_index]

        # Apply day filter if specified
        if day != 'all':
            df = df[df['day_of_week'] == day]

        # Remove outliers if requested
        if remove_outliers_flag:
            print("\nRemoving outliers from Trip Duration...")
            if 'Trip Duration' in df.columns:
                df = remove_outliers(df, 'Trip Duration', method='iqr')
            else:
                print("  Warning: 'Trip Duration' column not found.")

        print(f"Data loaded: {len(df)} records found")
        if initial_records != len(df):
            print(f"Filtered from {initial_records} total records.\n")
        else:
            print()
        
        return df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def time_stats(df):
    """
    Display statistics on the most frequent times of travel.
    Calculates and displays the most common month, day, and hour.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    if df.empty:
        print("No data available for time statistics.")
        return
    
    print('\n' + '=' * 50)
    print('Calculating The Most Frequent Times of Travel...')
    print('=' * 50)
    start_time = time.time()

    try:
        # Find and display most common month
        if df['month'].mode().empty:
            print("No month data available.")
        else:
            common_month = df['month'].mode()[0]
            month_name = VALID_MONTHS[common_month - 1].title()
            print(f'Most Common Month: {month_name}')

        # Find and display most common day of week
        if df['day_of_week'].mode().empty:
            print("No day of week data available.")
        else:
            common_day = df['day_of_week'].mode()[0]
            print(f'Most Common Day: {common_day.title()}')

        # Extract hour from Start Time and find most common
        df['hour'] = df['Start Time'].dt.hour
        if df['hour'].mode().empty:
            print("No hour data available.")
        else:
            common_hour = df['hour'].mode()[0]
            # Convert to 12-hour format for readability
            hour_12 = common_hour % 12 if common_hour % 12 != 0 else 12
            am_pm = 'AM' if common_hour < 12 else 'PM'
            print(f'Most Common Start Hour: {common_hour}:00 ({hour_12} {am_pm})')

        print(f"\nThis took {(time.time() - start_time):.4f} seconds.")
        print('-' * 50)
    
    except Exception as e:
        print(f"Error calculating time statistics: {e}")
        print('-' * 50)


def station_stats(df):
    """
    Display statistics on the most popular stations and trips.
    Shows most common start station, end station, and trip combination.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    if df.empty:
        print("No data available for station statistics.")
        return
    
    print('\n' + '=' * 50)
    print('Calculating The Most Popular Stations and Trip...')
    print('=' * 50)
    start_time = time.time()

    try:
        # Check for required columns
        required_cols = ['Start Station', 'End Station']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns - {', '.join(missing_cols)}")
            print('-' * 50)
            return

        # Find and display most common start station
        if df['Start Station'].mode().empty:
            print("No start station data available.")
        else:
            common_start = df['Start Station'].mode()[0]
            start_count = (df['Start Station'] == common_start).sum()
            print(f'Most Common Start Station: {common_start} ({start_count} trips)')

        # Find and display most common end station
        if df['End Station'].mode().empty:
            print("No end station data available.")
        else:
            common_end = df['End Station'].mode()[0]
            end_count = (df['End Station'] == common_end).sum()
            print(f'Most Common End Station: {common_end} ({end_count} trips)')

        # Create combination of start and end stations for trip analysis
        df['Start-End Combo'] = df['Start Station'] + " â†’ " + df['End Station']
        if df['Start-End Combo'].mode().empty:
            print("No trip combination data available.")
        else:
            common_trip = df['Start-End Combo'].mode()[0]
            trip_count = (df['Start-End Combo'] == common_trip).sum()
            print(f'Most Common Trip: {common_trip} ({trip_count} trips)')

        print(f"\nThis took {(time.time() - start_time):.4f} seconds.")
        print('-' * 50)
    
    except Exception as e:
        print(f"Error calculating station statistics: {e}")
        print('-' * 50)


def trip_duration_stats(df):
    """
    Display statistics on the total and average trip duration.
    Converts duration to human-readable format (days, hours, minutes).

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    if df.empty:
        print("No data available for trip duration statistics.")
        return
    
    print('\n' + '=' * 50)
    print('Calculating Trip Duration Statistics...')
    print('=' * 50)
    start_time = time.time()

    try:
        if 'Trip Duration' not in df.columns:
            print("Error: 'Trip Duration' column not found.")
            print('-' * 50)
            return

        # Calculate total travel time in seconds
        total_duration = df['Trip Duration'].sum()
        
        # Convert to human-readable format
        days = int(total_duration // (24 * 3600))
        hours = int((total_duration % (24 * 3600)) // 3600)
        minutes = int((total_duration % 3600) // 60)
        
        print(f'Total Travel Time: {days} days, {hours} hours, {minutes} minutes')
        print(f'Total Travel Time (seconds): {total_duration:,.0f}')

        # Calculate and display mean travel time
        mean_duration = df['Trip Duration'].mean()
        mean_minutes = int(mean_duration // 60)
        mean_seconds = int(mean_duration % 60)
        print(f'Average Trip Duration: {mean_minutes} minutes, {mean_seconds} seconds')
        print(f'Average Trip Duration (seconds): {mean_duration:.2f}')

        print(f"\nThis took {(time.time() - start_time):.4f} seconds.")
        print('-' * 50)
    
    except Exception as e:
        print(f"Error calculating trip duration statistics: {e}")
        print('-' * 50)


def user_stats(df):
    """
    Display statistics on bikeshare users.
    Shows user type distribution and demographic information (when available).
    
    Note: Washington dataset does not include Gender and Birth Year data.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    if df.empty:
        print("No data available for user statistics.")
        return
    
    print('\n' + '=' * 50)
    print('Calculating User Statistics...')
    print('=' * 50)
    start_time = time.time()

    try:
        # Display counts of user types
        if 'User Type' in df.columns:
            print('User Type Distribution:')
            user_types = df['User Type'].value_counts()
            if user_types.empty:
                print("  No user type data available.")
            else:
                for user_type, count in user_types.items():
                    print(f'  {user_type}: {count:,}')
        else:
            print("User Type data not available for this city.")

        # Display gender distribution if available
        if 'Gender' in df.columns:
            print('\nGender Distribution:')
            gender_counts = df['Gender'].value_counts()
            if gender_counts.empty:
                print("  No gender data available.")
            else:
                for gender, count in gender_counts.items():
                    print(f'  {gender}: {count:,}')
        else:
            print('\nGender data not available for this city.')

        # Display birth year statistics if available
        if 'Birth Year' in df.columns:
            print('\nBirth Year Statistics:')
            valid_years = df['Birth Year'].dropna()
            if valid_years.empty:
                print("  No birth year data available.")
            else:
                earliest = int(valid_years.min())
                recent = int(valid_years.max())
                common_year = int(valid_years.mode()[0]) if not valid_years.mode().empty else 0
                
                current_year = datetime.now().year
                print(f'  Earliest Birth Year: {earliest} (Age: {current_year - earliest})')
                print(f'  Most Recent Birth Year: {recent} (Age: {current_year - recent})')
                print(f'  Most Common Birth Year: {common_year} (Age: {current_year - common_year})')
        else:
            print('\nBirth year data not available for this city.')

        print(f"\nThis took {(time.time() - start_time):.4f} seconds.")
        print('-' * 50)
    
    except Exception as e:
        print(f"Error calculating user statistics: {e}")
        print('-' * 50)


def display_raw_data(df):
    """
    Display raw data upon user request.
    Shows 5 rows at a time until user decides to stop.
    
    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    if df.empty:
        print("No data available to display.")
        return
    
    row_index = 0
    show_data = input('\nWould you like to see 5 rows of raw data? Enter yes or no: ').lower().strip()
    
    while show_data == 'yes':
        print('\n', df.iloc[row_index:row_index + 5])
        row_index += 5
        
        # Check if there are more rows to display
        if row_index >= len(df):
            print('\nNo more data to display.')
            break
            
        show_data = input('\nWould you like to see 5 more rows? Enter yes or no: ').lower().strip()


def main():
    """
    Main function to run the bikeshare data analysis application.
    Orchestrates the entire analysis workflow and handles user interaction.
    """
    while True:
        # Get user filter preferences
        city, month, day = get_filters()
        
        # Ask if user wants to remove outliers
        remove_outliers_choice = input('\nRemove outliers from trip duration? (yes/no): ').lower().strip()
        remove_outliers_flag = (remove_outliers_choice == 'yes')
        
        # Load and filter data
        df = load_data(city, month, day, remove_outliers_flag)
        
        # Check if data loaded successfully
        if df is None:
            print("\nFailed to load data. Please try again.")
        elif df.empty:
            print("\nNo data available for the selected filters.")
            print("Please try different filter options.")
        else:
            # Display all statistics
            time_stats(df)
            station_stats(df)
            trip_duration_stats(df)
            user_stats(df)
            
            # Offer to display raw data
            display_raw_data(df)

        # Ask if user wants to restart
        print('\n' + '=' * 50)
        restart = input('Would you like to restart? Enter yes or no: ').lower().strip()
        if restart != 'yes':
            print('\nThank you for using the Bikeshare Data Analysis tool!')
            print('=' * 50)
            break


if __name__ == "__main__":
    main()