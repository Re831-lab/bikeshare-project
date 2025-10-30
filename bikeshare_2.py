import time
import pandas as pd
import numpy as np

CITY_DATA = {
    'chicago': 'chicago.csv',
    'new york city': 'new_york_city.csv',
    'washington': 'washington.csv'
}

def get_filters():
    """Ask user to specify a city, month, and day to analyze.

    Returns:
        tuple: (city, month, day)
            city (str): name of the city to analyze
            month (str): name of the month to filter by, or "all" for no filter
            day (str): name of the day of week to filter by, or "all" for no filter
    """
    print("Hello! Let's explore some US bikeshare data!")

    # City input
    cities = ['chicago', 'new york city', 'washington']
    while True:
        city = input("Enter city (Chicago, New York City, Washington): ").strip().lower()
        if city in cities:
            break
        else:
            print("Invalid city. Please try again.")

    # Month input
    months = ['january','february','march','april','may','june','all']
    while True:
        month = input("Enter month (January to June) or 'all': ").strip().lower()
        if month in months:
            break
        else:
            print("Invalid month. Please try again.")

    # Day input
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','all']
    while True:
        day = input("Enter day of week or 'all': ").strip().lower()
        if day in days:
            break
        else:
            print("Invalid day. Please try again.")

    print('-'*40)
    return city, month, day

def load_data(city, month, day):
    """Load data for the specified city and apply month and day filters.

    Args:
        city (str): name of the city to analyze
        month (str): month name to filter by, or "all"
        day (str): day of week to filter by, or "all"

    Returns:
        DataFrame: Pandas DataFrame containing filtered city data
    """
    df = pd.read_csv(CITY_DATA[city])

    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.day_name().str.lower()

    if month != 'all':
        months = ['january','february','march','april','may','june']
        month_index = months.index(month) + 1
        df = df[df['month'] == month_index]

    if day != 'all':
        df = df[df['day_of_week'] == day]

    return df

def time_stats(df):
    """Display the most frequent times of travel.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    common_month = df['month'].mode()[0]
    print('Most Common Month:', common_month)

    common_day = df['day_of_week'].mode()[0]
    print('Most Common Day:', common_day)

    df['hour'] = df['Start Time'].dt.hour
    common_hour = df['hour'].mode()[0]
    print('Most Common Start Hour:', common_hour)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def station_stats(df):
    """Display the most popular stations and trip.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    common_start = df['Start Station'].mode()[0]
    print('Most Common Start Station:', common_start)

    common_end = df['End Station'].mode()[0]
    print('Most Common End Station:', common_end)

    df['Start-End Combo'] = df['Start Station'] + " to " + df['End Station']
    common_trip = df['Start-End Combo'].mode()[0]
    print('Most Common Trip:', common_trip)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def trip_duration_stats(df):
    """Display statistics on trip duration.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    total_duration = df['Trip Duration'].sum()
    minutes = total_duration // 60
    hours = minutes // 60
    print(f'Total Travel Time: {hours} hours and {minutes % 60} minutes')

    mean_duration = df['Trip Duration'].mean()
    print('Mean Travel Time:', mean_duration)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def user_stats(df):
    """Display statistics on bikeshare users.

    Args:
        df (DataFrame): the DataFrame containing bikeshare data
    """
    print('\nCalculating User Stats...\n')
    start_time = time.time()

    user_types = df['User Type'].value_counts()
    print('Counts of User Types:\n', user_types)

    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        print('\nCounts of Gender:\n', gender_counts)

    if 'Birth Year' in df:
        earliest = int(df['Birth Year'].min())
        recent = int(df['Birth Year'].max())
        common_year = int(df['Birth Year'].mode()[0])
        print('\nEarliest Birth Year:', earliest)
        print('Most Recent Birth Year:', recent)
        print('Most Common Birth Year:', common_year)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def main():
    """Main function to run the bikeshare data analysis application."""
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
