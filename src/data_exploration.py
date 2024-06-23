import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from src.data_collection_and_processing import DataFrameProcessor  # Assuming DataFrameProcessor class is imported from data_collection_and_processing module

class EVDataVisualizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.plots_folder = 'plots'  # Folder to save plots
        self.check_plots_folder()   # Ensure the plots folder exists
        self.ev_make_distribution = None  # Initialize ev_make_distribution
        self.top_makes_data = None  # Initialize top_makes_data
    
    def check_plots_folder(self):
        # Create the plots folder if it does not exist
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
    
    def save_plot(self, fig, filename):
        # Save the plot to the plots folder
        filepath = os.path.join(self.plots_folder, filename)
        fig.savefig(filepath)
        plt.close(fig)  # Close the figure after saving
    
    def plot_ev_adoption_over_time(self):
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(12, 6))
        ev_adoption_by_year = self.dataframe['Model Year'].value_counts().sort_index()
        sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
        plt.title('EV Adoption Over Time')
        plt.xlabel('Model Year')
        plt.ylabel('Number of Vehicles Registered')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        filename = 'ev_adoption_over_time.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_top_cities_in_top_counties(self):
        ev_county_distribution = self.dataframe['County'].value_counts()
        top_counties = ev_county_distribution.head(3).index
        
        top_counties_data = self.dataframe[self.dataframe['County'].isin(top_counties)]
        ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
        
        top_cities = ev_city_distribution_top_counties.head(10)
        
        plt.figure(figsize=(12, 5))
        sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
        plt.title('Top Cities in Top Counties by EV Registrations')
        plt.xlabel('Number of Vehicles Registered')
        plt.ylabel('City')
        plt.legend(title='County')
        plt.tight_layout()
        
        # Save the plot
        filename = 'top_cities_in_top_counties.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_ev_type_distribution(self):
        ev_type_distribution = self.dataframe['Electric Vehicle Type'].value_counts()
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette="rocket")
        plt.title('Distribution of Electric Vehicle Types')
        plt.xlabel('Number of Vehicles Registered')
        plt.ylabel('Electric Vehicle Type')
        plt.tight_layout()
        
        # Save the plot
        filename = 'ev_type_distribution.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_ev_make_distribution(self):
        self.ev_make_distribution = self.dataframe['Make'].value_counts().head(10)
        
        plt.figure(figsize=(12, 4))
        sns.barplot(x=self.ev_make_distribution.values, y=self.ev_make_distribution.index, palette="cubehelix")
        plt.title('Top 10 Popular EV Makes')
        plt.xlabel('Number of Vehicles Registered')
        plt.ylabel('Make')
        plt.tight_layout()
        
        # Save the plot
        filename = 'ev_make_distribution.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_top_models_in_top_makes(self):
        if self.ev_make_distribution is None:
            self.plot_ev_make_distribution()  # Ensure ev_make_distribution is calculated
        
        top_3_makes = self.ev_make_distribution.head(3).index
        self.top_makes_data = self.dataframe[self.dataframe['Make'].isin(top_3_makes)]
        ev_model_distribution_top_makes = self.top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
        top_models = ev_model_distribution_top_makes.head(10)
        
        plt.figure(figsize=(12, 5))
        sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette="viridis")
        plt.title('Top Models in Top 3 Makes by EV Registrations')
        plt.xlabel('Number of Vehicles Registered')
        plt.ylabel('Model')
        plt.legend(title='Make', loc='center right')
        plt.tight_layout()
        
        # Save the plot
        filename = 'top_models_in_top_makes.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_ev_range_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.dataframe['Electric Range'], bins=30, kde=True, color='royalblue')
        plt.title('Distribution of Electric Vehicle Ranges')
        plt.xlabel('Electric Range (miles)')
        plt.ylabel('Number of Vehicles')
        plt.axvline(self.dataframe['Electric Range'].mean(), color='red', linestyle='--', label=f'Mean Range: {self.dataframe["Electric Range"].mean():.2f} miles')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        filename = 'ev_range_distribution.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_avg_range_by_year(self):
        average_range_by_year = self.dataframe.groupby('Model Year')['Electric Range'].mean().reset_index()
        
        plt.figure(figsize=(12, 4))
        sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='green')
        plt.title('Average Electric Range by Model Year')
        plt.xlabel('Model Year')
        plt.ylabel('Average Electric Range (miles)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        filename = 'avg_range_by_year.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_top_range_models(self):
        if self.ev_make_distribution is None:
            self.plot_ev_make_distribution()  # Ensure ev_make_distribution is calculated
        
        average_range_by_model = self.top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()
        top_range_models = average_range_by_model.head(10)
        
        plt.figure(figsize=(12, 4))
        barplot = sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette="cool")
        plt.title('Top 10 Models by Average Electric Range in Top Makes')
        plt.xlabel('Average Electric Range (miles)')
        plt.ylabel('Model')
        plt.legend(title='Make', loc='center right')
        plt.tight_layout()
        
        # Save the plot
        filename = 'top_range_models.png'
        self.save_plot(plt.gcf(), filename)
    
    def plot_ev_market_forecast(self):
        ev_registration_counts = self.dataframe['Model Year'].value_counts().sort_index()
        filtered_years = ev_registration_counts[ev_registration_counts.index <= 2023]
        
        def exp_growth(x, a, b):
            return a * np.exp(b * x)
        
        x_data = filtered_years.index - filtered_years.index.min()
        y_data = filtered_years.values
        
        params, covariance = curve_fit(exp_growth, x_data, y_data)
        
        forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
        forecasted_values = exp_growth(forecast_years, *params)
        
        forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecasted_values))
        
        years = np.arange(filtered_years.index.min(), 2029 + 1)
        actual_years = filtered_years.index
        forecast_years_full = np.arange(2024, 2029 + 1)
        actual_values = filtered_years.values
        forecasted_values_full = [forecasted_evs[year] for year in forecast_years_full]
        
        plt.figure(figsize=(12, 5))
        plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
        plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')
        
        plt.title('Current & Estimated EV Market')
        plt.xlabel('Year')
        plt.ylabel('Number of EV Registrations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        filename = 'ev_market_forecast.png'
        self.save_plot(plt.gcf(), filename)

# Example of how to use the class (execution part will be in main.py):
if __name__ == "__main__":
    pass  # Execution will be handled in main.py
