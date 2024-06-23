import pandas as pd
from src.data_collection_and_processing import DataFrameProcessor
from src.data_exploration import EVDataVisualizer
import logging
import functools

# Set up logging
logging.basicConfig(level=logging.INFO)

def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished executing {func.__name__}")
        return result
    return wrapper

# Decorator usage
@log_execution
def main():
    # Initialize DataFrameProcessor with input file path
    input_file_path = 'data/Electric_Vehicle_Population_Data.csv'
    processor = DataFrameProcessor(input_file_path)
    
    # Collect EV data from input file
    ev_data = processor.collect_ev_data()
    
    # Drop missing values
    ev_data_cleaned = processor.drop_missing_values()

    # Step 2: Initialize EVDataVisualizer with the loaded dataframe
    visualizer = EVDataVisualizer(ev_data_cleaned)
    
    # Step 3: Execute all plotting methods
    visualizer.plot_ev_adoption_over_time()
    visualizer.plot_top_cities_in_top_counties()
    visualizer.plot_ev_type_distribution()
    visualizer.plot_ev_make_distribution()
    visualizer.plot_top_models_in_top_makes()
    visualizer.plot_ev_range_distribution()
    visualizer.plot_avg_range_by_year()
    visualizer.plot_top_range_models()
    visualizer.plot_ev_market_forecast()
    
    # All plots are saved in the 'plots' folder

if __name__ == "__main__":
    main()

