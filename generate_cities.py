import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def process_geonames(input_path, output_path):
    """
    Processes the GeoNames cities500.txt file and converts it into a GeoDataFrame.
    Filters towns with population >= 500 and converts to GeoJSON format.
    
    Parameters:
    - input_path: Path to the cities500.txt file.
    - output_path: Path to save the processed GeoDataFrame (optional).
    
    Returns:
    - GeoDataFrame containing towns with population >= 500.
    """
    # Define column names based on GeoNames documentation
    columns = [
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude',
        'longitude', 'feature class', 'feature code', 'country code',
        'cc2', 'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code',
        'population', 'elevation', 'dem', 'timezone', 'modification date'
    ]
    
    # Read the GeoNames file
    print("Reading GeoNames data...")
    df = pd.read_csv(input_path, sep='\t', header=None, names=columns, dtype={'population': float})
    
    # Filter out entries with missing population
    df = df.dropna(subset=['population'])
    
    # Convert population to integer
    df['population'] = df['population'].astype(int)
    
    # Filter towns with population >= 500
    print("Filtering towns with population >= 1000...")
    df = df[df['population'] >= 1000]
    
    # Create geometry column
    print("Creating geometry column...")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Select necessary columns
    gdf = gdf[['name', 'population', 'country code', 'geometry']]
    
    # Save to GeoJSON (optional)
    if output_path:
        print(f"Saving processed data to {output_path}...")
        gdf.to_file(output_path, driver='GeoJSON')
    
    print("Processing complete.")
    return gdf

def main():
    input_path = 'data/towns/cities500.txt'
    output_path = 'data/towns/cities500.geojson'  # Optional
    
    if not os.path.exists(input_path):
        print(f"Input file not found at {input_path}. Please download and extract it first.")
        return
    
    gdf = process_geonames(input_path, output_path)
    
    # Optional: Save as CSV for inspection
    gdf.to_csv('data/towns/cities500_processed.csv', index=False)
    print("Processed GeoNames data exported to 'data/towns/cities500_processed.csv'.")

if __name__ == "__main__":
    main()

