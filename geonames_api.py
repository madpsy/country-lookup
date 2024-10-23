from flask import Flask, request, jsonify
import geopandas as gpd
from shapely.geometry import Point
import logging
import pycountry
from sklearn.neighbors import BallTree
import numpy as np
from functools import lru_cache
from geopy.distance import geodesic
from shapely.ops import nearest_points

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to shapefiles
COUNTRIES_SHAPEFILE = 'data/countries/ne_10m_admin_0_countries.shp'
TOWNS_SHAPEFILE = 'data/towns/cities500.geojson'  # Updated to use GeoNames data

# Load countries data
try:
    logger.info("Loading countries shapefile...")
    countries = gpd.read_file(COUNTRIES_SHAPEFILE)
    logger.info("Countries shapefile loaded successfully.")

    # Ensure CRS is WGS84
    if countries.crs != "EPSG:4326":
        logger.info("Converting countries CRS to EPSG:4326...")
        countries = countries.to_crs("EPSG:4326")
        logger.info("Countries CRS conversion completed.")

    # Select necessary columns
    countries = countries[['ADMIN', 'ISO_A2', 'ISO_A3', 'ADM0_A3', 'geometry']]

    # Build spatial index (Rtree)
    logger.info("Building spatial index for countries...")
    countries_sindex = countries.sindex
    logger.info("Spatial index for countries built successfully.")

except Exception as e:
    logger.error(f"Error loading countries shapefile: {e}")
    raise e

# Load towns data
try:
    logger.info("Loading towns GeoJSON...")
    towns = gpd.read_file(TOWNS_SHAPEFILE)
    logger.info("Towns GeoJSON loaded successfully.")

    # Ensure CRS is WGS84
    if towns.crs != "EPSG:4326":
        logger.info("Converting towns CRS to EPSG:4326...")
        towns = towns.to_crs("EPSG:4326")
        logger.info("Towns CRS conversion completed.")

    # Extract town coordinates
    logger.info("Extracting town coordinates for BallTree...")
    town_coords = np.array([[geom.y, geom.x] for geom in towns.geometry])

    # Convert degrees to radians for BallTree
    logger.info("Converting town coordinates to radians...")
    town_coords_rad = np.radians(town_coords)

    # Build BallTree for nearest neighbour search using haversine metric
    logger.info("Building BallTree for towns...")
    ball_tree = BallTree(town_coords_rad, metric='haversine')
    logger.info("BallTree for towns built successfully.")

except Exception as e:
    logger.error(f"Error loading towns GeoJSON: {e}")
    raise e

# Precompute buffer zones (100 m) around all countries
logger.info("Creating buffer zones around countries...")
buffer_km = 0.1
buffer_degrees = buffer_km / 111  # Approximate conversion from km to degrees
countries_buffer = countries.copy()
countries_buffer['geometry'] = countries_buffer['geometry'].buffer(buffer_degrees)
logger.info("Buffer zones created successfully.")

# Build spatial index for buffer zones
logger.info("Building spatial index for buffer zones...")
countries_buffer_sindex = countries_buffer.sindex
logger.info("Spatial index for buffer zones built successfully.")

def calculate_geodesic_distance(lat1, lon1, lat2, lon2):
    """
    Calculate geodesic distance between two points.
    
    Returns:
        float: Distance in kilometers
    """
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def get_country_code_and_name(lat, lon, max_distance_km=10):
    """
    Given latitude and longitude, return the corresponding ISO_A2 country code, country name,
    and geodesic distance to the country's boundary.
    If the point is not within any country, search within a buffer of max_distance_km.
    
    Returns:
        tuple: (country_code (str or None), country_name (str or None), distance_km (float or None))
    """
    try:
        point = Point(lon, lat)
        
        # Check if the point is within any country
        possible_matches_index = list(countries_sindex.intersection(point.bounds))
        possible_matches = countries.iloc[possible_matches_index]

        for idx, country in possible_matches.iterrows():
            if country['geometry'].contains(point):
                iso_a2 = country.get('ISO_A2', None)
                adm0_a3 = country.get('ADM0_A3', None)
                admin_name = country.get('ADMIN', None)
                
                if iso_a2 and iso_a2 != '-99':
                    return iso_a2, admin_name, 0.0
                elif adm0_a3 and adm0_a3 != '-99':
                    try:
                        country_obj = pycountry.countries.get(alpha_3=adm0_a3)
                        if country_obj:
                            return country_obj.alpha_2, country_obj.name, 0.0
                        else:
                            logger.warning(f"No ISO_A2 mapping found for ADM0_A3: {adm0_a3}")
                    except Exception as e:
                        logger.error(f"Error mapping ADM0_A3 to ISO_A2: {e}")
                else:
                    logger.warning(f"Invalid ISO_A2 and ADM0_A3 codes for country: {country['ADMIN']}")

        # If not within any country, use precomputed buffer
        logger.info("Point not within any country. Initiating buffer-based search using precomputed buffers...")
        possible_matches_index = list(countries_buffer_sindex.intersection(point.bounds))
        possible_matches = countries_buffer.iloc[possible_matches_index]

        nearest_country = None
        min_distance = float('inf')

        for idx, country in possible_matches.iterrows():
            try:
                # Calculate the nearest point on the country's geometry to the query point
                nearest_geom = nearest_points(country['geometry'], point)[0]
                nearest_coords = (nearest_geom.y, nearest_geom.x)  # (lat, lon)
                distance = calculate_geodesic_distance(lat, lon, nearest_coords[0], nearest_coords[1])
                
                if distance < min_distance and distance <= max_distance_km:
                    min_distance = distance
                    iso_a2 = country.get('ISO_A2', None)
                    adm0_a3 = country.get('ADM0_A3', None)
                    admin_name = country.get('ADMIN', None)
                    
                    if iso_a2 and iso_a2 != '-99':
                        nearest_country = (iso_a2, admin_name, distance)
                    elif adm0_a3 and adm0_a3 != '-99':
                        try:
                            country_obj = pycountry.countries.get(alpha_3=adm0_a3)
                            if country_obj:
                                nearest_country = (country_obj.alpha_2, country_obj.name, distance)
                            else:
                                logger.warning(f"No ISO_A2 mapping found for ADM0_A3: {adm0_a3}")
                        except Exception as e:
                            logger.error(f"Error mapping ADM0_A3 to ISO_A2: {e}")
                    else:
                        logger.warning(f"Invalid ISO_A2 and ADM0_A3 codes for country: {country['ADMIN']}")

            except Exception as e:
                logger.error(f"Error processing country geometry: {e}")
                continue  # Proceed to next country

        if nearest_country:
            return nearest_country
        else:
            logger.warning("No nearby country found within the specified distance.")
            return None, None, None

    except Exception as e:
        logger.error(f"Unexpected error in get_country_code_and_name: {e}")
        return None, None, None

def get_closest_town(lat, lon):
    """
    Given latitude and longitude, return the closest town with population >= 500 and its distance in km.
    Returns:
        tuple: (closest_town_name (str), distance_km (float))
    """
    try:
        # Convert query point to radians
        point_rad = np.radians([lat, lon])

        # Query BallTree for the nearest town
        dist, idx = ball_tree.query([point_rad], k=1)
        distance_km = dist[0][0] * 6371  # Earth's radius in kilometers

        closest_town = towns.iloc[idx[0][0]]['name']

        return closest_town, round(distance_km, 2)
    except Exception as e:
        logger.error(f"Error in get_closest_town: {e}")
        return None, None

@lru_cache(maxsize=10000)
def cached_get_country_code_and_name(lat, lon):
    return get_country_code_and_name(lat, lon)

@lru_cache(maxsize=10000)
def cached_get_closest_town(lat, lon):
    return get_closest_town(lat, lon)

@app.route('/api/country', methods=['POST'])
def country_endpoint():
    """
    API endpoint to get country code, country name, closest town, and distance from latitude and longitude.
    Expects 'lat' and 'lon' in JSON payload.
    """
    try:
        # Ensure the request has JSON content
        if not request.is_json:
            logger.warning("Invalid content type received.")
            return jsonify({'error': 'Invalid content type. Expected application/json.'}), 400

        data = request.get_json()

        # Extract latitude and longitude from JSON payload
        lat = data.get('lat', None)
        lon = data.get('lon', None)

        # Validate presence of latitude and longitude
        if lat is None or lon is None:
            logger.warning("Missing 'lat' or 'lon' in the request payload.")
            return jsonify({'error': 'Missing required JSON fields: lat and lon'}), 400

        # Validate data types
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            logger.warning("'lat' or 'lon' has invalid data type.")
            return jsonify({'error': 'Invalid data types for lat and lon. Expected numbers.'}), 400

        # Validate latitude and longitude ranges
        if not (-90 <= lat <= 90):
            logger.warning(f"Invalid latitude value: {lat}")
            return jsonify({'error': 'Invalid latitude value. Must be between -90 and +90.'}), 400
        if not (-180 <= lon <= 180):
            logger.warning(f"Invalid longitude value: {lon}")
            return jsonify({'error': 'Invalid longitude value. Must be between -180 and +180.'}), 400

        # Get country code, name, and distance
        country_code, country_name, distance_to_country = cached_get_country_code_and_name(lat, lon)

        logger.info(f"Country Code: {country_code}, Country Name: {country_name}, Distance: {distance_to_country}")

        if country_code is None:
            logger.warning("Country not found within the specified distance.")
            return jsonify({'error': 'Country not found within the specified distance'}), 404

        # Get closest town and distance
        closest_town, distance_km = cached_get_closest_town(lat, lon)

        # Ensure distance_closest_town_km is rounded to two decimal places
        if distance_km is not None:
            distance_km = round(distance_km, 2)

        response = {
            'country_code': country_code,
            'country_name': country_name,
            'closest_town': closest_town if closest_town else "Unknown",
            'distance_closest_town_km': distance_km if distance_km is not None else "Unknown"
        }

        # Include distance to the nearest country if the point was not within any country
        if distance_to_country and distance_to_country > 0:
            # Ensure distance_to_nearest_country_km is rounded to two decimal places
            distance_to_country = round(distance_to_country, 2)
            response['distance_to_nearest_country_km'] = distance_to_country

        logger.info(f"Response: {response}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        # Always return a valid JSON error response
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the Flask app
    # For production, use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=8000, threaded=True)

