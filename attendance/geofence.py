import math

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the Ray Casting algorithm.
    point: tuple (lat, lng)
    polygon: list of tuples [(lat, lng), ...]
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r

def check_geofence(site, lat, lng):
    """
    Check if the given coordinates are within the site's geofence.
    Returns True if inside, False otherwise.
    """
    if not site:
        return False

    # Check Polygon Geofence (KML)
    if site.coordinates:
        # Assuming site.coordinates is a list of [lat, lng] or {'lat': ..., 'lng': ...}
        # We need to normalize it to list of tuples
        polygon = []
        try:
            # Handle different JSON structures if necessary
            # The user said "geofence_coordinates (JSON array for polygons from KML)"
            # Let's assume it's a list of objects or lists
            raw_coords = site.coordinates
            if isinstance(raw_coords, list):
                for coord in raw_coords:
                    if isinstance(coord, dict):
                        polygon.append((float(coord.get('lat', 0)), float(coord.get('lng', 0))))
                    elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        polygon.append((float(coord[0]), float(coord[1])))
            
            if polygon and is_point_in_polygon((lat, lng), polygon):
                return True
        except Exception as e:
            print(f"Error checking polygon geofence: {e}")

    # Check Radius Geofence
    if site.geofence_lat is not None and site.geofence_lng is not None:
        distance = haversine_distance(lat, lng, site.geofence_lat, site.geofence_lng)
        radius = site.geofence_radius_meters or 100.0
        if distance <= radius:
            return True

    # If no geofence is defined, maybe default to True? 
    # Or if both are defined, does it need to satisfy one or both?
    # The user said "Checks if point is inside polygon OR within radius distance"
    # So if neither passed so far, return False.
    # BUT if no geofence is defined at all, we should probably return True (allow attendance)
    # or False (strict mode). 
    # Given the requirement "check if any employ attendance get marked out of site coordinates",
    # if no coordinates are set, maybe we can't say they are "out".
    # However, for now, if no geofence is set, let's assume it's valid (True) to avoid blocking.
    if not site.coordinates and site.geofence_lat is None:
        return True

    return False
