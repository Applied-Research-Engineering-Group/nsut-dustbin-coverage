"""
Single-file Flask app: NSUT Dustbin Coverage Dashboard

Features:
- Upload dustbins CSV (id,lat,lon) or use sample generated points
- Configure campus center, campus radius, zone size, coverage radius
- Runs coverage analysis (grid of square zones) without GeoPandas (uses shapely + pyproj)
- Shows interactive Folium map inline and provides downloadable CSV report

Requirements:
  pip install Flask pandas shapely pyproj folium numpy

Run:
  python nsut_flask_dustbin_dashboard.py
  Open http://127.0.0.1:5000
"""

from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string, flash
import os
import math
import pandas as pd
import numpy as np
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
from pyproj import CRS, Transformer
import folium
import tempfile

app = Flask(_name_)
app.secret_key = "replace_this_with_a_random_secret"

BASE_DIR = os.path.abspath(os.path.dirname(_file_))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# default NSUT center
NSUT_CENTER_LAT = 28.606497222222224
NSUT_CENTER_LON = 77.03583333333333

# projection transformers
crs_wgs84 = CRS.from_epsg(4326)
crs_merc = CRS.from_epsg(3857)
transformer_to_merc = Transformer.from_crs(crs_wgs84, crs_merc, always_xy=True)
transformer_to_wgs = Transformer.from_crs(crs_merc, crs_wgs84, always_xy=True)

def lonlat_to_merc(lon, lat):
    return transformer_to_merc.transform(lon, lat)

def merc_to_lonlat(x, y):
    return transformer_to_wgs.transform(x, y)

# core analysis function (no geopandas required)
def analyze_coverage(dustbins_df,
                     campus_center_lat=NSUT_CENTER_LAT, campus_center_lon=NSUT_CENTER_LON,
                     campus_radius_m=600, zone_size_m=20, coverage_radius_m=80):
    # ensure dustbins_df has lat, lon
    if dustbins_df is None or dustbins_df.empty:
        dustbins_df = pd.DataFrame(columns=['id','lat','lon'])

    # convert campus center to merc
    cx, cy = lonlat_to_merc(campus_center_lon, campus_center_lat)
    campus_center_pt = Point(cx, cy)
    campus_poly_merc = campus_center_pt.buffer(campus_radius_m)

    # create grid
    minx, miny, maxx, maxy = campus_poly_merc.bounds
    nx = int(math.ceil((maxx - minx) / zone_size_m))
    ny = int(math.ceil((maxy - miny) / zone_size_m))
    cells = []
    for i in range(nx):
        for j in range(ny):
            x0 = minx + i*zone_size_m
            y0 = miny + j*zone_size_m
            x1 = x0 + zone_size_m
            y1 = y0 + zone_size_m
            cells.append(box(x0, y0, x1, y1))

    # filter cells that intersect campus polygon
    cells_in = [c for c in cells if c.intersects(campus_poly_merc)]

    # prepare dustbin points in merc
    dustbin_points_merc = []
    dustbin_ids = []
    for idx, row in dustbins_df.iterrows():
        try:
            lon = float(row['lon'])
            lat = float(row['lat'])
            x, y = lonlat_to_merc(lon, lat)
            dustbin_points_merc.append(Point(x, y))
            dustbin_ids.append(row.get('id', idx))
        except Exception:
            continue

    results = []
    for i, cell in enumerate(cells_in):
        centroid = cell.centroid
        covered = False
        nearest_id = None
        nearest_dist = float('inf')
        # check each bin
        for k, dbp in enumerate(dustbin_points_merc):
            d = centroid.distance(dbp)
            if d <= coverage_radius_m:
                covered = True
                nearest_id = dustbin_ids[k]
                nearest_dist = d
                break
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = dustbin_ids[k]
        # suggested placement = centroid
        slon, slat = merc_to_lonlat(centroid.x, centroid.y)
        results.append({
            'zone_index': i,
            'covered': covered,
            'nearest_bin': nearest_id,
            'nearest_dist_m': nearest_dist if nearest_dist != float('inf') else None,
            'suggested_lat': slat,
            'suggested_lon': slon,
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'cell_bounds': cell.bounds
        })

    res_df = pd.DataFrame(results)
    total_zones = len(res_df)
    covered_zones = int(res_df['covered'].sum()) if total_zones>0 else 0
    uncovered_zones = total_zones - covered_zones
    coverage_pct = covered_zones/total_zones*100 if total_zones>0 else 0

    summary = {
        'total_zones': total_zones,
        'covered_zones': covered_zones,
        'uncovered_zones': uncovered_zones,
        'coverage_pct': coverage_pct
    }

    return res_df, summary, campus_poly_merc

# helper to build folium map from results
def build_map(res_df, dustbins_df, campus_center_lat, campus_center_lon, campus_poly_merc, zone_size_m, coverage_radius_m):
    m = folium.Map(location=[campus_center_lat, campus_center_lon], zoom_start=16)

    # campus boundary (convert campus_poly_merc to lonlat polygon points)
    poly_coords = [(merc_to_lonlat(x, y)[1], merc_to_lonlat(x, y)[0]) for x, y in list(campus_poly_merc.exterior.coords)]
    folium.Polygon(locations=poly_coords, color='blue', weight=2, fill=False, tooltip='Campus boundary').add_to(m)

    # dustbins
    if dustbins_df is not None and not dustbins_df.empty:
        for _, row in dustbins_df.iterrows():
            folium.CircleMarker(location=(row['lat'], row['lon']), radius=5, popup=str(row.get('id','')), tooltip='Existing bin').add_to(m)

    # zones: show centroids colored
    for _, row in res_df.iterrows():
        lat = row['suggested_lat']
        lon = row['suggested_lon']
        if row['covered']:
            color = 'green'
        else:
            color = 'red'
        folium.CircleMarker(location=(lat, lon), radius=3, color=color, fill=True, fill_opacity=0.7,
                            popup=f"Zone {row['zone_index']} - {'Covered' if row['covered'] else 'Uncovered'}").add_to(m)

    # suggested placements for uncovered zones
    for _, row in res_df[~res_df['covered']].iterrows():
        folium.Marker(location=(row['suggested_lat'], row['suggested_lon']),
                      icon=folium.Icon(color='blue', icon='trash', prefix='fa'),
                      popup=f"Suggest zone {row['zone_index']}").add_to(m)

    # add legend (simple)
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 120px; 
                 border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:8px;">
     <b>Legend</b><br>
     <i class="fa fa-circle" style="color:green"></i>&nbsp;Covered zone<br>
     <i class="fa fa-circle" style="color:red"></i>&nbsp;Uncovered zone<br>
     <i class="fa fa-map-marker" style="color:blue"></i>&nbsp;Suggested placement<br>
     <i class="fa fa-circle" style="color:black"></i>&nbsp;Existing bin<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# Flask routes
INDEX_HTML = '''
<!doctype html>
<title>NSUT Dustbin Coverage Dashboard</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/modern-normalize/1.1.0/modern-normalize.min.css">
<div style="max-width:1000px;margin:20px auto;font-family:Arial,Helvetica,sans-serif">
  <h2>NSUT Dustbin Coverage Dashboard</h2>
  <p>Upload a CSV with columns <code>id,lat,lon</code> (latitude, longitude). If none provided, sample points will be used.</p>
  <form method=post enctype=multipart/form-data action="/run">
    <label>CSV file: <input type=file name=file></label><br><br>
    <label>Campus center lat: <input name=campus_lat value="{{campus_lat}}"></label>
    <label> lon: <input name=campus_lon value="{{campus_lon}}"></label><br><br>
    <label>Campus radius (meters): <input name=campus_radius value="600"></label><br>
    <label>Zone size (meters): <input name=zone_size value="20"></label><br>
    <label>Coverage radius (meters): <input name=coverage value="80"></label><br><br>
    <button type=submit>Run analysis</button>
  </form>
  <hr>
  {% if summary %}
    <h3>Summary</h3>
    <ul>
      <li>Total zones: {{summary.total_zones}}</li>
      <li>Covered zones: {{summary.covered_zones}}</li>
      <li>Uncovered zones: {{summary.uncovered_zones}}</li>
      <li>Coverage: {{summary.coverage_pct|round(1)}}%</li>
    </ul>
    <p><a href="/download/report.csv">Download zone_coverage_report.csv</a></p>
    <div style="height:600px">{{map_html|safe}}</div>
  {% endif %}
  <hr>
  <p>Sample CSV format:</p>
  <pre>id,lat,lon\nB1,28.6069,77.0362\nB2,28.6071,77.0351</pre>
</div>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, campus_lat=NSUT_CENTER_LAT, campus_lon=NSUT_CENTER_LON, summary=None)

@app.route('/run', methods=['POST'])
def run_analysis():
    file = request.files.get('file')
    campus_lat = float(request.form.get('campus_lat', NSUT_CENTER_LAT))
    campus_lon = float(request.form.get('campus_lon', NSUT_CENTER_LON))
    campus_radius = float(request.form.get('campus_radius', 600))
    zone_size = float(request.form.get('zone_size', 20))
    coverage = float(request.form.get('coverage', 80))

    dustbins_df = None
    if file and file.filename:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        try:
            dustbins_df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Failed to read CSV: {e}")
            return redirect(url_for('index'))
    else:
        # create sample dustbins within campus (increased from 12 -> 30)
        rng = np.random.default_rng(42)
        sample_points = []
        cx, cy = lonlat_to_merc(campus_lon, campus_lat)
        for _ in range(30):
            r = campus_radius * math.sqrt(rng.random())
            theta = rng.random()*2*math.pi
            px = cx + r*math.cos(theta)
            py = cy + r*math.sin(theta)
            lon, lat = merc_to_lonlat(px, py)
            sample_points.append({'id': f'S{len(sample_points)+1}', 'lat': lat, 'lon': lon})
        dustbins_df = pd.DataFrame(sample_points)

    res_df, summary, campus_poly_merc = analyze_coverage(dustbins_df,
                                                         campus_center_lat=campus_lat,
                                                         campus_center_lon=campus_lon,
                                                         campus_radius_m=campus_radius,
                                                         zone_size_m=zone_size,
                                                         coverage_radius_m=coverage)

    # save report
    report_csv_path = os.path.join(OUTPUT_FOLDER, 'zone_coverage_report.csv')
    res_df.to_csv(report_csv_path, index=False)

    # build folium map and embed
    m = build_map(res_df, dustbins_df, campus_lat, campus_lon, campus_poly_merc, zone_size, coverage)
    # save to temporary html then read content
    tmp_map_fp = os.path.join(OUTPUT_FOLDER, 'nsut_dustbin_coverage_map.html')
    m.save(tmp_map_fp)
    with open(tmp_map_fp, 'r', encoding='utf-8') as f:
        map_html = f.read()

    return render_template_string(INDEX_HTML, campus_lat=campus_lat, campus_lon=campus_lon, summary=summary, map_html=map_html)

@app.route('/download/<path:filename>')
def download_file(filename):
    if filename == 'report.csv':
        return send_from_directory(OUTPUT_FOLDER, 'zone_coverage_report.csv', as_attachment=True)
    return 'Not found', 404

if __name__ == '__main__':
    app.run(debug=True)