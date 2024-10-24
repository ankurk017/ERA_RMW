import cartopy
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def convert_to_polar(variable, radius=5, resolution=None, coords=('hurricane_radial_distance_x', 'hurricane_radial_distance_y')):
    """
    Convert Cartesian coordinates to polar coordinates and interpolate values.

    This function takes a variable in Cartesian coordinates and converts it to polar coordinates.
    It then interpolates the values onto the new polar grid.

    Args:
        variable (xarray.DataArray): The variable to convert, typically wind speed or another meteorological variable.
        radius (float, optional): The maximum radius for polar coordinates in degrees. Defaults to 5.
        resolution (float, optional): The resolution of the polar grid. If None, it's calculated from the input data. Defaults to None.
        coords (tuple, optional): The names of the x and y coordinate variables in the input data. 
                                  Defaults to ('hurricane_radial_distance_x', 'hurricane_radial_distance_y').

    Returns:
        xarray.DataArray: A new DataArray with the variable interpolated onto a polar grid.
                          The dimensions are 'angle' (in radians) and 'radius' (in km).

    Note:
        - The function assumes that the input coordinates are in degrees and converts the radius to km (multiplied by 111.11).
        - The interpolation is performed using linear interpolation with NaN fill values for points outside the original grid.
    """
    if resolution is None:
        resolution = np.diff(variable[coords[0]])[0]
    r = np.arange(0, radius, resolution)
    ang = np.deg2rad(np.arange(0, 361))
    r_mesh, ang_mesh = np.meshgrid(r, ang)

    x_polar = r_mesh * np.cos(ang_mesh)
    y_polar = r_mesh * np.sin(ang_mesh)

    x_values = variable[coords[0]]
    y_values = variable[coords[1]]

    values = variable.values

    interp_func = RegularGridInterpolator(
        (x_values, y_values),
        values.T,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    polar_coords = np.column_stack((x_polar.ravel(), y_polar.ravel()))
    polar_values = interp_func(polar_coords).reshape(x_polar.shape)

    # Create an xarray DataArray with the polar coordinates and values
    polar_data = xr.DataArray(
        data=polar_values,
        dims=['angle', 'radius'],
        coords={
            'angle': ang,
            'radius': r*111.11,
        },
        attrs={
            'long_name': variable.attrs.get('long_name', 'Variable in polar coordinates'),
            'units': variable.attrs.get('units', 'km'),
        }
    )

    return polar_data

def convert_to_polar_pressure_levels(wind_speed_data, radius=5):
    """
    Convert Cartesian coordinates to polar coordinates and interpolate values for multiple pressure levels.

    This function applies the convert_to_polar function to wind speed data across multiple pressure levels.

    Args:
        wind_speed_data (xarray.DataArray): The wind speed data to convert. Expected to have a 'pressure_level' dimension.
        radius (float, optional): The maximum radius for polar coordinates in degrees. Defaults to 5.

    Returns:
        xarray.DataArray: Wind speed data in polar coordinates with pressure level as a coordinate.
                          The dimensions are 'pressure_level', 'angle' (in radians), and 'radius' (in km).

    Note:
        - This function assumes that the input wind_speed_data has a 'pressure_level' dimension.
        - It applies the convert_to_polar function to each pressure level separately and then concatenates the results.
    """
    data_polar = []
    for pressure_level in wind_speed_data.pressure_level.values:
        polar_xr = convert_to_polar(wind_speed_data.sel(pressure_level=pressure_level), radius=radius)
        polar_xr.coords['pressure_level'] = pressure_level
        data_polar.append(polar_xr)
    
    return xr.concat(data_polar, dim='pressure_level')

def calculate_rmw(data_polar):
    """
    Calculate the Radius of Maximum Winds (RMW) for each pressure level.
    
    This function computes the RMW by finding the radius at which the wind speed is maximum,
    averaged over all angles, for each pressure level.
    
    Parameters:
    data_polar (xarray.DataArray): Wind speed data in polar coordinates.
                                   Expected dimensions are 'pressure_level', 'angle', and 'radius'.
    
    Returns:
    xarray.DataArray: RMW for each pressure level. The dimension is 'pressure_level' and the values are in km.
    
    Note:
        - The function first averages the wind speed over all angles for each radius and pressure level.
        - Then it finds the radius with the maximum average wind speed for each pressure level.
        - The returned values are in the same units as the 'radius' coordinate of the input data (typically km).
    """
    return data_polar.mean(dim='angle').idxmax(dim='radius')

def plot_coast(axes: cartopy.mpl.geoaxes.GeoAxes, color='black', linewidth=2, gridlines_alpha=0.5, states=False) -> None:
    """
    Add coastlines, country borders, and optional state/provincial borders to a Cartopy GeoAxes.

    This function enhances a Cartopy GeoAxes by adding various geographical features and gridlines.

    Parameters:
    axes (cartopy.mpl.geoaxes.GeoAxes): The GeoAxes instance to plot on.
    color (str, optional): Color of the coastlines and borders. Default is 'black'.
    linewidth (int or float, optional): Width of the coastlines and borders. Default is 2.
    gridlines_alpha (float, optional): Transparency level of the gridlines. Default is 0.5.
    states (bool, optional): If True, include state/provincial borders. Default is False.

    Returns:
    cartopy.mpl.gridliner.Gridliner: The gridliner instance with longitude and latitude formatting.

    Example:
    --------
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_coast(ax, color='blue', linewidth=1.5, gridlines_alpha=0.7, states=True)
    plt.show()

    Note:
        - This function uses Natural Earth Features at 1:10m scale for country and state borders.
        - Gridlines are drawn using the Plate Carree projection.
        - The function configures gridline labels to appear only on the bottom and left sides of the plot.
    """

    countries = cfeature.NaturalEarthFeature(
        scale="10m", category="cultural", name="admin_0_countries", facecolor="none"
    )
    axes.add_feature(countries, edgecolor=color, linewidth=linewidth)

    if states:
        states = cfeature.NaturalEarthFeature(
            scale="10m",
            category="cultural",
            name="admin_1_states_provinces_lines",
            facecolor="none",
        )
        axes.add_feature(states, edgecolor=color, linewidth=linewidth)
    
    gl = axes.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=gridlines_alpha,
        linestyle="--",
    )
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return gl