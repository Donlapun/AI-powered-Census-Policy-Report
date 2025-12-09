"""Utility functions for Census Data Analysis and Report.

This module contains helper functions and utilities for the Census Data Analysis and Report
that don't require prompt engineering or agents.
"""

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st
import requests


load_dotenv()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
COUNTY_SUFFIXES = [" county", " parish", " borough"]


# normalize county name
def normalize_name(name):
    """
    Clean the county name for to match the census standard
    for data comparison
    Args:
        name (str): county name

    Returns:
        str: clean county name
    """
    name = name.lower().strip()
    for s in COUNTY_SUFFIXES:
        if name.endswith(s):
            name = name.replace(s, "")
    return name


@st.cache_data(ttl=3600)
def fetch_census_data(year, tables, geo_type, geo_name, fips):
    """
    Fetch Census data for specified tables and geographic entities.

    Args:
        year (int): Year of the ACS/Census data to retrieve.
        tables (list of str): List of Census table IDs to fetch (e.g., ["B02001_001E"]).
        geo_type (str): Type of geography, e.g., 'state', 'county', or 'tract'.
        geo_name (str): Name of the geographic location.
        fips (dict): Dictionary containing FIPS codes for the geographic location,
                     e.g., {"state": state_fips, "county": county_fips}.

    Returns:
        pandas.DataFrame: A DataFrame containing the requested Census data for the specified geography and tables.

    Raises:
        ValueError: If the requested tables, geography type, or FIPS codes are invalid.
        requests.RequestException: If there is a problem fetching data from the Census API.
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    res_df = pd.DataFrame()

    try:
        if geo_type == "state":
            # Fetch all counties in state
            url = f"{base}?get=NAME,{tables}&for=county:*&in=state:{fips['state']}&key={CENSUS_API_KEY}"
            res = requests.get(url)

            if res.status_code == 204:
                st.warning(f"No data returned for state {geo_name}")
                return pd.DataFrame()
            elif res.status_code != 200:
                st.error(f"Request failed ({res.status_code}): {res.text}")
                return pd.DataFrame()

            data = res.json()
            res_df = pd.DataFrame(data[1:], columns=data[0])
            res_df["GEOID"] = res_df["state"].str.zfill(2) + res_df["county"].str.zfill(
                3
            )

        elif geo_type == "county":
            # Fetch all tracts in county
            url = f"{base}?get=NAME,{tables}&for=tract:*&in=state:{fips['state']} county:{fips['county']}&key={CENSUS_API_KEY}"
            print("URL", url)
            res = requests.get(url)

            if res.status_code == 204:
                st.warning(f"No data returned for county {geo_name}")
                return pd.DataFrame()
            elif res.status_code != 200:
                st.error(f"Request failed ({res.status_code}): {res.text}")
                return pd.DataFrame()

            try:
                data = res.json()
                res_df = pd.DataFrame(data[1:], columns=data[0])
                res_df["GEOID"] = (
                    res_df["state"].str.zfill(2)
                    + res_df["county"].str.zfill(3)
                    + res_df["tract"].str.zfill(6)
                )
            except ValueError:
                st.warning(f"Invalid JSON returned for county {geo_name}")
                return pd.DataFrame()

        elif geo_type == "place":
            county_fips_list = fips.get("county", [])  # list of county FIPS
            all_dfs = []
            for county_fips in county_fips_list:
                url = f"{base}?get=NAME,{tables}&for=tract:*&in=state:{fips['state']}+county:{county_fips}&key={CENSUS_API_KEY}"
                print(url)
                res = requests.get(url)

                if res.status_code == 204:
                    st.warning(f"No data returned for place {geo_name}")
                    return pd.DataFrame()
                elif res.status_code != 200:
                    st.error(f"Request failed ({res.status_code}): {res.text}")
                    return pd.DataFrame()

                try:
                    data = res.json()
                    res_df = pd.DataFrame(data[1:], columns=data[0])
                    res_df["GEOID"] = (
                        res_df["state"].str.zfill(2)
                        + res_df["county"].str.zfill(3)
                        + res_df["tract"].str.zfill(6)
                    )
                    all_dfs.append(res_df)
                except ValueError:
                    st.warning(f"Invalid JSON returned for place {geo_name}")
                    return pd.DataFrame()
            if all_dfs:
                res_df = pd.concat(all_dfs, ignore_index=True)
            else:
                res_df = pd.DataFrame()

        else:
            raise ValueError(f"Invalid geo_type: {geo_type}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request exception: {e}")
        return pd.DataFrame()

    return res_df


def fetch_shapefile(geo_type, fips_state, fips_county, year):
    """
    Fetch Shapefile for census data

    Args:
        year (int): Year of the ACS/Census data to retrieve.
        geo_type (str): Type of geography, e.g., 'state', 'county', or 'tract'.
        geo_name (str): Name of the geographic location.
        fips (dict): Dictionary containing FIPS codes for the geographic location,
                     e.g., {"state": state_fips, "county": county_fips}.
    """
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{fips_state}_tract.zip"
    geo_df = gpd.read_file(url)
    if geo_type in ("state"):
        geo_df["GEOID"] = geo_df["GEOID"].str[:5]
    elif geo_type in ("county", "place"):
        geo_df = geo_df[geo_df["COUNTYFP"] == fips_county]
    else:
        pass

    geo_df = geo_df[geo_df["ALAND"] > 0]

    return geo_df


def format_tables(tables):
    """
    Append the suffix '_001E' to each table ID in the list
    (e.g., 'B02001' → 'B02001_001E') to get estimate number

    Args:
        tables (list of str): List of Census table base IDs without suffixes.

    Returns:
        str: string of formatted table IDs with '_001E' appended (suffix)
        separated by comma.

    """
    variables = [f"{t}_001E" for t in tables]
    variables_str = ",".join(variables)
    return variables_str
