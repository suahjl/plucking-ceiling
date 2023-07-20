from datetime import date, datetime, timedelta
import os
from typing import Literal
import time

from ceic_api_client.pyceic import Ceic
from google.cloud import storage
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()


def upload_gcs(bucket_name: str, source_file_name: str, cloud_file_name: str) -> str:
    """
    Upload file to bucket (GCS).

    :bucket_name `str`: Name of bucket in GCS\n
    :source_file_name `str`: Filename of the parquet locally\n
    :cloud_file_name `str`: Filename of the parquet in the bucket\n
    :return `str`: A success or failure message
    """
    try:
        time_start_temp = time.time()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(cloud_file_name)
        blob.upload_from_filename(f"../{bucket_name}/{source_file_name}")
        duration = "{:.1f}".format(time.time() - time_start_temp) + " seconds"
        res = f"SUCCESS ({duration}): {bucket_name}/{cloud_file_name}"
    except Exception as e:
        res = f"FAILURE: {source_file_name}\n\n{e}"

    return res


def write_csv_parquet(filepath: str, df: pd.DataFrame):
    """
    Write CSV and Parquet simultaneously.

    :filepath `str`: Filepath or location of where the CSV and Parquet will be outputted\n
    :df `pd.DataFrame`: A DataFrame instance
    """
    df.to_csv(f"../{filepath}.csv", index=False)
    df.to_parquet(f"../{filepath}.parquet", index=False, compression="brotli")
    print(f"Wrote CSV + Parquet: {filepath}")


def write_parquet(filepath: str, df: pd.DataFrame):
    """
    Write Parquet with customisation.

    :filepath `str`: Filepath or location of where the Parquet will be outputted\n
    :df `pd.DataFrame`: A DataFrame instance
    """
    df.to_parquet(f"../{filepath}.parquet", index=False, compression="brotli")
    print(f"Wrote Parquet: {filepath}")


def get_data_from_ceic(series_ids: list[float], start_date: date) -> pd.DataFrame:
    """
    Get CEIC data.
    Receive a list of series IDs (e.g., [408955597] for CPI inflation YoY) from CEIC
    and output a pandas data frame (for single entity time series).

    :series_ids `list[float]`: a list of CEIC Series IDs\n
    :start_date `date`: a date() object of the start date e.g. date(1991, 1, 1)\n
    :return `pd.DataFrame`: A DataFrame instance of the data
    """
    Ceic.login(username=os.getenv("CEIC_USERNAME"), password=os.getenv("CEIC_PASSWORD"))

    for _ in range(len(series_ids)):
        try:
            series_ids.remove(
                np.nan
            )  # brute force remove all np.nans from series ID list
        except:
            print("no more np.nan")
    k = 1
    frame_consol: pd.DataFrame = pd.DataFrame()

    for series_id in series_ids:
        series_result = Ceic.series(
            series_id=series_id, start_date=start_date
        )  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        time_points_dict = dict(
            (tp.date, tp.value) for tp in y[0].time_points
        )  # this is a list of 1 dictionary,
        series = pd.Series(
            time_points_dict
        )  # convert into pandas series indexed to timepoints

        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol = frame_consol.rename(columns={0: series_name})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame = frame.rename(columns={0: series_name})
            frame_consol = pd.concat(
                [frame_consol, frame], axis=1
            )  # left-right concat on index (time)
        elif k < 1:
            raise NotImplementedError
        k += 1
        frame_consol = frame_consol.sort_index()

    return frame_consol


def append_data(
        shading: Literal["recession"],
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Appends a column in a DataFrame to indicate date-based shading in the dashboard.

    :shading `"recession"`: Type of shading.\n
    :df `pd.DataFrame`: The DataFrame object to append the shading column. Must have a 'date' column.\n
    :return `pd.DataFrame`: A column-appended instance of the DataFrame
    """
    if shading == "recession":
        df[shading] = np.where(
            (
                    (df["date"] >= datetime.strptime("2001-02-01", "%Y-%m-%d"))
                    & (df["date"] <= datetime.strptime("2002-02-28", "%Y-%m-%d"))
            )
            | (
                    (df["date"] >= datetime.strptime("2008-01-01", "%Y-%m-%d"))
                    & (df["date"] <= datetime.strptime("2009-03-31", "%Y-%m-%d"))
            )
            | (
                    (df["date"] >= datetime.strptime("2020-02-01", "%Y-%m-%d"))
                    & (df["date"] <= datetime.strptime("2021-08-31", "%Y-%m-%d"))
            ),
            1,
            0,
        )

    return df
