"""
Data Processor Functions

Functions for processing data types and transforming values according to schema.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame by:
    - Converting column names to lowercase
    - Normalizing whitespace
    - Removing empty columns
    - Handling duplicate column names
    """
    def single_space(text):
        if isinstance(text, str):
            return re.sub(r'\s+', ' ', text)
        return text
    
    df.columns = [single_space(str(col)).lower() for col in df.columns]
    
    df = df.apply(
        lambda col: col.str.lower().apply(single_space) 
        if col.dtype == 'object' and col.apply(lambda x: isinstance(x, str)).all() 
        else col
    )
    
    df = df.map(single_space)
    df = df.dropna(axis=1, how='all')
    
    if any(df.columns.duplicated()):
        col_counts = {}
        new_columns = []
        for col in df.columns:
            if col in col_counts:
                col_counts[col] += 1
                new_columns.append(f"{col}.{col_counts[col]}")
            else:
                col_counts[col] = 0
                new_columns.append(col)
        df.columns = new_columns
    
    for col in df.columns:
        if df[col].dtype == 'object' and (df[col] == '').all():
            df = df.drop(col, axis=1)
    
    return df


def process_data_types(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """
    Process DataFrame columns to match schema data types.
    """
    df = df.copy()
    
    def get_column_safe(df, col_name):
        if col_name not in df.columns:
            return None
        try:
            col_data = df[col_name]
            if isinstance(col_data, pd.Series):
                return col_data
            elif isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            else:
                return pd.Series(col_data)
        except Exception as e:
            try:
                if col_name in df.columns:
                    col_idx = df.columns.get_loc(col_name)
                    if isinstance(col_idx, int):
                        return df.iloc[:, col_idx]
                    else:
                        return df.iloc[:, col_idx[0]]
            except Exception:
                pass
            print(f"[get_column_safe] ⚠️ Could not safely get column '{col_name}': {e}")
            return None
    
    for header in df.columns:
        base_header = header.split('.')[0] if '.' in header else header
        
        if base_header in schema.get('headers', {}) or header in schema.get('headers', {}):
            schema_header = base_header if base_header in schema.get('headers', {}) else header
            data_type = schema['headers'][schema_header].get("data_type")
            
            col_data = get_column_safe(df, header)
            if col_data is None:
                continue
            
            try:
                if data_type == "str":
                    df[header] = col_data.fillna('').astype(str)
                
                elif data_type == "int":
                    numeric_series = pd.to_numeric(col_data, errors='coerce')
                    int_series = numeric_series.round().astype('Int64')
                    int_series_obj = int_series.astype(object)
                    df[header] = int_series_obj.where(pd.notna(int_series_obj), '')
                
                elif data_type == "float":
                    numeric_series = pd.to_numeric(col_data, errors='coerce')
                    if "round" in schema['headers'][schema_header].get('transformations', {}):
                        round_places = schema['headers'][schema_header]['transformations']['round']
                        numeric_series = numeric_series.round(round_places)
                    df[header] = numeric_series.fillna('')
                
                elif data_type == "date":
                    date_series = pd.to_datetime(col_data, errors='coerce')
                    date_format = schema['headers'][schema_header].get('transformations', {}).get('date_format', "%Y%m%d")
                    df[header] = date_series.dt.strftime(date_format).fillna('')
            except Exception as e:
                print(f"[process_data_types] ⚠️ Failed to process column '{header}': {e}")
                continue
    
    return df


def transform_values(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """
    Transform column values according to schema mappings.
    """
    df = df.copy()
    
    for standard_header, details in schema.get('headers', {}).items():
        if standard_header in df.columns:
            if 'transformations' in details:
                for transformation, rule in details['transformations'].items():
                    if transformation == 'mappings':
                        for target_value, aliases in rule.items():
                            df[standard_header] = df[standard_header].replace(aliases, target_value)
    
    return df

