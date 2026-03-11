"""
MCF Generator Function

Generates MCF (Mapped Collateral Format) output from processed loan tapes.

This module contains the MCFProcessing class and create_mcf_data function.
Note: This is a pandas-compatible version (original uses PySpark).
"""

import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta
from .tag_generator import create_tags_df


def contains_at_least_one_date(series):
    """Check if series contains at least one valid date"""
    converted_series = pd.to_datetime(series, errors='coerce')
    return pd.notna(converted_series).any()


def contains_at_least_one_numeric(series):
    """Check if series contains at least one valid numeric value"""
    converted_series = pd.to_numeric(series, errors='coerce')
    return pd.notna(converted_series).any()


class MCFProcessing:
    """Pandas-compatible MCF processing class"""
    
    def __init__(self, tape_df):
        self.tape = tape_df.copy()
        if self.tape.columns.duplicated().any():
            seen = {}
            new_columns = []
            for col in self.tape.columns:
                if col not in seen:
                    seen[col] = 0
                    new_columns.append(col)
                else:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
            self.tape.columns = new_columns
    
    def _get_column_safe(self, col_name):
        if col_name not in self.tape.columns:
            return None
        col_data = self.tape[col_name]
        if isinstance(col_data, pd.DataFrame):
            return col_data.iloc[:, 0]
        return col_data
    
    def copy_columns(self, setup_df, column_headers):
        for tape_header, setup_header in column_headers:
            col_data = self._get_column_safe(tape_header)
            if col_data is not None:
                if isinstance(col_data, pd.Series):
                    setup_df[setup_header] = col_data.values
                else:
                    setup_df[setup_header] = pd.Series(col_data).values
            else:
                setup_df[setup_header] = ""
        return setup_df
    
    def copy_int_columns(self, setup_df, column_headers):
        for tape_header, setup_header in column_headers:
            col_data = self._get_column_safe(tape_header)
            if col_data is not None:
                if not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data)
                col = pd.to_numeric(col_data, errors='coerce', downcast='integer')
                col = col.astype('object')
                setup_df[setup_header] = col.where(col != 0, "")
            else:
                setup_df[setup_header] = ""
        return setup_df
    
    def copy_float_columns(self, setup_df, column_headers, decimals):
        for tape_header, setup_header in column_headers:
            col_data = self._get_column_safe(tape_header)
            if col_data is not None:
                if not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data)
                col = pd.to_numeric(col_data, errors='coerce')
                col = col.round(decimals)
                setup_df[setup_header] = col.where(pd.notna(col), "")
            else:
                setup_df[setup_header] = ""
        return setup_df
    
    def copy_percentage(self, setup_df, column_headers, round=6, as_decimal=False):
        for tape_header, setup_header in column_headers:
            col_data = self._get_column_safe(tape_header)
            if col_data is None:
                setup_df[setup_header] = ""
            else:
                if not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data)
                col = pd.to_numeric(col_data, errors='coerce')
                mean = col.mean()
                if as_decimal:
                    if mean is not None and mean >= 1:
                        col = col / 100
                    col = col.round(round)
                else:
                    if mean is not None and mean < 1:
                        col = col * 100
                    col = col.round(round)
                setup_df[setup_header] = col.where(pd.notna(col), "")
        return setup_df
    
    def copy_ranked_int_columns(self, setup_df, setup_header, columns):
        found = False
        for column in columns:
            if column in self.tape.columns and not found:
                col_data = self._get_column_safe(column)
                if col_data is not None:
                    if not isinstance(col_data, pd.Series):
                        col_data = pd.Series(col_data)
                    col = pd.to_numeric(col_data, errors='coerce', downcast='integer')
                    col = col.astype('object')
                    setup_df[setup_header] = col.where(col != 0, "")
                    found = True
        if not found:
            setup_df[setup_header] = ""
        return setup_df
    
    def set_amort_type(self, setup_df, setup_header, tape_arm_index, tape_step_rate="", ignore_values=[]):
        setup_df[setup_header] = "FIX"
        if tape_arm_index in self.tape.columns:
            col = self.tape[tape_arm_index].astype(str).str.upper()
            for value in ignore_values:
                col = col.where(col != str(value).upper(), "")
            col = col.str.replace(r'^\s*$|nan|\'|NaN', '', regex=True)
            setup_df[setup_header] = setup_df[setup_header].where(col == "", "ADJ")
        if tape_step_rate and tape_step_rate in self.tape.columns:
            col = pd.to_numeric(self.tape[tape_step_rate], errors='coerce')
            col = col.where(col == 0, col)
            setup_df[setup_header] = setup_df[setup_header].where(pd.isna(col), "STEP")
        return setup_df
    
    def create_eff_date(self, setup_df, eff_date):
        self.tape['EFF_DATE'] = pd.to_datetime(eff_date, format='%Y%m%d')
        setup_df['EFDATE'] = eff_date
        return setup_df
    
    def copy_age(self, setup_df, age_header, setup_header, orig_term_header, rem_term_header, first_pay_header, eff_date_header):
        if age_header in self.tape.columns:
            col = pd.to_numeric(self.tape[age_header], errors='coerce', downcast='integer')
            if rem_term_header in self.tape.columns and orig_term_header in self.tape.columns:
                orig_col = pd.to_numeric(self.tape[orig_term_header], errors='coerce', downcast='integer')
                rem_col = pd.to_numeric(self.tape[rem_term_header], errors='coerce', downcast='integer')
                col = col.where(col != 0, orig_col - rem_col)
            setup_df[setup_header] = col.where(pd.notna(col), 0)
            self.tape['CALC_AGE'] = setup_df[setup_header]
        elif rem_term_header in self.tape.columns and orig_term_header in self.tape.columns:
            orig_col = pd.to_numeric(self.tape[orig_term_header], errors='coerce', downcast='integer')
            rem_col = pd.to_numeric(self.tape[rem_term_header], errors='coerce', downcast='integer')
            setup_df[setup_header] = orig_col - rem_col
            self.tape['CALC_AGE'] = setup_df[setup_header]
        else:
            setup_df[setup_header] = ""
            self.tape['CALC_AGE'] = 0
        return setup_df
    
    def calc_wam(self, setup_df, setup_header, age_header, orig_term, rem_term):
        if orig_term in self.tape.columns:
            self.tape[orig_term] = pd.to_numeric(self.tape[orig_term], errors='coerce', downcast='integer').fillna(0)
            if rem_term in self.tape.columns:
                setup_df[setup_header] = np.where(
                    self.tape[orig_term] > 0,
                    self.tape[orig_term] - self.tape[age_header],
                    pd.to_numeric(self.tape[rem_term], errors='coerce', downcast='integer').fillna(0)
                )
            else:
                setup_df[setup_header] = self.tape[orig_term] - self.tape[age_header]
        elif rem_term in self.tape.columns:
            setup_df[setup_header] = pd.to_numeric(self.tape[rem_term], errors='coerce', downcast='integer').fillna(0)
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def set_amort_term(self, setup_df, setup_header, rem_amort, rem_io, orig_amort, orig_io, age):
        if rem_amort in self.tape.columns and rem_io in self.tape.columns:
            rem_amort_col = pd.to_numeric(self.tape[rem_amort], errors='coerce', downcast='integer').fillna(0).astype(int)
            rem_io_col = pd.to_numeric(self.tape[rem_io], errors='coerce', downcast='integer').fillna(0).astype(int)
            result = (rem_amort_col + rem_io_col).astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        elif orig_amort in self.tape.columns and orig_io in self.tape.columns and age in self.tape.columns:
            orig_amort_col = pd.to_numeric(self.tape[orig_amort], errors='coerce', downcast='integer').fillna(0).astype(int)
            orig_io_col = pd.to_numeric(self.tape[orig_io], errors='coerce', downcast='integer').fillna(0).astype(int)
            age_col = pd.to_numeric(self.tape[age], errors='coerce', downcast='integer').fillna(0).astype(int)
            result = (orig_amort_col + orig_io_col - age_col).astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        elif rem_amort in self.tape.columns:
            rem_amort_col = pd.to_numeric(self.tape[rem_amort], errors='coerce', downcast='integer').fillna(0).astype(int)
            result = rem_amort_col.astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        elif orig_amort in self.tape.columns and age in self.tape.columns:
            orig_amort_col = pd.to_numeric(self.tape[orig_amort], errors='coerce', downcast='integer').fillna(0).astype(int)
            age_col = pd.to_numeric(self.tape[age], errors='coerce', downcast='integer').fillna(0).astype(int)
            result = (orig_amort_col - age_col).astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def interest_only_term(self, setup_df, setup_header, orig_io_header, rem_io_header, age_header):
        if orig_io_header in self.tape.columns:
            orig_io_col = pd.to_numeric(self.tape[orig_io_header], errors='coerce', downcast='integer').fillna(0).astype(int)
            setup_df[setup_header] = orig_io_col
            if rem_io_header in self.tape.columns and age_header in self.tape.columns:
                rem_io_col = pd.to_numeric(self.tape[rem_io_header], errors='coerce', downcast='integer').fillna(0).astype(int)
                age_col = pd.to_numeric(self.tape[age_header], errors='coerce', downcast='integer').fillna(0).astype(int)
                setup_df[setup_header] = np.where(
                    (setup_df[setup_header] == 0) & (rem_io_col > 0),
                    rem_io_col + age_col,
                    setup_df[setup_header]
                )
            result = setup_df[setup_header].astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        elif rem_io_header in self.tape.columns and age_header in self.tape.columns:
            rem_io_col = pd.to_numeric(self.tape[rem_io_header], errors='coerce', downcast='integer').fillna(0).astype(int)
            age_col = pd.to_numeric(self.tape[age_header], errors='coerce', downcast='integer').fillna(0).astype(int)
            result = (rem_io_col + age_col).astype('object')
            setup_df[setup_header] = result.where(result != 0, "")
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def first_pay_date(self, setup_df, setup_header, age_header, date_header):
        if age_header in self.tape.columns:
            setup_df[setup_header] = self.tape.apply(
                lambda x: (x[date_header] - relativedelta(months=x[age_header] - 1)).strftime('%Y%m%d')
                if pd.notna(x[date_header]) and pd.notna(x[age_header]) else "",
                axis=1
            )
        return setup_df
    
    def first_rate_change_date(self, setup_df, setup_header, fixed_header, mos_till_rate_chang_header, age_header, eff_date_header, amort_indicator_header, amort_indicator):
        if fixed_header in self.tape.columns and amort_indicator_header in setup_df.columns:
            self.tape['ORIG_DATE'] = self.tape.apply(
                lambda x: (x[eff_date_header] - relativedelta(months=x[age_header]))
                if pd.notna(x[eff_date_header]) and pd.notna(x[age_header]) else pd.NaT,
                axis=1
            )
            self.tape[fixed_header] = pd.to_numeric(self.tape[fixed_header], errors='coerce', downcast='integer').fillna(0)
            self.tape[fixed_header] = np.where(setup_df[amort_indicator_header] != amort_indicator, 0, self.tape[fixed_header])
            setup_df[setup_header] = np.where(
                (setup_df[amort_indicator_header] == amort_indicator) & (self.tape[fixed_header] > 0),
                self.tape.apply(
                    lambda x: (pd.to_datetime(x['ORIG_DATE']) + relativedelta(months=int(float(x[fixed_header])))).strftime('%Y%m%d')
                    if pd.notna(x['ORIG_DATE']) and pd.notna(x[fixed_header]) else "",
                    axis=1
                ),
                ""
            )
            if mos_till_rate_chang_header in self.tape.columns:
                self.tape[mos_till_rate_chang_header] = pd.to_numeric(self.tape[mos_till_rate_chang_header], errors='coerce', downcast='integer').fillna(0)
                self.tape[mos_till_rate_chang_header] = np.where(setup_df[amort_indicator_header] != amort_indicator, 0, self.tape[mos_till_rate_chang_header])
                setup_df[setup_header] = np.where(
                    (setup_df[amort_indicator_header] == amort_indicator) & (setup_df[setup_header] == ""),
                    self.tape.apply(
                        lambda x: (pd.to_datetime(x[eff_date_header]) + relativedelta(months=int(float(x[mos_till_rate_chang_header])))).strftime('%Y%m%d')
                        if pd.notna(x[eff_date_header]) and pd.notna(x[mos_till_rate_chang_header]) else "",
                        axis=1
                    ),
                    setup_df[setup_header]
                )
        elif mos_till_rate_chang_header in self.tape.columns and amort_indicator_header in setup_df.columns:
            self.tape[mos_till_rate_chang_header] = pd.to_numeric(self.tape[mos_till_rate_chang_header], errors='coerce', downcast='integer').fillna(0)
            self.tape[mos_till_rate_chang_header] = np.where(setup_df[amort_indicator_header] != amort_indicator, 0, self.tape[mos_till_rate_chang_header])
            setup_df[setup_header] = np.where(
                (setup_df[amort_indicator_header] == amort_indicator) & (self.tape[mos_till_rate_chang_header] > 0),
                self.tape.apply(
                    lambda x: (pd.to_datetime(x[eff_date_header]) + relativedelta(months=int(float(x[mos_till_rate_chang_header])))).strftime('%Y%m%d')
                    if pd.notna(x[eff_date_header]) and pd.notna(x[mos_till_rate_chang_header]) else "",
                    axis=1
                ),
                ""
            )
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def next_chang_date(self, setup_df, setup_header, date_column, months_column, amort_indicator_header, amort_indicator):
        if date_column in self.tape.columns and months_column in self.tape.columns and amort_indicator_header in setup_df.columns:
            self.tape[months_column] = pd.to_numeric(self.tape[months_column], errors='coerce', downcast='integer').fillna(0)
            self.tape[months_column] = np.where(setup_df[amort_indicator_header] != amort_indicator, 0, self.tape[months_column])
            setup_df[setup_header] = np.where(
                setup_df[amort_indicator_header] == amort_indicator,
                self.tape.apply(
                    lambda x: (pd.to_datetime(x[date_column]) + relativedelta(months=int(float(x[months_column])))).strftime('%Y%m%d')
                    if pd.notna(x[date_column]) and pd.notna(x[months_column]) else "",
                    axis=1
                ),
                ""
            )
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def copy_fico(self, setup_df, setup_header, fico_column, fico_columns):
        if fico_column in self.tape.columns:
            col = pd.to_numeric(self.tape[fico_column], errors='coerce', downcast='integer')
            col = col.where(col != 0, np.nan)
            setup_df[setup_header] = col.apply(lambda x: int(x) if pd.notna(x) else "").fillna("")
        else:
            tape_ficos = []
            for column in fico_columns:
                if column in self.tape.columns:
                    self.tape[column] = pd.to_numeric(self.tape[column], errors='coerce', downcast='integer')
                    self.tape[column] = self.tape[column].where(self.tape[column] != 0, np.nan)
                    tape_ficos.append(column)
            if not tape_ficos:
                setup_df[setup_header] = ""
            else:
                setup_df[setup_header] = self.tape[tape_ficos].mean(axis=1)
                setup_df[setup_header] = setup_df[setup_header].apply(lambda x: int(x) if pd.notna(x) else "").fillna("")
        return setup_df
    
    def copy_zip_code(self, setup_df, setup_header, tape_header):
        if tape_header in self.tape.columns:
            self.tape[tape_header] = self.tape[tape_header].fillna(0)
            if self.tape[tape_header].astype(str).str.lower().str.contains('x').any():
                self.tape[tape_header] = self.tape[tape_header].astype(str).str.replace('X', '').str.replace('x', '')
            pattern = re.compile(r'[a-wy-zA-WY-Z]')
            contains_letters = self.tape[tape_header].astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notnull(x) else False)
            self.tape[tape_header] = np.where(contains_letters, 0, self.tape[tape_header])
            zip_int = pd.to_numeric(self.tape[tape_header], errors='coerce').fillna(0).astype(int)
            setup_df[setup_header] = np.where(
                zip_int <= 999,
                zip_int.astype(str).str.replace('.0', '').str.ljust(5, 'X'),
                zip_int.astype(str).str.replace('.0', '').str.rjust(5, '0')
            )
            setup_df[setup_header] = np.where(zip_int == 0, '', setup_df[setup_header])
        return setup_df
    
    def create_lien_columns(self, setup_df, lien_header, lien_header_prefix="LS"):
        if lien_header in self.tape.columns:
            lien_col = pd.to_numeric(self.tape[lien_header], errors='coerce')
            unique_positions = sorted(lien_col.dropna().unique())
            for pos in unique_positions:
                col_name = f"{lien_header_prefix}{int(pos)}"
                setup_df[col_name] = lien_col.apply(lambda x: 1 if x == pos else ('' if pd.isna(x) else 0))
        return setup_df
    
    def create_binary_value_columns(self, setup_df, value_header, column_prefix=""):
        if value_header in self.tape.columns:
            value_col = self.tape[value_header].fillna('')
            unique_values = sorted([val for val in value_col.unique() if val != ''])
            for val in unique_values:
                col_name = f"{column_prefix}{val}" if column_prefix else val
                setup_df[col_name] = value_col.apply(lambda x: 1 if x == val else ('' if x == '' else 0))
        return setup_df
    
    def calc_ppp_term(self, setup_df, setup_header, ppp_header, ppp_term_header):
        setup_df[setup_header] = 0
        for header in self.tape.columns:
            if ppp_header in header:
                self.tape[header] = pd.to_numeric(self.tape[header], errors='coerce', downcast='integer').fillna(0)
                setup_df[setup_header] = setup_df[setup_header] + self.tape[header]
        if ppp_term_header in self.tape.columns:
            ppp_term_col = pd.to_numeric(self.tape[ppp_term_header], errors='coerce', downcast='integer').fillna(0)
            setup_df[setup_header] = np.where(setup_df[setup_header] == 0, ppp_term_col, setup_df[setup_header])
        result = setup_df[setup_header].astype('object')
        setup_df[setup_header] = result.where(result != 0, "").fillna("")
        return setup_df
    
    def calc_ppp_int_months(self, setup_df, setup_header, int_months, interest_percentage):
        if int_months in self.tape.columns and interest_percentage in self.tape.columns:
            self.tape[interest_percentage] = pd.to_numeric(self.tape[interest_percentage], errors='coerce').replace(0, np.nan)
            mean = self.tape[interest_percentage].mean(skipna=True)
            if mean > 0:
                self.tape[interest_percentage] = self.tape[interest_percentage] / 100
                int_months_col = pd.to_numeric(self.tape[int_months], errors='coerce')
                setup_df[setup_header] = (int_months_col * self.tape[interest_percentage]).replace(np.nan, "")
            else:
                int_months_col = pd.to_numeric(self.tape[int_months], errors='coerce')
                setup_df[setup_header] = (int_months_col * self.tape[interest_percentage]).replace(np.nan, "")
        else:
            setup_df[setup_header] = ""
        return setup_df
    
    def set_steps(self, setup_df, setup_header, step_date_header, step_rate_header, eff_date_header, age_header, current_rate_header):
        step_date_headers = []
        step_rate_headers = []
        for header in self.tape.columns:
            if step_date_header in header:
                try:
                    step_date_headers.append((header, int(header.rsplit("_", 1)[1])))
                except:
                    pass
            elif step_rate_header in header:
                try:
                    step_rate_headers.append((header, int(header.rsplit("_", 1)[1])))
                except:
                    pass
        step_date_headers.sort(key=lambda x: x[1])
        step_rate_headers.sort(key=lambda x: x[1])
        
        for date_header, pos in step_date_headers:
            if contains_at_least_one_date(self.tape[date_header]):
                self.tape[date_header] = pd.to_datetime(self.tape[date_header], errors='coerce')
            elif contains_at_least_one_numeric(self.tape[date_header]):
                date_check = self.tape[date_header].fillna(0).astype(int).astype(str).replace("0", "")
                if contains_at_least_one_date(date_check):
                    self.tape[date_header] = pd.to_datetime(date_check, errors='coerce')
                else:
                    self.tape[date_header] = pd.to_numeric(self.tape[date_header], errors='coerce', downcast='integer')
            if pd.api.types.is_datetime64_any_dtype(self.tape[date_header]) and pd.isna(self.tape[date_header]).sum() < len(self.tape[age_header]):
                self.tape[date_header] = np.where(
                    self.tape[date_header].dt.day > 1,
                    self.tape[date_header] + pd.offsets.MonthEnd(0) + pd.offsets.Day(1),
                    self.tape[date_header]
                )
            elif pd.isna(self.tape[date_header]).sum() < len(self.tape[age_header]):
                self.tape[eff_date_header] = pd.to_datetime(self.tape[eff_date_header])
                self.tape[age_header] = pd.to_numeric(self.tape[age_header], errors='coerce', downcast='integer').fillna(0).astype(int)
                self.tape[date_header] = self.tape.apply(
                    lambda x: (x[eff_date_header] - pd.DateOffset(months=int(x[age_header]))) + pd.DateOffset(months=int(x[date_header]))
                    if pd.notna(x[date_header]) else pd.NaT,
                    axis=1
                )
            if pd.api.types.is_datetime64_any_dtype(self.tape[date_header]):
                self.tape[date_header] = self.tape[date_header] + pd.DateOffset(months=1)
        
        self.tape[current_rate_header] = pd.to_numeric(self.tape[current_rate_header], errors='coerce')
        incremental = False
        for rate_header, pos in step_rate_headers:
            self.tape[rate_header] = pd.to_numeric(self.tape[rate_header], errors='coerce')
            if pos == 1:
                mean_df = self.tape[self.tape[rate_header].notna()]
                step_rate_mean = mean_df[rate_header].replace(0, np.nan).mean(skipna=True)
                current_rate_mean = mean_df[current_rate_header].replace(0, np.nan).mean(skipna=True)
                if step_rate_mean < current_rate_mean:
                    incremental = True
                    self.tape[rate_header] = self.tape[rate_header] + self.tape[current_rate_header]
                    continue
            if incremental:
                self.tape[rate_header] = self.tape[rate_header] + self.tape[step_rate_headers[pos - 2][0]]
        
        step_headers = [(step_date_headers[i][0], step_rate_headers[i][0]) for i in range(0, min(len(step_date_headers), len(step_rate_headers)))]
        setup_df[setup_header] = ""
        for step_date, step_rate in step_headers:
            if self.tape[step_date].isnull().all():
                continue
            step_date_str = self.tape[step_date].dt.strftime('%Y%m%d').fillna("").astype(str) if pd.api.types.is_datetime64_any_dtype(self.tape[step_date]) else self.tape[step_date].astype(str)
            step_rate_str = self.tape[step_rate].round(6).fillna("").astype(str)
            step_string = np.where(
                (step_date_str != "") & (step_rate_str != ""),
                "STEP = " + step_date_str + " " + step_rate_str,
                ""
            )
            setup_df[setup_header] = np.where(
                step_string == "",
                setup_df[setup_header],
                setup_df[setup_header] + ", " + step_string
            )
        setup_df[setup_header] = setup_df[setup_header].apply(lambda x: x[2:] if x.startswith(", ") else x)
        return setup_df
    
    def clean_df(self, df):
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(r'\s+|\\n', ' ', regex=True)
            df[col] = df[col].str.replace(r"^ +| +$", "", regex=True)
            df[col] = df[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x) if isinstance(x, str) else x)
        return df
    
    def make_col_file(self, df):
        new_headers = []
        header_dict = {}
        for header in df.columns:
            length_list = []
            for _, row in df.iterrows():
                length = len(str(row[header]))
                length_list.append(length)
            max_length = max(length_list) if length_list else 0
            header_padded = header.ljust(max_length, ' ')
            header_formatted = f"[{header_padded}]"
            new_headers.append(header_formatted)
        for i in range(len(df.columns)):
            header_dict[df.columns[i]] = new_headers[i]
        df = df.rename(columns=header_dict)
        df = df.fillna("")
        return df


def create_mcf_data(df, eff_date, format='COL', acode="W", ppmcode="WQ"):
    """
    Create MCF data from processed DataFrame.
    
    Args:
        df: pandas DataFrame with processed loan data
        eff_date: Effective date in YYYYMMDD format
        format: Output format ('COL' or 'JSON', default: 'COL')
        acode: Agency code (default: 'W')
        ppmcode: Prepay model code (default: 'WQ')
    
    Returns:
        String containing MCF output in specified format
    """
    import traceback
    
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    if 'loan_number' not in df.columns:
        if 'LOANID' in df.columns:
            df = df.copy()
            df['loan_number'] = df['LOANID']
        else:
            possible_loan_cols = [col for col in df.columns if 'loan' in col.lower() or col.upper() == 'LOANID']
            if possible_loan_cols:
                df = df.copy()
                df['loan_number'] = df[possible_loan_cols[0]]
            else:
                raise ValueError(f"No loan ID column found. Expected 'loan_number' or 'LOANID'. Available columns: {list(df.columns)[:10]}")
    
    try:
        mcf = MCFProcessing(df)
        setup = pd.DataFrame()
        setup = mcf.copy_columns(setup, [('loan_number', 'LOANID')])
        
        if setup.empty or len(setup) == 0:
            raise ValueError(f"Setup DataFrame is empty after copying loan_number. Input DataFrame had {len(df)} rows.")
        
        setup['ACODE'] = acode
        setup['PPMCODE'] = ppmcode
        setup = mcf.create_eff_date(setup, eff_date)
        setup = mcf.set_amort_type(setup, 'LTYPE', 'arm_index', 'step_rate_1')
        setup = mcf.copy_age(setup, 'age', 'AGE', 'original_term', 'remaining_term', 'first_payment_date', 'EFF_DATE')
        setup = mcf.copy_int_columns(setup, [('original_amortization_term', 'OATERM')])
        setup = mcf.set_amort_term(setup, 'ATERM', 'remaining_amortization_term', 'remaining_interest_only_term', 'original_amortization_term', 'original_interest_only_term', 'CALC_AGE')
        setup = mcf.calc_wam(setup, 'WAM', 'CALC_AGE', 'original_term', 'remaining_term')
        setup = mcf.copy_int_columns(setup, [('original_term', 'OWAM')])
        setup = mcf.interest_only_term(setup, "IOTERM", "original_interest_only_term", "remaining_interest_only_term", "CALC_AGE")
        setup = mcf.copy_int_columns(setup, [('original_term', 'OTERM')])
        setup = mcf.copy_float_columns(setup, [('current_balance', 'BALANCE'), ('original_balance', 'OBALANCE'), ('deferment_balance', 'FORBEARANCE'), ('current_principal_and_interest', 'PANDI')], 2)
        setup = mcf.copy_percentage(setup, [('original_rate', 'OCOUPON'), ('current_rate', 'COUPON'), ('service_fee', 'SERVFEE')], 6)
        setup = mcf.copy_float_columns(setup, [('servicer_flat_fee', 'FEEAMT')], 2)
        setup = mcf.copy_columns(setup, [('day_count', 'DAYCOUNT')])
        setup = mcf.copy_columns(setup, [('arm_index', 'INDEX')])
        setup['INDEX'] = setup['INDEX'].astype(str).str.upper()
        setup = mcf.copy_int_columns(setup, [('arm_lookback_days', 'LOOKBACK')])
        setup = mcf.copy_percentage(setup, [('arm_margin', 'MARGIN'), ('arm_lifetime_cap', 'LICAP'), ('arm_lifetime_floor', 'LIFLOOR'), ('arm_initial_rate_cap', 'IRCAP'), ('arm_periodic_rate_cap', 'PICAP'), ('arm_periodic_rate_floor', 'PIFLOOR')])
        setup = mcf.copy_percentage(setup, [('arm_rounding_factor', 'ROUNDFACT')], 6)
        setup = mcf.copy_int_columns(setup, [('arm_rate_reset_frequency', 'ICFREQ')])
        setup = mcf.copy_ranked_int_columns(setup, 'PCFREQ', ['arm_payment_reset_frequency', 'arm_rate_reset_frequency'])
        setup = mcf.first_pay_date(setup, 'FPDATE', 'CALC_AGE', 'EFF_DATE')
        setup = mcf.first_rate_change_date(setup, 'FRDATE', 'arm_initial_fixed_rate_period', 'arm_months_until_next_rate_reset', 'CALC_AGE', 'EFF_DATE', 'LTYPE', 'ADJ')
        setup = mcf.next_chang_date(setup, 'ICDATE', 'EFF_DATE', 'arm_months_until_next_rate_reset', 'LTYPE', 'ADJ')
        setup = mcf.next_chang_date(setup, 'PCDATE', 'EFF_DATE', 'arm_months_until_next_payment_reset', 'LTYPE', 'ADJ')
        setup = mcf.copy_int_columns(setup, [('group', 'GCODE')])
        setup = mcf.copy_percentage(setup, [('current_loan_to_value', 'LTV'), ('original_loan_to_value', 'OLTV'), ('current_combined_loan_to_value', 'CLTV'), ('back_end_debt_to_income', 'DTI')], round=6, as_decimal=True)
        setup = mcf.copy_fico(setup, 'CSCORE', 'primary_borrower_original_fico', ['primary_borrower_original_fico_equifax', 'primary_borrower_original_fico_experian', 'primary_borrower_original_fico_transunion'])
        setup = mcf.copy_columns(setup, [('property_state', 'STATE')])
        setup = mcf.copy_zip_code(setup, 'ZIP', 'zip_code')
        setup = mcf.create_binary_value_columns(setup, 'occupancy')
        setup = mcf.create_binary_value_columns(setup, 'loan_purpose')
        setup = mcf.create_lien_columns(setup, 'lien')
        setup = mcf.copy_int_columns(setup, [('number_of_units', 'NUMUNITS')])
        setup = mcf.calc_ppp_term(setup, 'PETERM', 'prepayment_penalty_gross_percentage', 'prepayment_penalty_term')
        setup = mcf.calc_ppp_int_months(setup, 'PENINTEREST', 'prepayment_penalty_gross_interest_months', 'prepayment_penalty_gross_interest_percentage')
        setup = mcf.copy_columns(setup, [('prepayment_penalty_calculation', 'PENRATE')])
        setup = mcf.set_steps(setup, 'MODIFIERS', 'step_date', 'step_rate', 'EFF_DATE', 'CALC_AGE', 'current_rate')
        
        nqm_tags_tape = create_tags_df(mcf.tape, loan_number_header='loan_number')
        setup['LOANID'] = setup['LOANID'].astype(str)
        nqm_tags_tape['loan_number'] = nqm_tags_tape['loan_number'].astype(str)
        setup = pd.merge(setup, nqm_tags_tape, how='left', left_on='LOANID', right_on='loan_number')
        
        if 'loan_number' in setup.columns:
            setup.drop(columns=['loan_number'], inplace=True)
    
        setup = mcf.clean_df(setup)
        
        if setup.empty:
            raise ValueError("Setup DataFrame is empty after processing. Check input data and column mappings.")
        
        if format == 'COL':
            col_df = mcf.make_col_file(setup)
            output = col_df.to_string(index=False)
            if not output or len(output.strip()) == 0:
                raise ValueError(f"MCF COL format output is empty. Setup DataFrame has {len(setup)} rows and {len(setup.columns)} columns.")
            return output
        elif format == 'JSON':
            output = setup.to_json(orient='records')
            if not output or len(output.strip()) == 0:
                raise ValueError(f"MCF JSON format output is empty. Setup DataFrame has {len(setup)} rows and {len(setup.columns)} columns.")
            return output
        else:
            raise ValueError("Invalid format. Please choose either 'COL' or 'JSON'.")
    
    except Exception as e:
        error_msg = f"MCF generation failed: {str(e)}\n"
        error_msg += f"Input DataFrame shape: {df.shape}\n"
        error_msg += f"Input DataFrame columns: {list(df.columns)[:20]}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e

