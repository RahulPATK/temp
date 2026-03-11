"""
Tag Generator Function

Generates NQM tags for loan tapes.

This module contains all helper functions needed for tag generation.
"""

import numpy as np
import pandas as pd
import re


def standardize_flag_values(tape, tape_header, zero_errors=False, standardize_flag_values=[]):
    tape[tape_header] = tape[tape_header].astype(str).str.upper()
    tape[tape_header] = tape[tape_header].apply(lambda x: x.strip() if isinstance(x, str) else x)
    tape[tape_header] = np.where(tape[tape_header] == "N", "0", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "NO", "0", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "F", "0", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "FALSE", "0", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "Y", "1", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "YES", "1", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "T", "1", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "TRUE", "1", tape[tape_header])
    tape[tape_header] = np.where(tape[tape_header] == "99", "", tape[tape_header])
    if standardize_flag_values:
        for value in standardize_flag_values:
            tape[tape_header] = np.where(tape[tape_header] == value.upper(), "1", tape[tape_header])
    tape[tape_header] = pd.to_numeric(tape[tape_header], downcast='integer', errors='coerce')
    tape[tape_header] = tape[tape_header].apply(lambda x: int(x) if x == x else "").replace(np.nan, "")
    tape[tape_header] = tape[tape_header].apply(lambda x: int(x) if x == 0 or x == 1 else "").replace(np.nan, "")
    if zero_errors:
        tape[tape_header] = np.where(tape[tape_header] == 1, 1, 0)
    return tape


def create_tag(tape, tag_name, header, servicer, deal_code):
    if header in tape.columns:
        tape[tag_name + '_TAGS'] = servicer + "|" + deal_code + "|" + tape['loan_number'].astype(str) + f"|{tag_name}|" + tape[header].astype(str)
    return tape


def create_flag_tags(tape, servicer, deal_code, tag, tape_header):
    headers = duplicate_headers(tape, tape_header)
    if headers:
        tape = merge_standardize_flag_columns(tape, tag, headers)
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def merge_flag_columns(tape, tag, tape_headers):
    tape[tag] = ""
    for header in tape_headers:
        if header in tape.columns:
            tape[header] = pd.to_numeric(tape[header], downcast='integer', errors='coerce')
            tape[header] = tape[header].apply(lambda x: int(x) if x == x else "").replace(np.nan, "")
            tape[tag] = np.where(tape[tag] != 1, tape[header], tape[tag])
    return tape


def merge_float_columns(tape, tag, tape_headers, decimals=4, zeros=True):
    tape[tag] = 0
    for header in tape_headers:
        tape[header] = pd.to_numeric(tape[header], downcast='float', errors='coerce').replace(np.nan, 0)
        tape[tag] = np.where(tape[tag] > 0, tape[tag], tape[header])
    tape[tag] = tape[tag].round(decimals)
    if zeros is False:
        tape[tag] = np.where(tape[tag] > 0, tape[tag], "")
    return tape


def create_binary_tag_column(tape, tape_headers, tag, values):
    if tag not in tape.columns:
        for tape_header in tape_headers:
            tape[tape_header] = tape[tape_header].apply(lambda x: 1 if x in values else 0)
        tape = merge_flag_columns(tape, tag, tape_headers)
    return tape


def create_binary_value_column(tape, tag_name, extract_dict):
    for extract_column in extract_dict.keys():
        if tag_name in tape.columns:
            break
        tape_columns = duplicate_headers(tape, extract_column)
        if tape_columns:
            for column in tape_columns:
                tag_values = extract_dict.get(extract_column)
                for value in tag_values:
                    if value in set(tape[column]):
                        tape = create_binary_tag_column(tape, tape_columns, tag_name, tag_values)
                        break
    return tape


def duplicate_headers(tape, header):
    headers = []
    for tape_header in tape.columns:
        if tape_header == header:
            tape[f"copy_{tape_header}"] = tape[tape_header]
            headers.append(f"copy_{tape_header}")
        elif tape_header.startswith(f"{header}."):
            tape[f"copy_{tape_header}"] = tape[tape_header]
            headers.append(f"copy_{tape_header}")
    return headers


def merge_standardize_flag_columns(tape, tag, headers, flag_column_values=[]):
    for header in headers:
        tape = standardize_flag_values(tape, header, standardize_flag_values=flag_column_values)
    tape = merge_flag_columns(tape, tag, headers)
    return tape


def create_tag_flag_value(tape, servicer, deal_code, tag_name, flag_column, extract_dict, flag_column_values=[]):
    headers = duplicate_headers(tape, flag_column)
    if headers:
        tape = merge_standardize_flag_columns(tape, tag_name, headers, flag_column_values=flag_column_values)
    else:
        tape = create_binary_value_column(tape, tag_name, extract_dict)
    tape = create_tag(tape, tag_name, tag_name, servicer, deal_code)
    return tape


def create_binary_bal_column(tape, tag, bal_header, balance, operator='>'):
    operators = ['>', '<', '=', '<=', '>=', "!="]
    if operator not in operators:
        raise ValueError("Invalid operator type. Expected one of: %s" % operators)
    headers = duplicate_headers(tape, bal_header)
    if headers:
        for header in headers:
            tape[header] = pd.to_numeric(tape[header], downcast='float', errors='coerce')
            if operator == '>':
                tape[header] = np.where((tape[header].notna()) & (tape[header] > balance), 1, 0)
            elif operator == '<':
                tape[header] = np.where((tape[header].notna()) & (tape[header] < balance), 1, 0)
            elif operator == '=':
                tape[header] = np.where((tape[header].notna()) & (tape[header] == balance), 1, 0)
            elif operator == '<=':
                tape[header] = np.where((tape[header].notna()) & (tape[header] <= balance), 1, 0)
            elif operator == '>=':
                tape[header] = np.where((tape[header].notna()) & (tape[header] >= balance), 1, 0)
            elif operator == '!=':
                tape[header] = np.where((tape[header].notna()) & (tape[header] != balance), 1, 0)
        tape = merge_flag_columns(tape, tag, headers)
    return tape


def remove_non_numerics(s):
    return re.sub('[^0-9.]', '', s)


def create_percentage_tags(tape, servicer, deal_code, tag, tape_header, whole_num=True, cap=False, cap_value=5, mean_check=False):
    headers = duplicate_headers(tape, tape_header)
    if headers:
        for float_column in headers:
            tape[float_column] = tape[float_column].astype(str).apply(remove_non_numerics)
            tape[float_column] = pd.to_numeric(tape[float_column], downcast='float', errors='coerce')
            if mean_check:
                mean = tape[float_column].replace(0, np.nan).mean(skipna=True)
                if whole_num and mean < 1:
                    tape[float_column] = tape[float_column] * 100
                elif not whole_num and mean > 1:
                    tape[float_column] = tape[float_column] / 100
            if cap:
                tape[float_column] = tape[float_column].apply(lambda x: cap_value if x > cap_value else x)
            tape[float_column] = tape[float_column].round(decimals=4)
        tape = merge_float_columns(tape, tag, headers, decimals=4, zeros=False)
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def create_balance_tags(tape, servicer, deal_code, tag, tape_header):
    headers = duplicate_headers(tape, tape_header)
    if headers:
        tape[tag] = np.nan
        for header in headers:
            tape[header] = pd.to_numeric(tape[header], errors='coerce').round(2)
            tape[tag] = np.where(tape[tag].isna(), tape[header], tape[tag])
        tape[tag] = tape[tag].fillna("")
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def add_columns(tape, tag, columns):
    for column in columns:
        if column not in tape.columns:
            continue
        if tag not in tape.columns:
            tape[tag] = pd.to_numeric(tape[column], errors='coerce')
        else:
            tape[column] = pd.to_numeric(tape[column], errors='coerce')
            tape[tag] = tape.apply(lambda x: add_nan_logic(x[tag], x[column]), axis=1)
    return tape


def add_nan_logic(v1, v2):
    if np.isnan(v1) and np.isnan(v1):
        return np.nan
    elif np.isnan(v1):
        return v2
    elif np.isnan(v2):
        return v1
    else:
        return v1 + v2


def create_summed_tags(tape, servicer, deal_code, tag, columns):
    merged_columns = []
    for column in columns:
        headers = duplicate_headers(tape, column)
        if headers:
            first_column = headers[0]
            merged_columns.append(first_column)
            tape[first_column] = pd.to_numeric(tape[first_column], errors='coerce')
            for header in headers:
                tape[header] = pd.to_numeric(tape[header], errors='coerce')
                tape[first_column] = np.where(tape[first_column].isna(), tape[header], tape[first_column])
    if merged_columns:
        tape = add_columns(tape, tag, merged_columns)
        tape = create_balance_tags(tape, servicer, deal_code, tag, tag)
    return tape


def create_balance_summed_tags(tape, servicer, deal_code, tag, bal_header, columns):
    if bal_header in tape.columns:
        create_balance_tags(tape, servicer, deal_code, tag, bal_header)
    else:
        create_summed_tags(tape, servicer, deal_code, tag, columns)
    return tape


def export_tags(df):
    tag_columns = [c for c in df.columns if c.endswith("_TAGS")]
    columns_to_keep = ["loan_number"] + tag_columns
    df_filtered = df[columns_to_keep].copy()
    new_column_names = {}
    for col in df_filtered.columns:
        if col.endswith("_TAGS"):
            new_column_names[col] = col[:-5]
        else:
            new_column_names[col] = col
    df_filtered = df_filtered.rename(columns=new_column_names)
    for col in df_filtered.columns:
        if col != "loan_number":
            df_filtered[col] = df_filtered[col].astype(str).apply(
                lambda x: x.split('|')[-1] if '|' in x else x
            )
    return df_filtered


def create_value_tags(tape, servicer, deal_code, tag_name, extract_dict):
    tape = create_binary_value_column(tape, tag_name, extract_dict)
    tape = create_tag(tape, tag_name, tag_name, servicer, deal_code)
    return tape


def create_int_tags(tape, servicer, deal_code, tag, column, zero=True, max_int=False):
    headers = duplicate_headers(tape, column)
    if headers:
        tape[tag] = ""
        for header in headers:
            tape[header] = pd.to_numeric(tape[header], downcast='integer', errors='coerce')
            if max_int:
                tape[header] = np.where(tape[header] > max_int, np.nan, tape[header])
            tape[header] = tape[header].apply(lambda x: int(x) if x == x else "").replace("nan", "").fillna("")
            tape[tag] = np.where(tape[tag] == "", tape[header], tape[tag])
        if zero is not True:
            tape[tag] = tape[tag].replace(0, "")
        tape[tag] = tape[tag].replace("nan", "").fillna("")
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def create_larger_value_tags(tape, servicer, deal_code, tag, columns, max_value):
    headers = []
    for column in columns:
        header_list = duplicate_headers(tape, column)
        headers = headers + header_list
    if headers:
        for column in headers:
            if tag not in tape.columns:
                tape[tag] = pd.to_numeric(tape[column], errors='coerce').replace(np.nan, 0)
            else:
                tape[column] = pd.to_numeric(tape[column], errors='coerce').replace(np.nan, 0)
                tape[tag] = np.where(tape[column] > tape[tag], tape[column], tape[tag])
        tape[tag] = tape[tag].replace(0, "")
        tape[tag] = pd.to_numeric(tape[tag], errors='coerce').replace(np.nan, 0)
        tape[tag] = np.where(tape[tag] > max_value, max_value, tape[tag])
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def create_split_int_column(tape, servicer, deal_code, tag_name, extract_dict, max_value):
    for extract_column in extract_dict.keys():
        if tag_name in tape.columns:
            break
        re_search_values = extract_dict.get(extract_column)
        tape_columns = duplicate_headers(tape, extract_column)
        if tape_columns:
            tape[tag_name] = 0
            for column in tape_columns:
                tape[column] = tape[column].fillna("")
                tape[column] = tape[column].apply(lambda x: split_logic(x, re_search_values))
                tape[column] = pd.to_numeric(tape[column], downcast='integer', errors='coerce')
                tape[column] = tape[column].apply(lambda x: int(x) if x == x else np.nan).replace(np.nan, 0)
                tape[tag_name] = np.where(tape[column] > tape[tag_name], tape[column], tape[tag_name])
            tape = create_int_tags(tape, servicer, deal_code, tag_name, tag_name, zero=False, max_int=max_value)
            break
    return tape


def split_logic(x, re_search):
    split_values = []
    for pattern in re_search:
        split_search = re.search(r"(\d+)" + pattern, x)
        if split_search:
            split_values.append(int(split_search.groups(1)[0]))
    if split_values:
        split_value = max(split_values)
        return split_value
    return 0


def create_non_empty_tags(tape, servicer, deal_code, tag, columns):
    headers = []
    for column in columns:
        header_list = duplicate_headers(tape, column)
        headers = headers + header_list
    if headers:
        tape[tag] = ""
        for header in headers:
            tape[header] = tape[header]
            tape[header] = np.where(tape[header] is None, "", tape[header])
            tape[header] = np.where(tape[header].astype(str).str.upper() == "NONE", "", tape[header])
            tape[header] = tape[header].replace(np.nan, "")
            tape[header] = np.where(tape[header] == "", "", '1')
            tape[tag] = np.where(tape[tag] == "", tape[header], tape[tag])
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def create_sub_string_flag_tags(tape, servicer, deal_code, tag, column, values):
    headers = duplicate_headers(tape, column)
    if headers:
        tape[tag] = "0"
        for header in headers:
            tape[header] = tape[header].fillna("").astype(str)
            tape[header] = tape[header].apply(lambda x: sub_string_check_logic(x, values))
            tape[tag] = np.where(tape[tag] == "0", tape[header], tape[tag])
        tape[tag] = tape[tag].replace("0", "")
        tape = create_tag(tape, tag, tag, servicer, deal_code)
    return tape


def sub_string_check_logic(x, values):
    for value in values:
        if value in x:
            return "1"
    return "0"


def check_row(tape, row, criteria):
    for base_column, conditions in criteria.items():
        for column in tape.columns:
            if column == base_column or column.startswith(base_column + '.'):
                for value, exact_match in conditions:
                    cell_value = row[column]
                    if isinstance(cell_value, float) and cell_value.is_integer():
                        cell_value = int(cell_value)
                    if (exact_match and str(cell_value) == str(value)) or (not exact_match and str(value) in str(row[column])):
                        return 1
    return 0


def create_tags_multi_criteria(tape, servicer, deal_code, tag_name, *criteria_args, header_found=False, custom_values=None, value_to_change=False):
    if not custom_values:
        passed_value = 1
        not_passed = 0
    else:
        passed_value, not_passed = custom_values
    for criteria in criteria_args:
        for key in criteria:
            if key in tape.columns:
                header_found = True
                break
    if header_found:
        if tag_name not in tape.columns:
            tape[tag_name] = not_passed
        for index, row in tape.iterrows():
            if value_to_change is False:
                if str(tape.at[index, tag_name]) == str(passed_value):
                    continue
            if value_to_change is not False:
                if str(tape.at[index, tag_name]) not in value_to_change:
                    continue
            passed_all_stages = True
            for stage_criteria in criteria_args:
                if not check_row(tape, row, stage_criteria):
                    passed_all_stages = False
                    break
            if passed_all_stages:
                value = passed_value
            else:
                value = not_passed
            tape.at[index, tag_name] = value
        tape = create_tag(tape, tag_name, tag_name, servicer, deal_code)
    return tape


def create_tag_from_other_tags(tape, servicer, deal_code, tag_name, tags, tag_value=0):
    if tag_name not in tape.columns:
        tape[tag_name] = 0
    found_tags = []
    for tag in tags:
        if tag in tape.columns:
            found_tags.append(tag)
    if found_tags:
        tape[tag_name + 'tags_merged'] = tape[found_tags].max(axis=1)
        tape[tag_name + 'merged_reversed'] = 1 - tape[tag_name + 'tags_merged']
        tape[tag_name] = np.where(tape[tag_name] == tag_value, tape[tag_name + 'merged_reversed'], tape[tag_name])
    tape = create_tag(tape, tag_name, tag_name, servicer, deal_code)
    return tape


def create_tags_df(tape, servicer='SERVICER', deal_code='DEAL_CODE', loan_number_header='LOANID', lpad=0):
    """
    Create NQM tags DataFrame from loan tape.
    """
    tape = tape.copy()
    tape['loan_number'] = tape[loan_number_header].apply(lambda x: str(int(float(x))).zfill(lpad) if isinstance(x, float) else str(x).zfill(lpad))
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_SELF', {'primary_borrower_self_employment_flag': [(1, True), ('self_employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_SELF', {'primary_borrower_employment_detail': [('self_employed', True)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_EMPLOYED', {'primary_borrower_employment_flag': [(1, True), ('employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_EMPLOYED', {'primary_borrower_employment_detail': [('employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_EMPLOYED', {'primary_borrower_self_employment_flag': [('employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_EMPLOYED', {'documentation_type': [('w2', False), ('wvoe', False), ('vvoe', False), ('paystub ', False)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_RETIRED', {'primary_borrower_retirement_flag': [(1, True), ('retired', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_RETIRED', {'primary_borrower_employment_detail': [('retired', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_RETIRED', {'primary_borrower_self_employment_flag': [('retired', True)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNEMPLOYED', {'primary_borrower_unemployment_flag': [(1, True), ('unemployed', True), ('not_employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNEMPLOYED', {'primary_borrower_employment_detail': [('unemployed', True), ('not_employed', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNEMPLOYED', {'primary_borrower_self_employment_flag': [('unemployed', True), ('not_employed', True)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNKNOWN', {'primary_borrower_employment_unknown_flag': [(1, True), ('unknown', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNKNOWN', {'primary_borrower_employment_detail': [('unknown', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_UNKNOWN', {'primary_borrower_self_employment_flag': [('unknown', True)]}, header_found=True)
    tape = create_tag_from_other_tags(tape, servicer, deal_code, 'BRWR_EMP_UNKNOWN', ['BRWR_EMP_SELF', 'BRWR_EMP_EMPLOYED', 'BRWR_EMP_RETIRED', 'BRWR_EMP_UNEMPLOYED'])
    tape = create_binary_bal_column(tape, 'BRWR_EMP_DSCR', 'debt_service_coverage_ratio', 0, '>')
    if 'BRWR_EMP_DSCR' not in tape.columns:
        tape = create_binary_value_column(tape, 'BRWR_EMP_DSCR', {'documentation_type': ['debt_service_coverage_ratio', 'no_ratio']})
    tape = create_tag(tape, 'BRWR_EMP_DSCR', 'BRWR_EMP_DSCR', servicer, deal_code)
    tape = create_percentage_tags(tape, servicer, deal_code, 'UW_DSCR_VALUE', 'debt_service_coverage_ratio', cap=True, cap_value=5)
    tape = create_tag_flag_value(tape, servicer, deal_code, 'NQM_ATR_NON_QM', 'non_qualified_mortgage_flag', {'ability_to_repay_type': ['nqm']})
    tape = create_tag_flag_value(tape, servicer, deal_code, 'NQM_ATR_REBUTTABLE', 'non_qualified_mortgage_rebuttal_flag', {'ability_to_repay_type': ['rebuttal']})
    tape = create_tag_flag_value(tape, servicer, deal_code, 'NQM_ATR_EXEMPT', 'non_qualified_mortgage_exempt_flag', {'ability_to_repay_type': ['exempt', 'not_covered']})
    tape = create_tag_flag_value(tape, servicer, deal_code, 'NQM_ATR_SAFE_HARBOR', 'non_qualified_mortgage_safe_harbor_flag', {'ability_to_repay_type': ['safe_harbor', 'qm_safe_harbor']})
    tape = create_tag_flag_value(tape, servicer, deal_code, 'NQM_ATR_QUALIFIED', 'qualified_mortgage_flag', {'ability_to_repay_type': ['qm']})
    tape = create_flag_tags(tape, servicer, deal_code, 'UW_4506T', '4506-t_flag')
    citizen_criterias = [
        {'primary_borrower_us_citizen_flag': [(1, True), ('us_citizen', True), ('permanent_resident_alien', True)]},
        {'primary_borrower_citizenship': [('us_citizen', True), ('permanent_resident_alien', True)]},
        {'primary_borrower_foreign_national_flag': [(0, True)]},
        {'primary_borrower_foreign_national_flag': [('us_citizen', True), ('permanent_resident_alien', True)]}
    ]
    for criteria in citizen_criterias:
        tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_CITIZEN', criteria)
    tape = create_balance_tags(tape, servicer, deal_code, 'LOAN_JR_LIEN_BALANCE', 'junior_mortgage_balance')
    tape = create_binary_bal_column(tape, 'BRWR_BANKSTMT_PERSONAL', 'months_of_personal_bank_statements', 0, '>')
    tape = create_tag(tape, 'BRWR_BANKSTMT_PERSONAL', 'BRWR_BANKSTMT_PERSONAL', servicer, deal_code)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_PERSONAL', {'type_of_bank_statements': [('personal', False)]}, custom_values=(1, 0))
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_PERSONAL', {'documentation_type': [('personal_bank_statement', False)]}, custom_values=(1, 0), header_found=True)
    tape = create_binary_bal_column(tape, 'BRWR_BANKSTMT_BUSINESS', 'months_of_business_bank_statements', 0, '>')
    tape = create_tag(tape, 'BRWR_BANKSTMT_BUSINESS', 'BRWR_BANKSTMT_BUSINESS', servicer, deal_code)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_BUSINESS', {'type_of_bank_statements': [('business', False)]}, custom_values=(1, 0))
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_BUSINESS', {'documentation_type': [('business_bank_statement', False)]}, custom_values=(1, 0), header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_UNSPEC', {'type_of_bank_statements': [('unknown', True), ('unspecified', True)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_UNSPEC', {'documentation_type': [('personal_bank_statement', False), ('business_bank_statement', False)]}, custom_values=(0, 0))
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_UNSPEC', {'documentation_type': [('bank_statement', False)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_BANKSTMT_UNSPEC', {'primary_borrower_bank_statement_flag': [(1, False)]})
    tape = create_tag_from_other_tags(tape, servicer, deal_code, 'BRWR_BANKSTMT_UNSPEC', ['BRWR_BANKSTMT_PERSONAL', 'BRWR_BANKSTMT_BUSINESS'], tag_value=1)
    tape = create_flag_tags(tape, servicer, deal_code, 'UW_TILA_RESPA', 'tila_respa_integrated_disclosure_flag')
    tape = create_larger_value_tags(tape, servicer, deal_code, 'BRWR_BANKSTMT_COUNT', ['months_of_personal_bank_statements', 'months_of_business_bank_statements', 'months_of_bank_statements'], 60)
    if 'BRWR_BANKSTMT_COUNT' not in tape.columns:
        tape = create_split_int_column(tape, servicer, deal_code, 'BRWR_BANKSTMT_COUNT', {'documentation_type': ['_month_personal_bank_statement', '_month_business_bank_statement', '_month_bank_statement']}, 60)
    tape = create_balance_tags(tape, servicer, deal_code, 'UW_TILA_FEES', 'truth_in_lending_act_fee')
    create_summed_tags(tape, servicer, deal_code, 'BRWR_INCOME_WAGE', ['primary_borrower_wage_income', 'primary_borrower_bonus_income', 'primary_borrower_commission_income'])
    tape = create_balance_tags(tape, servicer, deal_code, 'BRWR_INCOME_OTHER', 'primary_borrower_other_income')
    tape = create_balance_tags(tape, servicer, deal_code, 'BRWR_INCOME_RESIDUAL', 'primary_borrower_residual_income')
    tape = create_balance_summed_tags(tape, servicer, deal_code, 'BRWR_INCOME_TOTAL', 'primary_borrower_total_income', ['primary_borrower_wage_income', 'primary_borrower_other_income', 'primary_borrower_bonus_income', 'primary_borrower_commission_income'])
    create_summed_tags(tape, servicer, deal_code, 'COBRWR_INCOME_WAGE', ['secondary_borrower_wage_income', 'secondary_borrower_bonus_income', 'secondary_borrower_commission_income'])
    tape = create_balance_tags(tape, servicer, deal_code, 'COBRWR_INCOME_OTHER', 'secondary_borrower_other_income')
    tape = create_balance_tags(tape, servicer, deal_code, 'COBRWR_INCOME_RESIDUAL', 'secondary_borrower_residual_income')
    tape = create_balance_summed_tags(tape, servicer, deal_code, 'COBRWR_INCOME_TOTAL', 'secondary_borrower_total_income', ['secondary_borrower_wage_income', 'secondary_borrower_other_income', 'secondary_borrower_bonus_income', 'secondary_borrower_commission_income'])
    tape = create_balance_tags(tape, servicer, deal_code, 'DSCR_RENTAL_INCOME', 'rental_income_debt_service_coverage_ratio')
    tape = create_int_tags(tape, servicer, deal_code, 'INCOME_VERIFICATION_LVL', 'primary_borrower_income_verification_level', zero=False, max_int=5)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'LOAN_CROSS_COLLAT', {'cross_collateralized_flag': [(1, True)]}, header_found=True)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_ASSET', {'primary_borrower_asset_depletion_flag': [(1, True)]})
    if 'BRWR_EMP_ASSET' not in tape.columns:
        tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_ASSET', {'documentation_type': [('asset', False)]})
    tape = create_balance_tags(tape, servicer, deal_code, 'BRWR_ASSETS', 'original_pledged_assets')
    tape = create_balance_tags(tape, servicer, deal_code, 'BRWR_BUSINESS_INCOME', 'rental_income_from_lease')
    if 'BRWR_BUSINESS_INCOME' not in tape.columns:
        tape = create_balance_tags(tape, servicer, deal_code, 'BRWR_BUSINESS_INCOME', 'rental_income_debt_service_coverage_ratio')
    tape = create_binary_bal_column(tape, 'BRWR_EMP_BUSINESS_1', 'months_of_business_bank_statements', 0, '>')
    tape = create_binary_bal_column(tape, 'BRWR_EMP_BUSINESS_2', 'debt_service_coverage_ratio', 0, '>')
    tape = merge_flag_columns(tape, 'BRWR_EMP_BUSINESS', ['BRWR_EMP_BUSINESS_1', 'BRWR_EMP_BUSINESS_2'])
    tape = create_tag(tape, 'BRWR_EMP_BUSINESS', 'BRWR_EMP_BUSINESS', servicer, deal_code)
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_BUSINESS', {'type_of_bank_statements': [('business', True)]})
    tape = create_tags_multi_criteria(tape, servicer, deal_code, 'BRWR_EMP_BUSINESS', {
        'documentation_type': [('p&l', False),
                               ('property_focused_investor_loan', True),
                               ('debt_service_coverage_ratio', True),
                               ('business_bank_statement', False)]}, header_found=True)
    tape = create_flag_tags(tape, servicer, deal_code, 'FIRST_HOME', 'first_time_buyer_flag')
    tape = export_tags(tape)
    return tape

