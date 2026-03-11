"""
LangChain Agents for TapeSmith — Bundled copy for MCP Server.

Adapted imports: uses processing.* instead of tapesmith_uc_functions.*.
Otherwise identical to the original tapesmith_model/agents.py.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Any, Optional, Tuple
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import from bundled processing package (MCP server self-contained)
try:
    from processing.schema_loader import load_schema
    from processing.data_processor import process_data_types, transform_values, sanitize_df
    from processing.tag_generator import create_tags_df
    from processing.mcf_generator import create_mcf_data
except ImportError:
    load_schema = None
    process_data_types = None
    transform_values = None
    sanitize_df = None
    create_tags_df = None
    create_mcf_data = None


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, config, llm):
        self.name = name
        self.config = config
        self.llm = llm
        self.history = []

    def log_interaction(self, input_data: Dict, output_data: Dict):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data,
        })


class HeaderMappingAgent(BaseAgent):
    """Agent for mapping tape headers to standard schema headers"""

    def __init__(self, config, schema: Dict, llm):
        super().__init__("HeaderMappingAgent", config, llm)
        self.schema = schema

    def _collect_unmapped_headers(self, tape_df, tape_headers, learned_corrections=None):
        mapped_headers = {}
        unmapped_headers = []
        corrections_applied = 0

        for tape_header in tape_headers:
            already_mapped = False
            for std_header, details in self.schema.get('headers', {}).items():
                if tape_header == std_header.lower() or tape_header in [h.lower() for h in details.get('headers', [])]:
                    mapped_headers[tape_header] = std_header
                    already_mapped = True
                    break

            if not already_mapped:
                original_header = [h for h in tape_df.columns if h.lower() == tape_header][0] if tape_header in [h.lower() for h in tape_df.columns] else tape_header
                sample_values = tape_df[original_header].dropna().unique().tolist()[:10] if original_header in tape_df.columns else []
                unmapped_headers.append({
                    'tape_header': tape_header,
                    'original_header': original_header,
                    'sample_values': sample_values,
                })

        # Apply learned corrections as OVERRIDES (after schema mapping)
        # This ensures user feedback always takes priority over schema defaults
        if learned_corrections:
            for tape_header in tape_headers:
                correction = learned_corrections.get(tape_header.lower())
                if correction:
                    old_mapping = mapped_headers.get(tape_header, "<unmapped>")
                    mapped_headers[tape_header] = correction
                    corrections_applied += 1
                    # Remove from unmapped if it was there
                    unmapped_headers = [u for u in unmapped_headers if u['tape_header'] != tape_header]
                    if old_mapping != correction:
                        print(f"[HeaderMappingAgent] 📝 Correction override: '{tape_header}' → '{correction}' (was '{old_mapping}')")

        if corrections_applied > 0:
            print(f"[HeaderMappingAgent] 📝 Applied {corrections_applied} learned correction(s) from past feedback")

        return mapped_headers, unmapped_headers

    def _build_batch_prompt(self, unmapped_headers, schema_context, corrections_context=""):
        headers_list = "\n".join([
            f"- {h['tape_header']}: Sample values = {h['sample_values'][:5]}{'...' if len(h['sample_values']) > 5 else ''}"
            for h in unmapped_headers
        ])

        corrections_section = ""
        if corrections_context:
            corrections_section = f"\n{corrections_context}\n\nUse these past corrections as guidance.\n\n"

        return f"""
You are a data mapping expert. Map the following tape headers to the most appropriate standard headers from the schema.

Schema (Standard Headers):
{schema_context}
{corrections_section}
Tape Headers to Map:
{headers_list}

Return ONLY a JSON object where:
- Top-level keys are tape header names (exact match from the list above)
- Each value is an object with:
  - "standard_header": the best matching standard header name (exact match from schema, or null if no match)
  - "confidence": confidence score (0-1)
  - "reasoning": brief explanation

Return mappings for ALL headers listed above.
"""

    def _parse_batch_response(self, response, unmapped_headers):
        suggestions = {}
        try:
            output_text = response.content if hasattr(response, 'content') else str(response)
            output_text = output_text.strip()
            if output_text.startswith("```"):
                first_newline = output_text.find("\n")
                if first_newline != -1:
                    output_text = output_text[first_newline + 1:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
            output_text = output_text.strip()
            if not output_text.startswith("{"):
                start = output_text.find("{")
                if start != -1:
                    end = output_text.rfind("}") + 1
                    output_text = output_text[start:end]
            if not output_text:
                return suggestions
            batch_results = json.loads(output_text)
            for header_info in unmapped_headers:
                tape_header = header_info['tape_header']
                if tape_header in batch_results:
                    suggestion = batch_results[tape_header]
                    if suggestion.get('standard_header') and suggestion.get('confidence', 0) > 0.5:
                        suggestions[tape_header] = suggestion
                elif tape_header.lower() in batch_results:
                    suggestion = batch_results[tape_header.lower()]
                    if suggestion.get('standard_header') and suggestion.get('confidence', 0) > 0.5:
                        suggestions[tape_header] = suggestion
        except json.JSONDecodeError as e:
            print(f"[HeaderMappingAgent] ⚠️ Failed to parse batch LLM response: {e}")
            raw = response.content if hasattr(response, 'content') else str(response)
            print(f"[HeaderMappingAgent] 🔍 Raw LLM response (first 300 chars): {raw[:300]}")
        except Exception as e:
            print(f"[HeaderMappingAgent] ⚠️ Error parsing batch response: {e}")
        return suggestions

    def execute(self, input_data: Dict) -> Dict:
        tape_df = input_data['tape_df']
        tape_headers = [h.lower() for h in list(tape_df.columns)]
        learned_corrections = input_data.get('learned_corrections', {})
        corrections_context = input_data.get('corrections_context', "")

        print(f"[HeaderMappingAgent] Processing {len(tape_headers)} headers...")

        schema_context = "\n".join([
            f"Standard Header: {key} | Data Type: {details.get('data_type', 'N/A')} | "
            f"Alternate Names: {', '.join(details.get('headers', []))} | "
            f"Description: {details.get('description', 'N/A')}"
            for key, details in self.schema.get('headers', {}).items()
        ])

        mapped_headers, unmapped_headers = self._collect_unmapped_headers(
            tape_df, tape_headers, learned_corrections
        )
        print(f"[HeaderMappingAgent] ✅ Already mapped: {len(mapped_headers)} headers")
        print(f"[HeaderMappingAgent] 🔍 Need LLM mapping: {len(unmapped_headers)} headers")

        suggestions = {}
        if unmapped_headers:
            print(f"[HeaderMappingAgent] 🤖 Calling LLM to map {len(unmapped_headers)} headers in batch...")
            try:
                batch_prompt = self._build_batch_prompt(unmapped_headers, schema_context, corrections_context)
                response = self.llm.invoke(batch_prompt)
                suggestions = self._parse_batch_response(response, unmapped_headers)
                for tape_header, suggestion in suggestions.items():
                    if suggestion.get('standard_header'):
                        mapped_headers[tape_header] = suggestion['standard_header']
                print(f"[HeaderMappingAgent] ✅ LLM mapped {len(suggestions)} headers successfully")
            except Exception as e:
                print(f"[HeaderMappingAgent] ❌ Batch LLM call failed: {e}")
        else:
            print(f"[HeaderMappingAgent] ℹ️ All headers already mapped - no LLM needed")

        result = {
            "status": "complete",
            "mapped_headers": mapped_headers,
            "suggestions": suggestions,
            "requires_human_approval": False,
        }
        self.log_interaction(input_data, result)
        return result


class ValueTransformationAgent(BaseAgent):
    """Agent for transforming column values"""

    def __init__(self, config, schema: Dict, llm):
        super().__init__("ValueTransformationAgent", config, llm)
        self.schema = schema

    def unmapped_column_values(self, df):
        unmapped_values = {}
        valid_headers = {
            header: details for header, details in self.schema.get('headers', {}).items()
            if header in df.columns and
            'transformations' in details and
            'mappings' in details['transformations']
        }
        for standard_header, details in valid_headers.items():
            mappings = details['transformations']['mappings']
            recognized_set = set(mappings.keys())
            for vals in mappings.values():
                if isinstance(vals, list):
                    recognized_set.update(vals)
                else:
                    recognized_set.add(vals)
            col_values = df[standard_header].values
            unique_vals = pd.Series(np.unique(col_values))
            mask = ~unique_vals.isin(recognized_set) & ~unique_vals.isna()
            if mask.any():
                unmapped_values[standard_header] = unique_vals[mask].tolist()
                if len(unmapped_values[standard_header]) > 1:
                    unmapped_values[standard_header].sort()
        return unmapped_values

    def get_llm_column_value_suggestions(self, unmapped_values, max_retries=3, retry_delay=2, corrections_context=""):
        if not unmapped_values:
            return {}
        suggestions = {}
        complete_schema_context = ""
        all_unmapped_data = {}
        for standard_header, texts in unmapped_values.items():
            header_schema = f"\nStandard Header: {standard_header}\n"
            if standard_header in self.schema.get('headers', {}):
                header_details = self.schema['headers'][standard_header]
                if 'transformations' in header_details and 'mappings' in header_details['transformations']:
                    mappings = header_details['transformations']['mappings']
                    for k, v in mappings.items():
                        header_schema += f"standard_mapping: {k}, mappings: {v}\n"
            complete_schema_context += header_schema
            all_unmapped_data[standard_header] = texts

        corrections_section = ""
        if corrections_context:
            corrections_section = f"\n{corrections_context}\n\nUse these past corrections as guidance.\n\n"

        prompt = f"""
You are an expert in mapping column values based on a provided schema.

Complete Schema:
{complete_schema_context}
{corrections_section}
Instructions:
1. Match each value to one of the standard mappings defined in the schema for that header.
2. If a value cannot be exactly matched, use your best judgment.
3. If the value is unclear, return it unchanged.
4. Return a JSON object where keys are standard headers, and each value maps input values to mapped values.
5. No extra formatting or explanation. Just the JSON object.

The provided data by standard header is: {json.dumps(all_unmapped_data)}
"""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[ValueTransformationAgent] 🤖 Invoking LLM (attempt {attempt}/{max_retries})...")
                response = self.llm.invoke(prompt)
                output_text = response.content if hasattr(response, 'content') else str(response)
                all_mappings = json.loads(output_text)
                total_mappings = sum(len(mapping) for mapping in all_mappings.values())
                print(f"[ValueTransformationAgent] ✅ LLM returned {total_mappings} mapping(s)")
                for standard_header in unmapped_values.keys():
                    if standard_header in all_mappings:
                        mapping = all_mappings[standard_header]
                        key_values = [{k: v} for k, v in mapping.items()]
                    else:
                        key_values = []
                    suggestions[standard_header] = key_values
                return suggestions
            except (json.JSONDecodeError, Exception) as e:
                if attempt < max_retries:
                    print(f"[ValueTransformationAgent] ⚠️ Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"[ValueTransformationAgent] ❌ All attempts failed: {e}")
                    return {header: [] for header in unmapped_values.keys()}
        return {header: [] for header in unmapped_values.keys()}

    def apply_llm_suggestions(self, df, suggestions):
        df = df.copy()
        for standard_header, mappings_list in suggestions.items():
            if standard_header in df.columns and mappings_list:
                for mapping_dict in mappings_list:
                    for original_value, mapped_value in mapping_dict.items():
                        df[standard_header] = df[standard_header].replace(original_value, mapped_value)
        return df

    def execute(self, input_data: Dict) -> Dict:
        tape_df = input_data['tape_df']
        learned_corrections = input_data.get('learned_corrections', {})
        corrections_context = input_data.get('corrections_context', "")

        print(f"[ValueTransformationAgent] Starting value transformation...")

        if transform_values:
            print(f"[ValueTransformationAgent] Applying rule-based transformations from schema...")
            transformed_df = transform_values(tape_df, self.schema)
        else:
            transformed_df = tape_df.copy()

        corrections_applied = 0
        if learned_corrections:
            for col in transformed_df.columns:
                for source_val, correct_val in learned_corrections.items():
                    mask = transformed_df[col].astype(str).str.lower() == source_val.lower()
                    if mask.any():
                        transformed_df.loc[mask, col] = correct_val
                        corrections_applied += mask.sum()
            if corrections_applied > 0:
                print(f"[ValueTransformationAgent] 📝 Applied {corrections_applied} learned correction(s)")

        print(f"[ValueTransformationAgent] Checking for unmapped values...")
        unmapped_values = self.unmapped_column_values(transformed_df)

        llm_suggestions = {}
        llm_called = False
        if unmapped_values:
            print(f"[ValueTransformationAgent] ✅ Found unmapped values in {len(unmapped_values)} column(s)")
            llm_called = True
            llm_suggestions = self.get_llm_column_value_suggestions(unmapped_values, corrections_context=corrections_context)
            if llm_suggestions:
                transformed_df = self.apply_llm_suggestions(transformed_df, llm_suggestions)
        else:
            print(f"[ValueTransformationAgent] ✅ No unmapped values found")
            print(f"[ValueTransformationAgent] ℹ️ LLM not needed for value transformation")

        result = {
            "status": "complete",
            "transformations_applied": True,
            "unmapped_values_detected": len(unmapped_values),
            "llm_called": llm_called,
            "llm_suggestions": llm_suggestions,
            "requires_human_approval": False,
        }
        input_data['tape_df'] = transformed_df
        self.log_interaction(input_data, result)
        return result


class ValidationAgent(BaseAgent):
    """Agent for validating processed data"""

    def __init__(self, config, schema: Dict, llm):
        super().__init__("ValidationAgent", config, llm)
        self.schema = schema

    def execute(self, input_data: Dict) -> Dict:
        tape_df = input_data['tape_df']
        validation_results = {
            "total_rows": len(tape_df),
            "total_columns": len(tape_df.columns),
            "missing_required_fields": [],
            "data_quality_issues": [],
        }
        required_fields = [h for h, details in self.schema.get('headers', {}).items()
                           if details.get('required', False)]
        for field in required_fields:
            if field not in tape_df.columns:
                validation_results["missing_required_fields"].append(field)

        result = {
            "status": "complete",
            "validation_results": validation_results,
            "is_valid": len(validation_results["missing_required_fields"]) == 0,
            "requires_human_approval": False,
        }
        self.log_interaction(input_data, result)
        return result


class MCFGenerationAgent(BaseAgent):
    """Agent for generating MCF output"""

    def __init__(self, config, llm, schema=None):
        super().__init__("MCFGenerationAgent", config, llm)
        self.schema = schema

    def create_llm_penalty_strings(self, df, header, max_retries=3, retry_delay=2,
                                   learned_corrections=None, corrections_context=""):
        if header not in df.columns:
            return df[header] if header in df.columns else pd.Series()

        unique_prepayment_values = df[header].unique()
        unique_prepayment_values = [str(val) for val in unique_prepayment_values if pd.notna(val) and str(val).strip()]
        if not unique_prepayment_values:
            return df[header]

        if learned_corrections:
            pre_corrections = {}
            remaining_values = []
            for val in unique_prepayment_values:
                correction = learned_corrections.get(val.strip().lower())
                if correction:
                    pre_corrections[val] = correction
                else:
                    remaining_values.append(val)
            if pre_corrections:
                print(f"[MCFGenerationAgent] 📝 Applied {len(pre_corrections)} learned penalty correction(s)")
                df = df.copy()
                for orig, corrected in pre_corrections.items():
                    df[header] = df[header].replace(orig, corrected)
            unique_prepayment_values = remaining_values
            if not unique_prepayment_values:
                print(f"[MCFGenerationAgent] ✅ All penalty strings resolved by learned corrections")
                return df[header]

        corrections_prompt_section = ""
        if corrections_context:
            corrections_prompt_section = f"\n{corrections_context}\n\nUse these past corrections as guidance.\n\n"

        prompt = f"""
You are an expert in transforming prepayment descriptions into formatted prepayment penalty strings.

Examples:
"5%, 4%, 3%" → "5:12/4:12/3:12"
"5%:36" → "5:36"
"1/0.5" → "1:12/0.5:12"
"5/4/3/2/1" → "5:12/4:12/3:12/2:12/1:12"
".03" → "3:12"

Instructions:
1. Transform each unique prepayment description into formatted penalty string.
2. Return a JSON object: key = current value, value = transformed value.
3. No extra formatting or explanation.
{corrections_prompt_section}
The provided values are: {unique_prepayment_values}
"""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[MCFGenerationAgent] 🤖 Invoking LLM for penalty strings (attempt {attempt}/{max_retries})...")
                response = self.llm.invoke(prompt)
                output_text = response.content if hasattr(response, 'content') else str(response)
                transforms = json.loads(output_text)
                print(f"[MCFGenerationAgent] ✅ LLM returned {len(transforms)} penalty string transformation(s)")
                return df[header].map(transforms).fillna(df[header])
            except Exception as e:
                if attempt < max_retries:
                    print(f"[MCFGenerationAgent] ⚠️ Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"[MCFGenerationAgent] ❌ All attempts failed: {e}")
                    return df[header]
        return df[header]

    def execute(self, input_data: Dict) -> Dict:
        tape_df = input_data['tape_df']
        effective_date = input_data.get('effective_date', '20250101')
        format_type = input_data.get('format', 'COL')
        learned_corrections = input_data.get('learned_corrections', {})
        corrections_context = input_data.get('corrections_context', "")

        if not create_mcf_data:
            result = {"status": "error", "error": "MCF generation function not available", "penalty_llm_called": False}
            self.log_interaction(input_data, result)
            return result

        try:
            tape_df = tape_df.copy()

            # Penalty string transformation
            print(f"[MCFGenerationAgent] Checking for penalty string columns...")
            penalty_columns = [col for col in tape_df.columns if 'prepayment_penalty' in col.lower() or 'penalty' in col.lower()]
            llm_called_for_penalty = False

            if penalty_columns:
                print(f"[MCFGenerationAgent] Found {len(penalty_columns)} potential penalty column(s): {penalty_columns}")
                for penalty_col in penalty_columns:
                    should_transform = False
                    if self.schema:
                        for std_header, details in self.schema.get('headers', {}).items():
                            if std_header == penalty_col or penalty_col in details.get('headers', []):
                                if 'transformations' in details and 'penalty_strings' in details['transformations']:
                                    should_transform = True
                                    break
                    if not should_transform and ('prepayment_penalty_calculation' in penalty_col.lower() or 'penrate' in penalty_col.lower()):
                        should_transform = True
                    if should_transform:
                        unique_values = tape_df[penalty_col].dropna().unique()
                        unique_values = [str(v) for v in unique_values if str(v).strip()]
                        if unique_values:
                            print(f"[MCFGenerationAgent] ✅ Found penalty strings in '{penalty_col}'")
                            llm_called_for_penalty = True
                            tape_df[penalty_col] = self.create_llm_penalty_strings(
                                tape_df, penalty_col,
                                learned_corrections=learned_corrections,
                                corrections_context=corrections_context,
                            )
                    else:
                        print(f"[MCFGenerationAgent] ℹ️ Column '{penalty_col}' does not require penalty string transformation")
                if not llm_called_for_penalty:
                    print(f"[MCFGenerationAgent] ℹ️ LLM not needed for penalty strings (no transformation required)")
            else:
                print(f"[MCFGenerationAgent] ✅ No penalty string columns found")

            # Ensure loan_number column
            def _get_col_values(df, col):
                c = df[col]
                return c.iloc[:, 0].values if isinstance(c, pd.DataFrame) else c.values

            if 'loan_number' not in tape_df.columns and 'LOANID' in tape_df.columns:
                tape_df['loan_number'] = _get_col_values(tape_df, 'LOANID')
            elif 'LOANID' not in tape_df.columns and 'loan_number' in tape_df.columns:
                tape_df['LOANID'] = _get_col_values(tape_df, 'loan_number')
            elif 'loan_number' not in tape_df.columns and 'LOANID' not in tape_df.columns:
                possible = [col for col in tape_df.columns if 'loan' in col.lower()]
                if possible:
                    tape_df['loan_number'] = _get_col_values(tape_df, possible[0])
                else:
                    raise ValueError(f"No loan ID column found. Available: {list(tape_df.columns)[:20]}")

            if 'loan_number' not in tape_df.columns and 'LOANID' in tape_df.columns:
                tape_df['loan_number'] = _get_col_values(tape_df, 'LOANID')
            if 'LOANID' not in tape_df.columns and 'loan_number' in tape_df.columns:
                tape_df['LOANID'] = _get_col_values(tape_df, 'loan_number')

            mcf_output = create_mcf_data(tape_df, effective_date, format_type)
            if not mcf_output or len(str(mcf_output).strip()) == 0:
                raise ValueError("MCF generation returned empty output.")

            result = {
                "status": "complete",
                "mcf_output": str(mcf_output),
                "format": format_type,
                "penalty_llm_called": llm_called_for_penalty,
                "requires_human_approval": False,
            }
        except Exception as e:
            import traceback
            error_details = f"{e}\n{traceback.format_exc()}"
            print(f"[MCFGenerationAgent] ❌ Error: {error_details}")
            result = {
                "status": "error",
                "error": str(e),
                "penalty_llm_called": llm_called_for_penalty if 'llm_called_for_penalty' in locals() else False,
            }
        self.log_interaction(input_data, result)
        return result

