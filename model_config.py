"""
Model Configuration

Configuration for the TapeSmith MLflow model.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class ModelConfig:
    """Configuration for TapeSmith model"""
    
    # Databricks Foundation Models
    use_databricks_models: bool = True
    databricks_model_name: str = "databricks-gpt-5-2"
    databricks_endpoint_name: Optional[str] = None
    
    # Azure OpenAI (fallback)
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_model: str = "gpt-4"
    
    # Schema and paths (resolved from env vars at runtime)
    schema_name: str = "nagy_clean"
    schema_path: Optional[str] = None  # Set via TAPESMITH_SCHEMA_PATH env var
    
    # UC Functions catalog
    uc_catalog: str = "tapesmith"
    uc_schema: str = "functions"
    
    # MLflow
    mlflow_experiment_name: str = "/tapesmith/experiments/orchestrator"
    track_metrics: bool = True
    
    # Model behavior
    auto_approve_llm_suggestions: bool = True
    enable_validation: bool = True
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Load configuration from environment variables and Databricks secrets"""
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            
            try:
                use_databricks = w.secrets.get_secret("tapesmith", "use_databricks_models")
                if use_databricks and use_databricks.lower() == "true":
                    return cls(
                        use_databricks_models=True,
                        databricks_model_name=w.secrets.get_secret("tapesmith", "databricks_model_name") or "databricks-gpt-5-2",
                        databricks_endpoint_name=w.secrets.get_secret("tapesmith", "databricks_endpoint_name"),
                    )
            except:
                pass
            
            try:
                return cls(
                    use_databricks_models=False,
                    azure_openai_api_key=w.secrets.get_secret("tapesmith", "azure_openai_api_key"),
                    azure_openai_endpoint=w.secrets.get_secret("tapesmith", "azure_openai_endpoint"),
                    azure_openai_api_version=w.secrets.get_secret("tapesmith", "azure_openai_api_version") or "2024-02-15-preview",
                    azure_openai_model=w.secrets.get_secret("tapesmith", "azure_openai_model") or "gpt-4",
                )
            except:
                pass
        except:
            pass
        
        use_databricks = os.getenv("USE_DATABRICKS_MODELS", "true").lower() == "true"
        
        if use_databricks:
            return cls(
                use_databricks_models=True,
                databricks_model_name=os.getenv("DATABRICKS_MODEL_NAME", "databricks-gpt-5-2"),
                databricks_endpoint_name=os.getenv("DATABRICKS_ENDPOINT_NAME"),
            )
        else:
            return cls(
                use_databricks_models=False,
                azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_openai_model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4"),
            )

