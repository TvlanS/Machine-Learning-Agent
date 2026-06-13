from typing import Type, List, Optional

from pydantic import BaseModel, Field
from pyprojroot import here
import sys
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import json
import warnings
#from new_tools import prediction
from src.data_modelling.tools.new_tools import prediction

sys.path.insert(0,str(here()))

from crewai.tools import BaseTool ,tool

import dython
from dython.nominal import associations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import pearsonr
import pandas as pd
import random          
import json
from datetime import datetime
import json



@tool("ask_user")
def ask_user(question: str) -> str:
            """Ask the human user a question and return their answer."""
            return input(f"\n[Agent asks] {question}\n> ")

#classification tool

class prediction_tool_Input(BaseModel):
    path: str = Field(
        description="Path to the CSV dataset containing the data for binary classification prediction."
    )
    target: str = Field(
        description="Name of the target column in the dataset (binary classification target)."
    )
    label0: str = Field(
        description="String label for the negative class (e.g., 'No Sepsis', 'Healthy', 'Class 0')."
    )
    label1: str = Field(
        description="String label for the positive class (e.g., 'Sepsis', 'Disease', 'Class 1')."
    )
    features: Optional[List[str]] = Field(
        default=[],
        description="List of feature column names to use for model training. If empty, all columns except the target will be used as features."
    )
    

class prediction_tool(BaseTool):
    name: str = "Prediction tool"
    description: str = (""" Perform prediction for categorical target features,
                        Includes XGboost as prediction modelling.
                        Each model is designed with and without hyperparameter tuning"""
    )
    args_schema: Type[BaseModel] = prediction_tool_Input

    def _run(self, path: str, target: str, label0:str, label1:str, features:Optional[list[str]] = []) -> str:
            
        
        clf = prediction(path=path,
            target= target,  
            label0= label0,
            label1= label1,
            features= features)

        results = clf.run()

        return json.dumps(results, indent=2)

class describe_dataset_tool_Input(BaseModel):
    dataset_path: str = Field(
        description="Path to the CSV dataset to profile and describe."
    )

class correlation_tool_Input(BaseModel):
    dataset_path: str = Field(
        description="Path to the CSV dataset to perform correlation analysis on."
    )
    target_feature: str = Field(
        description="The target column name to correlate all other features against."
    )
    excluded: list[str] = Field(
        default=[],
        description="List of column names to exclude from correlation analysis such as features that have no value for prediction modelling."
    )

class one_hot_encoding_tool_Input(BaseModel):
    path: str = Field(
        description="Path to the CSV dataset to apply one-hot encoding on."
    )
    target: str = Field(
        description="Name of the target column to keep in the final encoded dataset."
    )
    columns_drop: Optional[List[str]] = Field(
        default=None,
        description="list of column names to drop from the dataset before encoding , which includes continous features (strictly remove this), target variable and variables that have to value for prediction."
    )


class one_hot_encoding_tool(BaseTool):
    name: str = "One-Hot Encoding Tool"
    description: str = (
        "Applies one-hot encoding to categorical variables in a CSV dataset. "
        "Drops specified columns, encodes all remaining categorical columns, "
        "keeps the target column, and saves the encoded dataset to a timestamped CSV file. "
        "Returns the path to the saved encoded dataset."
    )
    args_schema: Type[BaseModel] = one_hot_encoding_tool_Input

    def _run(self, path: str, target: str, columns_drop: Optional[List[str]] = None) -> str:
        """
        Apply one-hot encoding to the dataset and save the result.
        
        Args:
            path: Path to the CSV dataset
            target: Target column name to keep
            columns_drop: Optional list of columns to drop before encoding
            
        Returns:
            str: Path to the saved encoded dataset
        """
        now = datetime.now()
        
        # Read the dataset
        df1 = pd.read_csv(path)
        
        # Handle columns_drop parameter
        if columns_drop is None:
            columns_drop = []

        valid_col = [col for col in columns_drop if col in df1.columns]

        # Drop specified columns and create a copy
        df = df1.drop(columns=valid_col).copy()
        
        # Get path name for output
        path_name = Path(path)
        
        # Initialize and apply one-hot encoder
        encoder = OneHotEncoder(
            sparse_output=False, 
            drop="if_binary", 
            handle_unknown="ignore"
        )
        
        if target in df.columns:
            feature_columns = [col for col in df.columns if col != target]
            X = df[feature_columns]

            X_encoded = encoder.fit_transform(X)
            X_encoded = pd.DataFrame(
                X_encoded, 
                columns=encoder.get_feature_names_out(feature_columns)
            )

            # Merge: encoded features + continuous (valid_col) + target
            encoded_merged = pd.concat([
                X_encoded.reset_index(drop=True),
                df1[valid_col].reset_index(drop=True),  # continuous cols dropped earlier
                df1[[target]].reset_index(drop=True)    # target col
            ], axis=1)

        else:
            # If target not in df (maybe it was dropped), encode everything
            X_encoded = encoder.fit_transform(df)
            X_encoded = pd.DataFrame(
                X_encoded, 
                columns=encoder.get_feature_names_out(df.columns)
            )

            encoded_merged = pd.concat([
                X_encoded.reset_index(drop=True),
                df1[valid_col].reset_index(drop=True)   # continuous cols dropped earlier
            ], axis=1)
        
        # Create timestamp for output filename
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = path_name.parent / f"{path_name.stem}_encoded_{timestamp}.csv"
        
        # Save the encoded dataset
        encoded_merged.to_csv(output_path, encoding="utf-8", index=False)
        
        # Print confirmation
        print(f"One-hot encoding completed successfully!")
        print(f"Original dataset: {path}")
        print(f"Encoded dataset saved to: {output_path}")
        print(f"Original columns: {len(df1.columns)}")
        print(f"Encoded columns: {len(encoded_merged.columns)}")
        print(f"Encoded columns list: {encoded_merged.columns.tolist()}")
        
        return str(output_path)

class describe_dataset_tool(BaseTool):
    name: str = "Dataset Description Tool"
    description: str = (
        "Profiles a dataset and returns a JSON summary including total dataset size "
        "and each column's data type, number of unique values, null count, and 3 random "
        "sample values. Use this BEFORE running Pearson correlation to decide which "
        "columns to include, encode, or skip."
    )
    args_schema: Type[BaseModel] = describe_dataset_tool_Input

    def _run(self, dataset_path: str) -> str:

        df = pd.read_csv(dataset_path)

        description = {
            "dataset_info": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "total_cells": int(df.size),
                "total_null_cells": int(df.isnull().sum().sum()),
                "null_pct_overall": round(df.isnull().mean().mean() * 100, 2)
            },
            "columns": {}
        }

        for col in df.columns:
            sample_values = df[col].dropna().tolist()
            random_samples = random.sample(sample_values, min(3, len(sample_values)))  # ✅ full line

            description["columns"][col] = {
                "dtype": str(df[col].dtype),
                "num_unique": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "random_samples": [str(s) for s in random_samples]
            }
        output = json.dumps(description, indent=2)
        print(output)
        return output
    
class drop_columns_tool_Input(BaseModel):
    path: str = Field(
        description="Path to the CSV dataset."
    )
    columns_to_drop: List[str] = Field(
        description="List of column names to drop from the dataset which dont bring value to prediction modelling for."
    )


class drop_columns_tool(BaseTool):
    name: str = "Drop Columns Tool"
    description: str = (
        "Drops specified columns from a CSV dataset. "
        "Use this tool when you need to remove unnecessary features, "
        "columns that don't carry value for prediction, or duplicate columns. "
        "Provide the dataset path and a list of column names to drop."
    )
    args_schema: Type[BaseModel] = drop_columns_tool_Input

    def _run(self, path: str, columns_to_drop: List[str]) -> str:
        """
        Drop specified columns from the dataset.
        
        Args:
            path: Path to the CSV dataset
            columns_to_drop: List of column names to drop
            
        Returns:
            str: Path to the saved cleaned dataset
        """
        now = datetime.now()
        
        # Load dataset
        df = pd.read_csv(path)
        path_name = Path(path)
        
        original_columns = len(df.columns)
        
        
        # Check which columns exist
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        missing_columns = [col for col in columns_to_drop if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: These columns not found in dataset: {missing_columns}")
        
        # Drop the columns
        if existing_columns:
            df = df.drop(columns=existing_columns)
            print(f"\nDropped {len(existing_columns)} columns: {existing_columns}")
        else:
            print(f"\nNo valid columns to drop")
        
        # Create output filename with timestamp
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = path_name.parent / f"{path_name.stem}_dropped.csv"
        
        # Save cleaned dataset
        df.to_csv(output_path, encoding='utf-8', index=False)
        
        return str(output_path)

class correlation_tool(BaseTool):
    name: str = "Correlation Tool"
    description: str = (
        "Performs correlation analysis to identify the correlation between a target feature "
        "and all other features in the dataset. Automatically selects the appropriate "
        "correlation method based on variable types (Pearson for numeric, Theil's U for nominal)."
    )
    args_schema: Type[BaseModel] = correlation_tool_Input

    def _run(
        self,
        dataset_path: str,
        target_feature: str,
        excluded: List[str] = None, 
    ) -> str:
        import warnings
        import sys

        # Safe default fallback for CrewAI args
        if excluded is None:
            excluded = []

        fallback_msg = ""  # Clean initialization for string construction later

        # --- Load ---
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            return f"Error: File not found at path '{dataset_path}'."
        except Exception as e:
            return f"Error reading CSV: {e}"

        # Guard — target must exist before processing
        if target_feature not in df.columns:
            return (
                f"Error: target_feature '{target_feature}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Drop columns that actually exist (skip silently if not found)
        cols_to_drop = [col for col in excluded if col in df.columns and col != target_feature]
        df = df.drop(columns=cols_to_drop)

        # Drop rows where target is null
        null_target_count = df[target_feature].isnull().sum()
        df = df.dropna(subset=[target_feature])

        if df.empty:
            return f"Error: No rows remain after dropping nulls in '{target_feature}'."

        # --- Dynamic Patch for Pydantic V2 / Dython Incompatibility ---
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results = associations(
                    df,
                    nominal_columns="auto",
                    nom_nom_assoc="theil",
                    plot=False,
                )
        except TypeError as te:
            if "skip_file_prefixes" in str(te):
                # Fallback Strategy: If dython is completely broken by Pydantic v2 in this env,
                # compute standard Pearson correlations for numeric values so the agent doesn't crash.
                numeric_df = df.select_dtypes(include=['number'])
                
                if target_feature in numeric_df.columns:  # ✅ Fixed: changed 'inside' to 'in'
                    corr_matrix_fallback = numeric_df.corr(method='pearson')
                    # Mock a results dict matching dython structure
                    results = {"corr": corr_matrix_fallback}
                    fallback_msg = " [Note: Falling back to standard Pearson correlation due to environment dython-pydantic library conflict]"
                else:
                    return (
                        f"Environment Error: Dython library is incompatible with Pydantic v2 "
                        f"installed by CrewAI (got {te}). Target feature is non-numeric, so automatic "
                        f"fallback correlation calculation could not be processed."
                    )
            else:
                return f"Error during associations computation: {te}"
        except Exception as e:
            return f"Error during associations computation: {e}"

        # --- Extract Matrix ---
        corr_matrix = results["corr"]
        if target_feature not in corr_matrix.columns:
            return (
                f"Error: '{target_feature}' not present in correlation matrix output. "
                f"It may have been excluded internally."
            )

        target_assoc = (
            corr_matrix[target_feature]
            .drop(labels=[target_feature], errors="ignore")  # Exclude self-correlation
            .sort_values(ascending=False)
        )

        # --- Output formatting ---
        header_lines = [
            f"Correlation with target: '{target_feature}'{fallback_msg}",
            f"Dataset rows used: {len(df)}" + (f" ({null_target_count} dropped due to null target)" if null_target_count else ""),
            f"Excluded columns: {cols_to_drop if cols_to_drop else 'none'}",
            "-" * 50,
        ]

        output = "\n".join(header_lines) + "\n" + target_assoc.to_string()

        paths = Path(dataset_path)
        parents = paths.parent

        now_ = datetime.now()
        timestamp = now_.strftime("%Y%m%d_%H%M%S")

        heatmap_path = parents/f"correlation_heatmap_{timestamp}.png"

        try:
            import numpy as np

            # Mask the upper triangle (including diagonal) to avoid redundancy
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Scale annotation font size down for larger matrices to prevent overlap
            n = corr_matrix.shape[0]
            annot_fontsize = max(6, 12 - n // 2)

            plt.figure(figsize=(10, 10))

            # Draw the heatmap using a clean color palette (coolwarm)
            sns.heatmap(
                corr_matrix,
                mask=mask,           # Hide upper triangle
                annot=True,          # Show the correlation numbers inside the squares
                fmt=".2f",           # Limit to 2 decimal places
                cmap="coolwarm",     # Red for positive correlation, blue for negative
                vmin=-1, vmax=1,     # Fix scale between -1 and 1
                square=True,         # Force square shapes
                linewidths=0.5,      # Small gap between squares
                annot_kws={"size": annot_fontsize}  # Scale font to matrix size
            )
            plt.title(f"Correlation Heatmap - Target: {target_feature}", fontsize=14)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            heatmap_output = f"Heatmap saved successfully to: '{heatmap_path}'"
            print(heatmap_output)
        except Exception as plot_error:
            heatmap_output = f"Failed to generate heatmap plot: {plot_error}"
            print(heatmap_output)

        return output


if __name__ == "__main__":
    path = r"C:\Users\tvlan\Documents\1.0 Python\data_modelling\datafolder\early_sepsis_full_simulated_dataset_dropped_encoded_20260607_195121.csv"

    tool = correlation_tool()
    result_path = tool._run(
        dataset_path = path,
        target_feature = "sepsis_risk",
        excluded = None
        )
    
    print(result_path)

