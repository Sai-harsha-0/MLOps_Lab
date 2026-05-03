"""Data ingestion and validation for MovieLens ratings."""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict
from src.config import RATINGS_SCHEMA, DATA_PATHS, EXPECTED_METRICS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RatingsSchemaValidator:
    """Validate ratings data against schema."""
    
    def __init__(self, schema: Dict = RATINGS_SCHEMA):
        self.schema = schema
        self.validation_report = {}
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        required = set(self.schema.keys())
        present = set(df.columns)
        
        if required != present:
            missing = required - present
            extra = present - required
            logger.error(f"Column mismatch. Missing: {missing}, Extra: {extra}")
            return False
        
        logger.info(f"✓ All required columns present: {list(df.columns)}")
        return True
    
    def validate_datatypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        errors = 0
        
        for col, rules in self.schema.items():
            target_dtype = rules['dtype']
            
            try:
                if target_dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
                elif target_dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    logger.warning(f"{col}: {nulls} invalid values")
                    errors += nulls
                
                logger.info(f"✓ {col}: dtype={target_dtype}")
            
            except Exception as e:
                logger.error(f"Conversion failed for {col}: {e}")
                return df, -1
        
        return df, errors
    
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        rows_before = len(df)
        
        for col, rules in self.schema.items():
            min_val = rules['min']
            max_val = rules['max']
            
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            invalid = (~mask).sum()
            
            if invalid > 0:
                logger.warning(f"{col}: {invalid} out-of-range values")
            
            df = df[mask]
        
        removed = rows_before - len(df)
        logger.info(f"✓ Range validation: {removed} rows removed")
        return df, removed
    
    def validate_nulls(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        rows_before = len(df)
        
        for col, rules in self.schema.items():
            if not rules['nullable']:
                df = df.dropna(subset=[col])
        
        removed = rows_before - len(df)
        return df, removed
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        logger.info(f"Validating {len(df)} rows...")
        
        if not self.validate_columns(df):
            return None, {'error': 'Missing columns'}
        
        df, dtype_errors = self.validate_datatypes(df)
        if dtype_errors < 0:
            return None, {'error': 'Datatype failed'}
        
        df, range_removed = self.validate_ranges(df)
        df, null_removed = self.validate_nulls(df)
        
        self.validation_report = {
            'total_errors': dtype_errors + range_removed + null_removed,
            'rows_retained': len(df)
        }
        
        logger.info(f"✓ Validation complete: {len(df)} rows")
        return df, self.validation_report


class RatingsLoader:
    """Load and process ratings."""
    
    def __init__(self, filepath: str = DATA_PATHS['raw']):
        self.filepath = filepath
        self.validator = RatingsSchemaValidator()
        self.raw_df = None
        self.clean_df = None
    
    def load(self) -> pd.DataFrame:
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"{self.filepath} not found")
        
        logger.info(f"Loading {self.filepath}...")
        
        self.raw_df = pd.read_csv(
            self.filepath,
            sep='\t'
        )
        
        logger.info(f"✓ Loaded {len(self.raw_df)} rows")
        return self.raw_df
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(['user_id', 'movie_id'], keep='last')
        return df
    
    def validate_and_clean(self):
        df = self.deduplicate(self.raw_df)
        df, report = self.validator.validate(df)
        self.clean_df = df
        return df, report
    
    def save(self):
        Path(DATA_PATHS['processed']).parent.mkdir(parents=True, exist_ok=True)
        self.clean_df.to_csv(DATA_PATHS['processed'], index=False)
        logger.info("✓ Saved clean data")
    
    def save_report(self):
        Path(DATA_PATHS['validation_report']).parent.mkdir(parents=True, exist_ok=True)
        
        with open(DATA_PATHS['validation_report'], 'w') as f:
            json.dump(self.validator.validation_report, f, indent=2)
        
        logger.info("✓ Saved validation report")


def main():
    logger.info("=== Ingestion Pipeline ===")
    
    loader = RatingsLoader()
    loader.load()
    
    df, report = loader.validate_and_clean()
    
    loader.save()
    loader.save_report()
    
    logger.info("✓ Pipeline completed")


if __name__ == "__main__":
    main()