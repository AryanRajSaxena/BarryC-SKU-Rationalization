import pandas as pd  # pip install pandas openpyxl
import numpy as np
from utils import (
    match_by_material_code,
    process_specifications,
    gower_similarity
)

def find_similar_materials(material_code: str, data_path: str, top_n: int = 10) -> pd.DataFrame:

    # Read and prepare the data
    active_cols = [
        'Material_Code', 'Material_Group', 'Base_Type', 'Moulding_Type',
        'Product_Type', 'components_Specifications', 'Legislation'
    ]
    
    try:
        # Read the data file
        df = pd.read_excel(data_path, usecols=active_cols)
        
        # Find matching materials by group attributes
        matches = match_by_material_code(df, material_code)
        if matches.empty:
            raise ValueError(f"No matches found for material code: {material_code}")
            
        # Process and expand specifications
        matches_expanded = process_specifications(matches, material_code, df)
        
        # Calculate similarity scores
        q_idx = df.index[df['Material_Code'] == material_code][0]
        scores = gower_similarity(
            matches_expanded,
            query_idx=q_idx,
            boost='count',
            normalize=True,
            exclude_cols=['Material_Code', 'Legislation']
        )
        
        # Get top N similar materials
        top_indices = scores.head(top_n).index
        similar_materials = df.loc[top_indices].copy()
        
        # Add similarity metrics to the results
        similar_materials = similar_materials.join(scores[['distance', 'similarity', 'score', 'used_count']])
        
        return similar_materials
        
    except Exception as e:
        print(f"Error processing material {material_code}: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    data_file = "/Users/aryanrajsaxena/Desktop/BarryC/data_analysis/data-files/Master Data - Part 1.xlsx"
    material_code = "YYW-PN-G300297-E15"
    
    try:
        similar_materials = find_similar_materials(material_code, data_file)
        print(f"\nTop similar materials for {material_code}:")
        print(similar_materials[['Material_Code', 'Material_Group', 'similarity', 'score', 'used_count']])
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error processing material {material_code}: {str(e)}")