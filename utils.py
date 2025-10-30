import pandas as pd
import numpy as np
import ast
from typing import Optional, Iterable, Union

def _parse_dict_cell(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return {}

def _flatten(d, parent=''):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else f"{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            # normalize list/tuple/set to a string so it's usable as a single value
            if isinstance(v, (list, tuple, set)):
                try:
                    v = ";".join(map(str, v))
                except Exception:
                    v = str(v)
            out[key] = v
    return out

def _strip_percent_to_float(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=['object']).columns
    for c in obj_cols:
        s = out[c]
        has_pct = s.astype(str).str.contains('%', na=False)
        if not has_pct.any():
            continue
        # strip %, commas, spaces; convert to numeric
        cleaned = s.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
        out[c] = pd.to_numeric(cleaned, errors='coerce')
    return out

def get_spec_keys_from_material(df, material_code, spec_col='components_Specifications'):
    """Get component specification keys from a specific material code"""
    material_idx = df.index[df['Material_Code'] == material_code][0]
    material_specs = df.loc[material_idx, spec_col]
    spec_dict = _parse_dict_cell(material_specs)
    return list(_flatten(spec_dict).keys())

def match_by_material_code(df: pd.DataFrame, material_code, code_col='Material_Code'):
    """
    Return rows whose (Material_Group, Base_Type, Moulding_Type, Product_Type)
    exactly match the values of the given material_code in df.
    If multiple rows share the material_code, the first match is used.
    """
    cols = ['Material_Group', 'Base_Type', 'Moulding_Type', 'Product_Type']
    required = [code_col] + cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ref_rows = df.loc[df[code_col] == material_code, cols]
    if ref_rows.empty:
        # No such material_code
        return df.iloc[0:0].copy()

    ref = ref_rows.iloc[0]  # use first occurrence
    mask = pd.Series(True, index=df.index)
    for c in cols:
        v = ref[c]
        mask &= (df[c].isna() if pd.isna(v) else df[c].eq(v))

    return df.loc[mask].copy()

def process_specifications(matches, material_code, df, spec_col='components_Specifications'):
    """Process and expand component specifications"""
    # Get the keys from the reference material code
    spec_keys = get_spec_keys_from_material(df, material_code)
    
    # Parse and flatten each row's dict, but only keep the keys from reference material
    parsed = matches[spec_col].apply(_parse_dict_cell).apply(_flatten)
    
    # Build a DataFrame with only the reference material's keys, NaN where missing
    spec_df = pd.DataFrame([{k: d.get(k, np.nan) for k in spec_keys} 
                           for d in parsed], index=matches.index)
    
    # Best-effort numeric coercion so numeric-looking strings become numbers
    def _convert_numeric(col: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(col)
        except (TypeError, ValueError):
            return col

    spec_df = spec_df.apply(_convert_numeric)
    
    # Join back and drop the original dict column
    matches_expanded = matches.drop(columns=[spec_col]).join(spec_df)
    
    # Convert percentage values to floats
    matches_expanded = _strip_percent_to_float(matches_expanded)
    
    return matches_expanded

def gower_similarity(
    df: pd.DataFrame,
    query_idx,
    weights: Optional[Union[dict, pd.Series]] = None,
    boost: str = 'count',            # 'count' or 'weight'
    normalize: bool = True,          # True -> final score kept in [0,1]
    exclude_cols: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Weighted Gower-like similarity with anchor-centric missing value handling:
    
    Case 1: Anchor NaN, candidate has value -> Column counts as used (15/15)
    Case 2: Anchor has value, candidate NaN -> Column counts as not used (14/15)
    Case 3: Both NaN -> Column counts as used (15/15)
    Case 4: Both have values -> Standard distance calculation
    """
    # Defensive copy
    X = df.copy()

    # Drop excluded columns
    if exclude_cols:
        exclude = [c for c in exclude_cols if c in X.columns]
        X = X.drop(columns=exclude)

    cols = X.columns.tolist()
    n = len(X)
    if len(cols) == 0:
        raise ValueError("No columns left after excluding columns.")

    # split numeric / categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    # build weight series (default 1.0)
    if weights is None:
        w = pd.Series(1.0, index=cols, dtype='float64')
    else:
        if isinstance(weights, pd.Series):
            w = pd.Series(1.0, index=cols, dtype='float64')
            for k, v in weights.items():
                if k in w.index:
                    w[k] = float(v)
        elif isinstance(weights, dict):
            w = pd.Series(1.0, index=cols, dtype='float64')
            for k, v in weights.items():
                if k in w.index:
                    w[k] = float(v)
        else:
            raise TypeError("weights must be None, dict, or pd.Series")

    # pick query row (by index label)
    q = X.loc[query_idx]

    # NUMERIC PART
    if num_cols:
        A = X[num_cols].to_numpy(dtype='float64')          # shape (n, m_num)
        qA = q[num_cols].to_numpy(dtype='float64')         # shape (m_num,)
        
        # Anchor-centric missing value handling
        anchor_nan = np.isnan(qA)                          # True where anchor is NaN
        data_nan = np.isnan(A)                             # True where data is NaN
        
        # Cases 1 & 3: Anchor NaN and (candidate has value OR candidate NaN) -> count as used
        # Case 2: Anchor has value, candidate NaN -> count as not used
        # Case 4: Both have values -> standard comparison
        used_num = (~anchor_nan & ~data_nan) | anchor_nan  # Case 4 OR (Case 1 & 3)
        
        # For distance calculation, only use where both have values (Case 4)
        valid_compare = ~anchor_nan & ~data_nan
        
        # ranges robust to all-NaN columns:
        col_max = np.nanmax(A, axis=0)
        col_min = np.nanmin(A, axis=0)
        ranges = col_max - col_min
        ranges = np.where(np.isnan(ranges) | (ranges == 0), 1.0, ranges)
        
        diff = np.abs(A - qA)                              # broadcast (n, m_num)
        comp_num = diff / ranges                           # scaled numeric difference
        comp_num[~valid_compare] = 0.0                     # zero distance for Case 1,2,3
        
        w_num = w[num_cols].to_numpy(dtype='float64')      
        num_sum = (comp_num * w_num).sum(axis=1)          
        num_used_w = (used_num * w_num).sum(axis=1)       # weight sum reflects anchor-centric logic
        num_used_cnt = used_num.sum(axis=1)               # count reflects anchor-centric logic
    else:
        num_sum = np.zeros(n, dtype='float64')
        num_used_w = np.zeros(n, dtype='float64')
        num_used_cnt = np.zeros(n, dtype='int64')

    # CATEGORICAL PART
    if cat_cols:
        B = X[cat_cols].astype(object)
        qB = q[cat_cols].astype(object)
        
        # Anchor-centric missing value handling for categorical
        anchor_miss = pd.isna(qB.values)                   # True where anchor is missing
        data_miss = B.isna().values                        # True where data is missing
        
        # Same logic as numeric part
        used_cat = (~anchor_miss & ~data_miss) | anchor_miss
        valid_compare = ~anchor_miss & ~data_miss
        
        # equality check only where both have values
        equal = (B.values == qB.values) & valid_compare
        comp_cat = (~equal).astype('float64')             # 1.0 if different, 0.0 if same or any NaN
        
        w_cat = w[cat_cols].to_numpy(dtype='float64')
        cat_sum = (comp_cat * w_cat).sum(axis=1)
        cat_used_w = (used_cat * w_cat).sum(axis=1)      # weight sum reflects anchor-centric logic
        cat_used_cnt = used_cat.sum(axis=1)              # count reflects anchor-centric logic
    else:
        cat_sum = np.zeros(n, dtype='float64')
        cat_used_w = np.zeros(n, dtype='float64')
        cat_used_cnt = np.zeros(n, dtype='int64')

    used_w = num_used_w + cat_used_w
    used_cnt = num_used_cnt + cat_used_cnt
    comp_sum = num_sum + cat_sum

    # distance calculation (now safer since we zero-out invalid comparisons)
    with np.errstate(invalid='ignore', divide='ignore'):
        dist = comp_sum / used_w
    dist = np.where(used_w == 0, np.nan, dist)    # no overlap -> NaN
    dist = np.clip(dist, 0.0, 1.0)                # clamp to [0,1]

    similarity = 1.0 - dist   

    # compute boost factor (now properly accounts for anchor-centric logic)
    total_weight = w.sum()
    total_count = len(cols)

    if boost == 'weight':
        if normalize:
            factor = np.where(total_weight > 0, used_w / total_weight, 0.0)
        else:
            factor = used_w.copy()
    else:  # 'count'
        if normalize:
            factor = used_cnt / total_count    # This now implements the 15/15, 14/15 logic
        else:
            factor = used_cnt.astype(float)

    score = similarity * factor

    out = pd.DataFrame({
        'distance': dist,
        'similarity': similarity,
        'score': score,
        'used_count': used_cnt,
        'used_weight': used_w
    }, index=X.index)

    out = out.sort_values(['score', 'similarity'], ascending=[False, False])
    return out