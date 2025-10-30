import gradio as gr
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from utils import (
    match_by_material_code,
    process_specifications,
    gower_similarity,
)


REQUIRED_COLUMNS = {
    "Material_Code",
    "Material_Group",
    "Base_Type",
    "Moulding_Type",
    "Product_Type",
    "components_Specifications",
}

STATUS_ORDER = {"Mismatch": 0, "Partial Match": 1, "Match": 2}


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise gr.Error(
            "The uploaded file is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "-"
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _classify_match(anchor: Any, candidate: Any) -> str:
    anchor_missing = pd.isna(anchor)
    candidate_missing = pd.isna(candidate)

    if anchor_missing and candidate_missing:
        return "Match"
    if anchor_missing or candidate_missing:
        return "Partial Match"

    if isinstance(anchor, (float, np.floating, int, np.integer)) and isinstance(
        candidate, (float, np.floating, int, np.integer)
    ):
        if np.isclose(float(anchor), float(candidate), atol=1e-6):
            return "Match"
        return "Mismatch"

    if str(anchor).strip().lower() == str(candidate).strip().lower():
        return "Match"
    return "Mismatch"


def load_dataset(file_path) -> Tuple[pd.DataFrame, Any, str]:
    if not file_path:
        raise gr.Error("Please upload an Excel data file.")

    if isinstance(file_path, (list, tuple)):
        if not file_path:
            raise gr.Error("Please upload an Excel data file.")
        file_path = file_path[0]

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as exc:
        raise gr.Error(f"Unable to read the uploaded file: {exc}") from exc

    _ensure_required_columns(df)

    if "Legislation" not in df.columns:
        df["Legislation"] = "Unknown"

    legislation_options = (
        ["All"]
        + sorted(
            {str(v).strip() for v in df["Legislation"].dropna().unique()} - {""}
        )
    )

    message = f"Loaded {len(df):,} rows with {df.shape[1]} columns."
    return df, gr.update(choices=legislation_options, value=legislation_options[0]), message


def _prepare_similarity(
    df: pd.DataFrame,
    material_code: str,
    top_n: int,
    legislation_filter: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any, str]:
    if df is None:
        raise gr.Error("Please load a data file before searching.")

    material_code = material_code.strip()
    if not material_code:
        raise gr.Error("Enter a material code to search.")

    if material_code not in df["Material_Code"].values:
        raise gr.Error(f"Material code '{material_code}' was not found in the dataset.")

    matches = match_by_material_code(df, material_code)
    if matches.empty:
        raise gr.Error(
            "No comparable SKUs share the required grouping attributes with the anchor material."
        )

    matches_expanded = process_specifications(matches, material_code, df)
    anchor_idx = matches_expanded.index[
        matches_expanded["Material_Code"] == material_code
    ][0]

    scores = gower_similarity(
        matches_expanded,
        query_idx=anchor_idx,
        boost="count",
        normalize=True,
        exclude_cols=["Material_Code", "Legislation"],
    )

    results = scores.join(
        df[
            [
                "Material_Code",
                "Legislation",
                "Material_Group",
                "Base_Type",
                "Moulding_Type",
                "Product_Type",
            ]
        ],
        how="left",
    )

    results = results.loc[results.index != anchor_idx]

    if legislation_filter and legislation_filter != "All":
        results = results[results["Legislation"].astype(str) == legislation_filter]

    results = results.sort_values(
        ["score", "similarity"], ascending=[False, False]
    ).head(int(top_n))

    if results.empty:
        empty_message = "No similar SKUs found for the selected criteria."
        empty_dropdown = gr.update(choices=[], value=None)
        return pd.DataFrame(), {}, empty_dropdown, empty_message

    display_df = results[
        [
            "Material_Code",
            "Legislation",
            "distance",
            "similarity",
            "score",
            "used_count",
        ]
    ].copy()
    display_df[["distance", "similarity", "score"]] = display_df[
        ["distance", "similarity", "score"]
    ].round(4)

    state = {
        "scores": scores,
        "matches_expanded": matches_expanded,
        "anchor_idx": anchor_idx,
        "anchor_code": material_code,
        "result_indices": results.index.tolist(),
    }

    candidate_codes = results["Material_Code"].tolist()
    message = f"Found {len(display_df)} similar SKUs."
    return (
        display_df.reset_index(drop=True),
        state,
        gr.update(choices=candidate_codes, value=candidate_codes[0]),
        message,
    )


def _build_comparison(
    search_state: Dict[str, Any], selected_code: str
) -> Tuple[str, pd.DataFrame]:
    if not search_state:
        return "Load results to compare SKUs.", pd.DataFrame()
    if not selected_code:
        return "Select a SKU to compare against the anchor.", pd.DataFrame()

    matches_expanded: pd.DataFrame = search_state["matches_expanded"]
    scores: pd.DataFrame = search_state["scores"]
    anchor_idx = search_state["anchor_idx"]
    anchor_code = search_state["anchor_code"]

    candidate_rows = matches_expanded[
        matches_expanded["Material_Code"] == selected_code
    ]
    if candidate_rows.empty:
        return "Selected SKU is not available for comparison.", pd.DataFrame()

    candidate_idx = candidate_rows.index[0]

    anchor_row = matches_expanded.loc[anchor_idx]
    candidate_row = matches_expanded.loc[candidate_idx]

    base_columns = [
        "Material_Group",
        "Base_Type",
        "Moulding_Type",
        "Product_Type",
        "Legislation",
    ]

    comparison_columns = base_columns + [
        c
        for c in matches_expanded.columns
        if c not in base_columns + ["Material_Code"]
    ]

    rows = []
    for col in comparison_columns:
        anchor_value = anchor_row.get(col, np.nan)
        candidate_value = candidate_row.get(col, np.nan)
        status = _classify_match(anchor_value, candidate_value)
        rows.append(
            {
                "Attribute": col,
                "Anchor Value": _format_value(anchor_value),
                "Candidate Value": _format_value(candidate_value),
                "Status": status,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df["Status"] = pd.Categorical(
        comparison_df["Status"],
        categories=["Mismatch", "Partial Match", "Match"],
        ordered=True,
    )
    comparison_df = comparison_df.sort_values("Status", key=lambda s: s.map(STATUS_ORDER))

    score = scores.loc[candidate_idx, "score"]
    similarity = scores.loc[candidate_idx, "similarity"]
    distance = scores.loc[candidate_idx, "distance"]
    used = scores.loc[candidate_idx, "used_count"]

    summary = (
        f"**{anchor_code} vs {selected_code}**  \n"
        f"Score: {score:.4f} • Similarity: {similarity:.4f} • Distance: {distance:.4f} \n"
        f"Evidence Columns Used: {int(used)}"
    )

    return summary, comparison_df.reset_index(drop=True)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="SKU Similarity Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            ## SKU Similarity Explorer
            Upload a master data file, choose an anchor SKU, and explore the most similar alternatives.
            Use the Legislation filter to focus your results, then drill into any candidate for a side-by-side comparison
            with the anchor SKU to understand alignment across attributes and component specifications.
            """
        )

        data_state = gr.State()
        search_state = gr.State()

        with gr.Column():
            with gr.Row():
                data_file = gr.File(
                    label="Master Data File (Excel)",
                    file_types=[".xlsx"],
                    type="filepath",
                    file_count="single",
                )
                load_button = gr.Button("Load Data", variant="primary")
            load_status = gr.Markdown("Upload your data file to begin.")

        legislation_filter = gr.Dropdown(
            label="Legislation Filter",
            choices=["All"],
            value="All",
        )

        with gr.Row():
            material_code_input = gr.Textbox(
                label="Anchor Material Code",
                placeholder="Enter the SKU to compare against",
            )
            topn_slider = gr.Slider(
                label="Number of Similar SKUs",
                minimum=1,
                maximum=50,
                value=10,
                step=1,
            )
            find_button = gr.Button("Find Similar SKUs", variant="primary")

        results_status = gr.Markdown()
        results_table = gr.Dataframe(
            headers=[
                "Material_Code",
                "Legislation",
                "distance",
                "similarity",
                "score",
                "used_count",
            ],
            datatype=["str", "str", "number", "number", "number", "number"],
            interactive=False,
            label="Similar SKUs",
        )

        candidate_selector = gr.Dropdown(
            label="Compare Candidate",
            choices=[],
            interactive=True,
        )

        comparison_summary = gr.Markdown("Select a candidate SKU to review the comparison.")
        comparison_table = gr.Dataframe(
            headers=["Attribute", "Anchor Value", "Candidate Value", "Status"],
            interactive=False,
            label="Attribute-Level Comparison",
        )

        load_button.click(
            fn=load_dataset,
            inputs=data_file,
            outputs=[data_state, legislation_filter, load_status],
        )

        find_event = find_button.click(
            fn=_prepare_similarity,
            inputs=[data_state, material_code_input, topn_slider, legislation_filter],
            outputs=[results_table, search_state, candidate_selector, results_status],
        )

        find_event.then(
            fn=_build_comparison,
            inputs=[search_state, candidate_selector],
            outputs=[comparison_summary, comparison_table],
        )

        candidate_selector.change(
            fn=_build_comparison,
            inputs=[search_state, candidate_selector],
            outputs=[comparison_summary, comparison_table],
        )

        gr.Markdown(
            """
            #### Tips
            - Ensure the uploaded file contains the required attributes listed in the documentation.
            - Use the Legislation filter to focus on products compliant with specific regions or standards.
            - Scores combine similarity with evidence coverage, so higher scores indicate both alignment and stronger data backing.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
