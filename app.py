#!/usr/bin/env python3
"""
Interactive Shortcut Detection Dashboard
Gradio-based web interface for detecting shortcuts in embedding spaces
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError, ParserError
from PIL import Image

# On macOS, ensure Homebrew system libraries (glib, pango) are discoverable
# so weasyprint PDF generation works inside conda/venv environments.
if sys.platform == "darwin":
    try:
        _brew_prefix = subprocess.check_output(
            ["brew", "--prefix"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        _brew_lib = os.path.join(_brew_prefix, "lib")
        _fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if _brew_lib not in _fallback:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{_brew_lib}:{_fallback}" if _fallback else _brew_lib
            )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shortcut_detect import (  # noqa: E402
    AdversarialDebiasing,
    BackgroundRandomizer,
    ContrastiveDebiasing,
    ExplanationRegularization,
    GradCAMHeatmapGenerator,
    HuggingFaceEmbeddingSource,
    LastLayerRetraining,
    ModelComparisonRunner,
    ShortcutDetector,
    ShortcutMasker,
    SpRAyDetector,
    generate_parametric_shortcut_dataset,
    get_embedding_registry,
)
from shortcut_detect.reporting.comparison_report import ComparisonReportBuilder  # noqa: E402
from shortcut_detect.reporting.csv_export import export_comparison_to_csv  # noqa: E402

REPORT_CONTRAST_CSS = """
/* Force detection report to black text on light background (Gradio 5/6 use different wrappers) */
.gr-html .container, .gr-html .container *,
.gr-html .shortcut-report, .gr-html .shortcut-report *,
main .container, main .container *,
main .shortcut-report, main .shortcut-report *,
[class*="html"] .container, [class*="html"] .container *,
[class*="html"] .shortcut-report, [class*="html"] .shortcut-report * { color: #000 !important; }
.gr-html .container, .gr-html .shortcut-report,
main .container, main .shortcut-report,
[class*="html"] .container, [class*="html"] .shortcut-report { background: #fff !important; }
.gr-html .shortcut-report th, main .shortcut-report th, [class*="html"] .shortcut-report th { background: #1d4ed8 !important; color: #fff !important; }
.gr-html .shortcut-report code, main .shortcut-report code, [class*="html"] .shortcut-report code { background: unset !important; background-color: rgba(255, 255, 255, 1) !important; color: rgba(21, 87, 36, 1) !important; background-clip: unset; -webkit-background-clip: unset; box-shadow: none !important; }
.gr-html .shortcut-report::selection, .gr-html .shortcut-report *::selection,
main .shortcut-report::selection, main .shortcut-report *::selection,
[class*="html"] .shortcut-report::selection, [class*="html"] .shortcut-report *::selection { background: #b4d5fe; color: #000; }
"""


def find_data_dir():
    """Find the data directory in the project"""
    current = Path(__file__).parent
    for _ in range(5):
        data_path = current / "data"
        if data_path.exists():
            return data_path
        current = current.parent
    return None


def load_sample_data(use_real_embeddings: bool = False):
    """
    Load CheXpert data with optional real or synthetic embeddings.

    When use_real_embeddings=True and data/chest_*.npy exist, loads pre-computed
    embeddings. Otherwise uses synthetic embeddings with real CheXpert metadata
    and demographics (train.csv + CHEXPERT DEMO.xlsx).

    Returns: embeddings, task_labels, group_labels, extra_labels, attributes, metadata_df
    """
    data_dir = find_data_dir()

    if data_dir is None:
        raise FileNotFoundError("Could not find data directory")

    # Optionally use real pre-computed embeddings (chest_*.npy from benchmark)
    if use_real_embeddings:
        emb_path = data_dir / "chest_embeddings.npy"
        lbl_path = data_dir / "chest_labels.npy"
        grp_path = data_dir / "chest_group_labels.npy"
        if emb_path.exists() and lbl_path.exists() and grp_path.exists():
            embeddings = np.load(str(emb_path))
            task_labels = np.load(str(lbl_path))
            group_labels = np.load(str(grp_path))
            if embeddings.ndim != 2 or task_labels.ndim != 1 or group_labels.ndim != 1:
                raise ValueError("Chest embeddings must be 2D; labels/group_labels must be 1D")
            n = embeddings.shape[0]
            if task_labels.shape[0] != n or group_labels.shape[0] != n:
                raise ValueError("Chest arrays must have matching length")
            # Ensure binary labels
            task_labels = (np.asarray(task_labels).astype(float) > 0).astype(int)
            # Convert group_labels to strings if needed
            group_labels = np.asarray(group_labels, dtype=object)
            extra_labels = {"group": group_labels}
            codes = pd.Categorical(group_labels).codes
            attributes = {"group": np.asarray(codes)}
            return embeddings, task_labels, group_labels, extra_labels, attributes, None

    # Load real CheXpert metadata and demographics
    train_csv = data_dir / "train.csv"
    demo_xlsx = data_dir / "CHEXPERT DEMO.xlsx"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")

    # Load training data
    data_df = pd.read_csv(train_csv)

    # Extract patient ID from Path column
    # Path format: "CheXpert-v1.0-small/train/patient00001/study1/..."
    data_df["PATIENT"] = data_df["Path"].str.extract(r"(patient\d+)", expand=False)

    # Load demographics if available
    if demo_xlsx.exists():
        demo_df = pd.read_excel(demo_xlsx, engine="openpyxl")
        # Merge demographics
        data_df = data_df.merge(
            demo_df[["PATIENT", "PRIMARY_RACE", "GENDER"]], on="PATIENT", how="left"
        )

    # Process race labels
    if "PRIMARY_RACE" in data_df.columns:
        data_df["race"] = "OTHER"
        mask_asian = data_df.PRIMARY_RACE.str.contains("Asian", na=False)
        data_df.loc[mask_asian, "race"] = "ASIAN"

        mask_black = data_df.PRIMARY_RACE.str.contains("Black", na=False)
        data_df.loc[mask_black, "race"] = "BLACK/AFRICAN AMERICAN"

        mask_white = data_df.PRIMARY_RACE.str.contains("White", na=False)
        data_df.loc[mask_white, "race"] = "WHITE"
    else:
        # If no demographics, create synthetic groups
        data_df["race"] = np.random.choice(
            ["ASIAN", "BLACK/AFRICAN AMERICAN", "WHITE"], size=len(data_df)
        )

    # Sample 2000 random samples for lightweight demo
    if len(data_df) > 2000:
        data_df = data_df.sample(n=2000, random_state=42).reset_index(drop=True)

    n_samples = len(data_df)

    # Generate lightweight embeddings (512-dim instead of 2048)
    embedding_dim = 512
    shortcut_dims = 10  # First 10 dimensions contain shortcuts

    # Create task labels (binary classification: Cardiomegaly)
    if "Cardiomegaly" in data_df.columns:
        task_labels = data_df["Cardiomegaly"].fillna(0).astype(int).values
    else:
        # Synthetic task labels if column doesn't exist
        task_labels = np.random.randint(0, 2, size=n_samples)

    # Ensure binary labels (map {-1,0,1} and other positives -> {0,1})
    task_labels = (task_labels > 0).astype(int)

    # Generate embeddings with shortcuts correlated to race
    synthetic_dataset = generate_parametric_shortcut_dataset(
        n_samples=n_samples, embedding_dim=embedding_dim, shortcut_dims=shortcut_dims, seed=42
    )
    embeddings = synthetic_dataset.embeddings

    # Get group labels (race)
    group_labels = data_df["race"].values

    # Build extra_labels for intersectional analysis (race + gender when both present)
    extra_labels = None
    if "GENDER" in data_df.columns and "race" in data_df.columns:
        gender = data_df["GENDER"].fillna("Unknown").astype(str).str.strip()
        # Normalize common gender values
        gender = gender.replace({"M": "Male", "F": "Female", "m": "Male", "f": "Female"})
        # Exclude rows with missing/unknown for intersectional (detector will filter)
        extra_labels = {
            "race": data_df["race"].values,
            "gender": gender.values,
        }

    # Attributes for causal_effect: race (encoded) + synthetic second attribute
    race_codes = pd.Categorical(data_df["race"]).codes
    rng = np.random.default_rng(42)
    attr_gender = rng.integers(0, 2, size=n_samples)
    attributes = {"race": np.asarray(race_codes), "gender": attr_gender}

    return embeddings, task_labels, group_labels, extra_labels, attributes, data_df


def _parse_attr_columns(df, attr_cols):
    """Parse attr_* columns into attributes dict for causal_effect."""
    if not attr_cols:
        return None
    attrs = {}
    for c in attr_cols:
        name = c[5:].strip() if len(c) > 5 else ""
        if not name:
            continue  # skip malformed columns like "attr_" (no suffix)
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            arr = np.asarray(col.fillna(-1).astype(int))
        else:
            arr = np.asarray(pd.Categorical(col).codes)
        attrs[name] = arr
    return attrs


def _add_intersectional_extra_labels(df, attr_cols, group_labels, extra_labels):
    """Add demographic attributes to extra_labels for intersectional and multi-attribute analysis.

    Adds all attr_* columns and group_label (as "group") when present.
    Intersectional detector requires 2+ attributes; multi-attribute single-attribute
    methods use all attributes for per-attribute analysis.
    """
    extra = dict(extra_labels) if extra_labels else {}

    # Add group_label as "group" when present
    if group_labels is not None:
        extra["group"] = np.asarray(group_labels, dtype=object)

    # Add all attr_* columns (attr_race -> "race", attr_gender -> "gender", etc.)
    for c in attr_cols:
        name = c[5:].strip() if len(c) > 5 else ""
        if name:
            extra[name] = np.asarray(df[c].fillna("").astype(str).values)

    # Return updated if we added any demographic attributes
    if len([k for k in extra if k not in ("spurious", "early_epoch_reps")]) >= 1:
        return extra
    return extra_labels


def load_custom_csv(csv_file, is_raw_data=False):
    """
    Load custom CSV file uploaded by user

    If is_raw_data=False (embeddings mode):
        Expected columns: embedding_0, embedding_1, ..., task_label, group_label
    If is_raw_data=True (raw data mode):
        Expected columns: text, task_label, group_label
        Optional columns:
        spurious_label (for SSA extra supervision)
        attr_<name> (for causal_effect, intersectional, and multi-attribute: e.g., attr_race, attr_gender, attr_age)
    """
    if csv_file is None:
        raise ValueError("No CSV file provided")

    # Handle both Gradio file object and string path
    file_path = csv_file.name if hasattr(csv_file, "name") else csv_file
    try:
        df = pd.read_csv(file_path)
    except EmptyDataError as exc:
        raise ValueError(
            "CSV file is empty. Please upload a file with a header row and data rows."
        ) from exc
    except ParserError as exc:
        raise ValueError(
            "Unable to parse CSV format. Check that rows have consistent columns and values are properly quoted."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file must be UTF-8 encoded text.") from exc

    if df.empty:
        raise ValueError("CSV has headers but no data rows. Add at least one sample row.")

    # Common wrong-delimiter signal: entire header parsed as one column.
    if len(df.columns) == 1:
        header_col = str(df.columns[0])
        if ";" in header_col or "\t" in header_col or "|" in header_col:
            raise ValueError(
                "CSV appears to use the wrong delimiter. Please upload a comma-separated file (`,` delimiter)."
            )

    if is_raw_data:
        # Raw data mode: expect text column
        if "text" not in df.columns:
            raise ValueError("Raw data CSV must have a 'text' column containing the raw text data")

        # Get text data
        raw_texts = df["text"].astype(str).tolist()

        # Get labels
        if "task_label" not in df.columns:
            raise ValueError("CSV must have a 'task_label' column")

        task_labels = df["task_label"].values
        if pd.isna(df["task_label"]).any():
            raise ValueError(
                "Column 'task_label' contains empty values. Every row must include a label."
            )

        if "group_label" in df.columns:
            group_labels = df["group_label"].values
        else:
            group_labels = None
        if "group_label" in df.columns and pd.isna(df["group_label"]).any():
            raise ValueError(
                "Column 'group_label' contains empty values. Fill missing values or remove the column."
            )

        if df["text"].astype(str).str.strip().eq("").any():
            raise ValueError(
                "Column 'text' contains empty values. Every row must include non-empty text."
            )

        # Validate lengths match
        if len(raw_texts) != len(task_labels):
            raise ValueError(
                f"Length mismatch: {len(raw_texts)} text entries but {len(task_labels)} task labels. "
                "Each row must have both text and task_label."
            )
        if group_labels is not None and len(raw_texts) != len(group_labels):
            raise ValueError(
                f"Length mismatch: {len(raw_texts)} text entries but {len(group_labels)} group labels. "
                "Each row must have text, task_label, and group_label."
            )

        if "spurious_label" in df.columns:
            extra_labels = {"spurious": df["spurious_label"].values}

        attr_cols = [c for c in df.columns if c.startswith("attr_")]
        # Add demographic attributes for intersectional (need 2+ attr_* or group_label + attr_*)
        extra_labels = _add_intersectional_extra_labels(df, attr_cols, group_labels, extra_labels)

        attributes = _parse_attr_columns(df, attr_cols)

        return raw_texts, task_labels, group_labels, extra_labels, attributes, df
    else:
        # Embeddings mode: extract embedding columns (exclude attr_*)
        embedding_cols = [col for col in df.columns if col.startswith("embedding_")]

        if len(embedding_cols) == 0:
            # Try to use all numeric columns except labels and attr_*
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"task_label", "group_label"} | {
                c for c in df.columns if c.startswith("attr_")
            }
            embedding_cols = [col for col in numeric_cols if col not in exclude]

        if len(embedding_cols) == 0:
            raise ValueError(
                "No embedding columns found. Expected columns like 'embedding_0', 'embedding_1', etc."
            )

        if any(pd.isna(df[col]).any() for col in embedding_cols):
            raise ValueError(
                "Embedding columns contain empty values. Fill missing values in embedding features."
            )

        non_numeric_cols = [
            col for col in embedding_cols if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric_cols:
            bad_cols = ", ".join(non_numeric_cols[:3])
            if len(non_numeric_cols) > 3:
                bad_cols += ", ..."
            raise ValueError(
                f"Embedding columns must be numeric. Non-numeric columns detected: {bad_cols}"
            )

        embeddings = df[embedding_cols].values

        # Get labels
        if "task_label" not in df.columns:
            raise ValueError("CSV must have a 'task_label' column")

        task_labels = df["task_label"].values
        if pd.isna(df["task_label"]).any():
            raise ValueError(
                "Column 'task_label' contains empty values. Every row must include a label."
            )

        if "group_label" in df.columns:
            group_labels = df["group_label"].values
        else:
            group_labels = None
        if "group_label" in df.columns and pd.isna(df["group_label"]).any():
            raise ValueError(
                "Column 'group_label' contains empty values. Fill missing values or remove the column."
            )

        extra_labels = None
        if "spurious_label" in df.columns:
            extra_labels = {"spurious": df["spurious_label"].values}

        attr_cols = [c for c in df.columns if c.startswith("attr_")]
        extra_labels = _add_intersectional_extra_labels(df, attr_cols, group_labels, extra_labels)
        attributes = _parse_attr_columns(df, attr_cols)

        return embeddings, task_labels, group_labels, extra_labels, attributes, df


def _build_ssa_splits(
    n_samples: int, labeled_fraction: float, seed: int = 42
) -> dict[str, np.ndarray]:
    if n_samples < 2:
        raise ValueError("SSA requires at least 2 samples to create labeled/unlabeled splits.")
    if labeled_fraction <= 0 or labeled_fraction >= 1:
        raise ValueError("SSA labeled fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_labeled = int(round(n_samples * labeled_fraction))
    n_labeled = max(1, min(n_labeled, n_samples - 1))

    train_l = np.sort(indices[:n_labeled])
    train_u = np.sort(indices[n_labeled:])

    return {"train_l": train_l, "train_u": train_u}


def _load_torch_model(model_file) -> torch.nn.Module:
    """Load a serialized PyTorch module from a Gradio file upload."""
    if model_file is None:
        raise ValueError("Please upload a PyTorch model file (.pt or .pth).")

    path = model_file.name if hasattr(model_file, "name") else model_file
    # PyTorch 2.6+ requires weights_only=False for loading full models
    model = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(model, torch.nn.Module):
        return model
    raise TypeError(
        "Loaded object is not a torch.nn.Module. Please upload a serialized model (not a state_dict)."
    )


def _parse_head_identifier(value: str | None, fallback: str | int) -> str | int:
    """Convert user-provided head identifier text into str/int used by GradCAM."""
    if value is None:
        return fallback
    text = str(value).strip()
    if text == "":
        return fallback
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _parse_optional_int(value: int | float | None) -> int | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target indices must be integers.") from exc


def _preprocess_gradcam_image(image_file, image_size: int, color_mode: str):
    if image_file is None:
        raise ValueError("Please upload an image for GradCAM analysis.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    mode = "L" if color_mode == "Grayscale" else "RGB"
    with Image.open(image_file.name if hasattr(image_file, "name") else image_file) as img:
        img = img.convert(mode)
        img = img.resize((image_size, image_size))
        if mode == "L":
            arr = np.array(img, dtype=np.float32) / 255.0
            display = np.stack([arr, arr, arr], axis=-1)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            display = arr
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    tensor = tensor.float()
    tensor.requires_grad_(True)
    return display, tensor


def _overlay_heatmap(
    base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    import matplotlib

    base = np.clip(base_image, 0.0, 1.0)
    cmap = matplotlib.colormaps["jet"]
    heat = np.clip(heatmap, 0.0, 1.0)
    colored = cmap(heat)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def _colorize_heatmap(heatmap: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    heat = np.clip(heatmap, 0.0, 1.0)
    colored = cmap(heat)[..., :3]
    return (colored * 255).astype(np.uint8)


def _load_heatmap_file(file_obj):
    if file_obj is None:
        raise ValueError("Please upload a heatmap file (.npz or .npy).")
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "heatmaps" in data:
            heatmaps = data["heatmaps"]
        else:
            keys = list(data.keys())
            if not keys:
                raise ValueError("No arrays found in .npz file.")
            heatmaps = data[keys[0]]
        labels = data["labels"] if "labels" in data else None
        group_labels = data["group_labels"] if "group_labels" in data else None
        return heatmaps, labels, group_labels
    if path.endswith(".npy"):
        heatmaps = np.load(path, allow_pickle=True)
        return heatmaps, None, None
    raise ValueError("Unsupported file format. Use .npz or .npy.")


def _preprocess_gradcam_images(image_files, image_size: int, color_mode: str) -> torch.Tensor:
    if not image_files:
        raise ValueError("Please upload one or more images.")
    tensors = []
    for image_file in image_files:
        _, tensor = _preprocess_gradcam_image(image_file, image_size, color_mode)
        tensors.append(tensor)
    return torch.cat(tensors, dim=0)


def _preprocess_mask_images(mask_files, image_size: int) -> np.ndarray:
    if not mask_files:
        raise ValueError("Please upload one or more mask images.")
    masks = []
    for mask_file in mask_files:
        with Image.open(mask_file.name if hasattr(mask_file, "name") else mask_file) as img:
            img = img.convert("L")
            img = img.resize((image_size, image_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            masks.append(arr)
    return np.stack(masks, axis=0)


def run_gradcam_analysis(
    model_file,
    image_file,
    target_layer: str,
    disease_head: str,
    attribute_head: str,
    disease_target: int | float | None,
    attribute_target: int | float | None,
    threshold: float,
    image_size: int,
    color_mode: str,
):
    """Gradio callback that computes GradCAM heatmaps and overlap metrics."""
    if not target_layer or target_layer.strip() == "":
        error_html = """
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>GradCAM Error:</strong> Please provide a valid target layer path.
        </div>
        """
        return error_html, None, None, None
    try:
        model = _load_torch_model(model_file)
        base_image, tensor_input = _preprocess_gradcam_image(
            image_file, int(image_size), color_mode
        )

        disease_identifier = _parse_head_identifier(disease_head, "disease")
        attribute_identifier = _parse_head_identifier(attribute_head, "attribute")

        disease_idx = _parse_optional_int(disease_target)
        attribute_idx = _parse_optional_int(attribute_target)

        generator = GradCAMHeatmapGenerator(model, target_layer=target_layer.strip())
        result = generator.generate_attention_overlap(
            tensor_input,
            disease_target=disease_idx,
            attribute_target=attribute_idx,
            disease_head=disease_identifier,
            attribute_head=attribute_identifier,
            threshold=float(threshold),
        )
        generator.close()

        disease_overlay = _overlay_heatmap(base_image, result.disease_heatmap[0])
        attribute_overlay = _overlay_heatmap(base_image, result.attribute_heatmap[0])

        metrics = result.metrics
        html = f"""
        <div style="background-color:#e8f5e9;border:1px solid #c8e6c9;border-radius:5px;padding:12px;color:#155724;">
            <strong>GradCAM complete.</strong>
            <ul style="margin:8px 0 0 16px;padding:0;">
                <li>Dice overlap: {metrics.get('dice', 0.0):.3f}</li>
                <li>IoU overlap: {metrics.get('iou', 0.0):.3f}</li>
                <li>Cosine similarity: {metrics.get('cosine', 0.0):.3f}</li>
            </ul>
        </div>
        """
        return html, disease_overlay, attribute_overlay, metrics
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>GradCAM Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None, None


def run_gradcam_mask_overlap_analysis(
    model_file,
    image_files,
    mask_files,
    target_layer: str,
    head: str,
    target_index: int | float | None,
    threshold: float,
    mask_threshold: float,
    image_size: int,
    color_mode: str,
    batch_size: int,
):
    if not target_layer or target_layer.strip() == "":
        error_html = """
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>GradCAM Mask Overlap Error:</strong> Please provide a valid target layer path.
        </div>
        """
        return error_html, None

    try:
        model = _load_torch_model(model_file)
        images = image_files or []
        masks = mask_files or []
        if len(images) != len(masks):
            raise ValueError("Number of images must match number of masks.")

        tensor_inputs = _preprocess_gradcam_images(images, int(image_size), color_mode)
        mask_arrays = _preprocess_mask_images(masks, int(image_size))

        head_id = _parse_head_identifier(head, "logits")
        target_idx = _parse_optional_int(target_index)

        detector = ShortcutDetector(
            methods=["gradcam_mask_overlap"],
            gradcam_overlap_threshold=float(threshold),
            gradcam_mask_threshold=float(mask_threshold),
            gradcam_overlap_batch_size=int(batch_size),
        )

        def _loader():
            return {
                "inputs": tensor_inputs,
                "masks": mask_arrays,
                "model": model,
                "target_layer": target_layer.strip(),
                "head": head_id,
                "target_index": target_idx,
                "batch_size": int(batch_size),
            }

        detector.fit_from_loaders({"gradcam_mask_overlap": _loader})
        result = detector.get_results().get("gradcam_mask_overlap", {})
        metrics = result.get("metrics", {})
        report = result.get("report", {})

        html = f"""
        <div style="background-color:#e8f5e9;border:1px solid #c8e6c9;border-radius:5px;padding:12px;color:#155724;">
            <strong>GradCAM mask overlap complete.</strong>
            <ul style="margin:8px 0 0 16px;padding:0;">
                <li>Samples: {metrics.get('n_samples', 0)}</li>
                <li>Attention-in-mask (mean): {metrics.get('attention_in_mask_mean', 0.0):.3f}</li>
                <li>Dice (mean): {metrics.get('dice_mean', 0.0):.3f}</li>
                <li>IoU (mean): {metrics.get('iou_mean', 0.0):.3f}</li>
            </ul>
        </div>
        """
        payload = {
            "metrics": metrics,
            "top_samples": report.get("top_samples", []),
            "bottom_samples": report.get("bottom_samples", []),
        }
        return html, payload
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>GradCAM Mask Overlap Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None


def run_spray_analysis(
    spray_input_mode: str,
    heatmap_file,
    model_file,
    image_files,
    target_layer: str,
    head: str,
    target_index: int | float | None,
    image_size: int,
    color_mode: str,
    affinity: str,
    cluster_selection: str,
    n_clusters: int | float | None,
    min_clusters: int,
    max_clusters: int,
    downsample_size: int | float | None,
):
    try:
        if spray_input_mode == "Upload Heatmaps (.npz/.npy)":
            heatmaps, labels, group_labels = _load_heatmap_file(heatmap_file)
        else:
            if not target_layer or target_layer.strip() == "":
                raise ValueError("Please provide a target layer for GradCAM heatmap generation.")
            model = _load_torch_model(model_file)
            images = image_files or []
            tensor_inputs = _preprocess_gradcam_images(images, int(image_size), color_mode)
            generator = GradCAMHeatmapGenerator(model, target_layer=target_layer.strip())
            head_id = _parse_head_identifier(head, "logits")
            idx = _parse_optional_int(target_index)
            heatmaps = generator.generate_heatmap(
                tensor_inputs,
                head=head_id,
                target_index=idx,
            )
            generator.close()
            labels = None
            group_labels = None

        n_clusters_val = _parse_optional_int(n_clusters)
        if n_clusters_val is not None and n_clusters_val <= 0:
            n_clusters_val = None
        downsample_val = _parse_optional_int(downsample_size)
        if downsample_val is not None and downsample_val <= 0:
            downsample_val = None

        detector = SpRAyDetector(
            affinity=affinity,
            cluster_selection=cluster_selection,
            n_clusters=n_clusters_val,
            min_clusters=int(min_clusters),
            max_clusters=int(max_clusters),
            downsample_size=downsample_val,
            random_state=42,
        )
        detector.fit(heatmaps=heatmaps, labels=labels, group_labels=group_labels)
        report = detector.get_report()

        clever = report.get("report", {}).get("clever_hans", {})
        metrics = report.get("metrics", {})
        summary_html = f"""
        <div style="background-color:#e3f2fd;border:1px solid #bbdefb;border-radius:5px;padding:12px;color:#0d47a1;">
            <strong>SpRAy complete.</strong>
            <ul style="margin:8px 0 0 16px;padding:0;">
                <li>Shortcut detected: {clever.get('shortcut_detected')}</li>
                <li>Risk: {str(clever.get('risk_level', 'unknown')).title()}</li>
                <li>Flags: {', '.join(clever.get('flags', [])) or 'None'}</li>
                <li>Clusters: {metrics.get('n_clusters')}</li>
                <li>Silhouette: {metrics.get('silhouette')}</li>
            </ul>
        </div>
        """

        gallery = []
        for cluster_id, heatmap in sorted(detector.representative_heatmaps_.items()):
            gallery.append((_colorize_heatmap(heatmap), f"Cluster {cluster_id}"))

        return summary_html, report.get("report", {}), gallery
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>SpRAy Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, []


def _load_cav_bundle(bundle_file):
    """Load CAV concept bundle from .npz/.npy upload."""
    if bundle_file is None:
        raise ValueError("Please upload a CAV bundle (.npz or .npy).")

    path = bundle_file.name if hasattr(bundle_file, "name") else bundle_file
    ext = str(path).lower()
    concept_sets = {}
    random_set = None
    target_activations = None
    target_directional_derivatives = None

    if ext.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            for key in keys:
                if key.startswith("concept_"):
                    concept_name = key[len("concept_") :].strip() or "unnamed_concept"
                    concept_sets[concept_name] = np.asarray(data[key], dtype=float)
                elif key == "concept_sets":
                    maybe_dict = data[key]
                    if isinstance(maybe_dict, np.ndarray) and maybe_dict.dtype == object:
                        maybe_dict = maybe_dict.item()
                    if isinstance(maybe_dict, dict):
                        for name, arr in maybe_dict.items():
                            concept_sets[str(name)] = np.asarray(arr, dtype=float)
                elif key == "random_set":
                    random_set = np.asarray(data[key], dtype=float)
                elif key == "target_activations":
                    target_activations = np.asarray(data[key], dtype=float)
                elif key == "target_directional_derivatives":
                    target_directional_derivatives = np.asarray(data[key], dtype=float)
    elif ext.endswith(".npy"):
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object:
            payload = payload.item()
        if not isinstance(payload, dict):
            raise ValueError(".npy CAV bundle must store a dict payload.")
        raw_concepts = payload.get("concept_sets", {})
        if not isinstance(raw_concepts, dict):
            raise ValueError("CAV bundle field 'concept_sets' must be a dict.")
        for name, arr in raw_concepts.items():
            concept_sets[str(name)] = np.asarray(arr, dtype=float)
        random_set = payload.get("random_set")
        if random_set is not None:
            random_set = np.asarray(random_set, dtype=float)
        if payload.get("target_activations") is not None:
            target_activations = np.asarray(payload["target_activations"], dtype=float)
        if payload.get("target_directional_derivatives") is not None:
            target_directional_derivatives = np.asarray(
                payload["target_directional_derivatives"],
                dtype=float,
            )
    else:
        raise ValueError("Unsupported CAV bundle format. Please upload .npz or .npy.")

    if not concept_sets:
        raise ValueError(
            "No concepts found. Provide keys like 'concept_<name>' or a 'concept_sets' dictionary."
        )
    if random_set is None:
        raise ValueError("Missing 'random_set' in CAV bundle.")

    return {
        "concept_sets": concept_sets,
        "random_set": random_set,
        "target_activations": target_activations,
        "target_directional_derivatives": target_directional_derivatives,
    }


def run_vae_analysis(
    image_files,
    labels_csv,
    img_size: int,
    channels: int,
    num_classes: int,
    latent_dim: int,
    epochs: int,
    classifier_epochs: int,
):
    """Run VAE shortcut detection on uploaded images."""
    supported_img_sizes = {32, 64, 128}
    try:
        if not image_files:
            raise ValueError("Please upload one or more images.")
        if labels_csv is None:
            raise ValueError("Please upload a labels CSV with task_label column.")
        path = labels_csv.name if hasattr(labels_csv, "name") else labels_csv
        labels_df = pd.read_csv(path)
        if "task_label" not in labels_df.columns:
            raise ValueError("Labels CSV must have 'task_label' column.")
        labels = np.asarray(labels_df["task_label"].values)
        if isinstance(image_files, list):
            n_images = len(image_files)
        else:
            image_files = [image_files]
            n_images = 1
        if len(labels) != n_images:
            raise ValueError(
                f"Labels CSV has {len(labels)} rows but {n_images} images. "
                "Each image must have a corresponding task_label."
            )
        img_size = int(img_size)
        if img_size not in supported_img_sizes:
            raise ValueError(
                f"Unsupported image size {img_size}. "
                f"Choose one of: {sorted(supported_img_sizes)}."
            )
        channels = int(channels)
        num_classes = int(num_classes)
        images_list = []
        for f in image_files:
            p = f.name if hasattr(f, "name") else f
            with Image.open(p) as img:
                img = img.convert("L" if channels == 1 else "RGB")
                img = img.resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                if channels == 1:
                    arr = arr[np.newaxis, ...]
                else:
                    arr = np.transpose(arr, (2, 0, 1))
                images_list.append(arr)
        images_np = np.stack(images_list, axis=0)
        images_t = torch.from_numpy(images_np).float()
        loader = {
            "images": images_t,
            "labels": labels.astype(np.int64),
            "img_size": img_size,
            "channels": channels,
            "num_classes": num_classes,
        }
        detector = ShortcutDetector(
            methods=["vae"],
            vae_latent_dim=int(latent_dim),
            vae_epochs=int(epochs),
            vae_classifier_epochs=int(classifier_epochs),
            vae_batch_size=min(16, n_images),
        )
        detector.fit_from_loaders({"vae": loader})
        result = detector.get_results().get("vae", {})
        if not result.get("success"):
            raise RuntimeError(result.get("error", "VAE analysis failed"))
        metrics = result.get("metrics", {})
        report = result.get("report", {})
        n_flagged = metrics.get("n_flagged", 0)
        max_pred = metrics.get("max_predictiveness")
        max_pred_str = f"{max_pred:.3f}" if max_pred is not None else "N/A"
        risk_label = result.get("risk_label", "Unknown")
        summary_html = f"""
        <div style="background-color:#e8f5e9;border:1px solid #c8e6c9;border-radius:5px;padding:12px;color:#155724;">
            <strong>VAE complete.</strong><br/>
            Latent dims: {metrics.get('latent_dim', 0)} | Flagged: {n_flagged}<br/>
            Max predictiveness: {max_pred_str}<br/>
            Risk: {risk_label}
        </div>
        """
        return summary_html, {
            "metrics": metrics,
            "per_dimension": report.get("per_dimension", []),
            "risk": risk_label,
        }
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>VAE Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None


def run_cav_analysis(
    cav_bundle_file,
    cav_quality_threshold: float,
    cav_shortcut_threshold: float,
    cav_test_size: float,
    cav_min_examples: int,
):
    """Run CAV shortcut concept testing from precomputed activation bundle."""
    try:
        bundle = _load_cav_bundle(cav_bundle_file)

        detector = ShortcutDetector(
            methods=["cav"],
            cav_quality_threshold=float(cav_quality_threshold),
            cav_shortcut_threshold=float(cav_shortcut_threshold),
            cav_test_size=float(cav_test_size),
            cav_min_examples_per_set=int(cav_min_examples),
        )
        detector.fit_from_loaders({"cav": bundle})
        result = detector.get_results().get("cav", {})
        if not result.get("success"):
            raise RuntimeError(result.get("error", "CAV analysis failed"))

        metrics = result.get("metrics", {})
        report = result.get("report", {})
        per_concept = report.get("per_concept", [])
        risk_label = result.get("risk_label", "Unknown")
        risk_reason = result.get("risk_reason", "No risk reason available.")
        flagged = int(sum(1 for row in per_concept if row.get("flagged")))

        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>CAV complete.</strong><br/>
            Concepts: {metrics.get('n_concepts', 0)} (tested: {metrics.get('n_tested', 0)})<br/>
            Risk: {risk_label}<br/>
            Reason: {risk_reason}<br/>
            Flagged concepts: {flagged}
        </div>
        """
        return summary_html, {"metrics": metrics, "per_concept": per_concept, "risk": risk_label}
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>CAV Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None


def run_shortcut_masking_analysis(
    masking_mode: str,
    mask_use_last_detection: bool,
    detection_state,
    mask_images,
    mask_masks_or_heatmap_file,
    mask_strategy_img: str,
    mask_heatmap_threshold: float,
    mask_augment_fraction: float,
    mask_embeddings_csv,
    mask_dim_indices_text: str,
    mask_strategy_emb: str,
):
    """Run M01 Shortcut Feature Masking: produce augmented images or embeddings."""
    import io
    import zipfile

    try:
        if masking_mode == "Image":
            if not mask_images:
                raise ValueError("Please upload one or more images.")
            if mask_masks_or_heatmap_file is None:
                raise ValueError(
                    "Please upload shortcut masks (images) or a heatmap file (.npz/.npy)."
                )
            image_list = mask_images if isinstance(mask_images, list) else [mask_images]
            n_images = len(image_list)
            mask_path = (
                mask_masks_or_heatmap_file.name
                if hasattr(mask_masks_or_heatmap_file, "name")
                else mask_masks_or_heatmap_file
            )
            path_str = str(mask_path).lower()
            images_list = []
            for f in image_list:
                img_path = f.name if hasattr(f, "name") else f
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img, dtype=np.float64) / 255.0
                images_list.append(arr)
            images = np.stack(images_list, axis=0)
            h, w = images.shape[1], images.shape[2]
            if path_str.endswith(".npz") or path_str.endswith(".npy"):
                heatmaps, _, _ = _load_heatmap_file(mask_masks_or_heatmap_file)
                heatmaps = np.asarray(heatmaps, dtype=np.float64)
                if heatmaps.ndim == 2:
                    heatmaps = heatmaps[np.newaxis, ...]
                if (
                    heatmaps.shape[0] != n_images
                    or heatmaps.shape[1] != h
                    or heatmaps.shape[2] != w
                ):
                    raise ValueError(
                        f"Heatmaps shape {heatmaps.shape} must match image count {n_images} and size {h}x{w}. "
                        "Resize heatmaps or use matching mask images."
                    )
                shortcut_masks = None
                heatmaps_arg = heatmaps
            else:
                with Image.open(mask_path) as img:
                    img = img.convert("L")
                    img = img.resize((w, h))
                    single = np.array(img, dtype=np.float64) / 255.0
                mask_binary = (single > 0.5).astype(np.float64)
                shortcut_masks = np.tile(mask_binary[np.newaxis, ...], (n_images, 1, 1))
                heatmaps_arg = None

            masker = ShortcutMasker(
                strategy=mask_strategy_img,
                heatmap_threshold=float(mask_heatmap_threshold),
                augment_fraction=float(mask_augment_fraction),
                random_state=42,
            )
            if shortcut_masks is not None and heatmaps_arg is None:
                augmented = masker.mask_images(images, shortcut_masks=shortcut_masks)
            else:
                augmented = masker.mask_images(images, heatmaps=heatmaps_arg)

            aug_uint8 = (np.clip(augmented, 0, 1) * 255).astype(np.uint8)
            tmp_dir = tempfile.mkdtemp(prefix="shortcut_masking_")
            zip_path = os.path.join(tmp_dir, "augmented_images.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i in range(aug_uint8.shape[0]):
                    buf = io.BytesIO()
                    frame = aug_uint8[i]
                    if frame.ndim == 2:
                        img_out = Image.fromarray(frame)
                    else:
                        img_out = Image.fromarray(frame)
                    img_out.save(buf, format="PNG")
                    zf.writestr(f"augmented_{i:04d}.png", buf.getvalue())
            summary_html = f"""
            <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
                <strong>Shortcut masking (M01) complete.</strong><br/>
                Mode: Image | Strategy: {mask_strategy_img}<br/>
                Samples augmented: {n_images}<br/>
                Download the zip below.
            </div>
            """
            report = {"mode": "image", "n_samples": n_images, "strategy": mask_strategy_img}
            return summary_html, report, zip_path

        else:
            if mask_use_last_detection and detection_state:
                path = os.path.join(detection_state, "embeddings_for_mitigation.csv")
                if not os.path.exists(path):
                    raise ValueError(
                        "Embeddings file not found. Run detection first and ensure CSV export completed."
                    )
            elif mask_embeddings_csv is not None:
                path = (
                    mask_embeddings_csv.name
                    if hasattr(mask_embeddings_csv, "name")
                    else mask_embeddings_csv
                )
            else:
                raise ValueError(
                    "Upload an embeddings CSV or run detection first and check 'Use embeddings from last detection run'."
                )
            df = pd.read_csv(path)
            embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
            if not embedding_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude = {"task_label", "group_label"} | {
                    c for c in df.columns if c.startswith("attr_")
                }
                embedding_cols = [c for c in numeric_cols if c not in exclude]
            if not embedding_cols:
                raise ValueError(
                    "No embedding columns found. Use embedding_0, embedding_1, ... or numeric columns."
                )
            embeddings = df[embedding_cols].values.astype(np.float64)
            dim_text = (mask_dim_indices_text or "").strip()
            if not dim_text:
                raise ValueError("Please enter dimension indices to mask (e.g. 0,3,7).")
            parts = [p.strip() for p in dim_text.replace(" ", ",").split(",") if p.strip()]
            flagged_dims = [int(p) for p in parts if p.isdigit()]
            if not flagged_dims:
                raise ValueError("Provide at least one integer dimension index (e.g. 0,3,7).")
            masker = ShortcutMasker(
                strategy=mask_strategy_emb,
                augment_fraction=float(mask_augment_fraction),
                random_state=42,
            )
            augmented = masker.mask_embeddings(embeddings, flagged_dims)
            df_out = df.copy()
            for j, col in enumerate(embedding_cols):
                df_out[col] = augmented[:, j]
            tmp_dir = tempfile.mkdtemp(prefix="shortcut_masking_")
            csv_path = os.path.join(tmp_dir, "augmented_embeddings.csv")
            df_out.to_csv(csv_path, index=False)
            summary_html = f"""
            <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
                <strong>Shortcut masking (M01) complete.</strong><br/>
                Mode: Embedding | Strategy: {mask_strategy_emb}<br/>
                Samples: {embeddings.shape[0]} | Masked dims: {flagged_dims}<br/>
                Download the CSV below.
            </div>
            """
            report = {
                "mode": "embedding",
                "n_samples": embeddings.shape[0],
                "flagged_dims": flagged_dims,
                "strategy": mask_strategy_emb,
            }
            return summary_html, report, csv_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Shortcut Masking Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_background_randomizer_analysis(
    bg_images,
    bg_masks_or_heatmap_file,
    bg_heatmap_threshold: float,
    bg_augment_fraction: float,
):
    """Run M02 Background Randomization: swap foregrounds with random backgrounds."""
    import io
    import zipfile

    try:
        if not bg_images:
            raise ValueError("Please upload one or more images.")
        if bg_masks_or_heatmap_file is None:
            raise ValueError(
                "Please upload foreground masks (images) or a heatmap file (.npz/.npy)."
            )
        image_list = bg_images if isinstance(bg_images, list) else [bg_images]
        n_images = len(image_list)
        if n_images < 2:
            raise ValueError("Background randomization requires at least 2 images to swap.")
        mask_path = (
            bg_masks_or_heatmap_file.name
            if hasattr(bg_masks_or_heatmap_file, "name")
            else bg_masks_or_heatmap_file
        )
        path_str = str(mask_path).lower()
        images_list = []
        for f in image_list:
            img_path = f.name if hasattr(f, "name") else f
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                arr = np.array(img, dtype=np.float64) / 255.0
            images_list.append(arr)
        images = np.stack(images_list, axis=0)
        h, w = images.shape[1], images.shape[2]
        if path_str.endswith(".npz") or path_str.endswith(".npy"):
            heatmaps, _, _ = _load_heatmap_file(bg_masks_or_heatmap_file)
            heatmaps = np.asarray(heatmaps, dtype=np.float64)
            if heatmaps.ndim == 2:
                heatmaps = heatmaps[np.newaxis, ...]
            if heatmaps.shape[0] != n_images or heatmaps.shape[1] != h or heatmaps.shape[2] != w:
                raise ValueError(
                    f"Heatmaps shape {heatmaps.shape} must match image count {n_images} and size {h}x{w}. "
                    "Resize heatmaps or use matching mask images."
                )
            foreground_masks = (heatmaps >= float(bg_heatmap_threshold)).astype(np.float64)
        else:
            with Image.open(mask_path) as img:
                img = img.convert("L")
                img = img.resize((w, h))
                single = np.array(img, dtype=np.float64) / 255.0
            mask_binary = (single > 0.5).astype(np.float64)
            foreground_masks = np.tile(mask_binary[np.newaxis, ...], (n_images, 1, 1))

        randomizer = BackgroundRandomizer(
            augment_fraction=float(bg_augment_fraction),
            random_state=42,
        )
        augmented = randomizer.swap_foregrounds(images, foreground_masks)

        aug_uint8 = (np.clip(augmented, 0, 1) * 255).astype(np.uint8)
        tmp_dir = tempfile.mkdtemp(prefix="background_randomizer_")
        zip_path = os.path.join(tmp_dir, "augmented_images.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(aug_uint8.shape[0]):
                buf = io.BytesIO()
                frame = aug_uint8[i]
                img_out = Image.fromarray(frame)
                img_out.save(buf, format="PNG")
                zf.writestr(f"augmented_{i:04d}.png", buf.getvalue())
        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>Background randomization (M02) complete.</strong><br/>
            Samples augmented: {n_images}<br/>
            Download the zip below.
        </div>
        """
        report = {"mode": "background_randomization", "n_samples": n_images}
        return summary_html, report, zip_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Background Randomization Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_adversarial_debiasing_analysis(
    debias_use_last_detection: bool,
    detection_state,
    debias_embeddings_csv,
    debias_group_col: str,
    debias_hidden_dim: int,
    debias_adversary_weight: float,
    debias_n_epochs: int,
):
    """Run M04 Adversarial Debiasing: produce debiased embeddings."""
    try:
        if debias_use_last_detection and detection_state:
            path = os.path.join(detection_state, "embeddings_for_mitigation.csv")
            if not os.path.exists(path):
                raise ValueError(
                    "Embeddings file not found. Run detection first and ensure CSV export completed."
                )
        elif debias_embeddings_csv is not None:
            path = (
                debias_embeddings_csv.name
                if hasattr(debias_embeddings_csv, "name")
                else debias_embeddings_csv
            )
        else:
            raise ValueError(
                "Upload an embeddings CSV or run detection first and check 'Use embeddings from last detection run'."
            )
        df = pd.read_csv(path)
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        if not embedding_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"task_label", "group_label"} | {
                c for c in df.columns if c.startswith("attr_")
            }
            embedding_cols = [c for c in numeric_cols if c not in exclude]
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found. Use embedding_0, embedding_1, ... or numeric columns."
            )

        group_col = (debias_group_col or "group_label").strip()
        if group_col not in df.columns:
            raise ValueError(
                f"Group/protected column '{group_col}' not found. " f"Available: {list(df.columns)}"
            )
        protected_labels = df[group_col].values

        task_labels = None
        if "task_label" in df.columns:
            task_labels = df["task_label"].values

        embeddings = df[embedding_cols].values.astype(np.float32)
        debiaser = AdversarialDebiasing(
            hidden_dim=int(debias_hidden_dim) if debias_hidden_dim else None,
            adversary_weight=float(debias_adversary_weight),
            n_epochs=int(debias_n_epochs),
            batch_size=64,
            random_state=42,
        )
        debiaser.fit(embeddings, protected_labels, task_labels=task_labels)
        debiased = debiaser.transform(embeddings)

        n_samples, h_dim = debiased.shape
        df_out = df.copy()
        for j in range(h_dim):
            df_out[f"debiased_embedding_{j}"] = debiased[:, j]

        tmp_dir = tempfile.mkdtemp(prefix="adversarial_debiasing_")
        csv_path = os.path.join(tmp_dir, "debiased_embeddings.csv")
        df_out.to_csv(csv_path, index=False)
        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>Adversarial debiasing (M04) complete.</strong><br/>
            Samples: {n_samples} | Debiased dims: {h_dim}<br/>
            Zhang et al. 2018 – adversarial training to remove demographic encoding.<br/>
            Download the CSV below.
        </div>
        """
        report = {
            "mode": "adversarial_debiasing",
            "n_samples": n_samples,
            "hidden_dim": h_dim,
            "group_col": group_col,
        }
        return summary_html, report, csv_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Adversarial Debiasing Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_last_layer_retraining_analysis(
    dfr_use_last_detection: bool,
    detection_state,
    dfr_embeddings_csv,
    dfr_group_col: str,
    dfr_C: float,
    dfr_penalty: str,
):
    """Run M06 Last Layer Retraining (DFR): retrain classifier on balanced subset."""
    try:
        if dfr_use_last_detection and detection_state:
            path = os.path.join(detection_state, "embeddings_for_mitigation.csv")
            if not os.path.exists(path):
                raise ValueError(
                    "Embeddings file not found. Run detection first and ensure CSV export completed."
                )
        elif dfr_embeddings_csv is not None:
            path = (
                dfr_embeddings_csv.name
                if hasattr(dfr_embeddings_csv, "name")
                else dfr_embeddings_csv
            )
        else:
            raise ValueError(
                "Upload an embeddings CSV or run detection first and check 'Use embeddings from last detection run'."
            )
        df = pd.read_csv(path)
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        if not embedding_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"task_label", "group_label"} | {
                c for c in df.columns if c.startswith("attr_")
            }
            embedding_cols = [c for c in numeric_cols if c not in exclude]
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found. Use embedding_0, embedding_1, ... or numeric columns."
            )

        if "task_label" not in df.columns:
            raise ValueError("task_label column required for Last Layer Retraining.")

        group_col = (dfr_group_col or "group_label").strip()
        if group_col not in df.columns:
            raise ValueError(
                f"Group/protected column '{group_col}' not found. " f"Available: {list(df.columns)}"
            )

        embeddings = df[embedding_cols].values.astype(np.float64)
        task_labels = df["task_label"].values
        group_labels = df[group_col].values

        dfr = LastLayerRetraining(
            C=float(dfr_C),
            penalty=dfr_penalty or "l1",
            class_weight="balanced",
            random_state=42,
        )
        dfr.fit(embeddings, task_labels, group_labels)
        predictions = dfr.predict(embeddings)

        n_samples = len(df)
        df_out = df.copy()
        df_out["dfr_prediction"] = predictions

        tmp_dir = tempfile.mkdtemp(prefix="last_layer_retraining_")
        csv_path = os.path.join(tmp_dir, "dfr_predictions.csv")
        df_out.to_csv(csv_path, index=False)

        accuracy = (predictions == task_labels).mean()
        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>Last Layer Retraining (M06 DFR) complete.</strong><br/>
            Samples: {n_samples} | Balanced subset: {dfr._n_balanced} | Groups: {dfr._n_groups}<br/>
            Prediction accuracy: {accuracy:.2%}<br/>
            Kirichenko et al. 2023 – retrain last layer on group-balanced subset.<br/>
            Download the CSV below.
        </div>
        """
        report = {
            "mode": "last_layer_retraining",
            "n_samples": n_samples,
            "n_balanced_subset": dfr._n_balanced,
            "n_groups": dfr._n_groups,
            "group_col": group_col,
            "accuracy": float(accuracy),
        }
        return summary_html, report, csv_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Last Layer Retraining (M06) Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_explanation_regularization_analysis(
    rrr_model_file,
    rrr_images,
    rrr_labels_csv,
    rrr_masks_or_heatmap,
    rrr_head: str,
    rrr_lambda: float,
    rrr_epochs: int,
    rrr_lr: float,
    rrr_batch_size: int,
    rrr_image_size: int,
    rrr_color_mode: str,
    rrr_heatmap_threshold: float,
):
    """Run M05 Explanation Regularization (RRR): fine-tune model with gradient penalty on shortcut regions."""
    try:
        if rrr_model_file is None:
            raise ValueError("Please upload a PyTorch model (.pt/.pth).")
        if not rrr_images:
            raise ValueError("Please upload one or more images.")
        if rrr_labels_csv is None:
            raise ValueError("Please upload a labels CSV with task_label column.")
        if rrr_masks_or_heatmap is None:
            raise ValueError("Please upload shortcut masks (images) or a heatmap file (.npz/.npy).")

        model = _load_torch_model(rrr_model_file)
        image_list = rrr_images if isinstance(rrr_images, list) else [rrr_images]
        n_images = len(image_list)

        tensor_inputs = _preprocess_gradcam_images(
            image_list, int(rrr_image_size), rrr_color_mode or "RGB"
        )
        tensor_inputs = tensor_inputs.detach()
        tensor_inputs.requires_grad_(False)

        df = pd.read_csv(rrr_labels_csv.name if hasattr(rrr_labels_csv, "name") else rrr_labels_csv)
        if "task_label" not in df.columns:
            raise ValueError(
                "Labels CSV must have a 'task_label' column. "
                "One row per image, same order as uploaded images."
            )
        labels = df["task_label"].values
        if len(labels) != n_images:
            raise ValueError(
                f"Labels CSV has {len(labels)} rows but {n_images} images uploaded. "
                "Rows must match image order."
            )

        mask_file = rrr_masks_or_heatmap
        if isinstance(mask_file, list):
            mask_file = mask_file[0] if mask_file else None
        if mask_file is None:
            raise ValueError("Please upload shortcut masks or a heatmap file.")
        mask_path = mask_file.name if hasattr(mask_file, "name") else mask_file
        path_str = str(mask_path).lower()
        _h_in, _w_in = tensor_inputs.shape[2], tensor_inputs.shape[3]

        if path_str.endswith(".npz") or path_str.endswith(".npy"):
            heatmaps, _, _ = _load_heatmap_file(mask_file)
            heatmaps = np.asarray(heatmaps, dtype=np.float32)
            if heatmaps.ndim == 2:
                heatmaps = heatmaps[np.newaxis, ...]
            if heatmaps.shape[0] != n_images:
                raise ValueError(
                    f"Heatmaps count {heatmaps.shape[0]} must match image count {n_images}."
                )
            shortcut_masks = (heatmaps >= float(rrr_heatmap_threshold)).astype(np.float32)
        else:
            mask_list = (
                rrr_masks_or_heatmap
                if isinstance(rrr_masks_or_heatmap, list)
                else [rrr_masks_or_heatmap]
            )
            if len(mask_list) != n_images and len(mask_list) != 1:
                raise ValueError(
                    f"Upload one mask per image ({n_images}) or a single mask, or use a heatmap file."
                )
            mask_arrays = _preprocess_mask_images(mask_list, int(rrr_image_size))
            if mask_arrays.shape[0] == 1 and n_images > 1:
                shortcut_masks = np.tile(mask_arrays, (n_images, 1, 1)).astype(np.float32)
            else:
                shortcut_masks = np.asarray(mask_arrays, dtype=np.float32)

        rrr = ExplanationRegularization(
            lambda_rrr=float(rrr_lambda),
            lr=float(rrr_lr),
            n_epochs=int(rrr_epochs),
            batch_size=int(rrr_batch_size),
            head=_parse_head_identifier(rrr_head, "logits"),
            random_state=42,
        )
        rrr.fit(model, tensor_inputs, labels, shortcut_masks)

        tmp_dir = tempfile.mkdtemp(prefix="explanation_regularization_")
        model_path = os.path.join(tmp_dir, "rrr_finetuned_model.pt")
        torch.save(model, model_path)

        last_h = rrr._history[-1] if rrr._history else {}
        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>Explanation Regularization (M05 RRR) complete.</strong><br/>
            Samples: {n_images} | Epochs: {rrr_epochs}<br/>
            Final CE loss: {last_h.get('ce_loss', 0):.4f} | Penalty: {last_h.get('penalty', 0):.4f}<br/>
            Ross et al. 2017 – Right for Right Reasons. Download the fine-tuned model below.
        </div>
        """
        report = {
            "mode": "explanation_regularization",
            "n_samples": n_images,
            "n_epochs": rrr_epochs,
            "lambda_rrr": float(rrr_lambda),
            "final_ce_loss": last_h.get("ce_loss"),
            "final_penalty": last_h.get("penalty"),
        }
        return summary_html, report, model_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Explanation Regularization (M05) Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_contrastive_debiasing_analysis(
    cnc_use_last_detection: bool,
    detection_state,
    cnc_embeddings_csv,
    cnc_group_col: str,
    cnc_hidden_dim: int,
    cnc_temperature: float,
    cnc_contrastive_weight: float,
    cnc_n_epochs: int,
):
    """Run M07 Contrastive Debiasing (Correct-n-Contrast): produce debiased embeddings."""
    try:
        if cnc_use_last_detection and detection_state:
            path = os.path.join(detection_state, "embeddings_for_mitigation.csv")
            if not os.path.exists(path):
                raise ValueError(
                    "Embeddings file not found. Run detection first and ensure CSV export completed."
                )
        elif cnc_embeddings_csv is not None:
            path = (
                cnc_embeddings_csv.name
                if hasattr(cnc_embeddings_csv, "name")
                else cnc_embeddings_csv
            )
        else:
            raise ValueError(
                "Upload an embeddings CSV or run detection first and check 'Use embeddings from last detection run'."
            )
        df = pd.read_csv(path)
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        if not embedding_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"task_label", "group_label"} | {
                c for c in df.columns if c.startswith("attr_")
            }
            embedding_cols = [c for c in numeric_cols if c not in exclude]
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found. Use embedding_0, embedding_1, ... or numeric columns."
            )

        if "task_label" not in df.columns:
            raise ValueError("task_label column required for Contrastive Debiasing (M07).")

        group_col = (cnc_group_col or "group_label").strip()
        if group_col not in df.columns:
            raise ValueError(
                f"Group/protected column '{group_col}' not found. " f"Available: {list(df.columns)}"
            )

        embeddings = df[embedding_cols].values.astype(np.float32)
        task_labels = df["task_label"].values
        group_labels = df[group_col].values

        cnc = ContrastiveDebiasing(
            hidden_dim=int(cnc_hidden_dim) if cnc_hidden_dim else None,
            temperature=float(cnc_temperature),
            contrastive_weight=float(cnc_contrastive_weight),
            use_task_loss=True,
            n_epochs=int(cnc_n_epochs),
            batch_size=64,
            random_state=42,
        )
        cnc.fit(embeddings, task_labels, group_labels)
        debiased = cnc.transform(embeddings)

        n_samples, h_dim = debiased.shape
        df_out = df.copy()
        for j in range(h_dim):
            df_out[f"debiased_embedding_{j}"] = debiased[:, j]

        tmp_dir = tempfile.mkdtemp(prefix="contrastive_debiasing_")
        csv_path = os.path.join(tmp_dir, "debiased_embeddings.csv")
        df_out.to_csv(csv_path, index=False)
        summary_html = f"""
        <div style="background-color:#e8f4fd;border:1px solid #bee3f8;border-radius:5px;padding:12px;color:#0c4a6e;">
            <strong>Contrastive debiasing (M07 CNC) complete.</strong><br/>
            Samples: {n_samples} | Debiased dims: {h_dim}<br/>
            Zhang et al. 2022 – Correct-n-Contrast: contrastive learning to separate shortcuts.<br/>
            Download the CSV below.
        </div>
        """
        report = {
            "mode": "contrastive_debiasing",
            "n_samples": n_samples,
            "hidden_dim": h_dim,
            "group_col": group_col,
        }
        return summary_html, report, csv_path
    except Exception as exc:
        error_html = f"""
        <div style="background-color:#f8d7da;border:1px solid #f5c6cb;border-radius:5px;padding:12px;color:#721c24;">
            <strong>Contrastive Debiasing (M07) Error:</strong> {str(exc)}
        </div>
        """
        return error_html, None, None


def run_detection(
    data_source,
    custom_csv=None,
    methods=None,
    input_mode="embeddings",
    hf_model_name=None,
    use_real_chexpert_embeddings=False,
    statistical_correction="fdr_bh",
    statistical_alpha=0.05,
    ssa_labeled_fraction=0.2,
    ssa_seed=42,
    eec_n_clusters=4,
    eec_n_epochs=1,
    eec_min_cluster_ratio=0.1,
    eec_entropy_threshold=0.7,
    gce_q=0.7,
    gce_loss_percentile_threshold=90.0,
    gce_max_iter=500,
    causal_effect_spurious_threshold=0.1,
    cav_bundle=None,
    cav_quality_threshold=0.7,
    cav_shortcut_threshold=0.6,
    cav_test_size=0.2,
    cav_min_examples=20,
    freq_top_percent=0.05,
    freq_tpr_threshold=0.5,
    freq_fpr_threshold=0.15,
    freq_probe_evaluation="train",
    condition_name="indicator_count",
):
    """
    Run shortcut detection on selected data source

    Args:
        data_source: "sample" or "custom"
        custom_csv: uploaded CSV file (if data_source is "custom")
        methods: list of detection methods to use
        input_mode: "embeddings" or "raw_data"
        hf_model_name: HuggingFace model name (required if input_mode is "raw_data")
        statistical_correction: Multiple testing correction method ('fdr_bh', 'bonferroni', etc.)
        statistical_alpha: Significance level for statistical tests
        ssa_labeled_fraction: Fraction of samples treated as labeled for SSA splits
        ssa_seed: Random seed for SSA split generation

    Returns:
        results_html: HTML report
        pdf_path: path to PDF report
        csv_dir: directory with CSV exports
    """
    if methods is None or len(methods) == 0:
        methods = [
            "hbac",
            "probe",
            "statistical",
            "geometric",
            "equalized_odds",
            "groupdro",
            "demographic_parity",
            "intersectional",
            "bias_direction_pca",
            "gce",
            "sis",
            "ssa",
        ]

    # causal_effect requires fit_from_loaders (attributes); run separately and merge
    embedding_methods = [m for m in methods if m != "causal_effect"]
    run_causal_effect = "causal_effect" in methods

    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "output" / f"dashboard_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = output_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)

        # Initialize detector with embedding-based methods (exclude causal_effect)
        detector = ShortcutDetector(
            methods=embedding_methods,
            condition_name=condition_name,
            statistical_correction=statistical_correction,
            statistical_alpha=statistical_alpha,
            eec_n_clusters=int(eec_n_clusters),
            eec_n_epochs=int(eec_n_epochs),
            eec_min_cluster_ratio=float(eec_min_cluster_ratio),
            eec_entropy_threshold=float(eec_entropy_threshold),
            gce_q=float(gce_q),
            gce_loss_percentile_threshold=float(gce_loss_percentile_threshold),
            gce_max_iter=int(gce_max_iter),
            freq_top_percent=float(freq_top_percent),
            freq_tpr_threshold=float(freq_tpr_threshold),
            freq_fpr_threshold=float(freq_fpr_threshold),
            freq_probe_evaluation=str(freq_probe_evaluation),
        )

        # Load data based on source and input mode
        if data_source == "Sample Data (CheXpert - Lightweight)":
            # Sample data: optionally use real embeddings (chest_*.npy) or synthetic with metadata
            embeddings, task_labels, group_labels, extra_labels, attributes, metadata_df = (
                load_sample_data(use_real_embeddings=bool(use_real_chexpert_embeddings))
            )
            n, d = embeddings.shape[0], embeddings.shape[1]
            emb_type = "real" if use_real_chexpert_embeddings else "synthetic"
            dataset_name = f"CheXpert Sample ({n} samples, {d}-dim embeddings, {emb_type})"

            splits = None
            if "ssa" in methods:
                splits = _build_ssa_splits(
                    len(task_labels), float(ssa_labeled_fraction), seed=int(ssa_seed)
                )

            # Fit detector with embeddings
            detector.fit(
                embeddings=embeddings,
                labels=task_labels,
                group_labels=group_labels,
                splits=splits,
                extra_labels=extra_labels,
            )
        else:  # Custom CSV
            if custom_csv is None:
                return "❌ Please upload a CSV file for custom data", None, None

            if input_mode == "raw_data":
                # Raw data mode: load text data and compute embeddings
                if not hf_model_name or hf_model_name.strip() == "":
                    return (
                        "❌ Please provide a HuggingFace model name for raw data mode (e.g., 'sentence-transformers/all-MiniLM-L6-v2')",
                        None,
                        None,
                    )

                raw_texts, task_labels, group_labels, extra_labels, attributes, metadata_df = (
                    load_custom_csv(custom_csv, is_raw_data=True)
                )

                # Create HuggingFace embedding source
                embedding_source = HuggingFaceEmbeddingSource(
                    model_name=hf_model_name.strip(), pooling="mean", batch_size=16
                )

                # Cache path for embeddings
                cache_path = str(output_dir / "embeddings_cache.npy")

                splits = None
                if "ssa" in methods:
                    splits = _build_ssa_splits(
                        len(task_labels), float(ssa_labeled_fraction), seed=int(ssa_seed)
                    )

                # Fit detector with raw inputs and embedding source
                detector.fit(
                    embeddings=None,  # Trigger embedding-only mode
                    labels=task_labels,
                    group_labels=group_labels,
                    raw_inputs=raw_texts,
                    embedding_source=embedding_source,
                    embedding_cache_path=cache_path,
                    splits=splits,
                    extra_labels=extra_labels,
                )

                embeddings = detector.embeddings_
                dataset_name = f"Custom Raw Data ({len(embeddings)} samples, {embeddings.shape[1]}-dim embeddings, model: {hf_model_name})"
            else:
                # Embeddings mode: load precomputed embeddings
                embeddings, task_labels, group_labels, extra_labels, attributes, metadata_df = (
                    load_custom_csv(custom_csv, is_raw_data=False)
                )
                dataset_name = f"Custom Embeddings ({len(embeddings)} samples, {embeddings.shape[1]}-dim embeddings)"

                splits = None
                if "ssa" in methods:
                    splits = _build_ssa_splits(
                        len(task_labels), float(ssa_labeled_fraction), seed=int(ssa_seed)
                    )

                # Fit detector with embeddings
                detector.fit(
                    embeddings=embeddings,
                    labels=task_labels,
                    group_labels=group_labels,
                    splits=splits,
                    extra_labels=extra_labels,
                )

        # Run causal_effect via fit_from_loaders if selected
        if run_causal_effect:
            if attributes is None or not attributes:
                return (
                    "❌ Causal Effect requires attribute columns (attr_&lt;name&gt;) in your CSV. "
                    "Add columns like attr_race, attr_gender.",
                    None,
                    None,
                )
            causal_detector = ShortcutDetector(
                methods=["causal_effect"],
                causal_effect_spurious_threshold=float(causal_effect_spurious_threshold),
            )
            loader = {
                "embeddings": detector.embeddings_,
                "labels": detector.labels_,
                "attributes": attributes,
            }
            causal_detector.fit_from_loaders({"causal_effect": loader})
            detector.results_["causal_effect"] = causal_detector.results_["causal_effect"]
            if "causal_effect" in causal_detector.detectors_:
                detector.detectors_["causal_effect"] = causal_detector.detectors_["causal_effect"]
            detector.methods = methods

        # Run CAV via fit_from_loaders if bundle provided (optional add-on)
        if cav_bundle is not None:
            try:
                bundle = _load_cav_bundle(cav_bundle)
                cav_detector = ShortcutDetector(
                    methods=["cav"],
                    cav_quality_threshold=float(cav_quality_threshold),
                    cav_shortcut_threshold=float(cav_shortcut_threshold),
                    cav_test_size=float(cav_test_size),
                    cav_min_examples_per_set=int(cav_min_examples),
                )
                cav_detector.fit_from_loaders({"cav": bundle})
                if cav_detector.results_.get("cav", {}).get("success"):
                    detector.results_["cav"] = cav_detector.results_["cav"]
                    if "cav" in cav_detector.detectors_:
                        detector.detectors_["cav"] = cav_detector.detectors_["cav"]
                    if "cav" not in detector.methods:
                        detector.methods = list(detector.methods) + ["cav"]
            except Exception:
                pass  # CAV is optional; do not fail the run

        # Generate HTML report
        html_path = output_dir / "report.html"
        detector.generate_report(
            output_path=str(html_path),
            format="html",
            include_visualizations=True,
            export_csv=True,
            csv_dir=str(csv_dir),
        )

        # Generate PDF report
        pdf_path = output_dir / "report.pdf"
        try:
            detector.generate_report(
                output_path=str(pdf_path), format="pdf", include_visualizations=True
            )
        except Exception as e:
            print(f"Warning: PDF generation failed: {e}")
            print("PDF export requires weasyprint: uv pip install weasyprint")
            pdf_path = None

        # Read HTML report
        with open(html_path) as f:
            html_content = f.read()

        # Inject contrast CSS into body so it applies when Gradio inlines content (head/style may be dropped)
        _inline_contrast_style = (
            "<style>.shortcut-report,.shortcut-report .container{background:#fff!important;color:#000!important}"
            ".shortcut-report *,.shortcut-report .container *{color:#000!important}"
            ".shortcut-report th{background:#1d4ed8!important;color:#fff!important}"
            ".shortcut-report code{background:unset!important;background-color:rgba(255,255,255,1)!important;color:rgba(21,87,36,1)!important;background-clip:unset;-webkit-background-clip:unset;box-shadow:none!important}"
            ".shortcut-report::selection,.shortcut-report *::selection{background:#b4d5fe;color:#000}</style>"
        )
        if "<body class='shortcut-report'>" in html_content:
            html_content = html_content.replace(
                "<body class='shortcut-report'>",
                "<body class='shortcut-report'>" + _inline_contrast_style,
                1,
            )

        # Create success message with dataset info
        info_html = f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
            <h3 style="color: rgba(21, 87, 36, 1); margin-top: 0;">✅ Detection Complete!</h3>
            <p style="color: rgba(21, 87, 36, 1); margin-bottom: 5px;"><strong style="color: rgba(21, 87, 36, 1);">Dataset:</strong> {dataset_name}</p>
            <p style="color: rgba(21, 87, 36, 1); margin-bottom: 5px;"><strong style="color: rgba(21, 87, 36, 1);">Methods:</strong> {', '.join(methods)}</p>
            <p style="color: rgba(21, 87, 36, 1); margin-bottom: 0;"><strong style="color: rgba(21, 87, 36, 1);">Reports saved to:</strong> {output_dir}</p>
        </div>
        """

        results_html = info_html + html_content

        return (
            results_html,
            str(pdf_path) if pdf_path else None,
            str(csv_dir),
        )

    except Exception as e:
        error_html = f"""
        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 15px;">
            <h3 style="color: #721c24; margin-top: 0;">❌ Error</h3>
            <p style="color: #721c24;">{str(e)}</p>
        </div>
        """
        return error_html, None, None


def run_comparison(
    data_source,
    custom_csv=None,
    comparison_csvs=None,
    comparison_model_ids=None,
    methods=None,
    input_mode="embeddings",
    statistical_correction="fdr_bh",
    statistical_alpha=0.05,
    ssa_labeled_fraction=0.2,
    ssa_seed=42,
):
    """
    Run model comparison: shortcut detection across multiple embedding models on the same data.

    Args:
        data_source: "Sample Data" or "Custom CSV Upload"
        custom_csv: Single CSV (raw data mode) or not used (embeddings mode uses comparison_csvs)
        comparison_csvs: List of uploaded CSV files (embeddings mode) - each CSV = one model
        comparison_model_ids: List of HuggingFace model names (raw data mode)
        methods: Detection methods to run
        input_mode: "embeddings" or "raw_data"
        statistical_correction: Multiple testing correction
        statistical_alpha: Significance level
        ssa_labeled_fraction: For SSA splits
        ssa_seed: For SSA splits

    Returns:
        results_html, comparison_html_path, csv_dir
    """
    if methods is None or len(methods) == 0:
        methods = [
            "hbac",
            "probe",
            "statistical",
            "geometric",
            "equalized_odds",
            "demographic_parity",
            "bias_direction_pca",
        ]

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "output" / f"comparison_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = output_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)

        # Load shared data (labels, group_labels, extra_labels)
        # Model comparison requires Custom CSV (Sample Data has single embedding set)
        if data_source == "Sample Data (CheXpert - Lightweight)":
            # Use sample data: generate 2 synthetic embedding variants for demo comparison
            emb_base, task_labels, group_labels, extra_labels, _, _ = load_sample_data()
            # Create two "models" by adding small noise (simulates different embedding models)
            rng = np.random.default_rng(42)
            emb1 = emb_base + rng.standard_normal(emb_base.shape).astype(np.float32) * 0.05
            emb2 = emb_base + rng.standard_normal(emb_base.shape).astype(np.float32) * 0.08
            model_sources = [
                ("Sample (base)", emb_base),
                ("Sample (variant A)", emb1),
                ("Sample (variant B)", emb2),
            ]
            raw_texts = None
        else:
            if input_mode == "raw_data":
                if custom_csv is None:
                    return (
                        "❌ Please upload a CSV file with text, task_label, group_label for raw data comparison.",
                        None,
                        None,
                    )
                if not comparison_model_ids or len(comparison_model_ids) < 2:
                    return (
                        "❌ Please provide at least 2 HuggingFace model names (e.g., sentence-transformers/all-MiniLM-L6-v2).",
                        None,
                        None,
                    )
                raw_texts, task_labels, group_labels, extra_labels, _, _ = load_custom_csv(
                    custom_csv, is_raw_data=True
                )
                registry = get_embedding_registry()
                model_sources = []
                for mid in comparison_model_ids:
                    if not mid or not str(mid).strip():
                        continue
                    mid = str(mid).strip()
                    model_sources.append((mid, registry.create(mid, pooling="mean", batch_size=16)))
            else:
                if not comparison_csvs or len(comparison_csvs) < 2:
                    return "❌ Please upload at least 2 embedding CSV files to compare.", None, None
                task_labels = None
                group_labels = None
                extra_labels = None
                model_sources = []
                for i, csv_file in enumerate(comparison_csvs):
                    if csv_file is None:
                        continue
                    try:
                        emb, tl, gl, el, _, _ = load_custom_csv(csv_file, is_raw_data=False)
                    except Exception as e:
                        return f"❌ Failed to load CSV {i + 1}: {e}", None, None
                    if task_labels is None:
                        task_labels, group_labels, extra_labels = tl, gl, el
                    else:
                        if len(tl) != len(task_labels):
                            return (
                                f"❌ CSV {i + 1} has {len(tl)} samples, expected {len(task_labels)}.",
                                None,
                                None,
                            )
                    model_id = getattr(csv_file, "name", "unknown")
                    if isinstance(model_id, str) and "/" in model_id:
                        model_id = model_id.split("/")[-1]
                    model_id = str(model_id).replace(".csv", "")[:50] or f"model_{i + 1}"
                    model_sources.append((model_id, emb))
                raw_texts = None

        if len(model_sources) < 2:
            return "❌ At least 2 models required for comparison.", None, None

        splits = None
        if "ssa" in methods and task_labels is not None:
            n_samples = len(task_labels)
            if n_samples > 0:
                splits = _build_ssa_splits(
                    n_samples, float(ssa_labeled_fraction), seed=int(ssa_seed)
                )

        # Run comparison
        runner = ModelComparisonRunner(
            methods=methods,
            seed=42,
            statistical_correction=statistical_correction,
            statistical_alpha=statistical_alpha,
        )
        result = runner.run(
            model_sources=model_sources,
            labels=task_labels,
            group_labels=group_labels,
            raw_inputs=raw_texts,
            extra_labels=extra_labels,
            splits=splits,
        )

        # Build comparison report
        report_builder = ComparisonReportBuilder(result)
        html_path = output_dir / "comparison_report.html"
        report_builder.to_html(str(html_path))

        # Export CSV
        export_comparison_to_csv(result, str(csv_dir))

        with open(html_path) as f:
            html_content = f.read()

        info_html = f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
            <h3 style="color: rgba(21, 87, 36, 1); margin-top: 0;">✅ Model Comparison Complete!</h3>
            <p style="color: rgba(21, 87, 36, 1);"><strong>Models:</strong> {', '.join(result.model_ids)}</p>
            <p style="color: rgba(21, 87, 36, 1);"><strong>Methods:</strong> {', '.join(methods)}</p>
            <p style="color: rgba(21, 87, 36, 1);">Reports saved to: {output_dir}</p>
        </div>
        """
        results_html = info_html + html_content

        return results_html, str(html_path), str(csv_dir)

    except Exception as e:
        error_html = f"""
        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 15px;">
            <h3 style="color: #721c24;">❌ Error</h3>
            <p style="color: #721c24;">{str(e)}</p>
        </div>
        """
        return error_html, None, None


# Create Gradio interface
with gr.Blocks(title="Shortcut Detection Dashboard", css=REPORT_CONTRAST_CSS) as demo:

    detection_state = gr.State(value=None)

    gr.Markdown(
        """
    # 🔍 Shortcut Detection Dashboard

    **Detect shortcuts and biases in machine learning embedding spaces**
    """
    )

    with gr.Tabs():
        with gr.Tab("Detection"):
            with gr.Row():
                # LEFT COLUMN - Results (wider, takes more space)
                with gr.Column(scale=3, min_width=420):
                    gr.Markdown("### 📊 Results")
                    results_output = gr.HTML(
                        label="Detection Results",
                        value="<p style='color: #374151; text-align: center; padding: 50px;'>Click 'Run Detection' to see results</p>",
                    )

                    gr.Markdown("### 📥 Download Reports")

                    with gr.Row():
                        pdf_download = gr.File(label="📄 PDF Report", interactive=False)

                        csv_download = gr.File(label="💾 CSV Exports (ZIP)", interactive=False)

                # RIGHT COLUMN - Controls (narrower, fixed width)
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown(
                        """
                    This dashboard detects shortcuts in ML embeddings using:
                    - 📊 Sample data from CheXpert medical imaging dataset
                    - 📁 Custom embeddings upload (CSV format)
                    - 📝 Raw text data with HuggingFace embedding models
                    - 🧬 HBAC, Probe, Statistical, Geometric, Fairness, and Robustness methods
                    """
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 1️⃣ Select Input Mode")

                    input_mode = gr.Radio(
                        choices=["Use Embeddings", "Use Raw Data"],
                        value="Use Embeddings",
                        label="Input Mode",
                        info="Choose whether to upload precomputed embeddings or raw data (text) that will be converted to embeddings",
                    )

                    hf_model_name = gr.Textbox(
                        label="HuggingFace Embedding Model",
                        placeholder="sentence-transformers/all-MiniLM-L6-v2",
                        value="",
                        visible=False,
                        info="Enter a HuggingFace model name to generate embeddings from raw text data",
                    )

                    gr.Markdown("### 2️⃣ Select Data Source")

                    data_source = gr.Radio(
                        choices=["Sample Data (CheXpert - Lightweight)", "Custom CSV Upload"],
                        value="Sample Data (CheXpert - Lightweight)",
                        label="Data Source",
                        info="Choose between sample data or upload your own CSV",
                    )

                    custom_csv = gr.File(label="Upload CSV", file_types=[".csv"], visible=False)

                    use_real_chexpert_embeddings = gr.Checkbox(
                        label="Use real CheXpert embeddings (data/chest_*.npy) when available",
                        value=False,
                        visible=True,
                        info="When checked and data/chest_embeddings.npy exist, uses pre-computed embeddings. Otherwise uses synthetic embeddings with real CheXpert metadata and demographics.",
                    )

                    with gr.Accordion("📖 Data Format Info", open=False):
                        gr.Markdown(
                            """
                        **Sample Data:** CheXpert medical imaging dataset
                        - Optionally use real embeddings (data/chest_*.npy) or synthetic with real metadata
                        - With metadata: 2000 samples, race + gender from CHEXPERT DEMO.xlsx
                        - Race groups: Asian, Black/African American, White, Other

                        **Custom CSV Format (Embeddings Mode):**
                        - Columns: `embedding_0`, `embedding_1`, ..., `task_label`, `group_label`
                        - Optional: `spurious_label` (for SSA extra supervision)
                        - Optional: `attr_<name>` (for Causal Effect, Intersectional, and **Multi-Attribute** analysis: e.g., attr_race, attr_gender, attr_age)
                        - With multiple `attr_*` columns (or group_label + attr_*), fairness methods run per attribute
                        - `task_label`: Binary classification labels (0 or 1)
                        - `group_label`: Demographic/group labels (e.g., race, gender)
                        - Validation checks: non-empty file, comma-separated format, numeric embedding columns, no missing values in required columns

                        **Custom CSV Format (Raw Data Mode):**
                        - Columns: `text`, `task_label`, `group_label`
                        - Optional: `spurious_label` (for SSA extra supervision)
                        - Optional: `attr_<name>` (for Causal Effect, Intersectional, and **Multi-Attribute**: e.g., attr_race, attr_gender)
                        - `text`: Raw text data (will be converted to embeddings using the specified HuggingFace model)
                        - `task_label`: Binary classification labels (0 or 1)
                        - `group_label`: Demographic/group labels (e.g., race, gender)
                        - Validation checks: non-empty file, comma-separated format, required columns present, no empty `text`/`task_label`/`group_label` values

                        **If the file format is invalid, the dashboard shows a red error card with the exact issue.**
                        """
                        )

                    gr.Markdown("### 3️⃣ Select Detection Methods")

                    methods = gr.CheckboxGroup(
                        choices=[
                            "hbac",
                            "probe",
                            "statistical",
                            "geometric",
                            "equalized_odds",
                            "groupdro",
                            "demographic_parity",
                            "intersectional",
                            "bias_direction_pca",
                            "early_epoch_clustering",
                            "gce",
                            "sis",
                            "ssa",
                            "causal_effect",
                            "frequency",
                            "generative_cvae",
                        ],
                        value=[
                            "hbac",
                            "probe",
                            "statistical",
                            "geometric",
                            "equalized_odds",
                            "groupdro",
                            "demographic_parity",
                            "intersectional",
                            "bias_direction_pca",
                            "early_epoch_clustering",
                        ],
                        label="Methods",
                        info="Select which methods to run",
                    )

                    with gr.Accordion("ℹ️ Method Descriptions", open=False):
                        gr.Markdown(
                            """
                        - **HBAC:** Hierarchical Bias-Aware Clustering
                        - **Probe:** Classifier-based information leakage test
                        - **Statistical:** Feature-wise group difference tests
                        - **Geometric:** Bias direction & prototype subspace analysis
                        - **Equalized Odds:** TPR/FPR gap analysis across demographic groups
                        - **GroupDRO:** Worst-group robustness analysis
                        - **Demographic Parity:** Positive rate gaps across groups
                        - **Intersectional:** Fairness across intersections of demographics (e.g., Black + Female); requires race + gender in data
                        - **Multi-Attribute:** When your CSV has group_label + attr_race, attr_gender, etc., fairness methods run per attribute
                        - **Bias Direction PCA:** Prototype PCA bias direction gaps
                        - **Early-Epoch Clustering (SPARE):** Cluster early-epoch representations to surface shortcut bias signals
                        - **GCE:** Generalized Cross Entropy bias detector — flags high-loss samples as minority/bias-conflicting
                        - **SIS:** Sufficient Input Subsets — finds minimal embedding dimensions for prediction (Carter et al. 2019)
                        - **SSA:** Spectral Shift Analysis (semi-supervised; placeholder)
                        - **Causal Effect:** Identifies spurious attributes via causal effect estimation; requires attr_* columns in CSV
                        - **Frequency:** Embedding-space frequency detector -- flags classes whose signal concentrates in few dimensions
                        - **Generative CVAE:** Conditional VAE counterfactual detector -- flips spurious attribute in embedding space and measures probe prediction shift
                        """
                        )

                    gr.Markdown("### 4️⃣ Overall Assessment Condition")
                    condition_name = gr.Dropdown(
                        choices=[
                            "indicator_count",
                            "majority_vote",
                            "weighted_risk",
                            "multi_attribute",
                            "meta_classifier",
                        ],
                        value="indicator_count",
                        label="Risk Aggregation Condition",
                        info="How method-level results are aggregated into the final risk summary",
                    )
                    with gr.Accordion("ℹ️ Condition Descriptions", open=False):
                        gr.Markdown(
                            """
                        - **indicator_count** (default): Counts total risk indicators across methods. 2+ = HIGH, 1 = MODERATE, 0 = LOW.
                        - **majority_vote**: Counts methods with at least one indicator as votes. Configurable threshold.
                        - **weighted_risk**: Weights each detector by evidence strength (probe accuracy above chance, statistical significance ratio, HBAC confidence, geometric effect size).
                        - **multi_attribute**: Cross-references risk across sensitive attributes. Escalates when multiple attributes independently flag shortcuts.
                        - **meta_classifier**: Heuristic ensemble of detector features, or trained sklearn model when provided.
                        """
                        )

                    with gr.Accordion("🧪 SSA Settings (Optional)", open=False):
                        gr.Markdown(
                            """
                        SSA requires labeled/unlabeled splits. The dashboard will automatically
                        create splits using the fraction below. **SSA is currently a placeholder**
                        in the library, so results are informational only.
                        """
                        )
                        ssa_labeled_fraction = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.2,
                            step=0.05,
                            label="Labeled Fraction",
                            info="Portion of samples treated as labeled for SSA",
                        )
                        ssa_seed = gr.Number(value=42, precision=0, label="SSA Split Seed")

                    with gr.Accordion("📊 Frequency Settings (Optional)", open=False):
                        gr.Markdown(
                            """
                        Configure the embedding frequency shortcut detector.
                        Detects whether class signals concentrate in few embedding dimensions.
                        """
                        )
                        freq_top_percent = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            value=0.05,
                            step=0.01,
                            label="Top Percent",
                            info="Fraction of top embedding dimensions to examine",
                        )
                        freq_tpr_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="TPR Threshold",
                            info="Per-class true-positive rate threshold for shortcut flagging",
                        )
                        freq_fpr_threshold = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            value=0.15,
                            step=0.01,
                            label="FPR Threshold",
                            info="Per-class false-positive rate threshold for shortcut flagging",
                        )
                        freq_probe_evaluation = gr.Dropdown(
                            choices=["train", "holdout"],
                            value="train",
                            label="Probe Evaluation Mode",
                            info="Evaluate probe on training data or holdout split",
                        )

                    with gr.Accordion("🌱 Early-Epoch Clustering (SPARE) Settings", open=False):
                        gr.Markdown(
                            """
                        Configure the SPARE detector that clusters early-epoch representations to flag shortcuts.
                        """
                        )
                        eec_n_clusters = gr.Slider(
                            minimum=2,
                            maximum=12,
                            value=4,
                            step=1,
                            label="Number of Clusters",
                            info="How many clusters to create on the early-epoch representations",
                        )
                        eec_n_epochs = gr.Number(
                            value=1,
                            precision=0,
                            label="Epoch Count",
                            info="Number of early epochs represented by the provided embeddings (for reporting only)",
                        )
                        eec_min_cluster_ratio = gr.Slider(
                            minimum=0.01,
                            maximum=0.30,
                            value=0.10,
                            step=0.01,
                            label="Minimum Cluster Ratio",
                            info="Threshold for flagging overly small clusters",
                        )
                        eec_entropy_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Entropy Threshold",
                            info="Entropy cutoff used when assessing cluster balance",
                        )

                    with gr.Accordion("📊 GCE (Generalized Cross Entropy) Settings", open=False):
                        gr.Markdown(
                            """
                        Configure the GCE bias detector. High-loss samples under GCE are flagged as minority/bias-conflicting.
                        """
                        )
                        gce_q = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="GCE q",
                            info="GCE parameter (q≈0.7 downweights easy samples, emphasizes hard/minority)",
                        )
                        gce_loss_percentile_threshold = gr.Slider(
                            minimum=70,
                            maximum=99,
                            value=90,
                            step=1,
                            label="Loss Percentile Threshold",
                            info="Samples with loss ≥ this percentile are flagged as minority/bias-conflicting",
                        )
                        gce_max_iter = gr.Number(
                            value=500,
                            precision=0,
                            label="Max Iterations",
                            info="Maximum iterations for training the GCE classifier",
                        )

                    with gr.Accordion("📐 Causal Effect Settings", open=False):
                        gr.Markdown(
                            """
                        Causal Effect detects spurious attributes by estimating causal effect on the task label.
                        Attributes with near-zero effect are flagged as shortcuts. Requires `attr_*` columns in your CSV.
                        """
                        )
                        causal_effect_spurious_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.01,
                            label="Spurious Threshold",
                            info="Attributes with |effect| < threshold are flagged as spurious",
                        )

                    with gr.Accordion("🧪 Include CAV (Concept Bundle)", open=False):
                        gr.Markdown(
                            """
                        Optionally include CAV concept testing in the Detection report.
                        Upload a precomputed .npz/.npy bundle with concept_sets and random_set.
                        """
                        )
                        detection_cav_bundle = gr.File(
                            label="CAV Bundle (.npz/.npy)",
                            file_types=[".npz", ".npy"],
                        )
                        detection_cav_quality_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.7,
                            step=0.01,
                            label="Concept Quality Threshold (AUC)",
                        )
                        detection_cav_shortcut_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.6,
                            step=0.01,
                            label="Shortcut Threshold (TCAV)",
                        )
                        detection_cav_test_size = gr.Slider(
                            minimum=0.1,
                            maximum=0.4,
                            value=0.2,
                            step=0.05,
                            label="Train/Test Split (test_size)",
                        )
                        detection_cav_min_examples = gr.Number(
                            label="Min Examples per Set",
                            value=20,
                            precision=0,
                        )

                    gr.Markdown("### 4️⃣ Statistical Correction (Optional)")

                    with gr.Accordion("📊 Multiple Testing Correction", open=False):
                        gr.Markdown(
                            """
                        When testing across many embedding dimensions, multiple testing correction
                        prevents inflated false positive rates. **FDR (Benjamini-Hochberg)** is
                        recommended for most use cases.
                        """
                        )

                        statistical_correction = gr.Dropdown(
                            choices=["fdr_bh", "bonferroni", "holm", "fdr_by"],
                            value="fdr_bh",
                            label="Correction Method",
                            info="FDR (fdr_bh) is recommended; Bonferroni is most conservative",
                        )

                        statistical_alpha = gr.Slider(
                            minimum=0.01,
                            maximum=0.10,
                            value=0.05,
                            step=0.01,
                            label="Significance Level (α)",
                            info="Threshold for statistical significance",
                        )

                    run_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")
                    gr.Markdown(
                        "**Tip:** Advanced visualizations (GradCAM/SpRAy/GT Mask Overlap) are available in the **Advanced Analysis** tab."
                    )

        with gr.Tab("Model Comparison"):
            gr.Markdown(
                """
            Compare shortcut detection across **multiple embedding models** side-by-side.
            For benchmarking papers: same data, different models.
            """
            )
            with gr.Row():
                with gr.Column(scale=3, min_width=420):
                    comparison_output = gr.HTML(
                        value="<p style='color: #374151; text-align: center; padding: 50px;'>Configure models and click 'Run Comparison' to see results</p>"
                    )
                    with gr.Row():
                        comparison_html_download = gr.File(
                            label="📄 Comparison Report (HTML)", interactive=False
                        )
                        comparison_csv_download = gr.File(
                            label="💾 Comparison CSV (ZIP)", interactive=False
                        )
                with gr.Column(scale=2, min_width=320):
                    comp_data_source = gr.Radio(
                        choices=["Sample Data (CheXpert - Lightweight)", "Custom CSV Upload"],
                        value="Sample Data (CheXpert - Lightweight)",
                        label="Data Source",
                    )
                    comp_custom_csv = gr.File(
                        label="Upload CSV (text, task_label, group_label)",
                        file_types=[".csv"],
                        visible=False,
                    )
                    comp_input_mode = gr.Radio(
                        choices=["Use Embeddings", "Use Raw Data"],
                        value="Use Embeddings",
                        label="Input Mode",
                    )
                    comp_comparison_csvs = gr.File(
                        label="Upload 2+ Embedding CSVs (each = one model)",
                        file_types=[".csv"],
                        file_count="multiple",
                        visible=False,
                    )
                    comp_model_ids = gr.Textbox(
                        label="HuggingFace Models (one per line)",
                        placeholder="sentence-transformers/all-MiniLM-L6-v2\nsentence-transformers/all-mpnet-base-v2",
                        value="",
                        visible=False,
                        lines=4,
                    )
                    comp_methods = gr.CheckboxGroup(
                        choices=[
                            "hbac",
                            "probe",
                            "statistical",
                            "geometric",
                            "equalized_odds",
                            "demographic_parity",
                            "bias_direction_pca",
                            "sis",
                        ],
                        value=[
                            "hbac",
                            "probe",
                            "statistical",
                            "geometric",
                            "equalized_odds",
                            "demographic_parity",
                        ],
                        label="Methods",
                    )
                    comp_run_btn = gr.Button("🚀 Run Comparison", variant="primary", size="lg")

            def _toggle_comp_inputs(data_source, input_mode):
                is_custom = data_source == "Custom CSV Upload"
                embeddings_vis = is_custom and input_mode == "Use Embeddings"
                raw_vis = is_custom and input_mode == "Use Raw Data"
                return (
                    gr.update(visible=raw_vis),
                    gr.update(visible=embeddings_vis),
                    gr.update(visible=raw_vis),
                )

            comp_data_source.change(
                fn=_toggle_comp_inputs,
                inputs=[comp_data_source, comp_input_mode],
                outputs=[comp_custom_csv, comp_comparison_csvs, comp_model_ids],
            )
            comp_input_mode.change(
                fn=_toggle_comp_inputs,
                inputs=[comp_data_source, comp_input_mode],
                outputs=[comp_custom_csv, comp_comparison_csvs, comp_model_ids],
            )

            def _run_comparison_and_downloads(
                data_source, custom_csv, comparison_csvs, comparison_model_ids, input_mode, methods
            ):
                model_ids = None
                if comparison_model_ids:
                    model_ids = [
                        x.strip()
                        for x in str(comparison_model_ids).strip().splitlines()
                        if x.strip()
                    ]
                results_html, html_path, csv_dir = run_comparison(
                    data_source=data_source,
                    custom_csv=custom_csv,
                    comparison_csvs=comparison_csvs,
                    comparison_model_ids=model_ids,
                    methods=methods,
                    input_mode="raw_data" if input_mode == "Use Raw Data" else "embeddings",
                )
                html_file = html_path if (html_path and os.path.exists(html_path)) else None
                csv_zip = None
                if csv_dir and os.path.exists(csv_dir):
                    import shutil

                    csv_zip_path = csv_dir.rstrip("/") + ".zip"
                    shutil.make_archive(csv_dir, "zip", csv_dir)
                    csv_zip = csv_zip_path if os.path.exists(csv_zip_path) else None
                return results_html, html_file, csv_zip

            comp_run_btn.click(
                fn=_run_comparison_and_downloads,
                inputs=[
                    comp_data_source,
                    comp_custom_csv,
                    comp_comparison_csvs,
                    comp_model_ids,
                    comp_input_mode,
                    comp_methods,
                ],
                outputs=[comparison_output, comparison_html_download, comparison_csv_download],
            )

        with gr.Tab("Advanced Analysis"):
            gr.Markdown(
                """
            Optional visualization tools for deeper analysis. These do not need a successful detection run.
            """
            )

            with gr.Row():
                with gr.Column(scale=2, min_width=400):
                    with gr.Accordion("🧠 GradCAM Attention Overlap (Optional)", open=False):
                        gr.Markdown(
                            """
                        Compare disease vs. attribute prediction attention maps using the new GradCAM utility.
                        Upload a PyTorch model that outputs both predictions and specify the target layer to visualize.
                        """
                        )
                        gradcam_model = gr.File(label="Upload PyTorch Model (.pt/.pth)")
                        gradcam_image = gr.File(label="Upload Image")
                        gradcam_target_layer = gr.Textbox(
                            label="Target Layer (module path)",
                            placeholder="e.g., backbone.layer4",
                            info="Layer used to compute GradCAM activations",
                        )
                        gradcam_disease_head = gr.Textbox(
                            label="Disease Head Key or Index",
                            value="disease",
                            info="Name or index of the disease prediction logits",
                        )
                        gradcam_attribute_head = gr.Textbox(
                            label="Attribute Head Key or Index",
                            value="attribute",
                            info="Name or index of the attribute prediction logits",
                        )
                        gradcam_disease_target = gr.Number(
                            label="Disease Target Class (optional)", value=None, precision=0
                        )
                        gradcam_attribute_target = gr.Number(
                            label="Attribute Target Class (optional)", value=None, precision=0
                        )
                        gradcam_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Binarization Threshold (for Dice/IoU)",
                        )
                        gradcam_image_size = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=224,
                            step=32,
                            label="Image Resize (pixels)",
                        )
                        gradcam_color_mode = gr.Radio(
                            ["Grayscale", "RGB"], value="Grayscale", label="Input Color Mode"
                        )
                        gradcam_btn = gr.Button("Generate GradCAM Heatmaps", variant="secondary")

                    with gr.Accordion("🩺 GradCAM GT Mask Overlap (Optional)", open=False):
                        gr.Markdown(
                            """
                        Measure how much model attention overlaps with ground-truth segmentation masks.
                        Upload matching image/mask pairs and a PyTorch model.
                        """
                        )
                        gradcam_mask_model = gr.File(label="Upload PyTorch Model (.pt/.pth)")
                        gradcam_mask_images = gr.File(
                            label="Upload Images",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        gradcam_mask_masks = gr.File(
                            label="Upload Masks",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        gradcam_mask_target_layer = gr.Textbox(
                            label="Target Layer (module path)",
                            placeholder="e.g., backbone.layer4",
                            info="Layer used to compute GradCAM activations",
                        )
                        gradcam_mask_head = gr.Textbox(
                            label="Head Key or Index",
                            value="logits",
                            info="Name or index of the prediction logits",
                        )
                        gradcam_mask_target_index = gr.Number(
                            label="Target Class (optional)", value=None, precision=0
                        )
                        gradcam_mask_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Attention Threshold (for Dice/IoU)",
                        )
                        gradcam_mask_mask_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Mask Threshold (binarize GT masks)",
                        )
                        gradcam_mask_image_size = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=224,
                            step=32,
                            label="Image Resize (pixels)",
                        )
                        gradcam_mask_color_mode = gr.Radio(
                            ["Grayscale", "RGB"], value="Grayscale", label="Input Color Mode"
                        )
                        gradcam_mask_batch_size = gr.Number(
                            label="Batch Size", value=8, precision=0
                        )
                        gradcam_mask_btn = gr.Button("Run Mask Overlap", variant="secondary")

                    with gr.Accordion("🧠 SpRAy Heatmap Clustering (Optional)", open=False):
                        gr.Markdown(
                            """
                        Run SpRAy (spectral clustering of explanation heatmaps) to surface suspicious
                        attention patterns indicative of Clever Hans behavior.
                        """
                        )
                        spray_input_mode = gr.Radio(
                            choices=["Upload Heatmaps (.npz/.npy)", "Generate from Model + Images"],
                            value="Upload Heatmaps (.npz/.npy)",
                            label="SpRAy Input Mode",
                        )
                        spray_heatmaps = gr.File(
                            label="Heatmaps File (.npz/.npy)",
                            file_types=[".npz", ".npy"],
                            visible=True,
                        )
                        spray_model = gr.File(
                            label="Upload PyTorch Model (.pt/.pth)", visible=False
                        )
                        spray_images = gr.File(
                            label="Upload Images",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                            visible=False,
                        )
                        spray_target_layer = gr.Textbox(
                            label="Target Layer (module path)",
                            placeholder="e.g., backbone.layer4",
                            visible=False,
                        )
                        spray_head = gr.Textbox(
                            label="Head Key or Index",
                            value="logits",
                            visible=False,
                        )
                        spray_target_index = gr.Number(
                            label="Target Class (optional)",
                            value=None,
                            precision=0,
                            visible=False,
                        )
                        spray_image_size = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=224,
                            step=32,
                            label="Image Resize (pixels)",
                            visible=False,
                        )
                        spray_color_mode = gr.Radio(
                            ["Grayscale", "RGB"],
                            value="Grayscale",
                            label="Input Color Mode",
                            visible=False,
                        )
                        spray_affinity = gr.Dropdown(
                            choices=["cosine", "rbf", "nearest_neighbors"],
                            value="cosine",
                            label="Affinity",
                        )
                        spray_cluster_selection = gr.Dropdown(
                            choices=["auto", "eigengap", "fixed"],
                            value="auto",
                            label="Cluster Selection",
                        )
                        spray_n_clusters = gr.Number(
                            label="Number of Clusters (optional)",
                            value=None,
                            precision=0,
                        )
                        spray_min_clusters = gr.Slider(
                            minimum=2,
                            maximum=6,
                            value=2,
                            step=1,
                            label="Min Clusters (eigengap)",
                        )
                        spray_max_clusters = gr.Slider(
                            minimum=3,
                            maximum=12,
                            value=6,
                            step=1,
                            label="Max Clusters (eigengap)",
                        )
                        spray_downsample = gr.Number(
                            label="Downsample Size (optional)",
                            value=32,
                            precision=0,
                        )
                        spray_btn = gr.Button("Run SpRAy Clustering", variant="secondary")

                    with gr.Accordion("🧪 CAV Concept Testing (Optional)", open=False):
                        gr.Markdown(
                            """
                            Run Concept Activation Vector testing from a precomputed concept bundle.
                            Upload `.npz` with `concept_<name>` arrays and `random_set`, plus optional
                            `target_activations` and `target_directional_derivatives`.
                            """
                        )
                        cav_bundle = gr.File(
                            label="CAV Bundle (.npz/.npy)",
                            file_types=[".npz", ".npy"],
                        )
                        cav_quality_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.7,
                            step=0.01,
                            label="Concept Quality Threshold (AUC)",
                        )
                        cav_shortcut_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.6,
                            step=0.01,
                            label="Shortcut Threshold (TCAV)",
                        )
                        cav_test_size = gr.Slider(
                            minimum=0.1,
                            maximum=0.4,
                            value=0.2,
                            step=0.05,
                            label="Train/Test Split (test_size)",
                        )
                        cav_min_examples = gr.Number(
                            label="Min Examples per Set",
                            value=20,
                            precision=0,
                        )
                        cav_btn = gr.Button("Run CAV Analysis", variant="secondary")

                    with gr.Accordion("🖼️ VAE Image Shortcut Detection (Optional)", open=False):
                        gr.Markdown(
                            """
                        VAE-based shortcut detection on raw images. Upload images and labels CSV.
                        Images are resized to the specified size. Labels CSV must have `task_label` column (length = number of images).
                        """
                        )
                        vae_images = gr.File(
                            label="Upload Images",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        vae_labels_csv = gr.File(
                            label="Labels CSV (task_label column, one row per image)",
                            file_types=[".csv"],
                        )
                        vae_img_size = gr.Radio(
                            choices=[32, 64, 128],
                            value=64,
                            label="Image Size (height/width)",
                        )
                        vae_channels = gr.Radio(
                            choices=[1, 3],
                            value=3,
                            label="Channels",
                        )
                        vae_num_classes = gr.Number(
                            label="Number of Classes",
                            value=2,
                            precision=0,
                        )
                        vae_latent_dim = gr.Number(
                            label="Latent Dimension",
                            value=10,
                            precision=0,
                        )
                        vae_epochs = gr.Number(
                            label="VAE Epochs",
                            value=20,
                            precision=0,
                        )
                        vae_classifier_epochs = gr.Number(
                            label="Classifier Epochs",
                            value=10,
                            precision=0,
                        )
                        vae_btn = gr.Button("Run VAE Analysis", variant="secondary")

                    with gr.Accordion("🎭 Shortcut Feature Masking (M01)", open=False):
                        gr.Markdown(
                            """
                        **Data mitigation (Teso & Kersting 2019):** Mask or inpaint detected shortcut regions to produce augmented data for retraining.
                        Choose **Image** (images + masks/heatmaps) or **Embedding** (CSV + dimension indices).
                        """
                        )
                        masking_mode = gr.Radio(
                            choices=["Image", "Embedding"],
                            value="Image",
                            label="Mode",
                        )
                        with gr.Row():
                            mask_images = gr.File(
                                label="Images (Image mode)",
                                file_types=[".png", ".jpg", ".jpeg"],
                                file_count="multiple",
                            )
                            mask_masks_or_heatmap_file = gr.File(
                                label="Masks or heatmap file (.npz/.npy)",
                                file_types=[".png", ".jpg", ".jpeg", ".npz", ".npy"],
                            )
                        mask_strategy_img = gr.Radio(
                            choices=["zero", "randomize", "inpaint"],
                            value="randomize",
                            label="Image strategy",
                        )
                        mask_heatmap_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Heatmap threshold (binarize)",
                        )
                        mask_use_last_detection = gr.Checkbox(
                            label="Use embeddings from last detection run (Embedding mode only)",
                            value=False,
                            info="Run detection first; embeddings will be loaded from the CSV export.",
                        )
                        mask_embeddings_csv = gr.File(
                            label="Embeddings CSV (Embedding mode)",
                            file_types=[".csv"],
                        )
                        mask_dim_indices_text = gr.Textbox(
                            label="Dimension indices to mask (e.g. 0,3,7)",
                            placeholder="0, 3, 7",
                        )
                        mask_strategy_emb = gr.Radio(
                            choices=["zero", "randomize"],
                            value="zero",
                            label="Embedding strategy",
                        )
                        mask_augment_fraction = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Augment fraction",
                        )
                        mask_btn = gr.Button("Run Shortcut Masking", variant="secondary")

                    with gr.Accordion("🔄 Background Randomization (M02)", open=False):
                        gr.Markdown(
                            """
                        **Data mitigation (Kwon et al. 2023):** Swap foregrounds with random backgrounds to reduce background shortcuts.
                        Upload images and foreground masks (or heatmap file).
                        """
                        )
                        bg_images = gr.File(
                            label="Images",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        bg_masks_or_heatmap_file = gr.File(
                            label="Foreground masks or heatmap file (.npz/.npy)",
                            file_types=[".png", ".jpg", ".jpeg", ".npz", ".npy"],
                        )
                        bg_heatmap_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Heatmap threshold (binarize)",
                        )
                        bg_augment_fraction = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Augment fraction",
                        )
                        bg_btn = gr.Button("Run Background Randomization", variant="secondary")

                    with gr.Accordion("🛡️ Adversarial Debiasing (M04)", open=False):
                        gr.Markdown(
                            """
                        **Model mitigation (Zhang et al. 2018):** Adversarial training to remove demographic encoding from embeddings.
                        Upload embeddings CSV with a group/protected column (e.g. group_label), or use embeddings from last detection run.
                        """
                        )
                        debias_use_last_detection = gr.Checkbox(
                            label="Use embeddings from last detection run",
                            value=False,
                            info="Run detection first; embeddings will be loaded from the CSV export.",
                        )
                        debias_embeddings_csv = gr.File(
                            label="Embeddings CSV",
                            file_types=[".csv"],
                        )
                        debias_group_col = gr.Textbox(
                            label="Group/protected column name",
                            value="group_label",
                            placeholder="group_label",
                        )
                        debias_hidden_dim = gr.Number(
                            label="Hidden dimension (bottleneck)",
                            value=32,
                            precision=0,
                            minimum=4,
                            maximum=512,
                        )
                        debias_adversary_weight = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.5,
                            step=0.1,
                            label="Adversary weight",
                        )
                        debias_n_epochs = gr.Number(
                            label="Training epochs",
                            value=50,
                            precision=0,
                            minimum=5,
                            maximum=500,
                        )
                        debias_btn = gr.Button("Run Adversarial Debiasing", variant="secondary")

                    with gr.Accordion("🔄 Last Layer Retraining (M06 DFR)", open=False):
                        gr.Markdown(
                            """
                        **Model mitigation (Kirichenko et al. 2023):** Retrain last layer on group-balanced subset.
                        Upload embeddings CSV with task_label and group_label, or use embeddings from last detection run.
                        """
                        )
                        dfr_use_last_detection = gr.Checkbox(
                            label="Use embeddings from last detection run",
                            value=False,
                            info="Run detection first; embeddings will be loaded from the CSV export.",
                        )
                        dfr_embeddings_csv = gr.File(
                            label="Embeddings CSV",
                            file_types=[".csv"],
                        )
                        dfr_group_col = gr.Textbox(
                            label="Group/protected column name",
                            value="group_label",
                            placeholder="group_label",
                        )
                        dfr_C = gr.Slider(
                            minimum=0.01,
                            maximum=10.0,
                            value=1.0,
                            step=0.1,
                            label="Regularization C (inverse strength)",
                        )
                        dfr_penalty = gr.Dropdown(
                            choices=["l1", "l2"],
                            value="l1",
                            label="Penalty",
                        )
                        dfr_btn = gr.Button("Run Last Layer Retraining", variant="secondary")

                    with gr.Accordion("Contrastive Debiasing (M07 CNC)", open=False):
                        gr.Markdown(
                            """
                        **Model mitigation (Zhang et al. 2022):** Correct-n-Contrast – contrastive learning to separate shortcuts.
                        Upload embeddings CSV with task_label and group_label, or use embeddings from last detection run.
                        Requires at least 2 groups and 2 task classes.
                        """
                        )
                        cnc_use_last_detection = gr.Checkbox(
                            label="Use embeddings from last detection run",
                            value=False,
                            info="Run detection first; embeddings will be loaded from the CSV export.",
                        )
                        cnc_embeddings_csv = gr.File(
                            label="Embeddings CSV",
                            file_types=[".csv"],
                        )
                        cnc_group_col = gr.Textbox(
                            label="Group/protected column name",
                            value="group_label",
                            placeholder="group_label",
                        )
                        cnc_hidden_dim = gr.Number(
                            label="Hidden dimension (bottleneck)",
                            value=32,
                            precision=0,
                            minimum=4,
                            maximum=512,
                        )
                        cnc_temperature = gr.Slider(
                            minimum=0.01,
                            maximum=0.5,
                            value=0.05,
                            step=0.01,
                            label="Temperature (contrastive)",
                        )
                        cnc_contrastive_weight = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Contrastive weight (vs task CE)",
                        )
                        cnc_n_epochs = gr.Number(
                            label="Training epochs",
                            value=50,
                            precision=0,
                            minimum=5,
                            maximum=500,
                        )
                        cnc_btn = gr.Button("Run Contrastive Debiasing", variant="secondary")

                    with gr.Accordion("Explanation Regularization (M05 RRR)", open=False):
                        gr.Markdown(
                            """
                        **Model mitigation (Ross et al. 2017):** Penalize input gradients on shortcut regions.
                        Upload model, images, labels CSV, and shortcut masks (or heatmap file).
                        """
                        )
                        rrr_model_file = gr.File(
                            label="PyTorch Model (.pt/.pth)",
                            file_types=[".pt", ".pth"],
                        )
                        rrr_images = gr.File(
                            label="Images",
                            file_types=[".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        rrr_labels_csv = gr.File(
                            label="Labels CSV (task_label column)",
                            file_types=[".csv"],
                        )
                        rrr_masks_or_heatmap = gr.File(
                            label="Shortcut Masks (images or .npz/.npy heatmap)",
                            file_types=[".png", ".jpg", ".jpeg", ".npz", ".npy"],
                            file_count="multiple",
                        )
                        rrr_head = gr.Textbox(
                            label="Head (logits key or index)",
                            value="logits",
                            placeholder="logits",
                        )
                        rrr_lambda = gr.Slider(
                            minimum=0.01,
                            maximum=10.0,
                            value=1.0,
                            step=0.1,
                            label="Penalty weight (lambda)",
                        )
                        rrr_epochs = gr.Number(
                            label="Epochs",
                            value=10,
                            precision=0,
                            minimum=1,
                            maximum=200,
                        )
                        rrr_lr = gr.Number(
                            label="Learning rate",
                            value=1e-4,
                            minimum=1e-6,
                            maximum=1e-1,
                        )
                        rrr_batch_size = gr.Number(
                            label="Batch size",
                            value=8,
                            precision=0,
                            minimum=1,
                            maximum=64,
                        )
                        rrr_image_size = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=224,
                            step=32,
                            label="Image size (pixels)",
                        )
                        rrr_color_mode = gr.Radio(
                            ["Grayscale", "RGB"],
                            value="RGB",
                            label="Color mode",
                        )
                        rrr_heatmap_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Heatmap threshold (binarize)",
                        )
                        rrr_btn = gr.Button("Run Explanation Regularization", variant="secondary")

                with gr.Column(scale=3):
                    gr.Markdown("### 🔥 GradCAM Attention Maps")
                    gradcam_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload a model and image in the left panel to generate overlays.</p>"
                    )
                    with gr.Row():
                        gradcam_disease_image = gr.Image(
                            label="Disease Prediction Attention", interactive=False
                        )
                        gradcam_attribute_image = gr.Image(
                            label="Attribute Prediction Attention", interactive=False
                        )
                    gradcam_metrics = gr.JSON(label="Attention Overlap Metrics")

                    gr.Markdown("### 🩺 GT Mask Overlap")
                    gradcam_mask_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload image/mask pairs to compute overlap metrics.</p>"
                    )
                    gradcam_mask_metrics = gr.JSON(label="GT Mask Overlap Summary")

                    gr.Markdown("### 🧭 SpRAy Heatmap Clusters")
                    spray_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload heatmaps or images to run SpRAy clustering.</p>"
                    )
                    spray_report = gr.JSON(label="SpRAy Cluster Report")
                    spray_gallery = gr.Gallery(label="Cluster Representative Heatmaps", columns=3)

                    gr.Markdown("### 🧪 CAV Concept Testing")
                    cav_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload a CAV bundle to run concept shortcut testing.</p>"
                    )
                    cav_report = gr.JSON(label="CAV Results")

                    gr.Markdown("### 🖼️ VAE Image Shortcut Detection")
                    vae_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload images and labels CSV to run VAE shortcut detection.</p>"
                    )
                    vae_report = gr.JSON(label="VAE Results")

                    gr.Markdown("### 🎭 Shortcut Feature Masking (M01)")
                    masking_status = gr.HTML(
                        value="<p style='color:#374151;'>Choose Image or Embedding mode and upload data to produce augmented outputs.</p>"
                    )
                    masking_report = gr.JSON(label="Masking Summary")
                    masking_download = gr.File(label="Download augmented data (zip or CSV)")

                    gr.Markdown("### 🔄 Background Randomization (M02)")
                    bg_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload images and foreground masks to swap foregrounds with random backgrounds.</p>"
                    )
                    bg_report = gr.JSON(label="Background Randomization Summary")
                    bg_download = gr.File(label="Download augmented images (zip)")

                    gr.Markdown("### 🛡️ Adversarial Debiasing (M04)")
                    debias_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload embeddings CSV with group column to produce debiased embeddings.</p>"
                    )
                    debias_report = gr.JSON(label="Adversarial Debiasing Summary")
                    debias_download = gr.File(label="Download debiased embeddings (CSV)")

                    gr.Markdown("### 🔄 Last Layer Retraining (M06 DFR)")
                    dfr_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload embeddings CSV with task_label and group_label for retrained predictions.</p>"
                    )
                    dfr_report = gr.JSON(label="Last Layer Retraining Summary")
                    dfr_download = gr.File(label="Download predictions (CSV)")

                    gr.Markdown("### Contrastive Debiasing (M07 CNC)")
                    cnc_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload embeddings CSV with task_label and group_label to produce debiased embeddings.</p>"
                    )
                    cnc_report = gr.JSON(label="Contrastive Debiasing Summary")
                    cnc_download = gr.File(label="Download debiased embeddings (CSV)")

                    gr.Markdown("### Explanation Regularization (M05 RRR)")
                    rrr_status = gr.HTML(
                        value="<p style='color:#374151;'>Upload model, images, labels CSV, and shortcut masks to fine-tune with gradient penalty.</p>"
                    )
                    rrr_report = gr.JSON(label="Explanation Regularization Summary")
                    rrr_download = gr.File(label="Download fine-tuned model (.pt)")

            # Show/hide fields based on input mode and data source
            def toggle_input_fields(input_mode, data_source):
                """Show/hide fields based on input mode and data source"""
                # HuggingFace model only shown for raw data mode AND custom CSV
                show_hf_model = input_mode == "Use Raw Data" and data_source == "Custom CSV Upload"
                show_csv = data_source == "Custom CSV Upload"
                show_chexpert_opt = data_source == "Sample Data (CheXpert - Lightweight)"
                return (
                    gr.update(visible=show_hf_model),  # hf_model_name
                    gr.update(visible=show_csv),  # custom_csv
                    gr.update(visible=show_chexpert_opt),  # use_real_chexpert_embeddings
                )

    input_mode.change(
        fn=toggle_input_fields,
        inputs=[input_mode, data_source],
        outputs=[hf_model_name, custom_csv, use_real_chexpert_embeddings],
    )

    data_source.change(
        fn=toggle_input_fields,
        inputs=[input_mode, data_source],
        outputs=[hf_model_name, custom_csv, use_real_chexpert_embeddings],
    )

    # Run detection when button is clicked
    def run_and_prepare_downloads(
        detection_state,
        input_mode,
        data_source,
        custom_csv,
        methods,
        hf_model_name,
        use_real_chexpert_embeddings,
        stat_correction,
        stat_alpha,
        ssa_fraction,
        ssa_seed,
        eec_n_clusters,
        eec_n_epochs,
        eec_min_cluster_ratio,
        eec_entropy_threshold,
        gce_q,
        gce_loss_percentile_threshold,
        gce_max_iter,
        causal_effect_spurious_threshold,
        detection_cav_bundle,
        detection_cav_quality_threshold,
        detection_cav_shortcut_threshold,
        detection_cav_test_size,
        detection_cav_min_examples,
        freq_top_percent,
        freq_tpr_threshold,
        freq_fpr_threshold,
        freq_probe_evaluation,
        condition_name,
    ):
        # Convert input_mode to internal format
        mode = "raw_data" if input_mode == "Use Raw Data" else "embeddings"
        hf_model = hf_model_name if mode == "raw_data" else None

        results_html, pdf_path, csv_dir = run_detection(
            data_source=data_source,
            custom_csv=custom_csv,
            methods=methods,
            input_mode=mode,
            hf_model_name=hf_model,
            use_real_chexpert_embeddings=bool(use_real_chexpert_embeddings),
            statistical_correction=stat_correction,
            statistical_alpha=stat_alpha,
            ssa_labeled_fraction=ssa_fraction,
            ssa_seed=ssa_seed,
            eec_n_clusters=eec_n_clusters,
            eec_n_epochs=eec_n_epochs,
            eec_min_cluster_ratio=eec_min_cluster_ratio,
            eec_entropy_threshold=eec_entropy_threshold,
            gce_q=gce_q,
            gce_loss_percentile_threshold=gce_loss_percentile_threshold,
            gce_max_iter=gce_max_iter,
            causal_effect_spurious_threshold=causal_effect_spurious_threshold,
            cav_bundle=detection_cav_bundle,
            cav_quality_threshold=detection_cav_quality_threshold,
            cav_shortcut_threshold=detection_cav_shortcut_threshold,
            cav_test_size=detection_cav_test_size,
            cav_min_examples=detection_cav_min_examples,
            freq_top_percent=freq_top_percent,
            freq_tpr_threshold=freq_tpr_threshold,
            freq_fpr_threshold=freq_fpr_threshold,
            freq_probe_evaluation=freq_probe_evaluation,
            condition_name=condition_name,
        )

        # Prepare downloads
        pdf_file = pdf_path if pdf_path and os.path.exists(pdf_path) else None

        if pdf_file is None:
            pdf_notice = (
                "<p style='color:#92400e; background:#fff4e6; padding:10px; border-radius:6px; margin-bottom:12px;'>"
                "PDF report not generated. Install weasyprint for PDF export: <code>pip install weasyprint</code></p>"
            )
            results_html = pdf_notice + results_html

        # Create ZIP of CSV exports
        csv_zip = None
        if csv_dir and os.path.exists(csv_dir):
            import shutil

            csv_zip_path = csv_dir.replace("csv_exports", "csv_exports.zip")
            shutil.make_archive(csv_dir, "zip", csv_dir)
            csv_zip = csv_zip_path if os.path.exists(csv_zip_path) else None

        new_state = csv_dir if (csv_dir and os.path.exists(csv_dir)) else detection_state
        return results_html, pdf_file, csv_zip, new_state

    run_btn.click(
        fn=run_and_prepare_downloads,
        inputs=[
            detection_state,
            input_mode,
            data_source,
            custom_csv,
            methods,
            hf_model_name,
            use_real_chexpert_embeddings,
            statistical_correction,
            statistical_alpha,
            ssa_labeled_fraction,
            ssa_seed,
            eec_n_clusters,
            eec_n_epochs,
            eec_min_cluster_ratio,
            eec_entropy_threshold,
            gce_q,
            gce_loss_percentile_threshold,
            gce_max_iter,
            causal_effect_spurious_threshold,
            detection_cav_bundle,
            detection_cav_quality_threshold,
            detection_cav_shortcut_threshold,
            detection_cav_test_size,
            detection_cav_min_examples,
            freq_top_percent,
            freq_tpr_threshold,
            freq_fpr_threshold,
            freq_probe_evaluation,
            condition_name,
        ],
        outputs=[results_output, pdf_download, csv_download, detection_state],
    )

    gradcam_btn.click(
        fn=run_gradcam_analysis,
        inputs=[
            gradcam_model,
            gradcam_image,
            gradcam_target_layer,
            gradcam_disease_head,
            gradcam_attribute_head,
            gradcam_disease_target,
            gradcam_attribute_target,
            gradcam_threshold,
            gradcam_image_size,
            gradcam_color_mode,
        ],
        outputs=[gradcam_status, gradcam_disease_image, gradcam_attribute_image, gradcam_metrics],
    )

    gradcam_mask_btn.click(
        fn=run_gradcam_mask_overlap_analysis,
        inputs=[
            gradcam_mask_model,
            gradcam_mask_images,
            gradcam_mask_masks,
            gradcam_mask_target_layer,
            gradcam_mask_head,
            gradcam_mask_target_index,
            gradcam_mask_threshold,
            gradcam_mask_mask_threshold,
            gradcam_mask_image_size,
            gradcam_mask_color_mode,
            gradcam_mask_batch_size,
        ],
        outputs=[gradcam_mask_status, gradcam_mask_metrics],
    )

    def toggle_spray_fields(input_mode):
        use_upload = input_mode == "Upload Heatmaps (.npz/.npy)"
        return (
            gr.update(visible=use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
        )

    spray_input_mode.change(
        fn=toggle_spray_fields,
        inputs=[spray_input_mode],
        outputs=[
            spray_heatmaps,
            spray_model,
            spray_images,
            spray_target_layer,
            spray_head,
            spray_target_index,
            spray_image_size,
            spray_color_mode,
        ],
    )

    spray_btn.click(
        fn=run_spray_analysis,
        inputs=[
            spray_input_mode,
            spray_heatmaps,
            spray_model,
            spray_images,
            spray_target_layer,
            spray_head,
            spray_target_index,
            spray_image_size,
            spray_color_mode,
            spray_affinity,
            spray_cluster_selection,
            spray_n_clusters,
            spray_min_clusters,
            spray_max_clusters,
            spray_downsample,
        ],
        outputs=[spray_status, spray_report, spray_gallery],
    )

    cav_btn.click(
        fn=run_cav_analysis,
        inputs=[
            cav_bundle,
            cav_quality_threshold,
            cav_shortcut_threshold,
            cav_test_size,
            cav_min_examples,
        ],
        outputs=[cav_status, cav_report],
    )

    vae_btn.click(
        fn=run_vae_analysis,
        inputs=[
            vae_images,
            vae_labels_csv,
            vae_img_size,
            vae_channels,
            vae_num_classes,
            vae_latent_dim,
            vae_epochs,
            vae_classifier_epochs,
        ],
        outputs=[vae_status, vae_report],
    )

    mask_btn.click(
        fn=run_shortcut_masking_analysis,
        inputs=[
            masking_mode,
            mask_use_last_detection,
            detection_state,
            mask_images,
            mask_masks_or_heatmap_file,
            mask_strategy_img,
            mask_heatmap_threshold,
            mask_augment_fraction,
            mask_embeddings_csv,
            mask_dim_indices_text,
            mask_strategy_emb,
        ],
        outputs=[masking_status, masking_report, masking_download],
    )

    bg_btn.click(
        fn=run_background_randomizer_analysis,
        inputs=[
            bg_images,
            bg_masks_or_heatmap_file,
            bg_heatmap_threshold,
            bg_augment_fraction,
        ],
        outputs=[bg_status, bg_report, bg_download],
    )

    debias_btn.click(
        fn=run_adversarial_debiasing_analysis,
        inputs=[
            debias_use_last_detection,
            detection_state,
            debias_embeddings_csv,
            debias_group_col,
            debias_hidden_dim,
            debias_adversary_weight,
            debias_n_epochs,
        ],
        outputs=[debias_status, debias_report, debias_download],
    )

    dfr_btn.click(
        fn=run_last_layer_retraining_analysis,
        inputs=[
            dfr_use_last_detection,
            detection_state,
            dfr_embeddings_csv,
            dfr_group_col,
            dfr_C,
            dfr_penalty,
        ],
        outputs=[dfr_status, dfr_report, dfr_download],
    )

    cnc_btn.click(
        fn=run_contrastive_debiasing_analysis,
        inputs=[
            cnc_use_last_detection,
            detection_state,
            cnc_embeddings_csv,
            cnc_group_col,
            cnc_hidden_dim,
            cnc_temperature,
            cnc_contrastive_weight,
            cnc_n_epochs,
        ],
        outputs=[cnc_status, cnc_report, cnc_download],
    )

    rrr_btn.click(
        fn=run_explanation_regularization_analysis,
        inputs=[
            rrr_model_file,
            rrr_images,
            rrr_labels_csv,
            rrr_masks_or_heatmap,
            rrr_head,
            rrr_lambda,
            rrr_epochs,
            rrr_lr,
            rrr_batch_size,
            rrr_image_size,
            rrr_color_mode,
            rrr_heatmap_threshold,
        ],
        outputs=[rrr_status, rrr_report, rrr_download],
    )

    gr.Markdown(
        """
    ---
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h4>📖 About Shortcut Detection</h4>
        <p>This dashboard uses the <a href="https://github.com/Kqp1227/Shortcut_Detect" target="_blank">ShortKit-ML</a> to identify shortcuts and biases in ML embedding spaces.</p>
        <p><strong>Built with:</strong> Python • Gradio • PyTorch • Scikit-learn</p>
    </div>
    """
    )


if __name__ == "__main__":
    import socket

    def _pick_free_port(preferred: int = 7860) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                sock.bind(("127.0.0.1", 0))
                return sock.getsockname()[1]

    env_port = os.getenv("GRADIO_SERVER_PORT")
    try:
        server_port = int(env_port) if env_port else _pick_free_port(7860)
    except ValueError:
        server_port = _pick_free_port(7860)

    print("=" * 70)
    print("🔍 Starting Shortcut Detection Dashboard")
    print("=" * 70)
    print(f"\n📍 Running locally at: http://127.0.0.1:{server_port}")
    print("📁 Project root:", project_root)
    print("\n✨ Features:")
    print("  • Sample data from CheXpert dataset")
    print("  • Custom CSV upload (embeddings or raw text)")
    print("  • HuggingFace embedding generation from raw text")
    print("  • PDF and CSV exports")
    print("  • Detection methods: HBAC, Probe, Statistical, Geometric, Equalized Odds, GroupDRO,")
    print("    Demographic Parity, Bias Direction PCA, Causal Effect, SSA (placeholder)")
    print(
        "  • Optional: CAV (include in report), VAE, Shortcut Masking M01, Background Randomization M02, Adversarial Debiasing M04, Explanation Regularization M05, Last Layer Retraining M06, Contrastive Debiasing M07 (Advanced Analysis)"
    )
    print("\n" + "=" * 70 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        share=False,  # Local only, no public URL
        inbrowser=True,  # Automatically open browser
    )
