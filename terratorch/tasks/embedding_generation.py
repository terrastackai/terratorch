import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_bounds
from shapely.geometry import box

from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.utils import bounds_from_transform, infer_transform_and_bounds

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter("once", UserWarning)
logger = logging.getLogger("EmbeddingGenerationTask")


class EmbeddingGenerationTask(TerraTorchTask):
    """
    Task that runs inference over model backbone to generate and save embeddings.
    """

    def __init__(
        self,
        model_args: dict,
        output_dir: str = "embeddings",
        embed_file_key: str = "filename",
        layers: list[int] = [-1],
        output_format: str = "tiff",
        has_cls: bool = False,
        embedding_pooling: str | None = None,
        num_workers: int = 4,
        pixel_size: float = 10.0,
        freeze_backbone=True,
    ) -> None:
        """Constructor for EmbeddingGenerationTask

        Args:
            model (str): Model name from backbone registry.
            model_args (dict, optional): Arguments passed to the model factory. Defaults to None.
            output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".
            embed_file_key (str, optional): Identifier key for single file ids in input data, will be used as embedding identifiers. Defaults to "filename".
            layers (list[int], optional): List of layers to extract embeddings from. Defaults to [-1].
            output_format (str, optional): Format for saving embeddings ('tiff' for GeoTIFF, 'parquet' for GeoParquet). Defaults to "tiff".
            has_cls (bool): Whether the model has a CLS token. Defaults to False.
            embedding_pooling (str | None, optional): Pooling method for embeddings. Defaults to None.
            num_workers (int, optional): Number of workers for saving embeddings. Defaults to 4.
            pixel_size (float, optional): Pixel size in meters, only used for constructing georef bounding boxes. Defaults to 10.0.
        """
        self.output_format = output_format.lower()

        if self.output_format not in ("tiff", "parquet", "parquet_joint", "neuco_csv"):
            raise ValueError(
                f"Unsupported output format: {self.output_format}. "
                "Supported formats are 'tiff', 'parquet', 'parquet_joint', 'neuco_csv."
            )
        # For joint parquet/ neuco_csv, part files are written which are kept track off and are joint at the end
        if self.output_format in ("parquet_joint", "neuco_csv"):
            self._part_idx = {f"{i:02d}": None for i in range(len(layers))}
            self._part_dir = {f"{i:02d}": None for i in range(len(layers))}

        self.num_workers = num_workers
        self._config_saved = False
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.embed_file_key = embed_file_key
        self.has_cls = has_cls
        self.embedding_pooling = embedding_pooling
        self.embedding_indices = layers
        self.input_shape = None
        self.pixel_size = pixel_size

        model_args = model_args or {}
        if model_args.get("necks", None):
            logger.info(
                "EmbeddingGeneration is designed to automatically add necks based on the selected "
                "output format and aggregation settings. Since necks were provided explicitly, "
                "automatic neck insertion and embedding aggregation are skipped. "
                "This may cause incompatibilities with the chosen output format."
            )
        elif embedding_pooling in (None, "None", "keep"):
            model_args["necks"] = [{"name": "SelectIndices", "indices": self.embedding_indices}]

            if output_format == "tiff":
                neck_cfg = {
                    "name": "ReshapeTokensToImage",
                    "remove_cls_token": self.has_cls,
                }

                if (
                    model_args.get("backbone_use_temporal", False)
                    and model_args.get("backbone_temporal_pooling", "mean") == "keep"
                ):
                    neck_cfg["temporal_inputs"] = True

                model_args["necks"].append(neck_cfg)
                logger.info(
                    "GeoTIFF selected; 2D token embeddings (ViT) will be reshaped to "
                    "[C, sqrt(num_tokens), sqrt(num_tokens)] after dropping CLS if present."
                )

            elif self.output_format == "parquet_joint":
                logger.info(
                    "Joint Parquet output supports only pooled (1D) embeddings, dense embeddings will be flattened. "
                    "Please set an embedding pooling mode (e.g., mean, max, or cls) or choose a different output format."
                )
        elif embedding_pooling in ["mean", "max", "min", "cls"]:
            model_args["necks"] = []
            neck_cfg = {
                "name": "AggregateTokens",
                "pooling": embedding_pooling,
                "indices": self.embedding_indices,
                "drop_cls": has_cls,
            }
            if (
                model_args.get("backbone_use_temporal", False)
                and model_args.get("backbone_temporal_pooling", "mean") == "keep"
            ):
                neck_cfg["temporal_inputs"] = True
            model_args["necks"].append(neck_cfg)

            if self.output_format == "tiff":
                warnings.warn("GeoTIFF output not recommended with embedding pooling, saves 1D vectors as (C,1,1).")
        else:
            raise ValueError(f"EmbeddingPooling {embedding_pooling} is not supported.")

        self.model_args = model_args
        self.aux_heads = []
        self.model_factory = MODEL_FACTORY_REGISTRY.build("EncoderDecoderFactory")
        super().__init__(task="embedding_generation")

    def infer_BT(self, x: torch.Tensor | dict[str, torch.Tensor]) -> tuple[int, int]:
        """Infer (B, T). For 5D assume [B, C, T, H, W] as standardized by TemporalWrapper."""
        if isinstance(x, dict):
            v = next(iter(x.values()))
        else:
            v = x
        B = v.shape[0]
        T = v.shape[2] if v.ndim == 5 else 1
        return B, T

    def check_file_ids(
        self,
        file_ids: torch.Tensor | np.ndarray | list | tuple,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        """Validate `file_ids` matches (B,) or (B, T) inferred from `x`."""
        B, T = self.infer_BT(x)

        if isinstance(file_ids, (torch.Tensor, np.ndarray)):
            expected = (B,) if T == 1 else (B, T)
            if tuple(file_ids.shape) != expected:
                raise ValueError(f"`file_ids` shape mismatch: expected {expected}, got {tuple(file_ids.shape)}")
            return

        if isinstance(file_ids, (list, tuple)):
            if len(file_ids) != B:
                raise ValueError(f"`file_ids` length mismatch: expected {B}, got {len(file_ids)}")
            if T > 1 and isinstance(file_ids[0], (list, tuple, np.ndarray)) and len(file_ids[0]) != T:
                raise ValueError(f"`file_ids` must have inner length {T}, got {len(file_ids[0])}")
            return

        raise TypeError("`file_ids` must be a tensor/ndarray or a (nested) list/tuple")

    def save_configuration_summary(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        """
        Saves a JSON containing model, layer configuration, and output specs.
        """
        if self._config_saved:
            return

        outputs = self.model.encoder(x)

        if not isinstance(outputs, list):
            outputs = [outputs]
        n_outputs = len(outputs)

        resolved_indices = [(idx if idx >= 0 else n_outputs + idx) for idx in self.embedding_indices]

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        config_summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.output_path.absolute()),
            "output_format": self.output_format,
            "backbone": self.model_args["backbone"],
            "backbone_total_params_million": total_params,
            "has_cls": self.has_cls,
            "embedding_pooling": self.embedding_pooling,
            "model_layer_count": n_outputs,
            "n_layers_saved": len(self.embedding_indices),
            "layers": [
                {
                    "output_folder_name": f"layer_{i:02d}",
                    "requested_index": folder,
                    "layer_number": res + 1,
                    "layer_output_shape": list(outputs[res][0].shape),
                }
                for i, (folder, res) in enumerate(zip(self.embedding_indices, resolved_indices))
            ],
        }

        out_path = self.output_path / "configuration_summary.json"
        try:
            with open(out_path, "w") as f:
                json.dump(config_summary, f, indent=2)
            logger.info(f"Configuration summary saved to {out_path}")
        except OSError as e:
            logger.error(f"Failed to write configuration summary: {e}")

        self._config_saved = True

    @torch.no_grad()
    def predict_step(self, batch: dict) -> None:
        embed_file_key = self.embed_file_key
        x = batch["image"]

        # Extract input image size used for later georeferencing
        if self.input_shape is None:
            if isinstance(x, dict):
                self.input_shape = next(iter(x.values())).shape[-2:]
            else:
                self.input_shape = x.shape

        if isinstance(x, dict) and embed_file_key in x:
            file_ids = x.pop(embed_file_key)
            metadata = self.pull_metadata(x)
        else:
            file_ids = batch.get(embed_file_key)
            if file_ids is None:
                raise KeyError(f"Key '{embed_file_key}' not found in input dictionary.")
            if "metadata" in batch:
                metadata = self.pull_metadata(batch["metadata"])
            else:
                metadata = self.pull_metadata(batch)

        if isinstance(file_ids, dict):
            file_ids = next(iter(file_ids.values()))

        self.check_file_ids(file_ids, x)
        embeddings = self(x)
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        self.save_configuration_summary(x)
        for layer, embeddings_per_layer in enumerate(embeddings):
            self.save_embeddings(embeddings_per_layer, file_ids, metadata, layer)

    def on_predict_end(self) -> None:
        if self.output_format == "parquet_joint":
            self.join_parquet_files()
        elif self.output_format == "neuco_csv":
            self.join_neuco_parts_to_csv()

    def save_embeddings(
        self,
        embedding: torch.Tensor | dict[str, torch.Tensor],
        file_ids: list[str] | None,
        metadata: dict,
        layer: int,
    ) -> None:
        """Save embeddings for a given layer (per sample, optional per timestep and per modality)."""
        path = self.output_path / f"layer_{layer:02d}"
        if isinstance(embedding, dict):
            for modality, t in embedding.items():
                path = path / modality
                self.write_batch(t, file_ids, metadata, path)
        elif isinstance(embedding, torch.Tensor):
            self.write_batch(embedding, file_ids, metadata, path)
        else:
            raise TypeError(f"Unsupported embedding type: {type(embedding)}. Expected Tensor or dict of Tensors.")

    def pull_metadata(self, data: dict) -> dict:
        """Extract known metadata fields from `batch`, removing them from data and returning a metadata dict.
        Args:
            data (dict): Input data dictionary containing metadata.
        Returns:
            dict: Metadata dictionary.
        """

        def pop_first(d: dict, keys):
            for k in keys:
                if k in d:
                    return d.pop(k)
            return None

        # Aliases in priority order
        metadata_map = {
            "file_id": ("file_id",),
            "product_id": ("product_id",),
            "time": ("time", "time_", "timestamp"),
            "grid_cell": ("grid_cell",),
            "grid_row_u": ("grid_row_u",),
            "grid_col_r": ("grid_col_r",),
            "geometry": ("geometry",),
            "utm_footprint": ("utm_footprint",),
            "crs": ("crs", "utm_crs"),
            "pixel_bbox": ("pixel_bbox",),
            "bounds": ("bounds",),
            "geotransform": ("geotransform",),
            "raster_shape": ("raster_shape",),
            "center_lat": ("center_lat", "centre_lat", "lat"),
            "center_lon": ("center_lon", "centre_lon", "lon"),
        }

        metadata = {}

        for key, aliases in metadata_map.items():
            value = pop_first(data, aliases)
            if value is not None:
                metadata[key] = value

        return metadata

    def write_batch(
        self,
        embedding: torch.Tensor,
        file_ids: list[str],
        metadata: dict,
        dir_path: Path,
    ) -> None:
        """ " Write a batch (optionally with timesteps) to GeoTIFF/GeoParquet."""
        dir_path.mkdir(parents=True, exist_ok=True)

        is_temporal = isinstance(file_ids[0], (list, tuple, np.ndarray))
        emb_np = embedding.detach().cpu().numpy()

        if self.output_format == "parquet_joint":
            self.write_parquet_batch(emb_np, file_ids, metadata, is_temporal, dir_path)
            return

        if self.output_format == "neuco_csv":
            self.write_neuco_parquet_batch(emb_np, file_ids, dir_path)
            return

        tasks = list(self.iter_samples(emb_np, file_ids, metadata, is_temporal))
        if self.output_format == "tiff":
            writer = self.write_tiff
        elif self.output_format == "parquet":
            writer = self.write_parquet
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format!r}")

        max_workers = min(len(tasks), self.num_workers)

        def write_one(task):
            arr, filename, meta = task
            writer(arr, filename, meta, dir_path)

        if max_workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                write_one(task)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(write_one, tasks))

    def iter_samples(
        self,
        embedding: np.ndarray,
        file_ids: list[str] | list[list[str]],
        metadata: dict,
        is_temporal: bool,
    ) -> iter:
        """Yields (embedding_np, filename, metadata_sample) tuples."""
        B = len(file_ids)

        md = {}
        for k, v in (metadata or {}).items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            md[k] = v

        def _select_meta(v, b: int, t: int | None, B: int, T: int | None):
            if isinstance(v, (list, tuple)):
                if len(v) == B:
                    vb = v[b]
                    if t is not None and T is not None and isinstance(vb, (list, tuple)) and len(vb) == T:
                        return vb[t]
                    return vb
                return v

            if getattr(v, "ndim", 0) == 0:
                return v.item()

            if v.ndim == 1:
                return v[b] if v.shape[0] == B else v

            if v.shape[0] == B:
                if t is None:
                    return v[b]
                if v.shape[1] == 1:
                    return v[b, 0]
                return v[b, t]
            return v

        if is_temporal:
            for b in range(B):
                T = len(file_ids[b])
                for t in range(T):
                    filename = file_ids[b][t]
                    meta = {}

                    for k, v in md.items():
                        meta[k] = _select_meta(v, b=b, t=t, B=B, T=T)

                    arr = embedding[b, t, ...]
                    yield arr, filename, meta
        else:
            for b in range(B):
                filename = file_ids[b]
                meta = {}

                for k, v in md.items():
                    meta[k] = _select_meta(v, b=b, t=None, B=B, T=None)

                arr = embedding[b, ...]
                yield arr, filename, meta

    def write_tiff(
        self,
        arr: np.ndarray,
        filename: str,
        metadata: dict,
        dir_path: Path,
        suffix: str = "_embedding",
    ) -> None:
        """Write a single sample to GeoTIFF.

        Georeferencing priority (best -> fallback):
          1) geotransform + raster_shape (+ crs)
          2) bounds (+ crs)
          3) center_lat/center_lon + pixel_size + input_shape: inferred (approx, assumes north-up, EPSG:4326)
          4) otherwise: write non-georeferenced GeoTIFF
        """
        filename = Path(filename).stem
        out_path = dir_path / f"{filename}{suffix}.tif"

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1, 1)
        elif arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3:
            raise ValueError(f"Expected arr with ndim 1, 2 or 3 mapping to (C,H,W), got shape {arr.shape}")

        profile = {
            "driver": "GTiff",
            "height": int(arr.shape[1]),
            "width": int(arr.shape[2]),
            "count": int(arr.shape[0]),
            "dtype": arr.dtype,
        }

        crs = metadata.get("crs", None)
        transform = None

        # 1) geotransform + raster_shape (preferred for GeoTIFF)
        geotransform = metadata.get("geotransform", None)
        raster_shape = metadata.get("raster_shape", None)

        if isinstance(geotransform, np.ndarray):
            geotransform = geotransform.tolist()
        if isinstance(raster_shape, np.ndarray):
            raster_shape = raster_shape.tolist()

        if (
            crs is not None
            and isinstance(geotransform, (list, tuple))
            and len(geotransform) == 6
            and isinstance(raster_shape, (list, tuple))
            and len(raster_shape) == 2
        ):
            x0, px, rx, y0, ry, py = [float(v) for v in geotransform]
            src_transform = Affine(px, rx, x0, ry, py, y0)
            src_h, src_w = int(raster_shape[0]), int(raster_shape[1])
            left, bottom, right, top = bounds_from_transform(src_transform, src_w, src_h)

            # If embedding resolution differs from source, preserve extent and re-sample transform to output size
            transform = from_bounds(left, bottom, right, top, profile["width"], profile["height"])

        # 2) bounds (+ crs): reconstruct north-up transform for output size
        if transform is None and crs is not None:
            b = metadata.get("bounds", None)
            if isinstance(b, np.ndarray):
                b = b.tolist()
            if isinstance(b, (list, tuple)) and len(b) == 4:
                left, bottom, right, top = [float(v) for v in b]
                transform = from_bounds(left, bottom, right, top, profile["width"], profile["height"])

        # 3) center + pixel_size + input_shape fallback (approx, EPSG:4326)
        if transform is None:
            lat, lon = metadata.get("center_lat", None), metadata.get("center_lon", None)

            if lat is not None and lon is not None:
                crs = "EPSG:4326"
                b = infer_transform_and_bounds(lat, lon, *self.input_shape, self.pixel_size)
                if b is not None and isinstance(b, (list, tuple)) and len(b) == 4:
                    left, bottom, right, top = [float(v) for v in b]
                    transform = from_bounds(left, bottom, right, top, profile["width"], profile["height"])

        if crs is not None and transform is not None:
            profile["crs"] = crs
            profile["transform"] = transform

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
            dst.update_tags(**{k: str(v) for k, v in metadata.items()})

    def write_parquet(
        self,
        arr: np.ndarray,
        filename: str,
        metadata: dict,
        dir_path: Path,
        suffix: str = "_embedding",
    ) -> None:
        """Write a single sample to GeoParquet.

        Georeferencing priority (best -> fallback):
          1) bounds (+ crs)
          2) geotransform + raster_shape (+ crs)
          3) center_lat/center_lon + pixel_size + input_shape: infer bounds
          4) otherwise: write non-geo Parquet
        """
        filename = Path(filename).stem
        out_path = dir_path / f"{filename}{suffix}.parquet"

        row = {
            "file_id": filename,
            "embedding": arr.tolist(),
        }

        crs = metadata.get("crs", None)
        bounds = None

        # 1) bounds (+ crs)
        b = metadata.get("bounds", None)
        if isinstance(b, np.ndarray):
            b = b.tolist()
        if crs is not None and isinstance(b, (list, tuple)) and len(b) == 4:
            bounds = tuple(float(v) for v in b)

        # 2) geotransform + raster_shape (+ crs): derive bounds
        if bounds is None and crs is not None:
            geotransform = metadata.get("geotransform", None)
            raster_shape = metadata.get("raster_shape", None)

            if isinstance(geotransform, np.ndarray):
                geotransform = geotransform.tolist()
            if isinstance(raster_shape, np.ndarray):
                raster_shape = raster_shape.tolist()

            if (
                isinstance(geotransform, (list, tuple))
                and len(geotransform) == 6
                and isinstance(raster_shape, (list, tuple))
                and len(raster_shape) == 2
            ):
                x0, px, rx, y0, ry, py = [float(v) for v in geotransform]
                src_transform = Affine(px, rx, x0, ry, py, y0)
                src_h, src_w = int(raster_shape[0]), int(raster_shape[1])
                bounds = tuple(float(v) for v in bounds_from_transform(src_transform, src_w, src_h))

        # 3) center + pixel_size + input_shape fallback (approx, EPSG:4326)
        if bounds is None:
            lat, lon = metadata.get("center_lat", None), metadata.get("center_lon", None)

            if lat is not None and lon is not None:
                crs = "4326"
                b = infer_transform_and_bounds(lat, lon, *self.input_shape, self.pixel_size)
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    bounds = tuple(float(v) for v in b)

        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                v = v.item() if v.ndim == 0 else v.tolist()
            elif hasattr(v, "item"):
                v = v.item()
            row[k] = v

        if crs is not None and bounds is not None:
            left, bottom, right, top = bounds
            row["geometry"] = box(left, bottom, right, top)
            gdf = gpd.GeoDataFrame([row], geometry="geometry", crs=crs)
            gdf.to_parquet(out_path, index=False)
        else:
            pd.DataFrame([row]).to_parquet(out_path, index=False)

    def write_parquet_batch(
        self,
        emb_np: np.ndarray,
        file_ids: list[str],
        metadata: dict,
        is_temporal: bool,
        dir_path: Path,
    ) -> None:
        """Write a batch of samples to GeoParquet."""
        dir_path.mkdir(parents=True, exist_ok=True)

        # In the case of temporal data, use 'iter_samples' logic to flatten the temporal and batch dimension
        if is_temporal:
            tasks = list(self.iter_samples(emb_np, file_ids, metadata, is_temporal))
            filenames, emb_list = [], []
            meta_cols = {k: [] for k in metadata.keys()}

            for arr, filename, meta in tasks:
                filenames.append(Path(filename).stem)
                emb_list.append(arr.reshape(-1))
                for k, v in meta.items():
                    meta_cols[k].append(v)
            emb_2d = np.asarray(emb_list, np.float32)

        # In the non-temporal case we can directly treat batch dimension as row dimension
        else:
            filenames = [Path(f).stem for f in file_ids]
            n = len(filenames)
            meta_cols = {}
            for k, v in (metadata or {}).items():
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()

                if isinstance(v, np.ndarray):
                    v = v.item() if v.ndim == 0 else v.tolist()
                elif hasattr(v, "item"):
                    v = v.item()

                if isinstance(v, list) and len(v) == n:
                    meta_cols[k] = v
                else:
                    meta_cols[k] = [v] * n  # Scalar / global metadata -> broadcast

            emb_2d = emb_np.reshape(emb_np.shape[0], -1)

        n = len(filenames)
        if emb_2d.ndim != 2 or emb_2d.shape[0] != n:
            raise ValueError(f"Expected aggregated embedding (n,d) with n=batch_size. Got {emb_2d.shape}, n={n}")

        df = pd.DataFrame(meta_cols)
        df["file_id"] = filenames
        df["embedding"] = [row.tolist() for row in emb_2d]

        if "crs" in df.columns:
            # Ensure single CRS value for a valid GeoDataFrame
            if df["crs"].notna().any() and (df["crs"].nunique(dropna=True) == 1):
                crs = df["crs"].iloc[0]
            else:
                crs = None
                logger.info(
                    "Input CRS values are missing or inconsistent across rows; "
                    "writing non-georeferenced Parquet instead."
                )
        else:
            crs = None

        if crs is not None and "bounds" in df.columns and df["bounds"].notna().all():
            df["geometry"] = [box(*map(float, b)) for b in df["bounds"]]
            out = gpd.GeoDataFrame(df, geometry="geometry", crs=df["crs"].iloc[0])
        else:
            out = df

        # For joint geoparquet we collect info on the written part files and paths
        l_key = dir_path.as_posix().split("layer_")[1]
        if self._part_idx[l_key] is None:
            self._part_idx[l_key] = 0
            self._part_dir[l_key] = dir_path

        part_path = dir_path / f"embeddings_part_{self._part_idx[l_key]:06d}.parquet"
        self._part_idx[l_key] += 1
        out.to_parquet(part_path, index=False)

    def join_parquet_files(self) -> None:
        """Join part parquet files into one final parquet file."""

        for l_key in self._part_dir.keys():
            parts = sorted(self._part_dir[l_key].glob("embeddings_part_*.parquet"))
            out_path = self._part_dir[l_key] / "embeddings.parquet"

            if not parts:
                raise FileNotFoundError(
                    f"No part files found in {self._part_dir[l_key]} matching embeddings_part_*.parquet"
                )

            is_geo = False
            first_gdf = None
            try:
                first_gdf = gpd.read_parquet(parts[0])
                is_geo = True
            except Exception:
                is_geo = False

            if is_geo:
                gdfs = [first_gdf]
                base_crs = first_gdf.crs
                for p in parts[1:]:
                    gdf = gpd.read_parquet(p)
                    if gdf.crs != base_crs:
                        is_geo = False
                        break
                    gdfs.append(gdf)

            if is_geo:
                out = pd.concat(gdfs, ignore_index=True)
                out = gpd.GeoDataFrame(out, geometry="geometry", crs=base_crs)
                out.to_parquet(out_path, index=False)
            else:
                dfs = [pd.read_parquet(p) for p in parts]
                out = pd.concat(dfs, ignore_index=True)
                out.to_parquet(out_path, index=False)

            for p in parts:
                try:
                    p.unlink()
                except OSError:
                    pass

    def join_neuco_parts_to_csv(self) -> None:
        """Join NeuCo part parquet files into final CSV with exact format: id + 1..D columns."""
        for l_key in self._part_dir.keys():
            if self._part_dir[l_key] is None:
                continue

            parts = sorted(self._part_dir[l_key].glob("embeddings_part_*.parquet"))
            if not parts:
                raise FileNotFoundError(
                    f"No part files found in {self._part_dir[l_key]} matching embeddings_part_*.parquet"
                )

            dfs = [pd.read_parquet(p) for p in parts]
            out = pd.concat(dfs, ignore_index=True)

            # Ensure exact column order: 'id' then '1'..'D' (as strings)
            if "id" not in out.columns:
                raise ValueError("NeuCo output missing required 'id' column.")
            dim_cols = [c for c in out.columns if c != "id"]
            dim_cols_sorted = sorted(dim_cols, key=lambda s: int(s))  # '1','2',...
            out = out[["id"] + dim_cols_sorted]

            csv_path = self._part_dir[l_key] / "embeddings.csv"
            out.to_csv(csv_path, index=False)

            # cleanup parts
            for p in parts:
                try:
                    p.unlink()
                except OSError:
                    pass

    def write_neuco_parquet_batch(
        self,
        emb_np: np.ndarray,  # (B, D)
        file_ids: list[str],  # length B
        dir_path: Path,
    ) -> None:
        """Append/ Create NeuCo-Bench CSV (id + dim columns)."""
        dir_path.mkdir(parents=True, exist_ok=True)

        if emb_np.ndim != 2:
            raise ValueError(
                f"NeuCo-Bench CSV output requires a (B, D) array (Batch, 1D vector), got {emb_np.shape}. "
                "Provide 'embedding_pooling' to reduce to 1D."
            )
        B, D = emb_np.shape
        if len(file_ids) != B:
            raise ValueError(f"len(file_ids) must match B. Got {len(file_ids)} vs {B}")

        ids = [Path(Path(f).stem).stem for f in file_ids]
        emb_2d = np.asarray(emb_np, dtype=np.float32)

        df = pd.DataFrame(emb_2d, columns=[str(i) for i in range(1, D + 1)])
        df.insert(0, "id", ids)

        # track parts per layer dir (same pattern as parquet_joint)
        l_key = dir_path.as_posix().split("layer_")[1]
        if self._part_idx[l_key] is None:
            self._part_idx[l_key] = 0
            self._part_dir[l_key] = dir_path

        part_path = dir_path / f"embeddings_part_{self._part_idx[l_key]:06d}.parquet"
        self._part_idx[l_key] += 1
        df.to_parquet(part_path, index=False)
