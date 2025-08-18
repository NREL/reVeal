"""
Create demand curve grid and logic for characterizing the demand curve grid.
"""
from functools import cached_property
from pathlib import Path

from exactextract.exact_extract import exact_extract
import geopandas as gpd
import pandas as pd
from libpysal import graph
import rasterio as rio
from shapely.geometry import box
from shapely.ops import unary_union

from large_loads_2025 import HPC_REV_DATA, HPC_LARGE_LOAD_DATA


class DemandCurveGrid:
    """Methods for building a demand-curve table."""

    def __init__(self, grid_size, crs, bounds=None, template=None):
        """Initialize a DemandCurve grid object.

        Parameters
        ----------
        grid_size : int
            The size of the grid cells.
        crs : str
            The coordinate reference system for the grid.
        bounds : tuple, optional
            The spatial bounds for the grid, by default None
        template : str, optional
            Path to a template file for the grid, by default None
        """
        self.grid_size = grid_size
        self.crs = crs
        self.grid = None
        self.bounds = bounds if bounds else None
        self.template = template if template else None
        self.bounds = self._get_bounds() if template else None
        self.grid = self.create_grid() if (bounds or template) else None

    def __repr__(self):
        """Return DemandCurveGrid object representation string."""
        name = self.__class__.__name__
        skip_types = (gpd.geodataframe.GeoDataFrame, pd.core.frame.DataFrame)
        items = self.__dict__.items()
        items = {k: v for k, v in items if not isinstance(v, skip_types)}
        address = hex(id(self))
        msgs = [f"\n   {k}={v}" for k, v in items.items()]
        msg = " ".join(msgs)
        return f"<{name} object at {address}> {msg}"

    def _get_bounds(self):
        """Get the bounds for the grid."""
        ext = Path(self.template).suffix.lower()
        if ext in [".gpkg", ".geojson", ".shp"]:
            bounds = gpd.read_file(self.template).total_bounds
        elif ext in [".tif"]:
            with rio.open(self.template) as src:
                bounds = src.bounds
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return bounds

    def create_grid(self):
        """Create a grid based on bounds.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame representing the grid.
        """
        bounds: tuple[float, float, float, float] = (
            self.bounds if self.bounds is not None else self._get_bounds()
        )
        minx, miny, maxx, maxy = bounds

        self.grid = gpd.GeoDataFrame(
            geometry=[
                box(x, y, x + self.grid_size, y + self.grid_size)
                for x in range(int(minx), int(maxx), self.grid_size)
                for y in range(int(miny), int(maxy), self.grid_size)
            ],
            crs=self.crs,
        )
        self.grid["grid_id"] = self.grid.index
        return self.grid

    def _neighbor(self):
        """Create new geometry for grid that consists of its neighbors."""
        grid = self.grid.copy()
        grid = grid.set_index("grid_id")
        cont = graph.Graph.build_contiguity(grid, rook=False)
        adj = cont.adjacency
        mappings = {i: list(adj[i].keys()) for i in grid.index}

        grid["geometry"] = grid.index.to_series().map(
            lambda idx: unary_union(
                [grid.geometry.loc[idx]]
                + [grid.geometry.loc[n] for n in mappings.get(idx, [])]
            )
        )
        grid = grid.reset_index()
        return grid

    @cached_property
    def grid_neighbors(self):
        """Cached property: unioned neighbor geometries for each grid cell."""
        return None if self.grid is None else self._neighbor()

    def _get_grid(self, neighbor=False):
        """Get the grid, optionally with neighbor geometries."""
        if neighbor:
            grid = self.grid_neighbors.copy()
        else:
            grid = self.grid.copy()
        return grid

    def _vector_proximity(self, df, grid, stem):
        """Calculate proximity of vector data to grid cells."""
        joined = gpd.sjoin_nearest(
            grid, df, how="left", distance_col=f"proximity_{stem}"
        )
        joined = joined.drop_duplicates(subset="grid_id")
        grid[f"proximity_{stem}"] = joined[f"proximity_{stem}"].values
        return grid

    def _vector_length(self, df, grid, stem):
        """Calculate length of vector data within grid cells."""
        inter = gpd.overlay(
            grid[['grid_id', 'geometry']],
            df[['geometry']],
            how='intersection'
        )
        inter['seg_length'] = inter.geometry.length
        length_series = inter.groupby('grid_id')['seg_length'].sum()
        col_name = f"length_{stem}"
        grid[col_name] = (
            grid['grid_id']
            .map(length_series)
            .fillna(0)
        )
        return grid

    def _vector_count(self, df, grid, stem):
        """Count occurrences of vector data within grid cells."""
        joined = gpd.sjoin(grid, df, how="left", predicate="intersects")
        counts = joined.groupby("grid_id")["index_right"].count()
        count_col = f"count_{stem}"
        grid[count_col] = grid["grid_id"].map(counts).fillna(0).astype(int)
        return grid

    def _vector_aggregate(self, df, grid, stem, value_col=None, func="sum"):
        """Aggregate vector data within grid cells."""
        joined = gpd.sjoin(grid, df, how="left", predicate="intersects")
        grp = joined.groupby("grid_id")[value_col]
        if func == "sum":
            agg_series = grp.sum()
        elif func in ("mean", "avg"):
            agg_series = grp.mean()
        else:
            raise ValueError(f"Unsupported aggregation function: {func}")

        grid[f"{func}_{stem}_{value_col}"] = grid["grid_id"].map(agg_series)
        return grid

    def _vector_intersects(self, df, grid, stem):
        """Flag each grid cell True/False if it intersects any feature."""
        joined = gpd.sjoin(
            grid[['grid_id', 'geometry']],
            df[['geometry']],
            how='left',
            predicate='intersects'
        )

        intersects = joined.groupby('grid_id')['index_right'].count()
        col_name = f"intersects_{stem}"
        grid[col_name] = (grid['grid_id']
                          .map(intersects)
                          .fillna(0)
                          .astype(int) > 0)
        return grid

    def _aggregate_vector_within_grid(
        self, df_path, value_col=None, agg_func="sum",
        buffer=None, neighbor=False
    ):
        """Aggregate vector data within grid cells.

        Parameters
        ----------
        df_path : str
            Path to the vector data file.
        value_col : str, optional
            Name of the column to aggregate, by default None
        agg_func : str, optional
            Aggregation function, by default "sum"
            Supported functions are:
            "proximity", "count", "sum", "mean", "avg", "length",
            and "intersects".
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None
        neighbor : bool, optional
            Whether to include neighboring grid cells, by default False

        Returns
        -------
        gpd.GeoDataFrame
            The updated grid with aggregated values.
        """
        grid = self._get_grid(neighbor)

        if buffer is not None:
            grid["geometry"] = grid.geometry.buffer(buffer)

        df = gpd.read_file(df_path).to_crs(self.crs)
        stem = Path(df_path).stem
        func = agg_func.lower()

        if isinstance(value_col, float) and pd.isna(value_col):
            value_col = None
        if isinstance(value_col, str) and not value_col.strip():
            value_col = None

        if func == "proximity":
            grid = self._vector_proximity(df, grid, stem)
        elif func == "count":
            grid = self._vector_count(df, grid, stem)
        elif func in ("sum", "mean", "avg"):
            grid = self._vector_aggregate(df, grid, stem, value_col, func)
        elif func == "length":
            grid = self._vector_length(df, grid, stem)
        elif func == "intersects":
            grid = self._vector_intersects(df, grid, stem)
        else:
            raise ValueError(f"Unsupported aggregation function: {func}")

        return grid

    def _aggregate_raster_within_grid(
        self, raster_path, agg_func="sum", buffer=None, neighbor=False
    ):
        """Aggregate raster values within grid cells using exactextract.

        Parameters
        ----------
        raster_path : str
            Path to the raster file.
        agg_func : str, optional
            Aggregation function to use, by default "sum"
            Supported functions are: "sum", "mean", "avg", "min", "max".
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None
        neighbor : bool, optional
            Whether to include neighboring grid cells, by default False

        Returns
        -------
        gpd.GeoDataFrame
            The updated grid with aggregated values.
        """
        grid = self._get_grid(neighbor)

        if buffer is not None:
            grid["geometry"] = grid.geometry.buffer(buffer)

        with rio.open(raster_path) as src:
            grid = grid.to_crs(src.crs)

        stem = Path(raster_path).stem
        func = agg_func.lower()

        result = exact_extract(
            rast=raster_path,
            vec=grid,
            include_cols=["grid_id"],
            ops=[func],
            output="pandas",
        )

        col_name = f"{func}_{stem}"
        result.rename(columns={f"{func}": col_name}, inplace=True)

        return result

    def aggregate_within_grid(
        self, df_path, value_col=None, agg_func="sum",
        buffer=None, neighbor=False
    ):
        """Aggregate data within grid cells.

        Parameters
        ----------
        df_path : str
            Path to the vector data file.
        value_col : str, optional
            Name of the column to aggregate, by default None
        agg_func : str, optional
            Aggregation function to use, by default "sum"
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with aggregated values.
        """
        ext = Path(df_path).suffix.lower()
        if ext in [".tif", ".tiff"]:
            grid = self._aggregate_raster_within_grid(
                df_path, agg_func, buffer, neighbor
            )
        elif ext in [".gpkg", ".geojson", ".shp"]:
            grid = self._aggregate_vector_within_grid(
                df_path, value_col, agg_func, buffer, neighbor
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return grid

    def csv_to_spec(self, csv_path):
        """Convert a CSV file to a specification dictionary.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.

        Returns
        -------
        dict
            A dictionary with the specification.
        """
        df = pd.read_csv(csv_path)
        spec = {}
        for _, row in df.iterrows():
            path = row["path"]
            agg = row["operation"]
            value_col = row.get("value_col", None)
            buffer = row.get("buffer", None)
            neighbor = row.get("neighbor", False)

            spec[Path(path).stem] = {
                "path": path,
                "agg": agg,
                "value_col": value_col,
                "buffer": buffer,
                "neighbor": neighbor,
            }
        return spec

    def characterize_grid(self, spec):
        """Characterize the grid based on the provided specification.

        Parameters
        ----------
        spec : dict
            A dictionary with the specification.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the characterized grid.
        """
        out = self.grid.copy()

        for _, cfg in spec.items():
            path = cfg["path"]
            agg = cfg.get("agg", "sum")
            buf = cfg.get("buffer", None)
            val_col = cfg.get("value_col", None)
            neighbor = cfg.get("neighbor", False)

            layer = self.aggregate_within_grid(
                path, value_col=val_col, agg_func=agg,
                buffer=buf, neighbor=neighbor
            )

            merge_cols = [
                c for c in layer.columns if c not in ("geometry", "grid_id")]
            out = out.merge(layer[["grid_id"] + merge_cols],
                            on="grid_id", how="left")

        return out


if __name__ == "__main__":
    temp = HPC_REV_DATA.joinpath("rasters/templates/rev_template_fy25.tif")
    demand_grid = DemandCurveGrid(grid_size=10_000, crs="EPSG:5070",
                                  template=str(temp))
    SPEC_PATH = HPC_LARGE_LOAD_DATA.joinpath(
        "tables/misc/characterization_layers.csv")
    spec_dict = demand_grid.csv_to_spec(SPEC_PATH)
    characterized_grid = demand_grid.characterize_grid(spec_dict)
