import datamol as dm
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import umap
from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10, Category20
from MolBokeh import MolBokeh

class ChemSpaceViz:
    """Chemical space visualization class for molecular datasets.
    
    Loads molecules from files, computes fingerprints, performs dimensionality reduction
    and clustering, creates interactive visualizations using Bokeh.
    """
    
    def __init__(
        self,
        file_paths,
        reduction_method: str = "pca",
        reduction_params: dict | None = None,
        n_clusters: int | None = None,
        max_k: int = 10,
        point_size: int = 4,
        color_by: str = "dataset",
        perplexity: int = 30,
        subset_size: int | None = None,
        **load_kwargs
    ):
        """Initialize chemical space visualizer.
        
        Args:
            file_paths: File path or list of file paths containing molecules
            reduction_method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            reduction_params: Additional parameters for reduction method
            n_clusters: Number of clusters (if None, determined automatically)
            max_k: Maximum number of clusters for optimal search
            point_size: Size of points in plot
            color_by: Coloring scheme ('dataset' or 'cluster')
            perplexity: Perplexity parameter for t-SNE or n_neighbors for UMAP
            subset_size: Subset size for analysis (if None, use all molecules)
            **load_kwargs: Additional parameters for file loading
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        print("Loading data from files:")
        all_dfs = []
        for fp in file_paths:
            print(" ", fp)
            if fp.endswith('.smi'):
                df = pd.read_csv(fp, sep='\s+', header=None, names=['smiles'])
            else:
                df = dm.open_df(fp, **load_kwargs)
            if isinstance(df, list):
                df = df[0]
            smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES' or 'standard_smiles'
            if smiles_col not in df.columns:
                raise ValueError(f"DataFrame from {fp} must contain a column 'smiles' or 'SMILES'")
            df["dataset"] = fp
            all_dfs.append(df)
        self.df = pd.concat(all_dfs, ignore_index=True)
        print("Total molecules loaded:", len(self.df))
        
        print("Converting SMILES to molecules and filtering invalid ones...")
        smiles_col = 'smiles' if 'smiles' in self.df.columns else 'SMILES' or 'standard_smiles'
        mols = [dm.to_mol(smile) for smile in self.df[smiles_col]]
        valid_indices = [i for i, m in enumerate(mols) if m is not None]
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        if subset_size is not None and subset_size < len(self.df):
            print(f"Picking a diverse subset of size {subset_size} out of {len(self.df)} molecules...")
            mols_for_subset = [dm.to_mol(smile) for smile in self.df[smiles_col]]
            indices, picks = dm.pick_diverse(mols_for_subset, npick=subset_size)
            self.df = self.df.iloc[indices].reset_index(drop=True)

        print("Calculating fingerprints for unique molecules in parallel...")
        unique_smiles = list(set(self.df[smiles_col]))
        def _to_fp_from_smile(smile):
            mol = dm.to_mol(smile)
            return dm.to_fp(mol)
        unique_fps = dm.parallelized(
            _to_fp_from_smile, unique_smiles,
            arg_type="item",
            progress=True,
            total=len(unique_smiles)
        )
        fps_dict = dict(zip(unique_smiles, unique_fps))
        self.fps = [fps_dict[smile] for smile in self.df[smiles_col]]
        self.fps_array = np.array(self.fps)

        if reduction_params is None:
            reduction_params = {}
        self.reduction_params = reduction_params

        print("Reducing dimensions using", reduction_method.upper())
        self.coords = self.reduce_dimensions(self.fps_array, method=reduction_method, perplexity=perplexity)
        self.df["x"] = self.coords[:, 0]
        self.df["y"] = self.coords[:, 1]
        self.color_by = color_by.lower()
        
        if self.color_by == "cluster":
            if n_clusters is None:
                print("Determining optimal number of clusters using the Elbow Method...")
                self.optimal_k = self.find_optimal_k(self.coords, max_k)
                print("Optimal k =", self.optimal_k)
            else:
                self.optimal_k = n_clusters
                print("Using user-specified number of clusters:", self.optimal_k)
            print("Clustering data with KMeans...")
            self.cluster_labels = self.cluster_data(self.coords, self.optimal_k)
            self.df["cluster"] = self.cluster_labels
            self.assign_colors_cluster()
        elif self.color_by == "dataset":
            self.assign_colors_dataset()
        else:
            raise ValueError("color_by must be either 'dataset' or 'cluster'")

        self.point_size = point_size
        self.smiles_col = smiles_col

    def reduce_dimensions(self, X: np.ndarray, method: str = "pca", perplexity: int = 30) -> np.ndarray:
        """Reduce dimensionality of fingerprint matrix.
        
        Args:
            X: Input fingerprint matrix
            method: Reduction method ('pca', 'tsne', 'umap')
            perplexity: Perplexity for t-SNE or n_neighbors for UMAP
            
        Returns:
            2D coordinates array
        """
        method = method.lower()
        if method == "pca":
            reducer = PCA(n_components=2, **self.reduction_params)
            return reducer.fit_transform(X)
        elif method == "tsne":
            reducer = TSNE(n_components=2, n_jobs=-1, perplexity=perplexity, **self.reduction_params)
            return reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, n_jobs=-1, n_neighbors=perplexity, **self.reduction_params)
            return reducer.fit_transform(X)
        else:
            raise ValueError("Unsupported reduction method: " + method)
    
    def find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method.
        
        Args:
            X: Input coordinate matrix
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        inertias = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
        inertias = np.array(inertias)
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        optimal_index = np.argmax(second_deltas)
        optimal_k = list(k_range)[optimal_index + 1]
        return optimal_k

    def cluster_data(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering on coordinate data.
        
        Args:
            X: Input coordinate matrix
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels array
        """
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(X)
        return km.labels_

    def assign_colors_dataset(self):
        """Assign colors based on dataset labels."""
        unique_datasets = sorted(self.df["dataset"].unique())
        num_datasets = len(unique_datasets)
        if num_datasets <= 10:
            palette = Category10[10]
        elif num_datasets <= 20:
            palette = Category20[20]
        else:
            palette = Category20[20]
        self.color_map = {ds: palette[i % len(palette)] for i, ds in enumerate(unique_datasets)}
        self.df["color"] = self.df["dataset"].map(self.color_map)
        print("Datasets found:", unique_datasets)

    def assign_colors_cluster(self):
        """Assign colors based on cluster labels."""
        unique_clusters = sorted(self.df["cluster"].unique())
        num_clusters = len(unique_clusters)
        if num_clusters <= 10:
            palette = Category10[10]
        elif num_clusters <= 20:
            palette = Category20[20]
        else:
            palette = Category20[20]
        self.color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(unique_clusters)}
        self.df["color"] = self.df["cluster"].map(self.color_map)
        print("Clusters found:", unique_clusters)

    def plot(self, use_webgl: bool = True, show_molecules: bool = True):
        """Create interactive plot in Jupyter notebook.
        
        Args:
            use_webgl: Use WebGL backend for better performance
            show_molecules: Show molecular structures on hover
        """
        output_notebook()
        self.df["mol_id"] = range(len(self.df))
        source = ColumnDataSource(self.df)
        if use_webgl:
            p = figure(
                width=600, height=500,
                tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save,hover",
                title="Chemical Space", tooltips=None,
                output_backend="webgl"
            )
        else:
            p = figure(
                width=600, height=500,
                tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save,hover",
                title="Chemical Space", tooltips=None
            )

        legend_field = 'cluster' if self.color_by == 'cluster' else 'dataset'
        p.scatter(
            x='x', y='y',
            source=source, 
            size=self.point_size,
            color='color',
            alpha=0.6,
            legend_field=legend_field
        )
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        hover_info = ["mol_id"]
        if self.color_by == "cluster":
            hover_info.append("cluster")
        else:
            hover_info.append("dataset")

        if show_molecules:
            p = MolBokeh().add_molecule(
                fig=p,
                source=source,
                smilesColName='smiles',
                hoverAdditionalInfo=hover_info,
                molSize=(150,100)
            )
        show(p)

    def save_plot(self, file_path: str, show_molecules: bool = True):
        """Save plot to HTML file.
        
        Args:
            file_path: Output file path
            show_molecules: Show molecular structures on hover
        """
        output_file(file_path)
        self.df["mol_id"] = range(len(self.df))
        source = ColumnDataSource(self.df)
        p = figure(
            width=1200, height=1000,
            tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save,hover",
            title="Chemical Space", tooltips=None
        )
        legend_items = []
        for label, color in self.color_map.items():
            mask = (self.df["dataset" if self.color_by == "dataset" else "cluster"] == label)
            scatter = p.scatter(
                x=self.df.loc[mask, "x"], 
                y=self.df.loc[mask, "y"],
                size=self.point_size,
                color=color,
                alpha=0.6,
                legend_label=str(label)
            )
            legend_items.append((str(label), [scatter]))

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        hover_info = ["mol_id"]
        if self.color_by == "dataset":
            hover_info.append("dataset")
        else:
            hover_info.append("cluster")

        if show_molecules:
            p = MolBokeh().add_molecule(
                fig=p,
                source=source,
                smilesColName='smiles',
                hoverAdditionalInfo=hover_info,
                molSize=(150,100)
            )
        save(p)
        print("Plot saved to", file_path)


    def find_optimal_k(self, X: np.ndarray, max_k: int = 15) -> int:
        """Find optimal number of clusters using multiple metrics.
        
        Args:
            X: Input coordinate matrix
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters based on voting
        """
        print("Searching optimal k...")
        ks, inertia, sil, ch, db = [], [], [], [], []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            ks.append(k)
            inertia.append(km.inertia_)
            sil.append(silhouette_score(X, labels))
            ch.append(calinski_harabasz_score(X, labels))
            db.append(davies_bouldin_score(X, labels))
        scores = pd.DataFrame({
            "k": ks,
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        })
        print(scores)
        inertia_elbow = np.argmax(np.diff(np.diff(inertia))) + 3
        sil_best = scores.loc[scores["silhouette"].idxmax(), "k"]
        ch_best = scores.loc[scores["calinski_harabasz"].idxmax(), "k"]
        db_best = scores.loc[scores["davies_bouldin"].idxmin(), "k"]
        votes = pd.Series([inertia_elbow, sil_best, ch_best, db_best]).value_counts()
        k_opt = votes.idxmax()
        print(f"Elbow={inertia_elbow}, Silhouette={sil_best}, CH={ch_best}, DB={db_best} -> chosen k={k_opt}")
        return int(k_opt)
