from lightning import LightningDataModule
import torch
from random import Random
import torch_geometric as tg
import hashlib
from collections import defaultdict
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import warnings
import pandas as pd
from pathlib import Path
import numpy as np
import zipfile
import requests

ADMET_BENCHMARK_METADATA = {
    # Absorption
    "caco2_wang": {
        "task_type": "regression",
        "metric": "MAE",
        "num_outputs": 1,
        "unit": "cm/s",
        "size": 906,
        "group": "Absorption",
        "display_name": "Caco2",
    },
    "hia_hou": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 578,
        "group": "Absorption",
        "display_name": "HIA",
    },
    "pgp_broccatelli": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 1212,
        "group": "Absorption",
        "display_name": "Pgp",
    },
    "bioavailability_ma": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 640,
        "group": "Absorption",
        "display_name": "Bioav",
    },
    "lipophilicity_astrazeneca": {
        "task_type": "regression",
        "metric": "MAE",
        "num_outputs": 1,
        "unit": "log-ratio",
        "size": 4200,
        "group": "Absorption",
        "display_name": "Lipo",
    },
    "solubility_aqsoldb": {
        "task_type": "regression",
        "metric": "MAE",
        "num_outputs": 1,
        "unit": "log mol/L",
        "size": 9982,
        "group": "Absorption",
        "display_name": "AqSol",
    },
    # Distribution
    "bbb_martins": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 1975,
        "group": "Distribution",
        "display_name": "BBB",
    },
    "ppbr_az": {
        "task_type": "regression",
        "metric": "MAE",
        "num_outputs": 1,
        "unit": "%",
        "size": 1797,
        "group": "Distribution",
        "display_name": "PPBR",
    },
    "vdss_lombardo": {
        "task_type": "regression",
        "metric": "Spearman",
        "num_outputs": 1,
        "unit": "L/kg",
        "size": 1130,
        "group": "Distribution",
        "display_name": "VDss",
    },
    # Metabolism
    "cyp2c9_veith": {
        "task_type": "classification",
        "metric": "AUPRC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 12092,
        "group": "Metabolism",
        "display_name": "CYP2C9 Inhibition",
    },
    "cyp2d6_veith": {
        "task_type": "classification",
        "metric": "AUPRC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 13130,
        "group": "Metabolism",
        "display_name": "CYP2D6 Inhibition",
    },
    "cyp3a4_veith": {
        "task_type": "classification",
        "metric": "AUPRC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 12328,
        "group": "Metabolism",
        "display_name": "CYP3A4 Inhibition",
    },
    "cyp2c9_substrate_carbonmangels": {
        "task_type": "classification",
        "metric": "AUPRC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 666,
        "group": "Metabolism",
        "display_name": "CYP2C9 Substrate",
    },
    "cyp2d6_substrate_carbonmangels": {
        "task_type": "classification",
        "metric": "AUPRC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 664,
        "group": "Metabolism",
        "display_name": "CYP2D6 Substrate",
    },
    "cyp3a4_substrate_carbonmangels": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 667,
        "group": "Metabolism",
        "display_name": "CYP3A4 Substrate",
    },
    # Excretion
    "half_life_obach": {
        "task_type": "regression",
        "metric": "Spearman",
        "num_outputs": 1,
        "unit": "hr",
        "size": 667,
        "group": "Excretion",
        "display_name": "Half Life",
    },
    "clearance_hepatocyte_az": {
        "task_type": "regression",
        "metric": "Spearman",
        "num_outputs": 1,
        "unit": "uL.min-1.(10^6 cells)-1",
        "size": 1020,
        "group": "Excretion",
        "display_name": "CL-Hepa",
    },
    "clearance_microsome_az": {
        "task_type": "regression",
        "metric": "Spearman",
        "num_outputs": 1,
        "unit": "mL.min-1.g-1",
        "size": 1102,
        "group": "Excretion",
        "display_name": "CL-Micro",
    },
    # Toxicity
    "ld50_zhu": {
        "task_type": "regression",
        "metric": "MAE",
        "num_outputs": 1,
        "unit": "log(1/(mol/kg))",
        "size": 7385,
        "group": "Toxicity",
        "display_name": "LD50",
    },
    "herg": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 648,
        "group": "Toxicity",
        "display_name": "hERG",
    },
    "ames": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 7255,
        "group": "Toxicity",
        "display_name": "Ames",
    },
    "dili": {
        "task_type": "classification",
        "metric": "AUROC",
        "num_outputs": 1,  # Binary classification
        "unit": "%",
        "size": 475,
        "group": "Toxicity",
        "display_name": "DILI",
    },
}

ADMET_URL = "https://dataverse.harvard.edu/api/access/datafile/4426004"
ADMET_ARCHIVE_NAME = "admet_group.zip"
ADMET_ARCHIVE_MD5 = "e3e4dda2f7f62268448fc813ceefccd5"
CSV_COLUMN_MAP = {
    "idx": "Drug_ID",
    "smiles": "Drug",
    "label": "Y",
}
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']
NUMBER2ATOM_TYPES = {
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S',
    17: 'Cl', 35: 'Br', 53: 'I', 1: 'H'
}
HYBRIDIZATION_TYPES = [
    HybridizationType.SP, HybridizationType.SP2,
    HybridizationType.SP3, HybridizationType.SP3D,
    HybridizationType.SP3D2
]

class ADMETDataModule(LightningDataModule):
    """
    Data module for ADMET datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        batch_size: int = 64,
        seed: int = 42,
        split_type: str = "random",
        val_fraction: float = 0.1,
        with_hydrogen: bool = False,
        kekulize: bool = False,
        with_descriptors: bool = False,
        with_fingerprints: bool = False,
        use_node2vec: bool = False,
        node2vec_device: str = "cpu",
        only_use_atom_number: bool = False,
    ):
        """
        Initialize the ADMETDataModule.

        Args:
            dataset_name (str): Name of the dataset.
            data_dir (str): Directory to store the data. (default: "./data")
            batch_size (int): Batch size for data loaders. (default: 64)
            seed (int): Random seed for reproducibility. (default: 42)
            split_type (str): Type of data split ("random" or "scaffold"). (default: "random")
            val_fraction (float): Fraction of data to use for validation. (default: 0.1)
            with_hydrogen (bool): If set to True, will store hydrogens in the molecule graph. (default: False)
            kekulize (bool): If set to True, converts aromatic bonds to single/double bonds. (default: False)
            with_descriptors (bool): If set to True, will include molecular descriptors. (default: False)
            with_fingerprints (bool): If set to True, will include molecular fingerprints. (default: False)
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.split_type = split_type
        self.val_fraction = val_fraction
        self.with_hydrogen = with_hydrogen
        self.kekulize = kekulize
        self.with_descriptors = with_descriptors
        self.with_fingerprints = with_fingerprints
        self.use_node2vec = use_node2vec
        self.node2vec_device = node2vec_device
        self.only_use_atom_number = only_use_atom_number

        # Setup paths and metadata
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.admet_zip_archive = self.data_dir / ADMET_ARCHIVE_NAME
        self.admet_path = self.data_dir / "admet_group"
        self.dataset_path = self.admet_path / self.dataset_name
        self.train_val_csv = self.dataset_path / "train_val.csv"
        self.test_csv = self.dataset_path / "test.csv"
        self.dataset_metadata = ADMET_BENCHMARK_METADATA[self.dataset_name]
        

    def prepare_data(self):
        if not self.admet_zip_archive.exists():
            response = requests.get(ADMET_URL)
            with open(self.admet_zip_archive, "wb") as f:
                f.write(response.content)
        # Verify the MD5 checksum
        md5_hash = hashlib.md5()
        with open(self.admet_zip_archive, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        if md5_hash.hexdigest() != ADMET_ARCHIVE_MD5:
            raise ValueError(
                f"MD5 checksum mismatch for {self.admet_zip_archive}. Expected {ADMET_ARCHIVE_MD5}, got {md5_hash.hexdigest()}"
            )
        if not self.admet_path.exists():
            with zipfile.ZipFile(self.admet_zip_archive, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)
        if not (self.train_val_csv.exists() and self.test_csv.exists()):
            raise FileNotFoundError(
                f"Dataset files not found. Please check the path: {self.dataset_path}"
            )

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            train_val_df = pd.read_csv(
                self.train_val_csv, header=0, names=CSV_COLUMN_MAP
            )
            smiles_list = train_val_df["smiles"].tolist()
            target_list = train_val_df["label"].tolist()
            if self.val_fraction > 0:
                train_fraction = 1 - self.val_fraction
                if self.split_type == "random":
                    random = Random(self.seed)
                    all_indices = list(range(len(smiles_list)))
                    random.shuffle(all_indices)
                    split_index = int(len(smiles_list) * (train_fraction))
                    train_indices = all_indices[:split_index]
                    val_indices = all_indices[split_index:]
                elif self.split_type == "scaffold":
                    train_indices, val_indices = create_scaffold_split_indices(
                        smiles_list,
                        seed=self.seed,
                        split_fractions=[train_fraction, self.val_fraction],
                    )
            else:
                train_indices = list(range(len(smiles_list)))
                val_indices = []

            self.train_dataset = self.create_dataset(
                [smiles_list[i] for i in train_indices],
                [target_list[i] for i in train_indices],
            )
            self.val_dataset = self.create_dataset(
                [smiles_list[i] for i in val_indices],
                [target_list[i] for i in val_indices],
            )
            # If using Node2Vec, compute embeddings for the training dataset
            if self.use_node2vec:
                self.train_dataset = self.append_node2vec_embeddings(self.train_dataset)
                if self.val_dataset:
                    self.val_dataset = self.append_node2vec_embeddings(self.val_dataset)

        elif stage == "test":
            test_df = pd.read_csv(self.test_csv, header=0, names=CSV_COLUMN_MAP)
            self.test_dataset = self.create_dataset(
                test_df["smiles"].tolist(), test_df["label"].tolist()
            )
            # If using Node2Vec, compute embeddings for the test dataset
            if self.use_node2vec:
                self.test_dataset = self.append_node2vec_embeddings(self.test_dataset)
        else:
            raise ValueError(f"Invalid stage: {stage}. Use 'fit' or 'test'.")
        
        
    def atom_features(self, atom_number):
        """Generate a feature vector for each atom."""
        features = []

        # Atom type (one-hot)
        atom_type = NUMBER2ATOM_TYPES.get(atom_number, 'X')  # 'X' for unknown
        atom_type_onehot = [int(atom_type == s) for s in ATOM_TYPES]
        features.extend(atom_type_onehot)


        return torch.tensor(features, dtype=torch.float)

    def create_dataset(
        self, smiles_list: list[str], target_list: list[float | int]
    ) -> tg.data.InMemoryDataset:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors, Crippen

        RDLogger.DisableLog("rdApp.*")  # type: ignore

        # float if regression, int if classification
        dataset_dtype = (
            torch.float
            if self.dataset_metadata["task_type"] == "regression"
            else torch.long
        )

        data_list = []
        for smiles, target in zip(smiles_list, target_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            if self.with_hydrogen:
                mol = Chem.AddHs(mol)
            if self.kekulize:
                Chem.Kekulize(mol)

            data = tg.utils.smiles.from_rdmol(mol)
            # Convert atomic number (feature 0) to one-hot encoding
            atoms = data.x[:, 0].tolist()
            atom_onehot = torch.stack(
                [self.atom_features(atom) for atom in atoms], dim=0
            )
            if self.only_use_atom_number:
                # If only using atom number, keep only the first column (atom type)
                data.x = atom_onehot
            else:
                data.x = torch.hstack((atom_onehot, data.x[:, 1:]))  # Append one-hot features
            
            data.x = data.x.to(dtype=torch.float)
            data.smiles = smiles
            data.target = torch.tensor(target, dtype=dataset_dtype)

            if self.with_descriptors:
                data.descriptor = torch.tensor(
                    [
                        Descriptors.MolWt(mol),
                        Crippen.MolLogP(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.FractionCSP3(mol),
                        Descriptors.qed(mol),
                    ],
                    dtype=torch.float,
                ).unsqueeze(0)
            if self.with_fingerprints:
                data.fingerprint = torch.tensor(
                    list(
                        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
                    ),
                    dtype=torch.float,
                ).unsqueeze(0)

            data_list.append(data)

        dataset = tg.data.InMemoryDataset()
        dataset.data, dataset.slices = dataset.collate(data_list)
        return dataset

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise ValueError(
                "Training dataset not set up. Call setup(stage='fit') first."
            )
        return tg.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            raise ValueError(
                "Validation dataset not set up. Call setup(stage='fit') first."
            )
        return tg.loader.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            raise ValueError("Test dataset not set up. Call setup(stage='test') first.")
        return tg.loader.DataLoader(self.test_dataset, batch_size=self.batch_size)


def create_scaffold_split_indices(
    smiles_list: list[str],
    seed: int,
    split_fractions: list[float],
    allow_errors: bool = False,
) -> list[list[int]]:
    """Create a scaffold-based split and return lists of indices for each split.

    Args:
        smiles_list (list[str]): A list of SMILES strings to be split.
        seed (int): The random seed for reproducibility.
        split_fractions (list[float]): A list of fractions for each split.
                                      Must sum to 1.0. e.g., [0.8, 0.1, 0.1].
        allow_errors (bool): Whether to allow errors in SMILES parsing. Default is False.
                             If True, molecules that cannot be parsed will be assigned to their own scaffold.

    Returns:
        list[list[int]]: A list of lists, where each inner list contains the integer indices of the molecules for that split.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        raise ImportError("Please install rdkit to perform scaffold splitting.")

    if not abs(sum(split_fractions) - 1.0) < 1e-8:
        raise ValueError(f"Fractions must sum to 1, but got {sum(split_fractions)}")

    num_splits = len(split_fractions)
    random = Random(seed)

    scaffolds = defaultdict(set)
    for i, smiles in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except Exception:
            scaffolds[f"error_{i}"].add(i)

    num_errors = sum(
        len(scaffold) for scaffold in scaffolds.values() if "error_" in scaffold
    )
    if num_errors > 0 and not allow_errors:
        raise ValueError(
            "Some SMILES could not be parsed. Set allow_errors=True to assign them to their own scaffold."
        )
    elif num_errors > 0 and allow_errors:
        warnings.warn(
            f"{num_errors} SMILES could not be parsed. They will be assigned to their own scaffold."
        )

    # Sort scaffold groups by size (largest first) and shuffle
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=len, reverse=True)
    random.shuffle(scaffold_sets)

    # Calculate target split sizes
    total_molecules = len(smiles_list)
    split_sizes = [int(total_molecules * frac) for frac in split_fractions]
    split_sizes[-1] += total_molecules - sum(split_sizes[:-1])

    splits = [[] for _ in range(num_splits)]

    for s_set in scaffold_sets:
        target_split = -1  # if not split is found, assign to the last one
        for i in range(num_splits - 1):
            if len(splits[i]) + len(s_set) <= split_sizes[i]:
                target_split = i
                break

        splits[target_split].extend(s_set)

    # Warn if any split is more than 5% off from the target size and the report the difference in percentage
    for i, split in enumerate(splits):
        actual_split_fraction = len(split) / total_molecules
        if abs(actual_split_fraction - split_fractions[i]) >= 0.01:
            warnings.warn(
                f"Split {i} is {abs(actual_split_fraction - split_fractions[i]) * 100:.2f}% off from the target size."
            )

    return splits

