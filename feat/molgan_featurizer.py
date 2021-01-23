import logging
import numpy as np
import rdkit.Chem as Chem

from deepchem.utils.typing import RDKitBond, RDKitMol, List
from deepchem.feat.base_classes import MolecularFeaturizer

logger = logging.getLogger(__name__)


class GraphMatrix:
    """
    This is class used to store data for MolGAN neural networks.

    Parameters
    ----------
    node_features: np.ndarray
      Node feature matrix with shape [num_nodes, num_node_features]
    edge_features: np.ndarray,
      Edge feature matrix with shape [num_nodes, num_nodes]

    Returns
    -------
    graph: GraphMatrix
      A molecule graph with some features.
    """

    def __init__(self, adjacency_matrix: np.ndarray, node_features: np.ndarray):
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features


class MolGanFeaturizer(MolecularFeaturizer):
    """This class implements featurizer used with MolGAN de-novo molecular generation based on:
    `MolGAN: An implicit generative model for small molecular graphs`<https://arxiv.org/abs/1805.11973>`_.
    The default representation is in form of GraphMatrix object.
    It is wrapper for two matrices containing atom and bond type information.
    The class also provides reverse capabilities"""

    def __init__(
        self,
        max_atom_count: int = 9,
        kekulize: bool = True,
        bond_labels: List[RDKitBond] = None,
        atom_labels: List[int] = None,
    ):
        """
        Parameters
        ----------
        max_atom_count: int, default 9
            Maximum number of atoms used for creation of adjacency matrix.
            Molecules cannot have more atoms than this number; implicit hydrogens do not count.
        kekulize: bool, default True
            Should molecules be kekulized; solves number of issues with defeaturization when used.
        bond_labels: List[RDKitBond]
            List containing types of bond used for generation of adjacency matrix
        atom_labels: List[int]
            List of atomic numbers used for generation of node features
        """
        self.max_atom_count = max_atom_count
        self.kekulize = kekulize

        # bond labels
        if bond_labels is None:
            self.bond_labels = [
                Chem.rdchem.BondType.ZERO,
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ]
        else:
            self.bond_labels = bond_labels

        # atom labels
        if atom_labels is None:
            self.atom_labels = [0, 6, 7, 8, 9]  # C,N,O,F
        else:
            self.atom_labels = atom_labels

        # create bond encoders and decoders
        self.bond_encoder = {l: i for i, l in enumerate(self.bond_labels)}
        self.bond_decoder = {i: l for i, l in enumerate(self.bond_labels)}
        # create atom encoders and decoders
        self.atom_encoder = {l: i for i, l in enumerate(self.atom_labels)}
        self.atom_decoder = {i: l for i, l in enumerate(self.atom_labels)}

    def _featurize(self, mol: RDKitMol) -> GraphMatrix:
        """Calculate adjacency matrix and nodes features for RDKitMol.

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphMatrix
          A molecule graph with some features.
        """
        if self.kekulize:
            Chem.Kekulize(mol)

        A = np.zeros(shape=(self.max_atom_count, self.max_atom_count), dtype=np.float32)
        bonds = mol.GetBonds()

        begin, end = [b.GetBeginAtomIdx() for b in bonds], [
            b.GetEndAtomIdx() for b in bonds
        ]
        bond_type = [self.bond_encoder[b.GetBondType()] for b in bonds]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)
        X = np.array(
            [self.atom_encoder[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (self.max_atom_count - mol.GetNumAtoms()),
            dtype=np.int32,
        )
        graph = GraphMatrix(A, X)

        return graph if (degree > 0).all() else None

    def _defeaturize(
        self, graph_matrix: GraphMatrix, sanitize: bool = True, cleanup=True
    ) -> RDKitMol:
        """Recreate RDKitMol from GraphMatrix object. Same object needs to be used for featurization and defeaturization.

        Parameters
        ----------
        graph_matrix: GraphMatrix
            GraphMatrix object.
        sanitize: bool, default True
            Should RDKit sanitization be included in the process.
        cleanup: bool, default True
            Splits salts and removes compounds with "*" atom types

        Returns
        -------
        mol: RDKitMol object
            RDKitMol object representing molecule.
        """

        node_labels = graph_matrix.node_features
        edge_labels = graph_matrix.adjacency_matrix

        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(
                    int(start), int(end), self.bond_decoder[edge_labels[start, end]]
                )

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None

        if cleanup:
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles = max(smiles.split("."), key=len)
                if "*" not in smiles:
                    mol = Chem.MolFromSmiles(smiles)
                else:
                    mol = None
            except Exception:
                mol = None

        return mol

    def defeaturize(self, graphs, log_every_n=1000) -> np.ndarray:
        """Calculates molecules from correspoing GraphMatrix objects.
        Parameters
        ----------
        graphs: GraphMatrix / iterable
          GraphMatrix object or corresponding iterable
        log_every_n: int, default 1000
          Logging messages reported every `log_every_n` samples.
        Returns
        -------
        features: np.ndarray
          A numpy array containing RDKitMol objext.
        """
        # Special case handling of single molecule
        if isinstance(graphs, GraphMatrix):
            graphs = [graphs]
        else:
            # Convert iterables to list
            graphs = list(graphs)

        molecules = []
        for i, gr in enumerate(graphs):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)

            try:
                molecules.append(self._defeaturize(gr))
            except Exception as e:
                logger.warning(
                    "Failed to defeaturize datapoint %d, %s. Appending empty array",
                    i,
                    gr,
                )
                logger.warning("Exception message: {}".format(e))
                molecules.append(np.array([]))

        molecules = np.asarray(molecules)
        return molecules
