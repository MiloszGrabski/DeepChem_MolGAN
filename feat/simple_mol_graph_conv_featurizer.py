from typing import List, Tuple
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot

def _construct_atom_feature(
    atom: RDKitAtom) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.
    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
    Returns
    -------
    np.ndarray
    A one-hot vector of the atom feature.
    """
    atom_type = get_atom_type_one_hot(atom)
    return atom_type


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.
    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
    Returns
    -------
    np.ndarray
    A one-hot vector of the bond feature.
    """
    bond_type = get_bond_type_one_hot(bond)
    return bond_type

class SimpleMolGraphConvFeaturizer(MolecularFeaturizer):
    """This class is a simple featurizer of general graph convolution networks for molecules.
    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = SimpleMolGraphConvFeaturizer()
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11
    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.
    Notes
    -----
    This class requires RDKit to be installed.
    """

    def __init__(self,
               use_edges: bool = True):
        """
        Parameters
        ----------
        use_edges: bool, default True
          Whether to use edge features or not.
        """
        try:
            from rdkit.Chem import AllChem  # noqa
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")

        self.use_edges = use_edges

    def _featurize(self, mol: RDKitMol) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphData
          A molecule graph with some features.
        """
        # construct atom (node) feature
        atom_features = [_construct_atom_feature(atom) for atom in mol.GetAtoms()]
        

        # construct edge (bond) index
        src, dest = [], []
        for bond in mol.GetBonds():
          # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.append([start, end])
            dest.append([end, start])
        edge_index = [src, dest]


        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            bond_features = []
            for bond in mol.GetBonds():
                bond_features.append(_construct_bond_feature(bond))
               

                
        return atom_features, edge_index, bond_features