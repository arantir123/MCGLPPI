from .chebnet import ChebyshevConvolutionalNetwork
from .gcn import GraphConvolutionalNetwork, RelationalGraphConvolutionalNetwork
from .gat import GraphAttentionNetwork
from .gin import GraphIsomorphismNetwork
from .schnet import SchNet
from .mpnn import MessagePassingNeuralNetwork
from .neuralfp import NeuralFingerprint
from .infograph import InfoGraph, MultiviewContrast
from .flow import GraphAutoregressiveFlow
from .esm import EvolutionaryScaleModeling
from .embedding import TransE, DistMult, ComplEx, RotatE, SimplE
from .neurallp import NeuralLogicProgramming
from .kbgat import KnowledgeBaseGraphAttentionNetwork
from .cnn import ProteinConvolutionalNetwork, ProteinResNet
from .lstm import ProteinLSTM
from .bert import ProteinBERT
from .statistic import Statistic
from .physicochemical import Physicochemical
from .gearnet import GeometryAwareRelationalGraphNeuralNetwork
# extra hyperparameter
from cg_steps.cg_models import CG22_GeometryAwareRelationalGraphNeuralNetwork, CG22_GearNetIEConv

# alias
ChebNet = ChebyshevConvolutionalNetwork
GCN = GraphConvolutionalNetwork
GAT = GraphAttentionNetwork
RGCN = RelationalGraphConvolutionalNetwork
GIN = GraphIsomorphismNetwork
MPNN = MessagePassingNeuralNetwork
NFP = NeuralFingerprint
GraphAF = GraphAutoregressiveFlow
ESM = EvolutionaryScaleModeling
NeuralLP = NeuralLogicProgramming
KBGAT = KnowledgeBaseGraphAttentionNetwork
ProteinCNN = ProteinConvolutionalNetwork
GearNet = GeometryAwareRelationalGraphNeuralNetwork
# extra ones
CG22_GearNet = CG22_GeometryAwareRelationalGraphNeuralNetwork
CG22_GearNetIEConv = CG22_GearNetIEConv

__all__ = [
    "ChebyshevConvolutionalNetwork", "GraphConvolutionalNetwork", "RelationalGraphConvolutionalNetwork",
    "GraphAttentionNetwork", "GraphIsomorphismNetwork", "SchNet", "MessagePassingNeuralNetwork",
    "NeuralFingerprint",
    "InfoGraph", "MultiviewContrast",
    "GraphAutoregressiveFlow",
    "EvolutionaryScaleModeling", "ProteinConvolutionalNetwork", "GeometryAwareRelationalGraphNeuralNetwork",
    "Statistic", "Physicochemical",
    "TransE", "DistMult", "ComplEx", "RotatE", "SimplE",
    "NeuralLogicProgramming", "KnowledgeBaseGraphAttentionNetwork",
    "ChebNet", "GCN", "GAT", "RGCN", "GIN", "MPNN", "NFP",
    "GraphAF", "ESM", "NeuralLP", "KBGAT",
    "ProteinCNN", "ProteinResNet", "ProteinLSTM", "ProteinBERT", "GearNet",
    "CG22_GearNet", "CG22_GeometryAwareRelationalGraphNeuralNetwork", "CG22_GearNetIEConv"
]