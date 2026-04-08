"""Tests for graph_data.mat_loader."""

import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pytest
import torch

from priorf_reasoner_slm.graph_data.mat_loader import (
    load_mat_dataset,
    compute_hsd,
    _detect_dataset,
    _sparse_to_edge_index,
)


def _make_yelp_mat(path: Path, n: int = 200, d: int = 32, fraud_ratio: float = 0.14):
    """Create a synthetic YelpChi .mat file for testing."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n, d)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    fraud_n = max(1, int(n * fraud_ratio))
    labels[rng.choice(n, fraud_n, replace=False)] = 1

    data_dict = {"features": features, "label": labels.reshape(-1, 1)}
    for key in ("net_rur", "net_rtr", "net_rsr"):
        rows = rng.choice(n, 800)
        cols = rng.choice(n, 800)
        data_dict[key] = sp.csr_matrix((np.ones(800), (rows, cols)), shape=(n, n))

    sio.savemat(str(path), data_dict)


def _make_amazon_mat(path: Path, n: int = 150, d: int = 25):
    """Create a synthetic Amazon .mat file for testing."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n, d)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    labels[:max(1, int(n * 0.1))] = 1

    data_dict = {"features": features, "label": labels.reshape(-1, 1)}
    for key in ("net_upu", "net_usu", "net_uvu"):
        rows = rng.choice(n, 600)
        cols = rng.choice(n, 600)
        data_dict[key] = sp.csr_matrix((np.ones(600), (rows, cols)), shape=(n, n))

    sio.savemat(str(path), data_dict)


class TestDetectDataset:
    def test_yelp(self, tmp_path):
        p = tmp_path / "test.mat"
        _make_yelp_mat(p)
        mat = sio.loadmat(str(p))
        assert _detect_dataset(mat) == "yelpchi"

    def test_amazon(self, tmp_path):
        p = tmp_path / "test.mat"
        _make_amazon_mat(p)
        mat = sio.loadmat(str(p))
        assert _detect_dataset(mat) == "amazon"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            _detect_dataset({"features": np.array([])})


class TestSparseToEdgeIndex:
    def test_removes_self_loops(self):
        adj = sp.csr_matrix(np.array([[1, 1], [1, 1]]))
        ei = _sparse_to_edge_index(adj)
        assert ei.shape[0] == 2
        # Only off-diagonal edges remain
        assert not torch.any(ei[0] == ei[1])


class TestComputeHSD:
    def test_empty_graph(self):
        x = torch.randn(10, 5)
        ei = torch.empty((2, 0), dtype=torch.long)
        hsd = compute_hsd(x, ei)
        assert hsd.shape == (10,)
        assert torch.all(hsd == 0)

    def test_shape_and_no_nan(self):
        x = torch.randn(10, 5)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
        hsd = compute_hsd(x, ei)
        assert hsd.shape == (10,)
        assert not torch.any(torch.isnan(hsd))


class TestLoadMatDataset:
    def test_yelp_loads(self, tmp_path):
        p = tmp_path / "YelpChi.mat"
        _make_yelp_mat(p)
        data = load_mat_dataset(p)
        assert data["dataset_name"] == "yelpchi"
        assert data["x"].shape[1] == 32
        assert data["y"].shape[0] == 200
        assert len(data["relations"]) == 3
        assert "RUR" in data["relations"]
        assert "RTR" in data["relations"]
        assert "RSR" in data["relations"]
        assert data["hsd"].shape == (200,)
        assert data["homo"].shape[0] == 2

    def test_amazon_loads(self, tmp_path):
        p = tmp_path / "Amazon.mat"
        _make_amazon_mat(p)
        data = load_mat_dataset(p)
        assert data["dataset_name"] == "amazon"
        assert data["x"].shape[1] == 25
        assert len(data["relations"]) == 3
        assert "UPU" in data["relations"]

    def test_node_ids(self, tmp_path):
        p = tmp_path / "YelpChi.mat"
        _make_yelp_mat(p)
        data = load_mat_dataset(p)
        expected = torch.arange(200)
        assert torch.equal(data["node_ids"], expected)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_mat_dataset("/nonexistent/path.mat")

    def test_hsd_invert(self, tmp_path):
        p = tmp_path / "YelpChi.mat"
        _make_yelp_mat(p)
        data_normal = load_mat_dataset(p)
        data_inverted = load_mat_dataset(p, hsd_invert=True)
        # Inverted HSD should be max - original
        expected = data_normal["hsd"].max() - data_normal["hsd"]
        assert torch.allclose(data_inverted["hsd"], expected)
