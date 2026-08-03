import pytest
from src.data.loader import generate_synthetic, train_test_split


def test_generate_synthetic_shape():
    df = generate_synthetic(n=100)
    assert len(df) == 100
    assert "ds" in df.columns and "y" in df.columns


def test_train_test_split():
    df = generate_synthetic(n=100)
    train, test = train_test_split(df, test_periods=20)
    assert len(train) == 80
    assert len(test) == 20
    assert train["ds"].max() < test["ds"].min()
