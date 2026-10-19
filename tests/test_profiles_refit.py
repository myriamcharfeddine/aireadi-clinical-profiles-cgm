"""D3: rebuild the clinical profiles in all four strata from the data and compare with the frozen record."""
import numpy as np
import pandas as pd

from cgmprofiles import io, profiles
from cgmprofiles.config import results_dir, restricted_dir

from conftest import needs_data


@needs_data
def test_refit_reproduces_frozen_centroids_and_labels():
    split, clinical = io.load_split(), io.load_clinical_frame()
    labels, models = profiles.build_profiles(clinical, split)
    centroid_dir = results_dir("profile_derivation", "centroids")
    assert max(profiles.compare_with_frozen_centroids(models, centroid_dir).values()) == 0.0
    for stratum in io.STRATA:
        frozen = profiles.frozen_model_from_files(stratum, models[stratum], centroid_dir)
        sub = labels[labels.canonical_stratum == stratum]
        refit_labels = sub.label.to_numpy()
        frozen_labels = profiles.assign_to_frozen_centroids(profiles.transform(sub, frozen), frozen).label.to_numpy()
        assert int((refit_labels != frozen_labels).sum()) == 0, stratum
    saved = restricted_dir() / "profile_labels.parquet"
    if saved.exists():
        previous = pd.read_parquet(saved)
        assert np.array_equal(previous.label.to_numpy(), labels.label.to_numpy())
