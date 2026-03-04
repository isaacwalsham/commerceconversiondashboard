from commerceconversiondashboard.data import bundled_dataset_path, list_bundled_datasets


def test_bundled_datasets_exist_and_resolve_paths():
    datasets = list_bundled_datasets()

    assert len(datasets) >= 1

    first = datasets[0]
    assert first.path.exists()
    assert bundled_dataset_path(first.dataset_id).exists()
