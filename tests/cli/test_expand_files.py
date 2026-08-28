from xvr.cli.register import _expand_files


def test_a_plain_file_passes_through(tmp_path):
    path = tmp_path / "a.dcm"
    path.touch()
    assert _expand_files([path]) == [path]


def test_a_directory_expands_to_its_sorted_dicoms(tmp_path):
    """`xvr register ... some/dir` registers every frame in the directory, in order."""
    for name in ["c.dcm", "a.dcm", "b.dcm"]:
        (tmp_path / name).touch()
    (tmp_path / "notes.txt").touch()
    assert [p.name for p in _expand_files([tmp_path])] == ["a.dcm", "b.dcm", "c.dcm"]


def test_files_and_directories_mix(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    (folder / "b.dcm").touch()
    loose = tmp_path / "a.dcm"
    loose.touch()
    assert [p.name for p in _expand_files([loose, folder])] == ["a.dcm", "b.dcm"]


def test_an_empty_directory_contributes_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert _expand_files([tmp_path / "empty"]) == []
