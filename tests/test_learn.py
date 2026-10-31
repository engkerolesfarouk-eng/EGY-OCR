import importlib
import os


def test_teach_and_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from arocr import learn
    importlib.reload(learn)
    assert learn.apply("لا يكنه ذلك") == "لا يكنه ذلك"
    learn.add("يكنه", "يمكنه")
    assert learn.apply("لا يكنه ذلك") == "لا يمكنه ذلك"
    assert learn.apply("يكنهم") == "يكنهم"              # whole words only
    learn.add("يكنه", "يمكنُه")                          # taught again: replaces
    assert learn.apply("يكنه") == "يمكنُه"
    assert len(learn.rules()) == 1
    assert os.path.exists(learn.path())
