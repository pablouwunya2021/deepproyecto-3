import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


@pytest.fixture(scope="session")
def cliente(tmp_path_factory):
    os.environ["LUMEN_DB"] = str(tmp_path_factory.mktemp("db") / "pruebas.sqlite")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
