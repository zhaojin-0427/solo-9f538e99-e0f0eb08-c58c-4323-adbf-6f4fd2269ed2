import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.web import create_app  # noqa: E402
from app.service import Service  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture()
def svc(tmp_db):
    return Service(tmp_db, synchronous=True)


@pytest.fixture()
def client(tmp_db):
    app = create_app(tmp_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, app.extensions["service"]
    app.extensions["service"].close()


@pytest.fixture()
def svc(tmp_db):
    s = Service(tmp_db, synchronous=True)
    yield s
    s.close()


def make_csv(rows, headers=("图像号", "患者号", "类别")):
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return "\n".join(lines) + "\n"


def make_dup(edges):
    lines = ["图像号A,图像号B"]
    for a, b in edges:
        lines.append(f"{a},{b}")
    return "\n".join(lines) + "\n"


def upload(client, rows, quota, edges=None, dup_text=None):
    image_csv = make_csv(rows)
    if dup_text is None:
        dup_text = make_dup(edges or [])
    data = {
        "image_csv": (_bytes(image_csv), "images.csv"),
        "dup_csv": (_bytes(dup_text), "dups.csv"),
    }
    for cls, n in quota.items():
        data[f"quota::{cls}"] = str(n)
    return client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def _bytes(text):
    from io import BytesIO

    return BytesIO(text.encode("utf-8"))
