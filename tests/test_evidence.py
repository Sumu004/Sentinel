import hashlib
import sqlite3
from pathlib import Path

from evidence.custody import CustodyLog
from evidence.merkle import MerkleTree


def test_merkle_tree_all_leaves_verify():
    leaves = [hashlib.sha256(f"clip{i}".encode()).hexdigest() for i in range(5)]
    tree = MerkleTree(leaves)
    for leaf in leaves:
        proof = tree.prove(leaf)
        assert proof.verify(tree.root)


def test_merkle_tree_rejects_wrong_root():
    leaves = [hashlib.sha256(f"clip{i}".encode()).hexdigest() for i in range(4)]
    tree = MerkleTree(leaves)
    proof = tree.prove(leaves[0])
    assert not proof.verify("f" * 64)


def test_merkle_tree_single_leaf():
    leaf = hashlib.sha256(b"only-clip").hexdigest()
    tree = MerkleTree([leaf])
    assert tree.root
    proof = tree.prove(leaf)
    assert proof.verify(tree.root)


def test_custody_log_records_and_verifies(tmp_path: Path):
    log = CustodyLog(db_path=tmp_path / "custody.db")
    log.record("clip1.mp4", "captured")
    log.record("clip1.mp4", "signed")
    log.record("clip1.mp4", "viewed", actor="operator1")

    history = log.history("clip1.mp4")
    assert len(history) == 3
    assert history[-1].actor == "operator1"
    assert log.verify_chain()


def test_custody_log_detects_tampering(tmp_path: Path):
    db_path = tmp_path / "custody.db"
    log = CustodyLog(db_path=db_path)
    log.record("clip1.mp4", "captured")
    log.record("clip1.mp4", "signed")
    assert log.verify_chain()

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE custody SET actor = 'attacker' WHERE seq = 1")
    conn.commit()
    conn.close()

    tampered_log = CustodyLog(db_path=db_path)
    assert not tampered_log.verify_chain()
