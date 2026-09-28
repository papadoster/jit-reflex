import sys

sys.path.insert(0, "src")
import a2c2_fix  # noqa: E402
import probe  # noqa: E402


def test_experts_cover_every_level_and_match_the_mac_diagnostic():
    assert sorted(a2c2_fix.EXPERTS) == sorted(probe.LEVELS)
    # docs/results/b2b5-checks.md (a2) used these: the spec's rule must pick the same experts
    assert a2c2_fix.EXPERTS["worlds/l/trampoline.json"] == (2, 840)
    assert a2c2_fix.EXPERTS["worlds/l/mjc_walker.json"] == (7, 580)
    assert a2c2_fix.EXPERTS["worlds/l/car_launch.json"] == (0, 920)


def test_expert_path_and_url():
    p, u = a2c2_fix.expert_path("worlds/l/trampoline.json"), a2c2_fix.expert_url("worlds/l/trampoline.json")
    assert p == "checkpoints/expert/seed_2_step_840_worlds_l_trampoline.pkl"
    assert u == "https://storage.googleapis.com/rtc-assets/expert/seed_2/840/policies/worlds_l_trampoline.pkl"


def test_latency_writes_head_numbers(tmp_path):
    a2c2_fix.latency(out_dir=str(tmp_path), repeats=3, warmup=1)
    info = __import__("json").loads((tmp_path / "head_latency.json").read_text())
    assert info["in_dim"] == 4 * 679 + 6 + 6 + 2
    assert info["head_ms_cpu"] > 0 and info["head_ms_gpu"] > 0
    # 2 x (2730 x 512 + 512 x 512 + 512 x 6) multiply-adds, LayerNorm and ReLU on top: about 3.3 MFLOP
    assert 0.003 < info["head_gflop"] < 0.004
