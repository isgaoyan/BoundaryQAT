"""阶段二固定校准清单生成规则测试。"""

import pytest

from scripts.create_calibration_manifest import select_subject_balanced_records


def make_records(subject_count: int, samples_per_subject: int) -> list[dict[str, str]]:
    """构造只含 LPW 训练样本的测试记录。"""
    return [
        {"sample_id": f"lpw__{subject:02d}_{index:03d}", "subject": f"{subject:02d}", "source": "lpw"}
        for subject in range(subject_count)
        for index in range(samples_per_subject)
    ]


def test_calibration_selection_is_reproducible_and_subject_balanced() -> None:
    """验证同一随机种子结果稳定，且整轮抽样覆盖全部受试者。"""
    records = make_records(subject_count=4, samples_per_subject=10)

    first = select_subject_balanced_records(records, sample_count=12, seed=7)
    second = select_subject_balanced_records(records, sample_count=12, seed=7)

    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert {item["subject"] for item in first} == {"00", "01", "02", "03"}
    assert {subject: sum(item["subject"] == subject for item in first) for subject in {"00", "01", "02", "03"}} == {
        "00": 3,
        "01": 3,
        "02": 3,
        "03": 3,
    }


def test_calibration_selection_rejects_non_lpw_source() -> None:
    """验证辅助数据不会被误用于正式 PTQ 校准。"""
    records = make_records(subject_count=2, samples_per_subject=2)
    records[0]["source"] = "hmep"

    with pytest.raises(ValueError, match="只包含 LPW"):
        select_subject_balanced_records(records, sample_count=2, seed=7)
