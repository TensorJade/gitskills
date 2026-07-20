from src.pipeline import maintenance_advice, risk_probability


def test_risk_probability_decreases_with_rul() -> None:
    assert risk_probability(10) > risk_probability(80)


def test_maintenance_advice_levels() -> None:
    assert maintenance_advice(10)[0] == "严重"
    assert maintenance_advice(25)[0] == "高风险"
    assert maintenance_advice(45)[0] == "关注"
    assert maintenance_advice(90)[0] == "正常"
