import pytest

from skat.rewards import outcome_features, victory_bonus, victory_status


@pytest.mark.parametrize("points", [61, 70, 80, 90])
def test_same_winning_tier_has_no_point_bonus(points):
    assert outcome_features(points, 120 - points) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("points", [31, 40, 50, 60])
def test_same_losing_tier_has_no_point_penalty(points):
    assert outcome_features(points, 120 - points) == (-1.0, 0.0, 0.0)


def test_retains_existing_asymmetric_thresholds():
    assert outcome_features(60, 60)[0] == -1.0
    assert outcome_features(61, 59)[0] == 1.0
    assert outcome_features(90, 30)[1] == 0.0
    assert outcome_features(91, 29)[1] == 1.0
    assert outcome_features(30, 90)[1] == -1.0
    assert outcome_features(31, 89)[1] == 0.0


def test_schwarz_features_and_overbid():
    assert outcome_features(120, 0) == (1.0, 1.0, 1.0)
    assert outcome_features(0, 120) == (-1.0, -1.0, -1.0)
    assert outcome_features(100, 20, overbid=True) == (-1.0, 0.0, 0.0)


def test_null_depends_on_tricks_not_points():
    assert outcome_features(20, 100, is_null=True) == (1.0, 0.0, 0.0)
    assert outcome_features(0, 120, is_null=True, soloist_took_trick=True) == (
        -1.0,
        0.0,
        0.0,
    )


def test_bonus_is_positive_even_without_schneider_and_increases_with_points():
    assert 0 < victory_bonus(61, True) < victory_bonus(80, True)
    assert victory_bonus(91, True) - victory_bonus(90, True) > 0.2
    assert victory_bonus(90, False) - victory_bonus(89, False) > 0.2
    assert victory_bonus(120, True) == 1


def test_victory_must_be_secured_by_actual_points():
    assert victory_status(55, 40) == 0
    assert victory_status(61, 40) == 1
    assert victory_status(40, 60) == -1
    assert victory_status(100, 20, overbid=True) == -1
