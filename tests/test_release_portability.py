"""Release configuration must use checkpoint paths that survive relocation."""

from pathlib import Path

from zbot_rl_mjlab.experiments.cards import load_card
from zbot_rl_mjlab.teachers import SNAKE_CHECKPOINT


def test_deployment_teacher_uses_the_same_archive_without_host_absolute_path():
    card = load_card('universal_snake_leg_deploy')
    configured = Path(card['agent']['algorithm']['teacher_checkpoints']['snake'])
    assert not configured.is_absolute()
    project = Path(__file__).resolve().parents[1]
    assert (project / configured).resolve() == Path(SNAKE_CHECKPOINT).resolve()
    assert (project / configured).is_file()
