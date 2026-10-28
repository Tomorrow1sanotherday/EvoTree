import pytest

from evotree.office_observation import office_observation_command


@pytest.mark.parametrize('app', ['libreoffice_calc','libreoffice_impress'])
def test_observation_commands_are_task_independent_readers(app):
    command = office_observation_command(app)
    compile(command, app, 'exec')
    assert not any(token in command for token in ('setActiveSheet(', 'setString(', 'setPropertyValue(',
                                                  'Extract43', 'Records23', 'source_role', 'target_role'))


def test_impress_reads_real_font_runs_and_slide_identity():
    command = office_observation_command('libreoffice_impress')
    assert "getPropertyValue('CharHeight')" in command
    assert "'current_slide_index'" in command
    assert office_observation_command('unsupported') is None
