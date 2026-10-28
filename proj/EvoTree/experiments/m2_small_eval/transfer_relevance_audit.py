"""Frozen development probes for workflow relevance and local state alignment.

Labels below are only read by the scorer. These are diagnostic examples, not
official task success measurements or held-out evidence.
"""
from __future__ import annotations

from evotree.situation_summary import SituationSummary


def summary(sentence, unknowns=()):
    return SituationSummary(sentence, (sentence,), tuple(unknowns))


def state_probes():
    """Predeclared positive, negative and insufficient-evidence cases."""
    color_goal = 'Set every text box on slide 5 to yellow.'
    old_goal = 'Set three text boxes on slide 1 to yellow, red and green in order.'
    return [
        dict(id='color-page-and-count-change', expected='compatible',
             current_goal=color_goal, historical_goal=old_goal,
             current=summary('The target slide 5 is active. The first target text box is selected; its measured font color is black, different from its required yellow.'),
             historical=summary('The target slide 1 is active. The first target text box is selected; its measured font color is black, different from its required yellow.')),
        dict(id='color-target-value-change', expected='compatible',
             current_goal='Set the selected text box to blue.', historical_goal='Set the selected text box to red.',
             current=summary('The target text box is selected with measured font color black, not its required blue.'),
             historical=summary('The target text box is selected with measured font color white, not its required red.')),
        dict(id='different-repetition-index', expected='compatible',
             current_goal='Color all four text boxes yellow.', historical_goal='Color all two text boxes green.',
             current=summary('The third target text box is selected and black, not yellow. Earlier text boxes have the requested yellow color.'),
             historical=summary('The second target text box is selected and black, not green. Earlier text boxes have the requested green color.')),
        dict(id='text-vs-background-property', expected='conflict',
             current_goal='Set text color to yellow.', historical_goal='Set slide background to yellow.',
             current=summary('The selected object is a text box with black font. The required property is its font color.'),
             historical=summary('The selected object is the slide page. The required property is the page background color.')),
        dict(id='source-vs-destination-role', expected='conflict',
             current_goal='Copy source values into a destination sheet.', historical_goal='Copy source values into a destination sheet.',
             current=summary('The blank destination sheet is active. Source values are on another sheet.'),
             historical=summary('The populated source sheet is active. The destination is another blank sheet.')),
        dict(id='known-complete-vs-incomplete', expected='conflict',
             current_goal='Set selected text to yellow.', historical_goal='Set selected text to red.',
             current=summary('The selected target text has measured color yellow, which satisfies its goal.'),
             historical=summary('The selected target text has measured color black, which does not satisfy its red goal.')),
        dict(id='missing-current-property', expected='unknown',
             current_goal=color_goal, historical_goal=old_goal,
             current=summary('Target slide 5 is active with two text boxes.', ['Text colors and local completion are unobserved.']),
             historical=summary('Target slide 1 is active and the first target text box is selected with black font, not yellow.')),
        dict(id='successful-call-not-proof', expected='unknown',
             current_goal='Color the selected target blue.', historical_goal='Color the selected target red.',
             current=summary('A blue text-color call returned without error.', ['Current target text color is unobserved.']),
             historical=summary('The selected target text has measured red font, satisfying its goal.')),
        dict(id='missing-selection', expected='unknown',
             current_goal='Format the selected source cell.', historical_goal='Format the selected source cell.',
             current=summary('A worksheet is open.', ['Active sheet role and selected cell are unobserved.']),
             historical=summary('The source sheet is active with source cell B2 selected and its required formatting not applied.')),
        dict(id='copy-destination-range-variation', expected='compatible',
             current_goal='Copy source A1:B4 into destination D1:E4.', historical_goal='Copy source C1:D8 into destination H1:I8.',
             current=summary('The populated source range is selected. The destination range is measured blank and no copy has occurred.'),
             historical=summary('The populated source range is selected. The destination range is measured blank and no copy has occurred.')),
        dict(id='blocked-dialog', expected='conflict',
             current_goal='Edit the target document.', historical_goal='Edit the target document.',
             current=summary('A modal confirmation dialog blocks the target document.'),
             historical=summary('The target document has focus with no blocking dialog.')),
        dict(id='different-content-same-local-phase', expected='compatible',
             current_goal='Make a selected title bold.', historical_goal='Make a selected title bold.',
             current=summary('The title text Sales is selected; measured font weight is normal, not bold.'),
             historical=summary('The title text Research is selected; measured font weight is normal, not bold.')),
    ]


def real_candidate_review(rows):
    """Review the 27 archived yellow comparisons without inferring hidden color.

    The shared color workflow is relevant reference material. Its exact local
    action suitability remains unknown because current colors are not measured.
    """
    result = []
    for row in rows:
        for index, comparison in enumerate(row.get('summary_comparisons', [])):
            template = comparison['template']
            is_color = template.startswith('Change_the_text_color_')
            background = template.startswith('Move_to_slide_')
            verdict = 'unknown' if is_color else 'conflict'
            reason = ('Shared text-color workflow is transferable reference material; current target '
                      'color and local completion are not measured, so executable state compatibility '
                      'cannot be confirmed. Page number or object count alone is not a conflict.'
                      if is_color else
                      'Different target property: slide background or font family, rather than text color.')
            result.append(dict(id=f"turn-{row['turn']}-candidate-{index}", turn=row['turn'],
                template=template, node=comparison['node'], action=comparison['action'],
                old_verdict=comparison['verdict'], old_reason=comparison['reason'],
                workflow_relevance='related' if is_color else 'unrelated',
                expected_local_verdict=verdict, reviewer_reason=reason,
                current=row['current_situation'], historical=comparison['historical']))
    return result
