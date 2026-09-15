import pytest

from scripts.prepare_stf import overlap_report, read_split


def test_split_preserves_frame_zero_padding_and_rejects_bad_identifiers(tmp_path):
    path = tmp_path/'split.txt'
    path.write_text('2018-02-06_14-25-51,00400\n')
    assert read_split(path)[0]['sample_id'] == '2018-02-06_14-25-51_00400'
    for text in ('../outside,00400\n', '', '2018-02-06_14-25-51,00400\n'*2):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_split(path)


def test_overlap_audit_distinguishes_duplicate_frames_from_shared_recordings():
    report = overlap_report({
        'train': ['2018-02-06_14-25-51_00400', '2018-02-06_14-25-51_00500'],
        'rain': ['2018-02-06_14-25-51_00400'],
        'clear_test': ['2018-02-06_14-25-51_00600'],
    })
    pair = next(row for row in report if row['left'] == 'rain' and row['right'] == 'train')
    assert pair['shared_frames'] == ['2018-02-06_14-25-51_00400']
    pair = next(row for row in report if row['left'] == 'clear_test' and row['right'] == 'train')
    assert pair['shared_frames'] == []
    assert pair['shared_recording_prefixes'] == ['2018-02-06_14-25-51']
