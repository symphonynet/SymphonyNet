from collections import Counter
import itertools, copy
from more_itertools import split_before
import os, traceback, time, warnings, sys
import multiprocessing
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Instrument
from miditoolkit.midi.containers import Note as mtkNote
from chorder import Dechorder

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoding import pit2str, pos2str, bom2str, dur2str, trk2str, ins2str, pit2alphabet

WORKERS = 32


def measure_calc_chord(evt_seq):
    assert evt_seq[0][1] == 'BOM', "wrong measure for chord"
    bom_tick = evt_seq[0][0]
    ts = min(evt_seq[0][-1], 8)
    chroma = Counter()
    mtknotes = []
    for evt in evt_seq[1:-1]:
        assert evt[1] == 'ON', "wrong measure for chord: " + evt[1] + evt_seq[-1][1]
        if evt[3] == 128:  # exclude drums
            continue
        o, p, d = evt[0] - bom_tick, evt[2], evt[-1]
        if p < 21 or p > 108:  # exclude unusual pitch
            continue
        if o < 8:
            note = mtkNote(60, p, o, o + d if o > 0 else 8)
            mtknotes.append(note)
        else:
            break

    chord, score = Dechorder.get_chord_quality(mtknotes, start=0, end=ts)
    if score < 0:
        return [bom_tick, 'CHR', None, None, None, None, 'NA']
    return [bom_tick, 'CHR', None, None, None, None,
            pit2alphabet[chord.root_pc] + (chord.quality if chord.quality != '7' else 'D7')]


def merge_drums(p_midi):  # merge all percussions
    drum_0_lst = []
    new_instruments = []
    for instrument in p_midi.instruments:
        if not len(instrument.notes) == 0:
            # --------------------
            if instrument.is_drum:
                for note in instrument.notes:
                    drum_0_lst.append(note)
            else:
                new_instruments.append(instrument)
    if len(drum_0_lst) > 0:
        drum_0_lst.sort(key=lambda x: x.start)
        # remove duplicate
        drum_0_lst = list(k for k, _ in itertools.groupby(drum_0_lst))

        drum_0_instrument = Instrument(program=0, is_drum=True, name="drum")
        drum_0_instrument.notes = drum_0_lst
        new_instruments.append(drum_0_instrument)

    p_midi.instruments = new_instruments


def merge_sparse_track(p_midi, CANDI_THRES=50, MIN_THRES=5):  # merge track has too less notes
    good_instruments = []
    bad_instruments = []
    good_instruments_idx = []
    for instrument in p_midi.instruments:
        if len(instrument.notes) < CANDI_THRES:
            bad_instruments.append(instrument)
        else:
            good_instruments.append(instrument)
            good_instruments_idx.append((instrument.program, instrument.is_drum))

    for bad_instrument in bad_instruments:
        if (bad_instrument.program, bad_instrument.is_drum) in good_instruments_idx:
            # find one track to merge
            for instrument in good_instruments:
                if bad_instrument.program == instrument.program and \
                        bad_instrument.is_drum == instrument.is_drum:
                    instrument.notes.extend(bad_instrument.notes)
                    break
        # no track to merge
        else:
            if len(bad_instrument.notes) > MIN_THRES:
                good_instruments.append(bad_instrument)
    p_midi.instruments = good_instruments


def limit_max_track(p_midi, MAX_TRACK=40):  # merge track with least notes and limit the maximum amount of track to 40

    good_instruments = p_midi.instruments
    good_instruments.sort(
        key=lambda x: (not x.is_drum, -len(x.notes)))  # place drum track or the most note track at first
    assert good_instruments[0].is_drum == True or len(good_instruments[0].notes) >= len(
        good_instruments[1].notes), tuple(len(x.notes) for x in good_instruments[:3])
    # assert good_instruments[0].is_drum == False, (, len(good_instruments[2]))
    track_idx_lst = list(range(len(good_instruments)))

    if len(good_instruments) > MAX_TRACK:
        new_good_instruments = copy.deepcopy(good_instruments[:MAX_TRACK])

        # print(midi_file_path)
        for id in track_idx_lst[MAX_TRACK:]:
            cur_ins = good_instruments[id]
            merged = False
            new_good_instruments.sort(key=lambda x: len(x.notes))
            for nid, ins in enumerate(new_good_instruments):
                if cur_ins.program == ins.program and cur_ins.is_drum == ins.is_drum:
                    new_good_instruments[nid].notes.extend(cur_ins.notes)
                    merged = True
                    break
            if not merged:
                pass  # print('Track {:d} deprecated, program {:d}, note count {:d}'.format(id, cur_ins.program, len(cur_ins.notes)))
        good_instruments = new_good_instruments
        # print(trks, probs, chosen)

    assert len(good_instruments) <= MAX_TRACK, len(good_instruments)
    for idx, good_instrument in enumerate(good_instruments):
        if good_instrument.is_drum:
            good_instruments[idx].program = 128
            good_instruments[idx].is_drum = False

    p_midi.instruments = good_instruments


def get_init_note_events(p_midi):  # extract all notes in midi file

    note_events, note_on_ticks, note_dur_lst = [], [], []
    for track_idx, instrument in enumerate(p_midi.instruments):
        # track_idx_lst.append(track_idx)
        for note in instrument.notes:
            note_dur = note.end - note.start

            # special case: note_dur too long
            max_dur = 4 * p_midi.ticks_per_beat
            if note_dur / max_dur > 1:

                total_dur = note_dur
                start = note.start
                while total_dur != 0:
                    if total_dur > max_dur:
                        note_events.extend([[start, "ON", note.pitch, instrument.program,
                                             instrument.is_drum, track_idx, max_dur]])

                        note_on_ticks.append(start)
                        note_dur_lst.append(max_dur)

                        start += max_dur
                        total_dur -= max_dur
                    else:
                        note_events.extend([[start, "ON", note.pitch, instrument.program,
                                             instrument.is_drum, track_idx, total_dur]])
                        note_on_ticks.append(start)
                        note_dur_lst.append(total_dur)

                        total_dur = 0

            else:
                note_events.extend(
                    [[note.start, "ON", note.pitch, instrument.program, instrument.is_drum, track_idx, note_dur]])

                # for score analysis and beat estimating when score has no time signature
                note_on_ticks.append(note.start)
                note_dur_lst.append(note.end - note.start)

    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[5], x[4], x[3], x[2], x[-1]))
    note_events = list(k for k, _ in itertools.groupby(note_events))
    return note_events, note_on_ticks, note_dur_lst


def calculate_measure(p_midi, first_event_tick,
                      last_event_tick):  # calculate measures and append measure symbol to event_seq

    measure_events = []
    time_signature_changes = p_midi.time_signature_changes

    if not time_signature_changes:  # no time_signature_changes, estimate it
        raise AssertionError("No time_signature_changes")
    else:
        if time_signature_changes[0].time != 0 and \
                time_signature_changes[0].time > first_event_tick:
            raise AssertionError("First time signature start with None zero tick")

        # clean duplicate time_signature_changes
        temp_sig = []
        for idx, time_sig in enumerate(time_signature_changes):
            if idx == 0:
                temp_sig.append(time_sig)
            else:
                previous_timg_sig = time_signature_changes[idx - 1]
                if not (previous_timg_sig.numerator == time_sig.numerator
                        and previous_timg_sig.denominator == time_sig.denominator):
                    temp_sig.append(time_sig)
        time_signature_changes = temp_sig
        # print("time_signature_changes", time_signature_changes)
        for idx in range(len(time_signature_changes)):
            # calculate measures, eg: how many ticks per measure
            numerator = time_signature_changes[idx].numerator
            denominator = time_signature_changes[idx].denominator
            ticks_per_measure = p_midi.ticks_per_beat * (4 / denominator) * numerator

            cur_tick = time_signature_changes[idx].time

            if idx < len(time_signature_changes) - 1:
                next_tick = time_signature_changes[idx + 1].time
            else:
                next_tick = last_event_tick + int(ticks_per_measure)

            if ticks_per_measure.is_integer():
                for measure_start_tick in range(cur_tick, next_tick, int(ticks_per_measure)):
                    if measure_start_tick + int(ticks_per_measure) > next_tick:
                        measure_events.append([measure_start_tick, "BOM", None, None, None, None, 0])
                        measure_events.append([next_tick, "EOM", None, None, None, None, 0])
                    else:
                        measure_events.append([measure_start_tick, "BOM", None, None, None, None, 0])
                        measure_events.append(
                            [measure_start_tick + int(ticks_per_measure), "EOM", None, None, None, None, 0])
            else:
                assert False, "ticks_per_measure Error"
    return measure_events


def quantize_by_nth(nth_tick, note_events):
    # Eg. Quantize by 32th note

    half = nth_tick / 2
    split_score = list(split_before(note_events, lambda x: x[1] == "BOM"))
    measure_durs = []
    eom_tick = 0
    for measure_id, measure in enumerate(split_score):
        bom_tick = measure[0][0]
        assert bom_tick == eom_tick, 'measure time error {bom_tick} {eom_tick}'
        eom_tick = measure[-1][0]
        mea_dur = eom_tick - bom_tick
        if mea_dur < nth_tick:  # measure duration need to be quantized
            measure_durs.append(1)
        else:
            if mea_dur % nth_tick < half:  # quantize to left
                measure_durs.append(mea_dur // nth_tick)
            else:
                measure_durs.append(mea_dur // nth_tick + 1)

        for evt in measure[1:-1]:
            assert evt[1] == 'ON', f'measure structure error {evt[1]}'
            rel_tick = evt[0] - bom_tick
            if rel_tick % nth_tick <= half:
                rel_tick = min(rel_tick // nth_tick, measure_durs[-1] - 1)
            else:
                rel_tick = min(rel_tick // nth_tick + 1, measure_durs[-1] - 1)
            evt[0] = rel_tick

    final_events = []
    lasteom = 0
    for measure_id, measure in enumerate(split_score):
        measure[0][0] = lasteom
        measure[-1][0] = measure[0][0] + measure_durs[measure_id]
        lasteom = measure[-1][0]

        for event in measure[1:-1]:
            event[0] += measure[0][0]

            if event[-1] < nth_tick:  # duration too short, quantize to 1
                event[-1] = 1
            else:
                if event[-1] % nth_tick <= half:
                    event[-1] = event[-1] // nth_tick
                else:
                    event[-1] = event[-1] // nth_tick + 1

        final_events.extend(measure)
    return final_events


def prettify(note_events, ticks_per_beat):
    fist_event_idx = next(i for i in (range(len(note_events))) if note_events[i][1] == "ON")
    last_event_idx = next(i for i in reversed(range(len(note_events))) if note_events[i][1] == "ON")

    assert note_events[fist_event_idx - 1][1] == "BOM", "measure_start Error"
    assert note_events[last_event_idx + 1][1] == "EOM", "measure_end Error"

    # remove invalid measures on both sides
    note_events = note_events[fist_event_idx - 1: last_event_idx + 2]

    # check again
    assert note_events[0][1] == "BOM", "measure_start Error"
    assert note_events[-1][1] == "EOM", "measure_end Error"

    # -------------- zero start tick -----------------
    start_tick = note_events[0][0]
    if start_tick != 0:
        for event in note_events:
            event[0] -= start_tick

    from fractions import Fraction
    ticks_32th = Fraction(ticks_per_beat, 8)

    note_events = quantize_by_nth(ticks_32th, note_events)

    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2], x[-1]))
    note_events = list(k for k, _ in itertools.groupby(note_events))

    # -------------------------check measure duration----------------------------------------------
    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2], x[-1]))
    split_score = list(split_before(note_events, lambda x: x[1] == "BOM"))

    check_measure_dur = [0]

    for measure_idx, measure in enumerate(split_score):
        first_tick = measure[0][0]
        last_tick = measure[-1][0]
        measure_dur = last_tick - first_tick
        if measure_dur > 100:
            raise AssertionError("Measure duration error")
        split_score[measure_idx][0][-1] = measure_dur

        if measure_dur in check_measure_dur:
            # print(measure_dur)
            raise AssertionError("Measure duration error")
    return split_score


def get_pos_and_cc(split_score):
    new_event_seq = []
    for measure_idx, measure in enumerate(split_score):
        measure.sort(key=lambda x: (x[1] == "EOM", x[1] == "ON", x[1] == 'CHR', x[1] == "BOM", x[-2]))
        bom_tick = measure[0][0]

        # split measure by track
        track_nmb = set(map(lambda x: x[-2], measure[2:-1]))
        tracks = [[y for y in measure if y[-2] == x] for x in track_nmb]

        # ---------- calculate POS for each track / add CC
        new_measure = []
        for track_idx, track in enumerate(tracks):
            pos_lst = []
            trk_abs_num = -1
            for event in track:
                if event[1] == "ON":
                    assert trk_abs_num == -1 or trk_abs_num == event[
                        -2], "Error: found inconsistent trackid within same track"
                    trk_abs_num = event[-2]
                    mypos = event[0] - bom_tick
                    pos_lst.append(mypos)
                    pos_lst = list(set(pos_lst))

            for pos in pos_lst:
                tracks[track_idx].append([pos + bom_tick, "POS", None, None, None, None, pos])
            tracks[track_idx].insert(0, [bom_tick, "CC", None, None, None, None, trk_abs_num])
            tracks[track_idx].sort(
                key=lambda x: (x[0], x[1] == "ON", x[1] == "POS", x[1] == "CC", x[5], x[4], x[3], x[2]))

        new_measure.append(measure[0])
        new_measure.append(measure[1])
        for track in tracks:
            for idx, event in enumerate(track):
                new_measure.append(event)

        new_event_seq.extend(new_measure)

    return new_event_seq


def event_seq_to_str(new_event_seq):
    char_events = []

    for evt in new_event_seq:
        if evt[1] == 'ON':
            char_events.append(pit2str(evt[2]))  # pitch
            char_events.append(dur2str(evt[-1]))  # duration
            char_events.append(trk2str(evt[-2]))  # track
            char_events.append(ins2str(evt[3]))  # instrument
        elif evt[1] == 'POS':
            char_events.append(pos2str(evt[-1]))  # type (time position)
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'BOM':
            char_events.append(bom2str(evt[-1]))
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'CC':
            char_events.append('NT')
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'CHR':
            char_events.append('H' + evt[-1])
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        else:
            assert False, ("evt type error", evt[1])
    return char_events


# abs_pos type pitch program is_drum track_id duration/rela_pos
def midi_to_event_seq_str(midi_file_path, readonly=False):
    p_midi = MidiFile(midi_file_path)
    for ins in p_midi.instruments:
        ins.remove_invalid_notes(verbose=False)

    merge_drums(p_midi)

    if not readonly:
        merge_sparse_track(p_midi)

    limit_max_track(p_midi)

    note_events, note_on_ticks, _ = get_init_note_events(p_midi)

    measure_events = calculate_measure(p_midi, min(note_on_ticks), max(note_on_ticks))
    note_events.extend(measure_events)
    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2]))

    split_score = prettify(note_events, p_midi.ticks_per_beat)

    for measure_idx, measure in enumerate(split_score):  # calculate chord for every measure
        chord_evt = measure_calc_chord(measure)
        split_score[measure_idx].insert(1, chord_evt)

    new_event_seq = get_pos_and_cc(split_score)

    char_events = event_seq_to_str(new_event_seq)

    return char_events


def mp_worker(file_path):
    try:
        event_seq = midi_to_event_seq_str(file_path)
        return event_seq
    except (OSError, EOFError, ValueError, KeyError) as e:
        print(file_path)
        traceback.print_exc(limit=0)
        print()
        return "error"

    except AssertionError as e:
        if str(e) == "No time_signature_changes":
            return "error"
        elif str(e) == "Measure duration error":
            # print("Measure duration error", file_path)
            return "error"
        else:
            print("Other Assertion Error", str(e), file_path)
            return "error"

    except Exception as e:
        print(file_path)
        traceback.print_exc(limit=0)
        print()
        return "error"


def mp_handler(file_paths):
    start = time.time()

    broken_counter = 0
    good_counter = 0

    event_seq_res = []
    chord_cnter = Counter()
    print(f'starts processing {len(file_paths)} midis with {WORKERS} processes')

    with multiprocessing.Pool(WORKERS) as p:
        for event_seq in p.imap(mp_worker, file_paths):
            if isinstance(event_seq, str):
                broken_counter += 1
            elif len(event_seq) > 0:
                event_seq_res.append(event_seq)
                good_counter += 1
            else:
                broken_counter += 1

    print(
        f"MIDI data preprocessing takes: {time.time() - start}s, {good_counter} samples collected, {broken_counter} broken.")

    # ----------------------------------------------------------------------------------
    txt_start = time.time()
    if not os.path.exists('data/preprocessed/'):
        os.makedirs('data/preprocessed/')

    with open("data/preprocessed/raw_corpus.txt", "w", encoding="utf-8") as f:
        for idx, piece in enumerate(event_seq_res):
            f.write(' '.join(piece) + '\n')

    print("Create txt file takes: ", time.time() - txt_start)
    # ----------------------------------------------------------------------------------


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    folder_path = "data/midis"
    file_paths = []
    for path, directories, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                file_path = path + "/" + file
                file_paths.append(file_path)

    # run multi-processing midi extractor
    mp_handler(file_paths)
