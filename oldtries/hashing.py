import time
import numpy as np
import matplotlib.pyplot as plt

timbre_bins = 2
timbre_coefficients = np.array([0, 1, 2, 3, 4, 5, 6])
timbre_segments = 1
pitch_segments = 1
min_beats = 2
loudness_segments = 1
loudness_bins = 2

SEGMENT_DTYPE = [('confidence', 'f4'), ('duration', 'f4'), ('loudness_max', 'f4'),
                 ('loudness_start', 'f4'), ('pitches', 'f4', 12), ('timbre', 'f4', 12)]

BEAT_DTYPE = [('index', 'i4'), ('index_in_parent', 'i4'), ('start_segment', 'i4'),
              ('num_segments', 'i4')]

MATCH_DTYPE = [('beat_i', 'i4'), ('beat_j', 'i4'), ('distance', 'f4')]

NULL_SEGMENT = np.array((0, 0, -60, -60, np.zeros(12), np.full(12, -10000)), dtype=SEGMENT_DTYPE)

# compute and store the timbre edges creating "timbre_bins" once.
timbre_edges = None
loudness_max_edges = None
loudness_start_edges = None

# Plot Matches
def plot_matches(tp, fn, fp, speedup):
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    score = speedup * recall * precision
    all_positives = len(tp) + len(fn)
    plt.figure(figsize=(9, 9))
    plt.plot(fn['beat_i'], fn['beat_j'], marker='.', linestyle='none', c='red',
            alpha=0.5, label='FN')
    plt.plot(fp['beat_i'], fp['beat_j'], marker='.', linestyle='none', c='orange',
            alpha=0.5, label='FP')
    plt.plot(tp['beat_i'], tp['beat_j'], marker='.', linestyle='none', c='green',
            alpha=0.5, label='TP')
    plt.title(f'{all_positives} Matches: {len(tp)} TP, {len(fn)} FN, {len(fp)} FP, '
            f'recall={recall:.3f}, precision={precision:.3f}, speedup={speedup:.1f}, 'f'score={score:.1f}')
    plt.xlabel('beat index')
    plt.ylabel('beat index')
    plt.axis('square')
    plt.legend()

    # plt.savefig('original.pdf')
    # plt.savefig('original.png')
    plt.show()

# Find Matches 
def find_matches(beats, segments, correct_matches):
    """
    Use a hash function to find the matching beats. Provide a threshold
    if using your own beat distance function. Provide the correct
    matches if you are using the simulated beat matching
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :param correct_matches: a record array of all the correct matches =
    ['beat_i', 'beat_j', 'distance']
    :return: all the matches based on the hash function
    """

    d, t = robust_time(populate_dictionary, beats, segments)
    print(f'populate {[len(di) for di in d]} hashes in {t:.6} seconds')
    matches = []
    seconds = t
    for i, di in enumerate(d):
        for h in di.keys():
            beats = di[h]
            # make sure beats are unique
            beats = np.unique(beats)
            m, t = beat_matches_sim(beats, correct_matches)
            seconds += t
            matches.append(m)
            print(f'{h} compare {len(beats)} beats in {t:.6} seconds')
    if len(matches) > 0:
        matches = np.concatenate(matches)
        matches = np.unique(matches)
    else:
        matches = np.array([])

    return matches, seconds

# Robust Timing
def robust_time(f, *args):
    t0 = time.time()
    times = []
    while True:
        t1 = time.time()
        r = f(*args)
        t2 = time.time()
        times.append(t2 - t1)
        if time.time() - t0 > 5:
            break
    return r, min(times)

# Beat Matches
def beat_matches_sim(beats, correct_matches):
    """
    Simulate matching beats by looking for all the correct matches that would have
    been discovered in this array of beats. Also, fake the timing to make it perform
    similarly
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param correct_matches: the correct record array for matches =
    ['beat_i', 'beat_j', 'distance']
    :return: a record array of matches within this array of beats
    """
    i = np.isin(correct_matches['beat_i'], beats['index'])
    j = np.isin(correct_matches['beat_j'], beats['index'])
    matches = correct_matches[i & j]
    seconds = timing(len(beats))
    return matches, seconds

# Populate Dictionary
def populate_dictionary(beats, segments):
    """
    Use each hash function to create a dictionary of {hash: [beats]}
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a list of dictionaries with hash:list of beats, one for each hash function
    """
    d = []
    for i, hashes in enumerate(hash_functions(beats, segments)): # for each hash function
        d.append({})
        for h, beat in zip(hashes, beats): # for each beat and its hash
            if beat['num_segments'] > 0:
                d[i].setdefault(tuple(h), []).append(beat)
    for di in d:
        for k in list(di.keys()):
            if len(di[k]) >= min_beats:
                di[k] = np.array(di[k])
            else:
                del di[k]
    return d


# Combined with OR
def hash_functions(beats, segments):
    """ 
    Return list of numpy arrays, one for each hash function. Beats match if they
    hash to the same bin in ANY of the hash functions
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a list of 2D matrix hashes, one matrix per hash function, one row per beat
    When matching, beats that match any hash function are compared
    """
    any_of_these = [hash_function_1, hash_function_2]
    arrays = [hf(beats, segments) for hf in any_of_these]
    return arrays

# Example hash functions
# To be combined with OR
def hash_function_1(beats, segments):
    """
    Return a numpy arrays containing the combined hash for each beat. Beats match
    if they match on all of these elemental hash functions
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a list of 2D matrix hashes, one matrix per hash function, one row per beat
    When matching, beats that match any hash function are compared
    """
    # assemble list of 2D arrays (len(beats)-by-?) each
    all_of_these = [
        hf_bib(beats, segments),
        hf_digitize_timbre(beats, segments),
    ]

    # stack into 2D array of integers (len(beats)-by-?)
    all_of_these = np.hstack(all_of_these).astype(np.int32)
    return all_of_these


def hash_function_2(beats, segments):
    """
    Return a numpy arrays containing the combined hash for each beat. Beats match
    if they match on all of these elemental hash functions
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =[confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a list of 2D matrix hashes, one matrix per hash function, one row per beat
    When matching, beats that match any hash function are compared
    """
    # assemble list of 2D arrays (len(beats)-by-?) each
    all_of_these = [
        hf_bib(beats, segments),
        hf_binarize_pitch(beats, segments)
    ]

    # stack into 2D array of integers (len(beats)-by-?)
    all_of_these = np.hstack(all_of_these).astype(np.int32)
    return all_of_these

# Example elemental hash functions
def hf_bib(beats, *_):
    """
    Elemental hash function for the "beat in bar"
    :param beats: numpy structured array of beats with fields =
        [index, index_in_parent, start_segment, num_segments]
    :return: numpy 2d array, each row a hash for the corresponding beat
    """
    index_in_parent = beats['index_in_parent'].reshape(-1, 1)
    return index_in_parent

def hf_num_segments_gt_0(beats, *_):
    """
    Elemental hash function for nonempty beats
    :param beats: numpy structured array of beats with fields =
        [index, index_in_parent, start_segment, num_segments]
    :return: numpy 2d array, each row a hash for the corresponding beat
    """
    num_segments_gt_0 = (beats['num_segments'] > 0).reshape(-1, 1)
    return num_segments_gt_0


# Helper "get segment" function
def get_timbre(beats, segments, segment_index, coefficients):
    """
     the designated "timbre_coefficients" for the "segment_index"-th segment in every beat.
    If a beat doesn't have enough segments return a vector full of -1.0 for that beat
    :param beats: numpy structured array of beats with fields =
        [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
        [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :param segment_index: which segment in the beat
    :param coefficients: which timbre coefficients to use (list of indexes 0..11)
    :return: numpy array num_beats-by-num_coefficients
    """
    start_segment = beats['start_segment']
    num_segments = beats['num_segments']
    index = (start_segment >= 0) & (num_segments > segment_index)
    timbre = np.full((len(beats), len(coefficients)), -1.0)
    timbre[index, :] = segments['timbre'][start_segment[index]+segment_index][:, coefficients]
    return timbre

def get_pitches(beats, segments, segment_index):
    """
    Return the pitches for the "segment_index"-th segment in every beat
    :param beats: numpy structured array of beats with fields =
        [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
        [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :param segment_index: which segment in the beat
    :return: numpy array num_beats-by-12
    """
    start_segment = beats['start_segment']
    num_segments = beats['num_segments']
    index = (start_segment >= 0) & (num_segments > segment_index)
    pitches = np.zeros((len(beats), 12))
    pitches[index, :] = segments['pitches'][start_segment[index]+segment_index]
    return pitches

# Timing
def timing(num_beats):
    """
    Estimate the time it would take an optimized beat distance computation to
    complete as a function of the number of beats to compare. Linearly interpolate
    the time per beat for a wide enough range
    :param num_beats: the number of beats to estimate the timing for
    :return: the estimated number of seconds
    """
    if num_beats == 0:
        return 3.076085864793597e-05

    t = np.array([
        1.31562138e-04, 7.59354485e-05, 7.93140652e-05, 6.10690933e-05, 5.23384175e-05,
        5.71063410e-05, 4.83461126e-05, 4.69709007e-05, 4.19671422e-05, 3.90646607e-05,
        3.61373253e-05, 3.39626489e-05, 3.47090948e-05, 2.96418386e-05, 2.55532008e-05,
        2.39602049e-05, 2.27489299e-05, 2.14544404e-05, 2.14664928e-05, 2.04831630e-05,
        2.28323566e-05, 2.00652120e-05, 1.88255464e-05, 1.77982765e-05, 1.62767751e-05,
        2.11544489e-05, 2.32914490e-05, 2.31238129e-05, 2.44980405e-05, 2.32798780e-05,
        2.29734778e-05, 2.18681438e-05, 2.07696430e-05, 2.03732824e-05, 1.92473409e-05,
        1.85563350e-05, 1.78633090e-05, 1.74770455e-05, 1.67262186e-05, 1.62830348e-05,
        1.54916607e-05, 1.59443028e-05, 1.64246766e-05, 1.46354617e-05, 1.40358171e-05,
        1.35766840e-05, 1.33510842e-05, 1.37398099e-05, 1.38773824e-05, 1.37820333e-05,
        1.37208186e-05, 1.41587948e-05, 1.43888458e-05, 1.44925300e-05, 1.47417469e-05,
        1.60327640e-05, 1.65774134e-05, 1.72393427e-05, 1.79293738e-05, 1.87104605e-05,
        1.96194572e-05, 1.98287359e-05, 2.12823988e-05, 2.23649793e-05, 2.85204513e-05,
        2.78176235e-05, 2.35765503e-05, 2.83723042e-05, 2.90892403e-05, 3.07639001e-05,
        3.05521677e-05, 3.28489888e-05, 3.47152306e-05, 3.70995752e-05, 4.28254210e-05,
        4.48108532e-05, 4.62172346e-05, 4.90306695e-05, 5.23632378e-05, 5.56539435e-05,
        5.92192535e-05, 6.28580333e-05, 6.70145152e-05, 7.16888624e-05, 7.27917068e-05,
        7.65491169e-05, 8.99500993e-05, 9.18378531e-05, 9.56893453e-05, 1.04352544e-04,
        1.10937583e-04, 1.19821321e-04, 1.28658646e-04, 1.54131216e-04, 1.68394472e-04,
        1.74502002e-04, 1.89593849e-04, 2.05887920e-04, 2.15343464e-04, 2.34867739e-04,
        2.54400766e-04, 2.70099259e-04, 2.83453813e-04, 3.07213741e-04, 3.45013163e-04,
        3.76968045e-04, 4.06214028e-04, 4.20146965e-04, 4.48110267e-04, 4.81686018e-04,
        5.24179722e-04, 5.58758549e-04, 6.00116012e-04, 6.42522785e-04, 7.03837199e-04,
        7.44835365e-04, 8.41028113e-04, 8.57581996e-04, 9.24117471e-04])

    n = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 24, 25, 27, 29, 32,
        34, 36, 39, 42, 45, 48, 51, 55, 59, 64, 68, 73, 78, 84, 90, 97, 103, 111, 119, 128, 137,
        147, 157, 168, 181, 194, 207, 222, 238, 256, 274, 294, 315, 337, 362, 388, 415, 445, 477,
        512, 548, 588, 630, 675, 724, 776, 831, 891, 955, 1024, 1097, 1176, 1260, 1351, 1448, 1552,
        1663, 1782, 1910, 2048, 2194, 2352, 2521, 2702, 2896, 3104, 3326, 3565, 3821, 4096, 4389,
        4705, 5042, 5404, 5792, 6208, 6653, 7131, 7643, 8192, 8779, 9410, 10085, 10809, 11585,
        12416, 13307, 14263, 15286, 16384, 17559, 18820, 20171, 21618])

    slope = (t[-1] - t[-2]) / (n[-1] - n[-2])

    # right = extrapolate from last value
    return np.interp(num_beats, n, t, left=3.076085864793597e-05, right=t[-1]+slope*(
            num_beats-n[-1])) * num_beats


def get_timbre_edges(segments):
    global timbre_edges
    if timbre_edges is None:
        timbre_edges = np.percentile(
                a=segments['timbre'], q=np.linspace(0, 100, timbre_bins + 1),
                axis=0)[1:-1].T
    return timbre_edges

def hash_func(beats, segments):
    """
    Return the hash of each beat in a list or array
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a 2D matrix of hashes, one row for each beat
    """
    # assemble list of 2D arrays (len(beats)-by-?) each
    index_in_parent = beats['index_in_parent'].reshape(-1, 1)
    arrays = [index_in_parent]
    # add more 2D arrays for hashing here
    # arrays.append(other_array)
    # stack into 2D array of integers (len(beats)-by-sum(Mi))
    arrays = np.hstack(arrays).astype(np.int32)
    return arrays

def hf_digitize_timbre(beats, *_):
    pass

def hf_binarize_pitch(beats, *_):
    pass

# Main Function for testing
def main():
    global timbre_edges
    # load the data
    data = np.load('beats.npz')
    beats = data['beats']
    segments = data['segments']

    timbre_edges = get_timbre_edges(segments)

    # load correct matches
    threshold = 50
    data = np.load(f'matches-{threshold}.npz')
    elapsed_max = float(data['elapsed'])
    correct_matches = data['matches']

    # Use student populate_hashes to find matches.
    student_matches, elapsed = find_matches(beats, segments, correct_matches)
    print(f'Found matches in {elapsed} seconds')

    if len(student_matches) > 0:
        f = ['beat_i', 'beat_j']
        tp = student_matches[np.isin(student_matches[f], correct_matches[f])]
        fp = student_matches[~np.isin(student_matches[f], correct_matches[f])]
        fn = correct_matches[~np.isin(correct_matches[f], student_matches[f])]
        precision = len(tp) / (len(tp) + len(fp))
        recall = len(tp) / (len(tp) + len(fn))
        speedup = elapsed_max / elapsed
        score = speedup * recall * precision

        print(f'recall={recall:.3f}, precision={precision:.3f}, speedup={speedup:.1f}, '
                f'score={score:.1f}')
        plot_matches(tp, fn, fp, speedup)

if __name__ == '__main__':
    main()

