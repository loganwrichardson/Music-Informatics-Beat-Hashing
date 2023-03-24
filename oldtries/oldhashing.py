import time
import numpy as np
import matplotlib.pyplot as plt

"""
Universal Hashing Example h(k) = [(ak+b)mod p] mod m
"""

def main():
    # load the data
    data = np.load('beats.npz')
    beats = data['beats']
    segments = data['segments']

    # load correct matches
    threshold = 50
    data = np.load(f'matches-{threshold}.npz')
    elapsed_max = float(data['elapsed'])
    correct_matches = data['matches']

    t0 = time.time()
    student_matches = find_matches(beats, segments, correct_matches=correct_matches)
    # student_matches = find_matches(beats, segments, threshold=threshold)
    elapsed = time.time() - t0
    print(f'Found matches in {elapsed} seconds')

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

    plt.figure(figsize=(9, 9))
    plt.plot(fn['beat_i'], fn['beat_j'], marker='.', linestyle='none', c='red',
            alpha=0.5, label='FN')
    plt.plot(fp['beat_i'], fp['beat_j'], marker='.', linestyle='none', c='orange',
            alpha=0.5, label='FP')
    plt.plot(tp['beat_i'], tp['beat_j'], marker='.', linestyle='none', c='green',
            alpha=0.5, label='TP')
    plt.title(f'{len(correct_matches)} Matches: {len(tp)} TP, {len(fn)} FN, {len(fp)} FP, '
            f'recall={recall:.3f}, precision={precision:.3f}, speedup={speedup:.1f}, '
            f'score={score:.1f}')
    plt.xlabel('beat index')
    plt.ylabel('beat index')
    plt.axis('square')
    plt.legend()
    plt.savefig('original.pdf')
    plt.savefig('original.png')
    plt.show()

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

def populate_dictionary(beats, segments):
    """
    Use the hash function to create a dictionary of {hash: [beats]}
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :return: a dictionary with hash:list of beats
    """
    d = {}
    for h, beat in zip(hash_func(beats, segments), beats):
        d.setdefault(tuple(h), []).append(beat)
    for k in d:
        d[k] = np.array(d[k])
    return d


def beat_matches(beats, segments, threshold):
    """
    Find the beat matches using your optimized beat distance function
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :param threshold: the distance threshold below which we consider matches
    :return: a record array for matches = ['beat_i', 'beat_j', 'distance']
    """
    dist = beat_dist_opt(beats, segments)
    i, j = np.where(dist <= threshold)
    d = dist[i, j]
    i = beats[i]['index']
    j = beats[j]['index']
    matches = np.rec.fromarrays(np.vstack((i, j, d)), dtype=MATCH_DTYPE)
    return matches

# For those who did not sufficiently optimize their beat distance calculations, you can use this simulated version.

def beat_matches_sim(beats, correct_matches):
    """
    Simulate matching beats by looking for all the correct matches that would have
    been discovered in this array of beats. Also, fake the timing to make it perform
    similarly
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param correct_matches: the correct record array for all matches =
    ['beat_i', 'beat_j', 'distance']
    :return: a record array of matches within this array of beats
    """
    t0 = time.time()
    i = np.isin(correct_matches['beat_i'], beats['index'])
    j = np.isin(correct_matches['beat_j'], beats['index'])
    matches = correct_matches[i & j]
    seconds = timing(len(beats))
    sleep = seconds - (time.time() - t0)
    if sleep > 0:
        time.sleep(sleep)
    return matches


# This function helps beat_match_sim perform in about the same time as an optimized beat_matches.
def timing(num_beats):
    """
    Estimate the time it would take an optimized beat distance computation to
    complete as a function of the number of beats to compare. Linearly interpolate
    the time per beat for a wide enough range.
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
    return np.interp(
            num_beats, n, t, left=3.076085864793597e-05, right=t[-1]+slope*(num_beats-n[-1])
    ) * num_beats


def find_matches(beats, segments, threshold=None, correct_matches=None):
    """
    Use a hash function to find the matching beats. Provide a threshold
    if using your own beat distance function. Provide the correct
    matches if you are using the simulated beat matching
    :param beats: numpy structured array of beats with fields =
    [index, index_in_parent, start_segment, num_segments]
    :param segments: numpy structured array of segments with fields =
    [confidence, duration, loudness_max, loudness_start, pitches, timbre]
    :param threshold: the distance threshold below which we consider matches
    :param correct_matches: a record array of all the correct matches =
    ['beat_i', 'beat_j', 'distance']
    :return: all the matches based on the hash function
    """
    t0 = time.time()
    d = populate_dictionary(beats, segments)
    t1 = time.time()
    print(f'populate {len(d)} hashes in {t1 - t0:.6} seconds')
    matches = []
    for h, beats in d.items():
        if threshold is not None:
            m = beat_matches(beats, segments, threshold)
        elif correct_matches is not None:
            m = beat_matches_sim(beats, correct_matches)
        else:
            raise ValueError('Must provide threshold or correct_matches')
        print(f'compare {len(beats)} beats in {time.time() - t1:.6} seconds')
        t1 = time.time()
        matches.append(m)
    return np.concatenate(matches)

if __name__ == '__main__':
    main()
