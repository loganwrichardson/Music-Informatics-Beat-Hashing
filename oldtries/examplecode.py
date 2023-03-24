def plot_ssm(ssm, name, markers=False):
    ssm = np.array(ssm)
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(ssm, cmap='gray_r', origin='lower')
    if markers:
        ax.plot(*np.where(ssm > -80), linestyle='none', marker=',', color='cyan')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'{name} Similarity')
    ax.set_xlabel(f'{name} Index')
    ax.set_ylabel(f'{name} Index')

#naive pitch ssm
def pitch_ssm(analysis):
    ssm = []
    for seg1 in analysis['segments']:
        row = []
        pitches1 = seg1['pitches']
        for seg2 in analysis['segments']:
            pitches2 = seg2['pitches']
            p1_norm = sum([p1**2 for p1 in pitches1])**(1/2)
            p2_norm = sum([p2**2 for p2 in pitches2])**(1/2)
            dot_product = sum([p1*p2 for p1, p2 in zip(pitches1, pitches2)])
            cosine_similarity = dot_product / (p1_norm * p2_norm)
            row.append(cosine_similarity)
        ssm.append(row)
    return ssm

#opt pitch
def pitch_ssm_opt(analysis):
    pitches = np.array([segment['pitches'] for segment in analysis['segments']])
    return 1 - cdist(pitches, pitches, 'cosine')

#naive timbre
def timbre_ssm(analysis):
    ssm = []
    for seg1 in analysis['segments']:
        row = []
        timbre1 = seg1['timbre']
        for seg2 in analysis['segments']:
            timbre2 = seg2['timbre']
            squared_errors = [(t1 - t2)**2 for t1, t2 in zip(timbre1, timbre2)]
            sum_squared_error = sum(squared_errors)
            distance = sum_squared_error ** (1/2)
            row.append(-distance)
        ssm.append(row)
    return ssm

#opt timbre
def timbre_ssm_opt(analysis):
    timbre = np.array([segment['timbre'] for segment in analysis['segments']])
    return -cdist(timbre, timbre, 'euclidean')

#naive seg dist
def segment_dist(seg1, seg2):
    pitch_dist = 0
    for x1, x2 in zip(seg1['pitches'], seg2['pitches']):
        pitch_dist += (x1 - x2) ** 2
    pitch_dist **= (1/2)
    timbre_dist = 0
    for x1, x2 in zip(seg1['timbre'], seg2['timbre']):
        timbre_dist += (x1 - x2) ** 2
    timbre_dist **= (1/2)
    
    d = 10 * pitch_dist + timbre_dist
    d += abs(seg1['loudness_start'] - seg2['loudness_start'])
    d += abs(seg1['loudness_max'] - seg2['loudness_max'])
    d += 100 * abs(seg1['duration'] - seg2['duration'])
    d += abs(seg1['confidence'] - seg2['confidence'])
    return d
    
def segment_ssm(analysis):
    ssm = []
    for seg1 in analysis['segments']:
        row = []
        for seg2 in analysis['segments']:
            sim = -segment_dist(seg1, seg2)
            row.append(sim)
        ssm.append(row)
    ssm = np.array(ssm)
    return ssm
