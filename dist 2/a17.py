import json
wins = 0
total = 0
for i in range(55, 59):
    d = json.load(open(f'matches_bytefight/match ({i}).json'))
    a_final = d['a_points'][-1]
    b_final = d['b_points'][-1]
    result = d['result']
    errlog_a = d.get('errlog_a', '')
    errlog_b = d.get('errlog_b', '')
    if 'AKIR' in errlog_a or 'v5' in errlog_a or 'v4' in errlog_a or 'HMM' in errlog_a:
        us = 'A'
    elif 'AKIR' in errlog_b or 'v5' in errlog_b or 'v4' in errlog_b or 'HMM' in errlog_b:
        us = 'B'
    else:
        us = 'A'
    moves = d['left_behind']
    n = len(moves)
    if us == 'A':
        our_pts, their_pts = a_final, b_final
        our_time = d['a_time_left'][-1]
        won = result == 0
        our_moves = [moves[j] for j in range(0, n, 2)]
        our_catches = sum(1 for j, c in enumerate(d['rat_caught']) if c and j % 2 == 0)
    else:
        our_pts, their_pts = b_final, a_final
        our_time = d['b_time_left'][-1]
        won = result == 1
        our_moves = [moves[j] for j in range(1, n, 2)]
        our_catches = sum(1 for j, c in enumerate(d['rat_caught']) if c and j % 2 == 1)
    srch = sum(1 for m in our_moves if m == 'search')
    carp = sum(1 for m in our_moves if m == 'carpet')
    prime = sum(1 for m in our_moves if m == 'prime')
    plain = sum(1 for m in our_moves if m == 'plain')
    our_rat_pts = our_catches * 4 - (srch - our_catches) * 2
    tag = 'WIN ' if won else 'LOSE'
    if won: wins += 1
    total += 1
    print(f'{i}: Us={our_pts:3d} Albert={their_pts:3d} [{tag}] t={our_time:.0f}s us={us} | prime={prime:2d} carpet={carp:2d} plain={plain:2d} srch={srch:2d} catch={our_catches}/{srch} rat={our_rat_pts:+d} | errA:{errlog_a[:30]} errB:{errlog_b[:15]}')
print(f'\nRecord: {wins}-{total-wins} ({100*wins//total}%)')
