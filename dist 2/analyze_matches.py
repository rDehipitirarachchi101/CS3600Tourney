#!/usr/bin/env python3
"""Universal match analyzer — analyzes ALL json files in matches_bytefight/"""
import json, glob, os, sys

CPT = {1:-1, 2:2, 3:4, 4:6, 5:10, 6:15, 7:21}

files = sorted(glob.glob('matches_bytefight/match*.json') + glob.glob('matches_bytefight/match *.json'),
               key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit()) or '0'))

if not files:
    print("No match files found in matches_bytefight/")
    sys.exit(1)

wins, losses, ties = 0, 0, 0
print(f"{'#':>3} {'Us':>4} {'Opp':>4} {'Res':>4} {'Time':>5} {'P':>3} {'C':>3} {'Pl':>3} {'S':>3} {'Rat':>5} {'CarpPts':>7} {'OppCPts':>7} | {'Opponent'}")
print("-" * 110)

for f in files:
    d = json.load(open(f))
    num = ''.join(c for c in os.path.basename(f) if c.isdigit())
    ea, eb = d.get('errlog_a',''), d.get('errlog_b','')
    result = d['result']
    moves = d['left_behind']
    n = len(moves)

    # Detect our side
    if 'AKIR' in ea or 'HMM' in ea or 'v7' in ea or 'v6' in ea or 'v5' in ea or 'v4' in ea:
        us = 'A'
    elif 'AKIR' in eb or 'HMM' in eb or 'v7' in eb or 'v6' in eb or 'v5' in eb or 'v4' in eb:
        us = 'B'
    else:
        us = 'A'

    if us == 'A':
        our_pts, their_pts = d['a_points'][-1], d['b_points'][-1]
        our_time = d['a_time_left'][-1]
        won = result == 0
        lost = result == 1
        our_moves = [moves[j] for j in range(0, n, 2)]
        our_catches = sum(1 for j,c in enumerate(d['rat_caught']) if c and j%2==0)
        opp_catches = sum(1 for j,c in enumerate(d['rat_caught']) if c and j%2==1)
        opp_label = eb[:35]
    else:
        our_pts, their_pts = d['b_points'][-1], d['a_points'][-1]
        our_time = d['b_time_left'][-1]
        won = result == 1
        lost = result == 0
        our_moves = [moves[j] for j in range(1, n, 2)]
        our_catches = sum(1 for j,c in enumerate(d['rat_caught']) if c and j%2==1)
        opp_catches = sum(1 for j,c in enumerate(d['rat_caught']) if c and j%2==0)
        opp_label = ea[:35]

    srch = sum(1 for m in our_moves if m == 'search')
    carp = sum(1 for m in our_moves if m == 'carpet')
    prime = sum(1 for m in our_moves if m == 'prime')
    plain = sum(1 for m in our_moves if m == 'plain')

    our_clens, opp_clens = [], []
    for j, nc in enumerate(d['new_carpets']):
        if nc:
            l = len(nc)
            if (j%2==0 and us=='A') or (j%2==1 and us=='B'):
                our_clens.append(l)
            else:
                opp_clens.append(l)

    our_cpts = sum(CPT.get(l,0) for l in our_clens)
    opp_cpts = sum(CPT.get(l,0) for l in opp_clens)

    if won:
        tag = 'WIN'
        wins += 1
    elif lost:
        tag = 'LOSE'
        losses += 1
    else:
        tag = 'TIE'
        ties += 1

    rat_str = f"{our_catches}/{srch}"
    print(f"{num:>3} {our_pts:>4} {their_pts:>4} {tag:>4} {our_time:>5.0f} {prime:>3} {carp:>3} {plain:>3} {srch:>3} {rat_str:>5} {our_cpts:>+7} {opp_cpts:>+7} | {opp_label}")

print("-" * 110)
total = wins + losses + ties
pct = 100*wins/total if total else 0
print(f"Record: {wins}W-{losses}L-{ties}T ({pct:.0f}%) over {total} games")
