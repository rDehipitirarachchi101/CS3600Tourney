import json
import os

carpet_points_table = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

files = [
    "matches_bytefight/match (54).json",
    "matches_bytefight/match (55).json",
    "matches_bytefight/match (56).json",
    "matches_bytefight/match (57).json",
    "matches_bytefight/match (58).json",
    "matches_bytefight/match (59).json",
    "matches_bytefight/match (60).json",
    "matches_bytefight/match (61).json",
    "matches_bytefight/match (62).json",
    "matches_bytefight/match (63).json",
]

results = []

for fpath in files:
    with open(fpath) as f:
        d = json.load(f)
    
    match_num = fpath.split("(")[1].split(")")[0]
    errlog_a = d.get("errlog_a", "")
    errlog_b = d.get("errlog_b", "")
    
    # Determine which side is ours
    our_side = None
    opp_id = ""
    if "AKIR" in errlog_a.upper() or "SIMPLE" in errlog_a.upper() or "EXPECTIMAX" in errlog_a.upper():
        our_side = "a"
        opp_id = errlog_b
    elif "AKIR" in errlog_b.upper() or "SIMPLE" in errlog_b.upper() or "EXPECTIMAX" in errlog_b.upper():
        our_side = "b"
        opp_id = errlog_a
    
    if our_side == "a":
        our_score = d["a_points"][-1]
        opp_score = d["b_points"][-1]
        our_time = d["a_time_left"][-1]
        result_code = d["result"]
        # result: 0=A wins, 1=B wins, 2=tie
        if result_code == 0:
            outcome = "WIN"
        elif result_code == 1:
            outcome = "LOSS"
        else:
            outcome = "TIE"
    else:
        our_score = d["b_points"][-1]
        opp_score = d["a_points"][-1]
        our_time = d["b_time_left"][-1]
        result_code = d["result"]
        if result_code == 1:
            outcome = "WIN"
        elif result_code == 0:
            outcome = "LOSS"
        else:
            outcome = "TIE"
    
    # Count moves by type from left_behind
    left_behind = d["left_behind"]
    # A's moves are at even indices (0,2,4,...), B's at odd indices (1,3,5,...)
    move_counts = {"prime": 0, "carpet": 0, "plain": 0, "search": 0}
    opp_move_counts = {"prime": 0, "carpet": 0, "plain": 0, "search": 0}
    
    for i, move in enumerate(left_behind):
        if our_side == "a":
            if i % 2 == 0:  # A's turns
                move_counts[move] = move_counts.get(move, 0) + 1
            else:
                opp_move_counts[move] = opp_move_counts.get(move, 0) + 1
        else:
            if i % 2 == 1:  # B's turns
                move_counts[move] = move_counts.get(move, 0) + 1
            else:
                opp_move_counts[move] = opp_move_counts.get(move, 0) + 1
    
    # Count rat catches
    rat_caught = d["rat_caught"]
    our_rat_catches = 0
    opp_rat_catches = 0
    for i, caught in enumerate(rat_caught):
        if caught:
            if our_side == "a":
                if i % 2 == 0:
                    our_rat_catches += 1
                else:
                    opp_rat_catches += 1
            else:
                if i % 2 == 1:
                    our_rat_catches += 1
                else:
                    opp_rat_catches += 1
    
    # Compute carpet points
    new_carpets = d["new_carpets"]
    our_carpet_pts = 0
    opp_carpet_pts = 0
    for i, carpet_list in enumerate(new_carpets):
        if carpet_list:  # non-empty carpet placement
            size = len(carpet_list)
            pts = carpet_points_table.get(size, 0)
            if our_side == "a":
                if i % 2 == 0:
                    our_carpet_pts += pts
                else:
                    opp_carpet_pts += pts
            else:
                if i % 2 == 1:
                    our_carpet_pts += pts
                else:
                    opp_carpet_pts += pts
    
    results.append({
        "match": match_num,
        "our_side": our_side,
        "errlog_a": errlog_a,
        "errlog_b": errlog_b,
        "opp_id": opp_id,
        "our_score": our_score,
        "opp_score": opp_score,
        "outcome": outcome,
        "result_code": result_code,
        "our_time": our_time,
        "moves": move_counts,
        "opp_moves": opp_move_counts,
        "our_rat_catches": our_rat_catches,
        "opp_rat_catches": opp_rat_catches,
        "our_carpet_pts": our_carpet_pts,
        "opp_carpet_pts": opp_carpet_pts,
    })

# Print summary table
print(f"{'Match':<7} {'Side':<5} {'Opp':<8} {'Our':>4} {'Opp':>4} {'Result':<6} {'TimeLeft':>9} {'Prime':>6} {'Carpet':>7} {'Plain':>6} {'Search':>7} {'RatUs':>6} {'RatOp':>6} {'CarpPtsUs':>10} {'CarpPtsOp':>10}")
print("-" * 130)

wins = 0
losses = 0
ties = 0
total_our = 0
total_opp = 0

for r in results:
    m = r["moves"]
    print(f"{r['match']:<7} {r['our_side'].upper():<5} {r['opp_id']:<8} {r['our_score']:>4} {r['opp_score']:>4} {r['outcome']:<6} {r['our_time']:>9.1f} {m.get('prime',0):>6} {m.get('carpet',0):>7} {m.get('plain',0):>6} {m.get('search',0):>7} {r['our_rat_catches']:>6} {r['opp_rat_catches']:>6} {r['our_carpet_pts']:>10} {r['opp_carpet_pts']:>10}")
    if r["outcome"] == "WIN":
        wins += 1
    elif r["outcome"] == "LOSS":
        losses += 1
    else:
        ties += 1
    total_our += r["our_score"]
    total_opp += r["opp_score"]

total = wins + losses + ties
win_pct = wins / total * 100 if total > 0 else 0

print()
print(f"Overall Record: {wins}W - {losses}L - {ties}T  ({total} games)")
print(f"Win Percentage: {win_pct:.1f}%")
print(f"Total Points: Us {total_our} vs Opp {total_opp} (Avg: {total_our/total:.1f} vs {total_opp/total:.1f})")

# Aggregate move counts
total_moves = {"prime": 0, "carpet": 0, "plain": 0, "search": 0}
for r in results:
    for k, v in r["moves"].items():
        total_moves[k] = total_moves.get(k, 0) + v
print(f"\nTotal Our Moves: prime={total_moves['prime']}, carpet={total_moves['carpet']}, plain={total_moves['plain']}, search={total_moves['search']}")

total_our_carpet = sum(r["our_carpet_pts"] for r in results)
total_opp_carpet = sum(r["opp_carpet_pts"] for r in results)
total_our_rats = sum(r["our_rat_catches"] for r in results)
total_opp_rats = sum(r["opp_rat_catches"] for r in results)
print(f"Total Carpet Points: Us {total_our_carpet} vs Opp {total_opp_carpet}")
print(f"Total Rat Catches: Us {total_our_rats} vs Opp {total_opp_rats}")
