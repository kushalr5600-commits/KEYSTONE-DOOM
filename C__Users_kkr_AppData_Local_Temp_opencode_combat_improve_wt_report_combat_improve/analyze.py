import json, csv, hashlib, os, statistics
from pathlib import Path
root=Path(r'C:\Users\kkr\AppData\Local\Temp\opencode\combat_improve_wt')
report=root/'report_combat_improve'
base=json.load(open(root/'transfer/verify_candidate_full_v2.json'))  # DoorPolicyNet baseline
cand=json.load(open(root/'transfer/verify_combat_improve_full_v2.json'))

def key(r): return (r['map'], int(r['seed']))
b={key(r):r for r in base}
c={key(r):r for r in cand}
common=sorted(set(b)&set(c))
rows=[]
for k in common:
    br,cr=b[k],c[k]
    rows.append({
        'map':br['map'],'seed':br['seed'],
        'base_exit':br['exit'],'cand_exit':cr['exit'],'delta_exit':cr['exit']-br['exit'],
        'base_deaths':br['dead'],'cand_deaths':cr['dead'],'delta_deaths':cr['dead']-br['dead'],
        'base_damage':br['damage'],'cand_damage':cr['damage'],'delta_damage':cr['damage']-br['damage'],
        'base_keys':br['keys'],'cand_keys':cr['keys'],'delta_keys':cr['keys']-br['keys'],
        'base_kills':br['kills'],'cand_kills':cr['kills'],'delta_kills':cr['kills']-br['kills'],
        'base_timeout':1 if br.get('term_reason')=='trunc' else 0,'cand_timeout':1 if cr.get('term_reason')=='trunc' else 0,
        'base_p99_ms':br.get('p99_ms'),'cand_p99_ms':cr.get('p99_ms'),
        'base_max_ms':br.get('max_ms'),'cand_max_ms':cr.get('max_ms'),
    })
with open(report/'seed_level_paired.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

def summ(rs, prefix):
    n=len(rs)
    return {
        f'{prefix}_n':n,
        f'{prefix}_exits':sum(r[f'{prefix}_exit'] for r in rs),
        f'{prefix}_deaths':sum(r[f'{prefix}_deaths'] for r in rs),
        f'{prefix}_damage':sum(r[f'{prefix}_damage'] for r in rs),
        f'{prefix}_keys':sum(r[f'{prefix}_keys'] for r in rs),
        f'{prefix}_kills':sum(r[f'{prefix}_kills'] for r in rs),
        f'{prefix}_timeouts':sum(r[f'{prefix}_timeout'] for r in rs),
    }

# aggregate
agg={'n':len(rows)}
for p in ['base','cand']: agg.update(summ(rows,p))
for m in ['exit','deaths','damage','keys','kills']:
    deltas=[r[f'delta_{m}'] for r in rows]
    agg[f'delta_{m}_sum']=sum(deltas)
    agg[f'delta_{m}_mean']=sum(deltas)/len(deltas)
    agg[f'delta_{m}_nonzero']=sum(1 for d in deltas if d!=0)
print('AGG',json.dumps(agg))

# per-map
map_rows=[]
maps=sorted(set(r['map'] for r in rows))
for m in maps:
    rs=[r for r in rows if r['map']==m]
    mr={'map':m,'n':len(rs)}
    for p in ['base','cand']: mr.update(summ(rs,p))
    for k in ['exit','deaths','damage','keys','kills']:
        mr[f'delta_{k}']=sum(r[f'delta_{k}'] for r in rs)
    map_rows.append(mr)
with open(report/'per_map_summary.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(map_rows[0].keys()))
    w.writeheader(); w.writerows(map_rows)
for mr in map_rows:
    print(f"{mr['map'].split('/')[-1]:30s} b_ex {mr['base_exits']}/{mr['n']} c_ex {mr['cand_exits']}/{mr['n']} d_ex {mr['delta_exit']:+d} b_dmg {mr['base_damage']:6.1f} c_dmg {mr['cand_damage']:6.1f} d_dmg {mr['delta_damage']:+6.1f} b_k {mr['base_kills']} c_k {mr['cand_kills']} d_k {mr['delta_kills']:+d} b_de {mr['base_deaths']} c_de {mr['cand_deaths']}")

# combat-only
combat=[r for r in rows if 'combat_holdout' in r['map']]
print('COMBAT_ONLY',len(combat))
for p in ['base','cand']: print(p,summ(combat,p))
