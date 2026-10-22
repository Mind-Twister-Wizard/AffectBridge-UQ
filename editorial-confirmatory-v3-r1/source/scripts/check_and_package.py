"""Fail-closed paper export: requires 12 full and 30 paired runs plus isolation audits."""
import argparse,csv,hashlib,json,math,statistics,zipfile
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--engine',required=True);a=p.parse_args()
root=Path(a.engine).resolve(); out=root/'outputs'
records=json.loads((out/'all_results.json').read_text(encoding='utf8'))
expected_folds={'crema_d','ravdess','tess','savee'}
seeds={42,52,62}
full={(r['holdout'],r['seed']) for r in records if r['ablation']=='full_DIMAP-C'}
assert full=={(f,s) for f in expected_folds for s in seeds},f'Incomplete full-model runs: {len(full)}/12'
paired_names={'no_prototypes','global_prototype','no_domain_adversarial','no_speaker_adversarial','no_cross_corpus_mixup'}
abl={(r['holdout'],r['seed'],r['ablation']) for r in records if r['ablation'] in paired_names}
expect={(f,s,n) for f in ('crema_d','ravdess') for s in seeds for n in paired_names}
assert abl==expect,f'Incomplete paired ablations: {len(abl)}/30'
assert len(records)==42, f'Expected 42 total runs, got {len(records)}'
manifest=root/'data'/'metadata'/'manifest.csv'
rows=list(csv.DictReader(manifest.open(encoding='utf8',newline='')))
assert rows
seen=set()
for row in rows:
    key=(row['corpus'],row['content_sha256'])
    assert key not in seen,f'Duplicate recorded within corpus: {key[0]}'
    seen.add(key)
for row in records:
    assert row.get('protocol_version')=='editorial_confirmatory_v3_r1'
    assert row.get('implementation_version')=='DIMAP-C-v3-editorial-confirmatory-r1'
    assert all(k in row for k in ('ece_pre','ece_post','macro_f1','accuracy','uar'))
    assert all(math.isfinite(float(row[k])) for k in ('ece_pre','ece_post','macro_f1','accuracy','uar'))
    run=root/'outputs'/'runs'/f"{row['ablation']}__holdout_{row['holdout']}__seed_{row['seed']}"
    for name in ('split_audit.json','checkpoint.pt','metrics.json','predictions.csv','calibration.json'):
        assert (run/name).is_file(),f'Missing {run/name}'
    audit=json.loads((run/'split_audit.json').read_text(encoding='utf8'))
    assert audit.get('calibration_separate_from_model_selection') is True
    assert int(row['selection_count'])>0 and int(row['calibration_count'])>0
    assert audit['counts']['val']==row['selection_count']
    assert audit['counts']['cal']==row['calibration_count']
assert (out/'final_report.html').is_file()
bykey={(r['holdout'],r['seed'],r['ablation']):r for r in records}
deltas=[]
for fold,seed,name in sorted(expect):
    b=bykey[(fold,seed,'full_DIMAP-C')];r=bykey[(fold,seed,name)]
    deltas.append({'holdout':fold,'seed':seed,'ablation':name,'full_macro_f1':b['macro_f1'],
       'ablation_macro_f1':r['macro_f1'],'delta_full_minus_ablation':b['macro_f1']-r['macro_f1']})
with (out/'paired_ablation_seed_level.csv').open('w',newline='',encoding='utf8') as fh:
    wr=csv.DictWriter(fh,fieldnames=deltas[0]);wr.writeheader();wr.writerows(deltas)
agg=[]
for fold in ('crema_d','ravdess'):
 for name in sorted(paired_names):
    subset=[x['delta_full_minus_ablation'] for x in deltas if x['holdout']==fold and x['ablation']==name]
    agg.append({'holdout':fold,'ablation':name,'seeds':'42,52,62','delta_macro_f1_mean':statistics.mean(subset),
      'delta_macro_f1_sd':statistics.stdev(subset)})
with (out/'paired_ablation_summary.csv').open('w',newline='',encoding='utf8') as fh:
    wr=csv.DictWriter(fh,fieldnames=agg[0]);wr.writeheader();wr.writerows(agg)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
checks={'experiment_status':'COMPLETE_VERIFIED','protocol':'editorial_confirmatory_v3_r1',
'full_model_runs':12,'paired_ablation_runs':30,'dataset_manifest_rows':len(rows),
'deduplicated_manifest_sha256':sha(manifest),'code_patch_marker':json.loads((root/'.editorial_confirmatory_patch.json').read_text(encoding='utf8')),
'notes':['New numerical results only; never reuse pre-deduplication result tables',
'Conformal 90% guarantee not asserted under target shift',
'Human-review policy was evaluated offline; no live user study',
'Paired ablations only for CREMA-D and RAVDESS; no inference beyond these folds']}
(out/'VERIFIED_COMPLETE.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
selected=['VERIFIED_COMPLETE.json','final_report.html','paper_main_results.csv','results_summary.csv',
'paired_ablation_seed_level.csv','paired_ablation_summary.csv','dataset_summary.csv','ablation_results.csv','run_manifest.json']
archive=out/'EDITORIAL_RESULTS_FOR_MANUSCRIPT.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in selected:
    path=out/name
    if path.is_file():z.write(path,arcname=name)
 for path in sorted((out/'figures').glob('*.png')):z.write(path,arcname=f'figures/{path.name}')
print('SUCCESS: 12 complete corrected model runs, 30 paired ablation runs, no within-corpus duplicate hashes, separate calibration audit, paired exports.');print(archive)
