import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('editorial_splits',ROOT/'patches'/'splits.py')
splits=importlib.util.module_from_spec(spec);spec.loader.exec_module(splits)

def fake_manifest():
    rows=[]
    for name, n_speakers in [('crema_d',40),('ravdess',24),('tess',2),('savee',4)]:
        for s in range(n_speakers):
            for label in ['anger','disgust','fear','happiness','sadness','neutral']:
                for rep in range(5):
                    rows.append({'corpus':name,'speaker_id':f'{name}:{s}',
                                 'label':label,'is_known':1,
                                 'relative_path':f'{name}/{s}_{label}_{rep}.wav'})
        if name!='crema_d':
            for r in range(3):
                rows.append({'corpus':name,'speaker_id':f'{name}:unknown:{r}',
                             'label':'surprise','is_known':0,
                             'relative_path':f'{name}/unknown_{r}.wav'})
    return pd.DataFrame(rows)

def test_lodo_all_folds_and_seeds():
    df=fake_manifest()
    for holdout in df.corpus.unique():
        for seed in [42,52,62]:
            a=splits.make_lodo_splits(df,holdout,seed=seed)
            b=splits.make_lodo_splits(df,holdout,seed=seed)
            for part in ['train','val','cal','test']:
                assert np.array_equal(a[part],b[part])
            assert set(df.iloc[a['test']].corpus)=={holdout}
            for part in ['train','val','cal']:
                assert holdout not in set(df.iloc[a[part]].corpus)
            for i,p in enumerate(['train','val','cal','test']):
                for other in ['train','val','cal','test'][i+1:]:
                    assert not (set(a[p])&set(a[other]))
            assert a['methods'].get('tess') in (None,'source_utterance_stratified_fallback')
            if 'tess' in a['methods']:
                assert a['methods']['tess']=='source_utterance_stratified_fallback'

def test_fail_on_bad_fractions():
    df=fake_manifest()
    try:
        splits.make_lodo_splits(df,'ravdess',seed=42,validation_fraction=.8,calibration_fraction=.3)
    except ValueError:
        pass
    else:
        raise AssertionError('Bad partition fractions must fail')
