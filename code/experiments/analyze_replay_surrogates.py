"""Compare matched original FLAT parameter sets after ten-seed replay."""
import json
import numpy as np
import pandas as pd
import run_replay_surrogates as protocol


def main():
    out = protocol.OUT
    jobs, manifest = protocol.prepare()
    assert json.loads((out/'manifest.json').read_text(encoding='utf-8')) == manifest
    df = pd.read_csv(out/'per_replay.csv')
    assert len(df)==720 and (df.status=='ok').all()
    assert not df.duplicated(['scene','arm','opt_seed','sim_seed']).any()
    expected = {protocol.pilot.key(j):j for j in jobs}
    for row in df.to_dict('records'):
        job = expected[protocol.pilot.key(row)]
        assert row['source_sha256']==job['source_sha256']
    parameters = pd.DataFrame(manifest['parameter_sets']).drop(columns=['params'])
    per_opt = df.groupby(['scene','arm','opt_seed']).jb.mean().rename('replay_jb').reset_index()
    per_opt = per_opt.merge(parameters[['scene','arm','opt_seed','original_best_jb']],validate='one_to_one')
    per_opt['replay_minus_search'] = per_opt.replay_jb-per_opt.original_best_jb
    per_opt.to_csv(out/'original_vs_replay.csv',index=False)
    scene_rows = []
    for scene in protocol.SCENES:
        row = {'scene':scene}
        block = per_opt[per_opt.scene==scene]
        means = block.groupby('arm').replay_jb.mean()
        for arm,label in [('BF-SAC-GP','gp'),('BF-SAC-RF','rf'),('BF-SAC-MLP','mlp')]:
            vals = block[block.arm==arm]
            row[label+'_search_mean'] = vals.original_best_jb.mean()
            row[label+'_replay_mean'] = vals.replay_jb.mean()
            row[label+'_replay_sd'] = vals.replay_jb.std(ddof=1)
        row['best_backend'] = means.idxmin()
        row['max_minus_min'] = means.max()-means.min()
        row['max_relative_gap_pct'] = 100*(means.max()/means.min()-1)
        scene_rows.append(row)
    scenes = pd.DataFrame(scene_rows)
    scenes.to_csv(out/'scene_comparison.csv',index=False)
    global_rows = []
    for arm in protocol.ARMS:
        block = per_opt[per_opt.arm==arm]
        scene_means = block.groupby('scene').replay_jb.mean()
        first = df[(df.arm==arm)&(df.sim_seed<=10005)].groupby(['scene','opt_seed']).jb.mean().mean()
        second = df[(df.arm==arm)&(df.sim_seed>10005)].groupby(['scene','opt_seed']).jb.mean().mean()
        row = dict(arm=arm,search_mean=block.original_best_jb.mean(),
                   replay_mean=scene_means.mean(),first5_mean=first,second5_mean=second,
                   mean_replay_minus_search=block.replay_minus_search.mean())
        # Monte Carlo variation with the selected parameter sets held fixed.
        seed_means = df[df.arm==arm].groupby('sim_seed').jb.mean()
        row['conditional_mc_se'] = seed_means.std(ddof=1)/np.sqrt(10)
        global_rows.append(row)
    global_table = pd.DataFrame(global_rows)
    global_table.to_csv(out/'global_comparison.csv',index=False)
    gaps = []
    by_seed = df.groupby(['sim_seed','arm']).jb.mean().unstack()
    for a,b in [('BF-SAC-GP','BF-SAC-RF'),('BF-SAC-GP','BF-SAC-MLP'),('BF-SAC-RF','BF-SAC-MLP')]:
        d = by_seed[b]-by_seed[a]
        gaps.append(dict(reference=a,comparison=b,mean_difference=float(d.mean()),
                         conditional_gap_mc_se=float(d.std(ddof=1)/np.sqrt(10))))
    gm = global_table.replay_mean
    analysis = dict(protocol='matched original parameter replays',
                    n_scenes=6,n_original_runs_per_backend_per_scene=4,n_replay_seeds=10,
                    global_results=global_rows,conditional_pairwise_precision=gaps,
                    max_global_backend_gap_pct=float(100*(gm.max()/gm.min()-1)),
                    scope='Only GP/RF/MLP original runs with common complete parameters; external baselines and objective ablation unavailable for full original replay.',
                    selection='optimization seeds 101,202,303,404 fixed by parameter availability before replay',
                    statistics='No significance claims; conditional MC errors describe simulation-seed precision, not uncertainty across traffic scenes.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    print(scenes.to_string(index=False))
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    main()
