"""Reproduce Appendix A scores and audit response-scale effects; no SUMO calls."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import analyze_landscape_learnability as learn
from predictive_scores import _nll, _crps_gaussian

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/results/probability_score_audit'


def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads((ROOT/'outputs/results/landscape_calibration_v2.json').read_text())
    rows,predictions,splits = [],[],[]
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('ignore')
        for scene in ('XAM-N6','Xian'):
            X,Yb = learn.load_xy(scene,'jb')
            Xs,Ys = learn.load_xy(scene,'js')
            assert np.array_equal(X,Xs) and len(X)==400
            train,test = learn.split_fixed(len(X),100,learn.BASE_SEED)
            for obj,Y in [('jb',Yb),('js',Ys)]:
                for rep in range(5):
                    sub = np.random.default_rng(learn.BASE_SEED+100*17+rep).choice(train,100,replace=False)
                    model = learn.fit_flat(scene,'gp',X[sub],Y[sub],run_seed=learn.BASE_SEED+rep)
                    mu,sigma = model.predict(X[test],return_std=True)
                    location,scale = float(np.mean(Y[sub])),float(np.std(Y[sub]))
                    assert scale>0
                    y_z,mu_z,sigma_z = (Y[test]-location)/scale,(mu-location)/scale,sigma/scale
                    nll,crps = _nll(Y[test],mu,sigma),_crps_gaussian(Y[test],mu,sigma)
                    nll_z,crps_z = _nll(y_z,mu_z,sigma_z),_crps_gaussian(y_z,mu_z,sigma_z)
                    assert np.isclose(nll_z,nll-np.log(scale),atol=1e-10)
                    assert np.isclose(crps_z,crps/scale,atol=1e-10)
                    q=np.array([.5,.8,.9,.95])
                    coverage=(np.abs(Y[test]-mu)[:,None] <= sigma[:,None]*norm.ppf((1+q)/2)).mean(axis=0)
                    rows.append(dict(scene=scene,objective=obj,rep=rep,train_mean=location,train_std=scale,
                                     nll_raw=nll,crps_raw=crps,nll_standardized=nll_z,crps_standardized=crps_z,
                                     mace=float(np.abs(coverage-q).mean()),coverage95=float(coverage[-1])))
                    splits.append(dict(scene=scene,objective=obj,rep=rep,train_indices=sub.tolist(),test_indices=test.tolist()))
                    predictions.extend(dict(scene=scene,objective=obj,rep=rep,index=int(idx),y=float(y),mu=float(m),sigma=float(s))
                                       for idx,y,m,s in zip(test,Y[test],mu,sigma))
    reps=pd.DataFrame(rows)
    metrics=['nll_raw','crps_raw','nll_standardized','crps_standardized','mace','coverage95']
    summary={}
    for (scene,obj),g in reps.groupby(['scene','objective']):
        block={metric:dict(mean=float(g[metric].mean()),std=float(g[metric].std(ddof=0))) for metric in metrics}
        for new,old in [('nll_raw','nll'),('crps_raw','crps')]:
            assert abs(block[new]['mean']-original['scenes'][scene][obj][old]['mean'])<1e-8
        summary.setdefault(scene,{})[obj]=block
    reps.to_csv(OUT/'per_subsample.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    provenance={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [ROOT/'outputs/results/landscape_full10d_XAM-N6.json',
                          ROOT/'outputs/results/landscape_full10d_Xian.json',
                          ROOT/'code/calibration/gp_surrogate.py',Path(__file__)]}
    output=dict(normalization='each training subset response mean and population SD; same transform applied to held-out y and predictions; no model refitting changes',
                aggregation='five subsamples, population SD as in original Appendix A',
                checks='original raw NLL/CRPS reproduced; exact affine score transformation verified',
                source_hashes=provenance,scenes=summary)
    (OUT/'metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
