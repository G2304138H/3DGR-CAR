import json
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
from evaluate_stage2_npz import NpzIndex, projection_case_identity
from gcp_view_direction_robustness import resolve_robustness_plan, run_view_direction_robustness


def test_nested_cases_are_distinct_and_match_original_splits(tmp_path):
    for n in (1, 23):
        p=tmp_path/'rca'/str(n)/'vessel_code_step6.npz'
        p.parent.mkdir(parents=True)
        np.savez(p, images=np.ones((2,8,8)))
    index=NpzIndex(tmp_path/'rca')
    for n in (1,23):
        p=index.resolve([f'rca_{n:04d}'], 'projection')
        assert p.parent.name==str(n)
        assert projection_case_identity(p)==(f'rca_{n:04d}','rca',str(n))
    np.savez(p,images=np.ones((2,8,8)),case_id='99')
    with pytest.raises(ValueError,match='Conflicting case_id'):
        projection_case_identity(p)


def test_scalar_sweep_runs_without_baseline_and_saves_masks(tmp_path):
    resolved=dict(eval_output_dir=str(tmp_path/'out'),checkpoint_path=str(tmp_path/'best.pt'),
        checkpoint_choice='best',experiment_dir=str(tmp_path),eval_split='test',
        effective_view_sweep=[dict(eval_num_views=2,view_indices=[0,1])],
        save_prediction_npz_files=True,
        view_direction_robustness=dict(error_degrees=[.2,.5,1.,2.],
            accurate_baseline_summary=None,require_accurate_baseline=False,save_mask_npz_files=True))
    calls=[]
    def run(command, **kwargs):
        child=json.loads(Path(command[-1]).read_text());calls.append(child)
        assert child['paper_metric_save_masks'] and child['save_prediction_npz_files']
        angles=child['evaluation_view_directions']
        assert angles['theta_change_deg']==0 and not angles['accurate']
        out=Path(child['eval_output_dir']);out.mkdir(parents=True)
        (out/'cases.json').write_text(json.dumps([dict(case_name='rca_0023',split='test',eval_num_views=2)]))
        (out/'performance_summary.json').write_text(json.dumps(dict(
            comparison_condition=dict(checkpoint=resolved['checkpoint_path'],evaluation_split='test',effective_view_sweep=resolved['effective_view_sweep']),
            roles_by_view_count={'k2':{'optimized':{'metrics':{'masked_dice_3d':{'mean':.8}}}}},
            per_case_metrics_file='cases.json')))
        return mock.Mock(returncode=0)
    with mock.patch('gcp_view_direction_robustness.subprocess.run',side_effect=run):
        summary=run_view_direction_robustness(resolved=resolved,config_path=tmp_path/'config.json',evaluator_path=tmp_path/'evaluate.py')
    data=json.loads(summary.read_text())
    assert data['num_conditions']==4 and data['status']=='complete'
    assert data['quadratic_response_surfaces'] is None
    assert [c['evaluation_view_directions']['phi_change_deg'] for c in calls]==[.2,.5,1.,2.]
    assert [Path(c['eval_output_dir']).name for c in calls]==['error_0p2deg','error_0p5deg','error_1deg','error_2deg']


def test_scalar_validation():
    for errors in ([],[-1],[float('nan')],[181],[True]):
        with pytest.raises(ValueError):
            resolve_robustness_plan({'view_direction_robustness':{'error_degrees':errors}})
    with pytest.raises(ValueError):
        resolve_robustness_plan({'view_direction_robustness':{'error_degrees':[1],'axis_degrees':[2]}})


@pytest.mark.parametrize("artery", ["rca", "lca"])
def test_ready_configs_preserve_checkpoint_and_optimization(artery):
    from evaluate_gcp import resolve_view_sweep
    configs=Path(__file__).parent/'configs'
    original=json.loads((configs/f'eval_gcp_paper_metric_{artery}_val_test.json').read_text())
    for kind in ('seg_err_artifacts_normal','view_direction'):
        c=json.loads((configs/f'eval_gcp_{artery}_{kind}_test.json').read_text())
        assert c['model']==original['model']
        assert c['gaussian_optimization']==original['gaussian_optimization']
        assert c['split_json_path']==original['split_json_path']
        assert c['eval_split']=='test'
        assert resolve_view_sweep(c)==[(2,[0,1])]
        assert c['paper_metric_save_masks'] and c['save_prediction_npz_files']
        assert c['eval_output_dir']!=original['eval_output_dir']
        if kind=='view_direction':
            plan=resolve_robustness_plan(c)
            assert plan['conditions']==[(0.,.2),(0.,.5),(0.,1.),(0.,2.)]
            assert plan['save_mask_npz_files'] and not plan['require_accurate_baseline']
        else:
            assert c['evaluation_dataset_dir'].endswith('seg_err_artifacts_normal/'+artery)
