"""Grade the fixed N1 synthetic grid; numerical PASS never masks RSS failures."""
import argparse
import copy
import json
import math
from pathlib import Path

EXPECTED={(rhs,team,contiguous) for rhs in (1,5,20,30,31,32,60,250)
          for team in (1,2,8,16) for contiguous in (0,1)}


def grade(report):
    rows=report.get('rows',[])
    cells=[(r.get('rhs'),r.get('team'),r.get('contiguity')) for r in rows]
    coverage=len(cells)==64 and len(set(cells))==64 and set(cells)==EXPECTED
    numerical=[];resource=[]
    for row in rows:
        cell={key:row.get(key) for key in ('rhs','team','contiguity')}
        for arm in ('before','reference'):
            comparison=row.get('comparison',{}).get(arm,{})
            be=comparison.get('b_error',math.inf)
            ve=comparison.get('V_diagonal_scaled_error',math.inf)
            if not (math.isfinite(be) and be<=1e-9 and math.isfinite(ve) and ve<=1e-8
                    and comparison.get('same_sample') is True
                    and comparison.get('zero_covariance_equal') is True):
                numerical.append(dict(**cell,arm=arm))
        rss=row.get('rss_kib',{})
        if not all(isinstance(rss.get(arm),(int,float)) and math.isfinite(rss[arm])
                   and rss[arm]>0 for arm in ('candidate','before','reference')):
            numerical.append(dict(**cell,error='absent/non-finite resource receipt'))
            continue
        ratio=rss['candidate']/rss['reference']
        if ratio>1.15:
            resource.append(dict(**cell,rss_kib=rss,candidate_reference_ratio=ratio,
                candidate_before_ratio=rss['candidate']/rss['before']))
    if not coverage:status='COVERAGE_MISSING'
    elif numerical or report.get('failures'):status='FAIL_NUMERICAL_OR_FIT'
    elif resource:status='FAIL_RESOURCE_ACCEPTANCE'
    else:status='PASS_SCOPED_GRID'
    return dict(status=status,complete_grid=coverage,numerical_failures=numerical,
        rss_failures=resource,rss_limit=1.15,
        scope='Fixed synthetic Comparable grid only; neither mathematical oracle nor timing/release certification')


def selftest(report):
    result=grade(report)
    assert result['status']=='FAIL_RESOURCE_ACCEPTANCE' and len(result['rss_failures'])==4
    # Reject a misleading numerical PASS and an empty producer failure list.
    masked=copy.deepcopy(report);masked['status']='PASS';masked['memory_excess_cells']=[]
    assert grade(masked)['rss_failures']==result['rss_failures']
    missing=copy.deepcopy(report);missing['rows'].pop()
    assert grade(missing)['status']=='COVERAGE_MISSING'
    wrong=copy.deepcopy(report);wrong['rows'][0]['comparison']['before']['b_error']=1e-4
    assert grade(wrong)['status']=='FAIL_NUMERICAL_OR_FIT'
    nonfinite=copy.deepcopy(report);nonfinite['rows'][0]['rss_kib']['candidate']=math.nan
    assert grade(nonfinite)['status']=='FAIL_NUMERICAL_OR_FIT'
    print('SELFTEST PASS: masked RSS failures, missing row, bad estimate and non-finite receipt rejected')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args();report=json.loads(args.report.read_text())
    if args.selftest:selftest(report)
    result=grade(report)
    if args.output:
        with args.output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result))
    return result['status']!='PASS_SCOPED_GRID'


if __name__=='__main__':raise SystemExit(main())
