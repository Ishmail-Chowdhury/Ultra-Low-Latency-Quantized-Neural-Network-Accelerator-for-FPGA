"""Compare frozen-model float and fixed preprocessing; no training or selection."""
import argparse
import json
from pathlib import Path

from models.lob.data import read_snapshots, feature_window
from models.lob.fixed_features import ONE, read_fixed_snapshots
from models.lob.inference import LobPredictor
from models.lob.metrics import classification_metrics
from scripts.fetch_itch_sample import sha256_file


def validate(model, frontend, source, output, rtl_samples=0):
    floating = LobPredictor(model)
    fixed = LobPredictor(model, frontend=frontend)
    report = json.loads((model / 'report.json').read_text())
    if sha256_file(source) != report['source_sha256']:
        raise ValueError('Source differs from the frozen model dataset')
    rows = read_snapshots(source, floating.config.levels)
    integer_rows = read_fixed_snapshots(source, floating.config.levels)
    vectors = json.loads((model / 'vectors.json').read_text())['vectors']
    labels, original_predictions, fixed_predictions, inputs = [], [], [], []
    max_feature_error = max_normalized_error = 0.
    input_differences = input_count = logit_differences = 0
    max_input_difference = 0
    for vector in vectors:
        i = vector['dataset_index']
        raw = feature_window(rows, i, floating.config)
        integer_raw = fixed.frontend.feature_window(integer_rows, i)
        normalized = fixed.frontend.normalize(integer_raw)
        x = fixed.frontend.quantize(normalized)
        expected = floating.preprocess(rows, i)
        if expected != vector['inputs']:
            raise ValueError('Floating preprocessing does not reproduce saved vectors')
        max_feature_error = max(max_feature_error, max(abs(a / ONE - b) for a,b in zip(integer_raw,raw)))
        max_normalized_error = max(max_normalized_error, max(abs(a / ONE - b) for a,b in zip(normalized,floating.scaler.transform(raw))))
        differences = [abs(a-b) for a,b in zip(x,expected)]
        input_differences += sum(d != 0 for d in differences)
        input_count += len(x)
        max_input_difference = max(max_input_difference,max(differences))
        _, hidden = fixed.first.evaluate(x)
        _, logits = fixed.second.evaluate(hidden)
        logit_differences += logits != vector['outputs']
        labels.append(vector['label'])
        original_predictions.append(max(range(3),key=lambda k:vector['outputs'][k]))
        fixed_predictions.append(max(range(3),key=lambda k:logits[k]))
        inputs.append(x)
    result = {'schema_version':1, 'scope':'arithmetic fidelity on previously examined test data; no new forecasting claim',
              'samples':len(vectors),'source_sha256':sha256_file(source),
              'model_report_sha256':sha256_file(model/'report.json'),
              'frontend_sha256':sha256_file(frontend/'frontend.json'),
              'max_raw_feature_absolute_error':max_feature_error,
              'max_normalized_absolute_error':max_normalized_error,
              'int8_input_values_changed':input_differences,'int8_input_values':input_count,
              'max_int8_input_difference':max_input_difference,'vectors_with_changed_logits':logit_differences,
              'predictions_changed':sum(a!=b for a,b in zip(original_predictions,fixed_predictions)),
              'float_preprocessing_metrics':classification_metrics(labels,original_predictions),
              'fixed_preprocessing_metrics':classification_metrics(labels,fixed_predictions)}
    if rtl_samples:
        if not 2 <= rtl_samples <= len(inputs):
            raise ValueError('RTL sample count must be between two and held-out count')
        from tb.dense_runner import verify_mlp
        positions=[i*(len(inputs)-1)//(rtl_samples-1) for i in range(rtl_samples)]
        selected=[inputs[i] for i in positions]
        result['rtl']={'samples':rtl_samples,'positions':positions,'scope':'existing MLP RTL driven by software fixed features; frontend is not RTL',
                       'cases':{str(stalls):verify_mlp(fixed.first,fixed.second,selected,stalls=stalls,
                         memory_directory=model,hidden_lanes=min(4,fixed.first.outputs),output_lanes=min(3,fixed.second.outputs))
                                for stalls in (False,True)}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rtl',)},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('build/prediction-quality'))
    p.add_argument('--frontend',type=Path,default=Path('build/fixed-features/validated-frontend'))
    p.add_argument('--input',type=Path,default=Path('build/nasdaq-days/snapshots.csv'))
    p.add_argument('--output',type=Path,default=Path('build/fixed-features/validation.json'))
    p.add_argument('--rtl-samples',type=int,default=0)
    a=p.parse_args()
    validate(a.model,a.frontend,a.input,a.output,a.rtl_samples)


if __name__=='__main__':
    main()
