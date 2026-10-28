"""Frozen Wav2Vec-BERT baseline. No uploads; hidden evaluation is a separate command."""
import argparse
import json
import random
import wave
import time
from pathlib import Path

from baseline_data import ROOT, LABELS, load_split, classification_metrics, digest


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def dependencies():
    global torch, np, AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification


def freeze_encoder(model):
    for parameter in model.wav2vec2_bert.parameters():
        parameter.requires_grad_(False)
    model.wav2vec2_bert.eval()


def batch_inputs(rows, extractor, device):
    arrays = []
    for row in rows:
        with wave.open(row['audio_path'], 'rb') as stream:
            arrays.append(np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768.0)
    inputs = extractor(arrays, sampling_rate=16000, padding=True,
                       return_attention_mask=True, return_tensors='pt')
    return {k: v.to(device) for k, v in inputs.items()}


def pooled_encoder(model, inputs):
    if model.config.use_weighted_layer_sum or model.config.add_adapter:
        raise ValueError('Pooled cache requires the fixed last-layer encoder without adapters')
    hidden = model.wav2vec2_bert(**inputs).last_hidden_state
    mask = model._get_feature_vector_attention_mask(hidden.shape[1], inputs['attention_mask'])
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def extract_features(model, extractor, rows, batch_size, device, name):
    model.eval()
    features = {}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs = batch_inputs(batch, extractor, device)
            pooled = pooled_encoder(model, inputs)
            if start == 0:
                direct = model(**inputs).logits
                cached = model.classifier(model.projector(pooled))
                torch.testing.assert_close(direct, cached, atol=1e-5, rtol=1e-4)
            features.update({r['sample_id']: f.detach().cpu() for r,f in zip(batch, pooled)})
            if start % 200 == 0 or start + len(batch) == len(rows):
                print(f'features={name} samples={start+len(batch)}/{len(rows)} seconds={time.perf_counter()-started:.1f}', flush=True)
    return features


def head_logits(model, features, rows, device):
    vectors = torch.stack([features[r['sample_id']] for r in rows]).to(device)
    return model.classifier(model.projector(vectors))


def evaluate(model, extractor, rows, batch_size, device, features=None):
    model.eval()
    truth, predicted, records = [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            logits = (head_logits(model, features, batch, device) if features is not None
                      else model(**batch_inputs(batch, extractor, device)).logits)
            ids = logits.argmax(-1).cpu().tolist()
            truth.extend(LABELS.index(r['region']) for r in batch)
            predicted.extend(ids)
            records.extend(dict(sample_id=r['sample_id'], prediction=LABELS[p]) for r, p in zip(batch, ids))
    return classification_metrics(truth, predicted), records


def train(config, train_rows, validation_rows, hashes, smoke=False):
    dependencies()
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = ROOT / config['output_dir']
    # Do not silently overwrite an existing run or turn a smoke run into a baseline score.
    out.mkdir(parents=True, exist_ok=False)
    cache = str(ROOT / config['cache_dir'])
    extractor = AutoFeatureExtractor.from_pretrained(config['model_id'], revision=config['revision'], cache_dir=cache)
    model = Wav2Vec2BertForSequenceClassification.from_pretrained(
        config['model_id'], revision=config['revision'], cache_dir=cache,
        num_labels=5, label2id={v:i for i,v in enumerate(LABELS)},
        id2label=dict(enumerate(LABELS)), apply_spec_augment=False,
        use_weighted_layer_sum=False).to(device)
    freeze_encoder(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    import transformers
    run = dict(config=config, manifest_hashes=hashes, device=str(device),
               torch_version=torch.__version__, transformers_version=transformers.__version__,
               numpy_version=np.__version__, smoke_only=smoke,
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
               selection_metric='validation_macro_f1', global_speaker_independence=False)
    save_json(out / 'run.json', run)
    started = time.perf_counter()
    train_features = val_features = None
    if config.get('cache_frozen_features', False):
        train_features = extract_features(model, extractor, train_rows, config['batch_size'], device, 'train')
        val_features = extract_features(model, extractor, validation_rows, config['batch_size'], device, 'validation')
        torch.save(dict(train=train_features, validation=val_features, manifest_hashes=hashes,
                        model_revision=config['revision']), out / 'frozen_features.pt')
    run['feature_extraction_seconds'] = time.perf_counter() - started
    save_json(out / 'run.json', run)
    best, stale, history = -1.0, 0, []
    for epoch in range(config['epochs']):
        model.train()
        # requires_grad=False alone does not disable encoder dropout/layerdrop.
        model.wav2vec2_bert.eval()
        order = list(train_rows)
        random.Random(config['seed'] + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), config['batch_size']):
            batch = order[start:start + config['batch_size']]
            labels = torch.tensor([LABELS.index(r['region']) for r in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.cross_entropy(head_logits(model, train_features, batch, device), labels)
                    if train_features is not None else model(**batch_inputs(batch, extractor, device), labels=labels).loss)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
            if start % 200 == 0:
                print(f'epoch={epoch+1} samples={start+len(batch)}/{len(order)} loss={loss.item():.4f}', flush=True)
        metrics, _ = evaluate(model, extractor, validation_rows, config['batch_size'], device, val_features)
        history.append(dict(epoch=epoch + 1, train_loss=total_loss/len(order), validation=metrics))
        save_json(out / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            model.save_pretrained(out / 'best', safe_serialization=True)
            extractor.save_pretrained(out / 'best')
            save_json(out / 'best_metrics.json', history[-1])
        else:
            stale += 1
        if stale >= config['patience']:
            break
    run['total_training_seconds'] = time.perf_counter() - started
    run['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
    run['completed_epochs'] = len(history)
    save_json(out / 'run.json', run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'train', 'evaluate'])
    parser.add_argument('--config', default='configs/baseline_freeze.json')
    parser.add_argument('--split', choices=['validation', 'public_test', 'hidden_test'], default='public_test')
    parser.add_argument('--organizer', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One update on 2 training clips; not a performance result')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text('utf-8'))
    version = ROOT / config['version_dir']
    if args.command in ('preflight', 'train'):
        train_rows, train_hash = load_split(version, 'train', verify_audio=True)
        val_rows, val_hash = load_split(version, 'validation', verify_audio=True)
        for field in ('recording_key', 'audio_member_sha256', 'pcm_sha256'):
            if {r[field] for r in train_rows} & {r[field] for r in val_rows}:
                raise ValueError(f'Train/validation overlap: {field}')
        hashes = dict(train=train_hash, validation=val_hash)
        print(json.dumps(dict(train=len(train_rows), validation=len(val_rows), manifest_hashes=hashes)), flush=True)
        if args.command == 'preflight':
            return
        if args.smoke:
            config['output_dir'] += '_smoke'
            config['epochs'] = 1
            train_rows, val_rows = train_rows[:2], val_rows[:2]
        train(config, train_rows, val_rows, hashes, args.smoke)
    else:
        if args.smoke:
            parser.error('--smoke is only supported for training')
        if args.split == 'hidden_test' and not args.organizer:
            parser.error('Hidden evaluation requires the separate --organizer option')
        out = ROOT / config['output_dir']
        run = json.loads((out / 'run.json').read_text('utf-8'))
        if run['smoke_only']:
            raise ValueError('Smoke checkpoints cannot be used for benchmark evaluation')
        rows, manifest_hash = load_split(version, args.split, verify_audio=True)
        # Verify the evaluation version still has the exact training/validation manifests.
        for split, locked_hash in run['manifest_hashes'].items():
            if load_split(version, split)[1] != locked_hash:
                raise ValueError('Checkpoint dataset version mismatch')
        dependencies()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Wav2Vec2BertForSequenceClassification.from_pretrained(out / 'best', local_files_only=True).to(device)
        if model.config.label2id != {v:i for i,v in enumerate(LABELS)}:
            raise ValueError('Checkpoint label mapping mismatch')
        extractor = AutoFeatureExtractor.from_pretrained(out / 'best', local_files_only=True)
        metrics, predictions = evaluate(model, extractor, rows, config['batch_size'], device)
        destination = out / ('organizer' if args.split == 'hidden_test' else 'evaluation')
        destination.mkdir(exist_ok=True)
        save_json(destination / f'{args.split}_metrics.json', dict(**metrics, manifest_sha256=manifest_hash))
        save_json(destination / f'{args.split}_predictions.json', predictions)
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
