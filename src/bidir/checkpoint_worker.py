"""Generate with one full checkpoint in a disposable GPU process.

The orchestrator supplies identical messages/sampling for every checkpoint. No full
checkpoint is ever passed as a LoRARequest. Process exit releases GPU memory before
the next checkpoint or frozen scoring model is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

from bidir import engine as eng
from bidir.config import resolve_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    task = json.loads(Path(args.input).read_text())
    e = eng.get_engine(task['checkpoint'], task['engine'])
    # Render with the original model tokenizer for every weight condition. This
    # prevents checkpoint tokenizer metadata from changing the experiment prompt.
    from transformers import AutoTokenizer
    e._tokenizer = AutoTokenizer.from_pretrained(resolve_model(task['model'])['hf_id'])
    rendered = eng.render(e, task['messages'])
    raw, tokens = e.generate(rendered, task['sampling'], task['adapters'])
    Path(args.output).write_text(json.dumps(dict(
        raw=raw, tokens=tokens, engine_version=e.version(),
        rendered_sha256=hashlib.sha256(json.dumps(rendered).encode()).hexdigest())))
    return 0


if __name__ == '__main__':
    try:
        rc = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    eng.shutdown_and_exit(rc)
