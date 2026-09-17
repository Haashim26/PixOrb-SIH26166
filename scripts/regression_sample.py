"""Run the supplied team sample pair through the final PixOrb pipeline."""
import sys
from pathlib import Path
base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))
from pixorb.pipeline import run

ref = base/'data'/'sample_pair'/'pair1_reference.jpg'
src = base/'data'/'sample_pair'/'pair1_source_original.jpg'
result = run(ref, src, base/'results'/'sample_run')
import json
print(json.dumps(result['metrics'], indent=2, default=str))
