"""Create the same v3-r1 local experiment settings as the completed run."""
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument("--engine",required=True)
a=p.parse_args()
root=Path(a.engine).resolve()
base=yaml.safe_load((root/"config.yaml").read_text(encoding="utf-8"))
base["output_dir"]=str(root/"outputs_editorial_confirmatory")
base["metadata_dir"]=str(root/"metadata_editorial_confirmatory")
base["cache_dir"]=str(root/"cache_editorial_confirmatory")
base["raw_dir"]=str(root/"data"/"raw")
base["run_name"]="affectbridge_editorial_confirmatory_v3r1"
base["seeds"]=[42,52,62]
base["resume_completed"]=True
base["feature_batch_size"]=1
(root/"editorial_config.yaml").write_text(yaml.safe_dump(base,sort_keys=False),encoding="utf-8")
print("Wrote editorial_config.yaml (upstream load_config routes outputs to ENGINE/outputs)")
