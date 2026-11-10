"""loudness_process — loudness / dynamics conditioning for singing-voice datasets.

Audit -> normalise -> verify -> listening comparison, in one command line:

    python -m loudness_process audit   <in_dir>
    python -m loudness_process process <in_dir> <out_dir>
    python -m loudness_process run     <in_dir> <out_dir>
    python -m loudness_process verify  <in_dir> <out_dir>
    python -m loudness_process ab      <in_dir> <out_dir>

See README.md for the workflow and the parameter reference.
"""

__version__ = '1.0.0'

from .dataset import audit, discover, group_of, load_features  # noqa: F401
from .loudness import integrated_loudness, lufs, true_peak_db  # noqa: F401
from .normalize import Options, design_offset, plan_dataset, process_group  # noqa: F401
