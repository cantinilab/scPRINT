#!/bin/bash
export http_proxy=http://prodprox.idris.fr:3128
export https_proxy=http://prodprox.idris.fr:3128
VENV=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/.venv
UV=/linkhome/rech/gennth01/uat95fg/.local/bin/uv
$UV pip install -q pyfaidx -p $VENV && echo "pyfaidx ok"
exec $VENV/bin/python3 scripts/extract_genomic_seqs_local.py "$@"
