#!/bin/bash
export http_proxy=http://prodprox.idris.fr:3128
export https_proxy=http://prodprox.idris.fr:3128
export HTTP_PROXY=http://prodprox.idris.fr:3128
export HTTPS_PROXY=http://prodprox.idris.fr:3128
exec /lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/.venv/bin/python3 "$@"
