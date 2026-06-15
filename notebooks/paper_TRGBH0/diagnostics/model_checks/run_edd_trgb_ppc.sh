#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTDIR="$ROOT/notebooks/paper_TRGBH0/output/model_checks"
PYTHON="${CANDEL_PYTHON:-$ROOT/venv_candel/bin/python}"
PPC_FACTOR="${CANDEL_PPC_FACTOR:-10}"
# PPC_FACTOR=10
# MANTICORE_FIELD_INDEX="${CANDEL_MANTICORE_FIELD_INDEX:-10}"
MANTICORE_FIELD_INDEX=68
printf -v MANTICORE_FIELD_TAG "field%02d" "$MANTICORE_FIELD_INDEX"
B_MIN="${CANDEL_B_MIN:-10}"
B_MIN_TAG="${B_MIN//./p}"

mkdir -p "$OUTDIR"

# No reconstruction: Vext-only Gaussian cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode none \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_vext_only_bmin${B_MIN_TAG}_gaussian_ppc.pdf"

# # Carrick2015 reconstruction: Gaussian cz likelihood.
"$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
    --mode carrick \
    --b-min "$B_MIN" \
    --ppc-factor "$PPC_FACTOR" \
    --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_gaussian_ppc.pdf"

# Carrick2015 reconstruction: Student-t cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_cz-student_t_sel-TRGB_magnitude_Carrick2015_main.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_student_t_ppc.pdf"

# ManticoreLocalCOLA field reconstruction: Student-t cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode manticore \
#     --field-index "$MANTICORE_FIELD_INDEX" \
#     --posterior "$ROOT/results/TRGBH0_paper/single_fields/EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_${MANTICORE_FIELD_TAG}_single.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_manticore_${MANTICORE_FIELD_TAG}_bmin${B_MIN_TAG}_student_t_ppc.pdf"

# ManticoreLocalCOLA field reconstruction: Gaussian cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode manticore \
#     --field-index "$MANTICORE_FIELD_INDEX" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_manticore_${MANTICORE_FIELD_TAG}_bmin${B_MIN_TAG}_gaussian_ppc.pdf"
