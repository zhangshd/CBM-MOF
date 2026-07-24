#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="${REPO_ROOT}/release/zenodo_r1"
MODELS_ARCHIVE="cbm_mof_models.tar.gz"
DATA_ARCHIVE="cbm_mof_key_data.tar.gz"
ZENODO_API="https://zenodo.org/api"

MODEL_FILES=(
  "results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0150.pt"
  "results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt"
  "results/alignn/model_ep150/uq/uncertainty_trees.pkl"
)

DATA_FILES=(
  "data/alignn/train/id_prop.csv"
  "data/alignn/val/id_prop.csv"
  "data/alignn/test/id_prop.csv"
  "data/alignn/targets.txt"
  "data/alignn/transform_config.json"
  "data/processed/textural_screened/textural_screened_clustered_with_umap.csv"
  "data/processed/stratified_datasets/train_set.csv"
  "data/processed/stratified_datasets/val_set.csv"
  "data/processed/stratified_datasets/test_set.csv"
  "results/alignn/model_ep150/deployment/train_groundtruth.csv"
  "results/alignn/model_ep150/deployment/val_groundtruth.csv"
  "results/alignn/model_ep150/deployment/test_groundtruth.csv"
  "results/alignn/model_ep150/deployment/train_predictions.csv"
  "results/alignn/model_ep150/deployment/val_predictions.csv"
  "results/alignn/model_ep150/deployment/test_predictions.csv"
  "results/alignn/model_ep150/full_library_inference/full_library_with_api.csv"
  "results/alignn/model_ep150/top_candidates/full_library_stable_no_uq_filter.csv"
  "results/alignn/model_ep150/top_candidates/all_top_union.csv"
  "results/alignn/model_ep150/top_candidates/exp_top50_psa.csv"
  "results/alignn/model_ep150/top_candidates/exp_top50_vsa.csv"
  "results/alignn/model_ep150/top_candidates/hypo_top50_psa.csv"
  "results/alignn/model_ep150/top_candidates/hypo_top50_vsa.csv"
  "results/alignn/model_ep150/top_candidates/screening_funnel_stats.csv"
  "results/alignn/model_ep150/composition_sensitivity/candidate_mof_ids.csv"
  "results/alignn/model_ep150/composition_sensitivity/composition_sensitivity_results.csv"
  "results/alignn/model_ep150/composition_sensitivity/composition_rank_change_summary.csv"
  "results/alignn/model_ep150/composition_sensitivity/composition_sensitivity_hit_rates.csv"
  "results/alignn/model_ep150/process_candidates/gcmc_vs_ml_comparison.csv"
  "results/alignn/model_ep150/process_candidates/gcmc_ml_metrics.csv"
  "results/alignn/model_ep150/process_candidates/psa_beaters.csv"
  "results/alignn/model_ep150/process_candidates/vsa_beaters.csv"
  "results/alignn/model_ep150/process_candidates/top10_psa.csv"
  "results/alignn/model_ep150/process_candidates/top10_vsa.csv"
  "results/alignn/model_ep150/process_candidates/top20_combined.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_input/top20_pure_component.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_input/top20_pure_component_multitemp.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/best_isotherm_fits.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/ext_dsl_fits.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/ext_dsl_per_temp_fits.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/model_selection_summary.json"
  "results/alignn/model_ep150/psa_optimization/pareto_eval_results.csv"
  "results/alignn/model_ep150/psa_optimization/pareto_analysis.csv"
  "results/alignn/model_ep150/psa_optimization/material_ranking.csv"
  "results/alignn/model_ep150/psa_optimization/selected_knee_points.json"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_beater_details.csv"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_categorical_summary.csv"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_numeric_summary.csv"
  "results/alignn/model_ep150/uq/lsv_thresholds.json"
  "results/alignn/model_ep150/uq/k_sensitivity_sweep.json"
  "results/alignn/model_ep150/figures/Table_S3_model_metrics.csv"
  "results/alignn/model_ep150/figures/Figure07_feature_shift_summary.csv"
  "results/alignn/model_ep150/figures/Figure10_screening_scatter_summary.csv"
  "results/alignn/model_ep150/figures/Figure11_cluster_analysis_summary.csv"
  "results/alignn/model_ep150/figures/FigureS04_cluster_property_summary.csv"
  "results/alignn/model_ep150/figures/LSV_thresholds_ep150.csv"
)

DERIVED_SOURCE_FILES=(
  "data/processed/RAC_and_zeo_features_deduplicated.csv"
  "results/cbm_screening/inference/umap_coordinates_descriptor_with_metrics_ml.csv"
  "results/alignn/model_ep150/deployment/metrics_summary.json"
  "results/alignn/model_ep150/uq/uq_calibration.json"
)

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_files() {
  local file
  for file in "$@"; do
    [[ -f "${REPO_ROOT}/${file}" ]] || die "required file not found: ${file}"
  done
}

dotenv_token() {
  local env_file="${REPO_ROOT}/.env" line value
  [[ -f "${env_file}" ]] || return 0
  [[ "$(stat -c '%a' "${env_file}")" == "600" ]] ||
    die ".env must have owner-only permissions (run: chmod 600 .env)"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    case "${line}" in
      ZENODO_TOKEN=*)
        value="${line#ZENODO_TOKEN=}"
        value="${value%$'\r'}"
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
          value="${value:1:${#value}-2}"
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
          value="${value:1:${#value}-2}"
        fi
        printf '%s' "${value}"
        return 0
        ;;
    esac
  done < "${env_file}"
}

token() {
  if [[ -n "${ZENODO_TOKEN:-}" ]]; then
    printf '%s' "${ZENODO_TOKEN}"
    return
  fi
  local value
  value="$(dotenv_token)"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
    return
  fi
  read -r -s -p 'Zenodo token: ' value
  printf '\n' >&2
  [[ -n "${value}" ]] || die "empty Zenodo token"
  printf '%s' "${value}"
}

prepare() {
  require_command tar
  require_command sha256sum
  require_command python
  require_files "${MODEL_FILES[@]}" "${DATA_FILES[@]}" "${DERIVED_SOURCE_FILES[@]}" \
    "zenodo/README.md" "zenodo/metadata.json"

  local temp_dir audit_dir generated_dir model_stage flagged
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "${temp_dir}"' RETURN
  audit_dir="${temp_dir}/audit"
  generated_dir="${temp_dir}/generated"
  model_stage="${temp_dir}/model_stage"
  mkdir -p "${RELEASE_DIR}"
  rm -f "${RELEASE_DIR}/cbm_mof_models_r1.tar.gz" \
    "${RELEASE_DIR}/cbm_mof_key_data_r1.tar.gz"
  mkdir -p "${model_stage}/models"

  install -m 0644 "${REPO_ROOT}/${MODEL_FILES[0]}" "${model_stage}/models/inference_checkpoint.pt"
  install -m 0644 "${REPO_ROOT}/${MODEL_FILES[1]}" "${model_stage}/models/model_metadata.pt"
  install -m 0644 "${REPO_ROOT}/${MODEL_FILES[2]}" "${model_stage}/models/uncertainty_trees.pkl"

  python - "${REPO_ROOT}" "${generated_dir}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

repo = Path(sys.argv[1])
output = Path(sys.argv[2]) / "release_data"
output.mkdir(parents=True, exist_ok=True)

metrics_path = repo / "results/alignn/model_ep150/deployment/metrics_summary.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics["checkpoint"] = "models/inference_checkpoint.pt"
metrics["meta_checkpoint"] = "models/model_metadata.pt"
(output / "alignn_test_metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

uq_path = repo / "results/alignn/model_ep150/uq/uq_calibration.json"
uq_calibration = json.loads(uq_path.read_text(encoding="utf-8"))
uq_calibration.pop("input_dir", None)
(output / "uq_calibration.json").write_text(
    json.dumps(uq_calibration, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

gcmc_path = repo / "results/alignn/model_ep150/process_candidates/gcmc_vs_ml_comparison.csv"
gcmc_columns = [
    "mof_id",
    "QstCH4_gcmc",
    "QstCH4_gcmc_error",
    "QstN2_gcmc",
    "QstN2_gcmc_error",
    "group",
    "in_psa100",
    "in_vsa100",
]
gcmc = pd.read_csv(gcmc_path, usecols=gcmc_columns)

zeo_path = repo / "data/processed/RAC_and_zeo_features_deduplicated.csv"
zeo = pd.read_csv(zeo_path, usecols=["name", "Df"]).drop_duplicates("name")
figure8 = gcmc.merge(zeo, left_on="mof_id", right_on="name", how="left", validate="one_to_one")
if figure8["Df"].isna().any():
    missing = figure8.loc[figure8["Df"].isna(), "mof_id"].tolist()
    raise SystemExit(f"Missing PLD Df values for {len(missing)} validated candidates")
figure8 = figure8.drop(columns="name").rename(
    columns={
        "Df": "PLD_A",
        "QstCH4_gcmc": "QstCH4_kJ_mol",
        "QstCH4_gcmc_error": "QstCH4_error_kJ_mol",
        "QstN2_gcmc": "QstN2_kJ_mol",
        "QstN2_gcmc_error": "QstN2_error_kJ_mol",
    }
)
figure8.to_csv(output / "figure8_pld_qst_source.csv", index=False)

cluster_path = repo / "results/cbm_screening/inference/umap_coordinates_descriptor_with_metrics_ml.csv"
clusters = pd.read_csv(cluster_path, usecols=["CifId", "cluster"]).drop_duplicates("CifId")
figure7 = gcmc[["mof_id", "group", "in_psa100", "in_vsa100"]].merge(
    clusters,
    left_on="mof_id",
    right_on="CifId",
    how="left",
    validate="one_to_one",
)
if figure7["cluster"].isna().any():
    missing = figure7.loc[figure7["cluster"].isna(), "mof_id"].tolist()
    raise SystemExit(f"Missing cluster assignments for {len(missing)} validated candidates")
figure7.drop(columns="CifId").to_csv(
    output / "figure7_validated_candidate_clusters.csv",
    index=False,
)
PY

  tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${model_stage}" -czf "${temp_dir}/${MODELS_ARCHIVE}" models
  tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${REPO_ROOT}" -czf "${temp_dir}/${DATA_ARCHIVE}" "${DATA_FILES[@]}" \
    -C "${generated_dir}" release_data

  cp "${REPO_ROOT}/zenodo/README.md" "${temp_dir}/README.md"
  mkdir -p "${audit_dir}/models" "${audit_dir}/data"
  tar -xzf "${temp_dir}/${MODELS_ARCHIVE}" -C "${audit_dir}/models"
  tar -xzf "${temp_dir}/${DATA_ARCHIVE}" -C "${audit_dir}/data"
  cp "${temp_dir}/README.md" "${audit_dir}/README.md"
  flagged="$(grep -RIlEa '/home/|/Users/|ZENODO_TOKEN|Authorization: Bearer|access_token' "${audit_dir}" || true)"
  [[ -z "${flagged}" ]] || die "private path or credential marker found in archive: ${flagged}"
  (
    cd "${temp_dir}"
    sha256sum "${MODELS_ARCHIVE}" "${DATA_ARCHIVE}" README.md > SHA256SUMS
    sha256sum -c SHA256SUMS
    tar -tzf "${MODELS_ARCHIVE}" >/dev/null
    tar -tzf "${DATA_ARCHIVE}" >/dev/null
  )

  install -m 0644 "${temp_dir}/${MODELS_ARCHIVE}" "${RELEASE_DIR}/${MODELS_ARCHIVE}"
  install -m 0644 "${temp_dir}/${DATA_ARCHIVE}" "${RELEASE_DIR}/${DATA_ARCHIVE}"
  install -m 0644 "${temp_dir}/README.md" "${RELEASE_DIR}/README.md"
  install -m 0644 "${temp_dir}/SHA256SUMS" "${RELEASE_DIR}/SHA256SUMS"
  printf 'Prepared local release in %s\n' "${RELEASE_DIR}"
  (cd "${RELEASE_DIR}" && sha256sum -c SHA256SUMS && du -h "${MODELS_ARCHIVE}" "${DATA_ARCHIVE}" README.md SHA256SUMS)
}

validate_metadata() {
  require_command jq
  jq -e '
    .metadata.upload_type == "dataset" and
    .metadata.version == "1.0.0-r1" and
    .metadata.license == "cc-by-4.0" and
    (.metadata.creators | type == "array" and length > 0) and
    (all(.metadata.creators[]; (.name | type == "string" and length > 0)))
  ' "${REPO_ROOT}/zenodo/metadata.json" >/dev/null ||
    die "zenodo/metadata.json is incomplete; confirm the manuscript creator order before creating a draft"
}

set_restricted_visibility() {
  local deposition_id="$1" access_token="$2" draft_file payload_file response
  draft_file="$(mktemp)"
  payload_file="$(mktemp)"
  trap 'rm -f "${draft_file}" "${payload_file}"' RETURN

  curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -H 'Accept: application/vnd.inveniordm.v1+json' \
    -o "${draft_file}" \
    "${ZENODO_API}/records/${deposition_id}/draft"
  jq '{
    access: {record: "public", files: "restricted"},
    files: {
      enabled: .files.enabled,
      default_preview: .files.default_preview,
      order: .files.order
    },
    metadata,
    custom_fields,
    pids
  }' "${draft_file}" > "${payload_file}"

  response="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -H 'Accept: application/vnd.inveniordm.v1+json' \
    -H 'Content-Type: application/json' \
    -X PUT --data-binary "@${payload_file}" \
    "${ZENODO_API}/records/${deposition_id}/draft")"
  jq -e '
    .access.record == "public" and
    .access.files == "restricted" and
    (.access.embargo.active == false)
  ' <<<"${response}" >/dev/null || die "failed to set Visibility = Restricted"
  printf 'Set Visibility = Restricted for deposition %s\n' "${deposition_id}"
}

verify_restricted_visibility() {
  local deposition_id="$1" access_token="$2" submitted endpoint response
  submitted="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    "${ZENODO_API}/deposit/depositions/${deposition_id}" | jq -er '.submitted')"
  if [[ "${submitted}" == "true" ]]; then
    endpoint="${ZENODO_API}/records/${deposition_id}"
  else
    endpoint="${ZENODO_API}/records/${deposition_id}/draft"
  fi
  response="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -H 'Accept: application/vnd.inveniordm.v1+json' \
    "${endpoint}")"
  jq -e '
    .access.record == "public" and
    .access.files == "restricted" and
    (.access.embargo.active == false)
  ' <<<"${response}" >/dev/null || die "deposition visibility is not Restricted"
  printf 'Verified Visibility = Restricted (public metadata, restricted files)\n'
}

draft() {
  require_command curl
  require_command jq
  [[ -f "${RELEASE_DIR}/SHA256SUMS" ]] || die "run '$0 prepare' first"
  (cd "${RELEASE_DIR}" && sha256sum -c SHA256SUMS)
  validate_metadata

  local access_token response deposition_id bucket file
  access_token="$(token)"
  response="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -H 'Content-Type: application/json' \
    -X POST --data '{}' "${ZENODO_API}/deposit/depositions")"
  deposition_id="$(jq -er '.id' <<<"${response}")"
  bucket="$(jq -er '.links.bucket' <<<"${response}")"
  printf 'Created unpublished Zenodo deposition %s\n' "${deposition_id}"

  for file in "${MODELS_ARCHIVE}" "${DATA_ARCHIVE}" README.md SHA256SUMS; do
    printf 'Uploading %s\n' "${file}"
    curl --fail-with-body --silent --show-error \
      -H "Authorization: Bearer ${access_token}" \
      --upload-file "${RELEASE_DIR}/${file}" "${bucket}/${file}" >/dev/null
  done

  curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -H 'Content-Type: application/json' \
    -X PUT --data-binary "@${REPO_ROOT}/zenodo/metadata.json" \
    "${ZENODO_API}/deposit/depositions/${deposition_id}" >/dev/null
  set_restricted_visibility "${deposition_id}" "${access_token}"
  printf 'Draft ready: https://zenodo.org/deposit/%s\n' "${deposition_id}"
  printf 'Not published. Review the metadata and files before any publish command.\n'
}

verify() {
  [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || die "usage: $0 verify <deposition-id>"
  require_command curl
  require_command jq
  require_command md5sum
  local deposition_id="$1" access_token remote file expected_size expected_md5 actual_size actual_md5
  access_token="$(token)"
  remote="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    "${ZENODO_API}/deposit/depositions/${deposition_id}/files")"

  for file in "${MODELS_ARCHIVE}" "${DATA_ARCHIVE}" README.md SHA256SUMS; do
    [[ -f "${RELEASE_DIR}/${file}" ]] || die "local release file missing: ${file}"
    expected_size="$(stat -c '%s' "${RELEASE_DIR}/${file}")"
    expected_md5="$(md5sum "${RELEASE_DIR}/${file}" | awk '{print $1}')"
    actual_size="$(jq -er --arg name "${file}" '.[] | select((.filename // .name) == $name) | (.filesize // .size)' <<<"${remote}")"
    actual_md5="$(jq -er --arg name "${file}" '.[] | select((.filename // .name) == $name) | .checksum' <<<"${remote}" | sed 's/^md5://')"
    [[ "${actual_size}" == "${expected_size}" ]] || die "size mismatch for ${file}"
    [[ "${actual_md5}" == "${expected_md5}" ]] || die "MD5 mismatch for ${file}"
    printf 'Verified %s (%s bytes)\n' "${file}" "${expected_size}"
  done
  verify_restricted_visibility "${deposition_id}" "${access_token}"
}

publish() {
  [[ $# -eq 2 && "$1" =~ ^[0-9]+$ && "$2" == '--confirm-publish' ]] ||
    die "usage: $0 publish <deposition-id> --confirm-publish"
  require_command curl
  require_command jq
  local deposition_id="$1" access_token response doi
  access_token="$(token)"
  verify_restricted_visibility "${deposition_id}" "${access_token}"
  response="$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -X POST "${ZENODO_API}/deposit/depositions/${deposition_id}/actions/publish")"
  doi="$(jq -er '.doi' <<<"${response}")"
  printf 'Published Zenodo deposition %s (DOI: %s)\n' "${deposition_id}" "${doi}"
}

usage() {
  printf 'Usage:\n'
  printf '  %s prepare\n' "$0"
  printf '  %s draft\n' "$0"
  printf '  %s verify <deposition-id>\n' "$0"
  printf '  %s publish <deposition-id> --confirm-publish\n' "$0"
}

case "${1:-}" in
  prepare)
    [[ $# -eq 1 ]] || die "usage: $0 prepare"
    prepare
    ;;
  draft)
    [[ $# -eq 1 ]] || die "usage: $0 draft"
    draft
    ;;
  verify)
    shift
    verify "$@"
    ;;
  publish)
    shift
    publish "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
