#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="${REPO_ROOT}/release/zenodo_r1"
MODELS_ARCHIVE="cbm_mof_models_r1.tar.gz"
DATA_ARCHIVE="cbm_mof_key_data_r1.tar.gz"
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
  "results/alignn/model_ep150/full_library_inference/full_library_with_api.csv"
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
  "results/alignn/model_ep150/process_candidates/isotherm_input/top20_multitemp.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/best_isotherm_fits.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/ext_dsl_fits.csv"
  "results/alignn/model_ep150/process_candidates/isotherm_fits/ext_dsl_per_temp_fits.csv"
  "results/alignn/model_ep150/psa_optimization/pareto_eval_results.csv"
  "results/alignn/model_ep150/psa_optimization/pareto_analysis.csv"
  "results/alignn/model_ep150/psa_optimization/material_ranking.csv"
  "results/alignn/model_ep150/psa_optimization/selected_knee_points.json"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_beater_details.csv"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_categorical_summary.csv"
  "results/alignn/model_ep150/structural_analysis/cluster8_beaters/cluster8_numeric_summary.csv"
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

token() {
  if [[ -n "${ZENODO_TOKEN:-}" ]]; then
    printf '%s' "${ZENODO_TOKEN}"
    return
  fi
  local value
  read -r -s -p 'Zenodo token: ' value
  printf '\n' >&2
  [[ -n "${value}" ]] || die "empty Zenodo token"
  printf '%s' "${value}"
}

prepare() {
  require_command tar
  require_command sha256sum
  require_files "${MODEL_FILES[@]}" "${DATA_FILES[@]}" "zenodo/README.md" "zenodo/metadata.json"

  local temp_dir audit_dir flagged
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "${temp_dir}"' RETURN
  audit_dir="${temp_dir}/audit"
  mkdir -p "${RELEASE_DIR}"

  tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${REPO_ROOT}" -czf "${temp_dir}/${MODELS_ARCHIVE}" "${MODEL_FILES[@]}"
  tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${REPO_ROOT}" -czf "${temp_dir}/${DATA_ARCHIVE}" "${DATA_FILES[@]}"

  cp "${REPO_ROOT}/zenodo/README.md" "${temp_dir}/README.md"
  mkdir -p "${audit_dir}/models" "${audit_dir}/data"
  tar -xzf "${temp_dir}/${MODELS_ARCHIVE}" -C "${audit_dir}/models"
  tar -xzf "${temp_dir}/${DATA_ARCHIVE}" -C "${audit_dir}/data"
  flagged="$(grep -RIlE '/home/|ZENODO_TOKEN|Authorization: Bearer|access_token' "${audit_dir}" || true)"
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
}

publish() {
  [[ $# -eq 2 && "$1" =~ ^[0-9]+$ && "$2" == '--confirm-publish' ]] ||
    die "usage: $0 publish <deposition-id> --confirm-publish"
  require_command curl
  local deposition_id="$1" access_token
  access_token="$(token)"
  curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${access_token}" \
    -X POST "${ZENODO_API}/deposit/depositions/${deposition_id}/actions/publish"
  printf '\nPublished Zenodo deposition %s\n' "${deposition_id}"
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
