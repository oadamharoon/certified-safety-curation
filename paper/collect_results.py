"""Harvest all eval_results JSONs into one snapshot for paper tables/figures.

Regenerating any table: rerun this script; it scans the experiment output
directories and writes data/results_snapshot.json keyed by task/config/seed.
"""
from __future__ import annotations
import json, glob, os, re
from collections import defaultdict

REPO = "/home/omniverse/workspace/safevlmcpl/vlm-with-cpl/new_data"
OUT = "/home/omniverse/workspace/safevlmcpl/iclr2027/data"

PATTERNS = {
    # config_name: filename regex (relative to eval_results_)
    "bc_all": r"^bc$",
    "bcsafe": r"^bcsafe_seed(\d+)$",
    "bcsafeseg": r"^bcsafeseg_seed(\d+)$",
    "vawr_5seed": r"^bc_v_ens_awr_final_5seed_seed(\d+)$",
    "vfilt_matchgt": r"^vfilt_matchgt_seed(\d+)$",
    "vfilt_q25": r"^vfilt_q25_seed(\d+)$",
    "vfilt_return": r"^vfilt_return_matchgt_seed(\d+)$",
    "vfilt_random": r"^vfilt_random_matchgt_seed(\d+)$",
    "calfilt_ltt_legacy": r"^calfilt_lttv2_seed(\d+)$",
    "calfilt_ltt": r"^calfilt_ltt_seed(\d+)$",
    "calfilt_lttR50_legacy": r"^calfilt_lttv2R50_seed(\d+)$",
    "calfilt_lttR50": r"^calfilt_lttR50_seed(\d+)$",
    "calfilt_pref": r"^calfilt_pref_seed(\d+)$",
    "xlab_oracle_step": r"^xlab_oracle_step_exp_k1_seed(\d+)$",
    "xlab_oracle_ctg": r"^xlab_oracle_ctg_exp_k1_seed(\d+)$",
    "xlab_rank": r"^xlab_learned_rank_k1_seed(\d+)$",
    "xlab_binary": r"^xlab_learned_binary_k1_seed(\d+)$",
    "xlab_exp_k1": r"^xlab_learned_exp_k1_seed(\d+)$",
    "xlab_k3": r"^xlab_learned_exp_k3_seed(\d+)$",
    "xlab_k5": r"^xlab_learned_exp_k5_seed(\d+)$",
    "xlab_k10": r"^xlab_learned_exp_k10_seed(\d+)$",
    "cf_v1": r"^cf_M8_seed(\d+)$",
    "cf_v2": r"^cfv2_M16_seed(\d+)$",
    "vawr_from_bcsafe": r"^xlab_vawr_from_bcsafe_seed(\d+)$",
    "cpl_gt": r"^cplgt_seed(\d+)$",
    "labels_only": r"^labelsonly_seed(\d+)$",
    "qfilt": r"^qfilt_seed(\d+)$",
    "labels_split_s50": r"^lblsplit_s50_seed(\d+)$",
    "labels_split_s100": r"^lblsplit_s100_seed(\d+)$",
    "labels_split_s200": r"^lblsplit_s200_seed(\d+)$",
    "labels_only_n50": r"^labelsonly_n50_seed(\d+)$",
    "labels_only_sf10": r"^labelsonly_sf10_seed(\d+)$",
    "labels_only_sf1": r"^labelsonly_sf1_seed(\d+)$",
    "labels_only_n100": r"^labelsonly_n100_seed(\d+)$",
    "labels_only_n400": r"^labelsonly_n400_seed(\d+)$",
    "cf_octg": r"^cf_octg_seed(\d+)$",
    "oracle_rtg": r"^xlab_oracle_rtg_exp_k1_seed(\d+)$",
    "calfilt_a10": r"^calfilt_a10_seed(\d+)$",
    "calfilt_a40": r"^calfilt_a40_seed(\d+)$",
    "calfilt_noise20": r"^calfilt_noise20_seed(\d+)$",
    "vfilt_retbot": r"^vfilt_retbot_seed(\d+)$",
    "labels_only_fixdraw": r"^lblonly_fixdraw_seed(\d+)$",
    "bc_a40d2": r"^bca40d2_seed(\d+)$",
    "bc_a40d3": r"^bca40d3_seed(\d+)$",
    "bc_a40d1": r"^bca40d1_seed(\d+)$",
    "bc_a40selq35": r"^bca40selq35_seed(\d+)$",
    "bc_a40selq55": r"^bca40selq55_seed(\d+)$",
    "bc_a40selq75": r"^bca40selq75_seed(\d+)$",
    "bc_a40selq80": r"^bca40selq80_seed(\d+)$",
    "bc_a40selq85": r"^bca40selq85_seed(\d+)$",
    "calfilt_tier2_legacy": r"^calfilt_t2_seed(\d+)$",
    "calfilt_tier2": r"^calfilt_tier2_seed(\d+)$",
    "bc_echo": r"^bcecho_carrun_seed(\d+)$",
    "wbc_q85": r"^wbcq85_seed(\d+)$",
    "wbc_q80": r"^wbcq80_seed(\d+)$",
    "wbc_q75": r"^wbcq75_seed(\d+)$",
    "wbc_q70": r"^wbcq70_seed(\d+)$",
    "wbc_q65": r"^wbcq65_seed(\d+)$",
    "wbc_echo": r"^wbcecho_seed(\d+)$",
    "wbc1_q85": r"^wc1q85_seed(\d+)$",
    "vfilt_calsafe": r"^vfilt_calsafe_seed(\d+)$",
    "vawr_b01": r"^vawr_b01_seed(\\d+)$",
    "vawr_b03": r"^vawr_b03_seed(\\d+)$",
    "vawr_b3": r"^vawr_b3_seed(\\d+)$",
    "vawr_c5": r"^vawr_c5_seed(\\d+)$",
    "vawr_c100": r"^vawr_c100_seed(\\d+)$",
    "xagg_min": r"^xagg_min_seed(\d+)$",
    "xagg_p10": r"^xagg_p10_seed(\d+)$",
    "calfilt_csf": r"^calfilt_csf_seed(\d+)$",
    "calfilt_hcsf": r"^calfilt_hcsf_seed(\d+)$",
    "mwbc1_q85": r"^mc1q85_seed(\d+)$",
    "mwbc2_q85": r"^mc2q85_seed(\d+)$",
    "mwbc3_q85": r"^mc3q85_seed(\d+)$",
    "mwbcs05_q85": r"^ms05q85_seed(\d+)$",
    "mwbcs2_q85": r"^ms2q85_seed(\d+)$",
    "mwbc1_q80": r"^mc1q80_seed(\d+)$",
    "mwbc2_q80": r"^mc2q80_seed(\d+)$",
    "mwbc3_q80": r"^mc3q80_seed(\d+)$",
    "mwbcs05_q80": r"^ms05q80_seed(\d+)$",
    "mwbcs2_q80": r"^ms2q80_seed(\d+)$",
    "mwbc1_q75": r"^mc1q75_seed(\d+)$",
    "mwbc2_q75": r"^mc2q75_seed(\d+)$",
    "mwbc3_q75": r"^mc3q75_seed(\d+)$",
    "mwbcs05_q75": r"^ms05q75_seed(\d+)$",
    "mwbcs2_q75": r"^ms2q75_seed(\d+)$",
    "mwbc1_q70": r"^mc1q70_seed(\d+)$",
    "mwbc2_q70": r"^mc2q70_seed(\d+)$",
    "mwbc3_q70": r"^mc3q70_seed(\d+)$",
    "mwbcs05_q70": r"^ms05q70_seed(\d+)$",
    "mwbcs2_q70": r"^ms2q70_seed(\d+)$",
    "mwbc1_q65": r"^mc1q65_seed(\d+)$",
    "mwbc2_q65": r"^mc2q65_seed(\d+)$",
    "mwbc3_q65": r"^mc3q65_seed(\d+)$",
    "mwbcs05_q65": r"^ms05q65_seed(\d+)$",
    "mwbcs2_q65": r"^ms2q65_seed(\d+)$",
    "mwbc1_echo": r"^mc1echo_seed(\d+)$",
    "mwbc2_echo": r"^mc2echo_seed(\d+)$",
    "mwbc3_echo": r"^mc3echo_seed(\d+)$",
    "mwbcs05_echo": r"^ms05echo_seed(\d+)$",
    "mwbcs2_echo": r"^ms2echo_seed(\d+)$",
    "wbc3_q85": r"^wc3q85_seed(\d+)$",
    "toph_q85": r"^thq85_seed(\d+)$",
    "wbc1_q80": r"^wc1q80_seed(\d+)$",
    "wbc3_q80": r"^wc3q80_seed(\d+)$",
    "toph_q80": r"^thq80_seed(\d+)$",
    "wbc1_q75": r"^wc1q75_seed(\d+)$",
    "wbc3_q75": r"^wc3q75_seed(\d+)$",
    "toph_q75": r"^thq75_seed(\d+)$",
    "wbc1_q70": r"^wc1q70_seed(\d+)$",
    "wbc3_q70": r"^wc3q70_seed(\d+)$",
    "toph_q70": r"^thq70_seed(\d+)$",
    "wbc1_q65": r"^wc1q65_seed(\d+)$",
    "wbc3_q65": r"^wc3q65_seed(\d+)$",
    "toph_q65": r"^thq65_seed(\d+)$",
    "wbc1_echo": r"^wc1echo_seed(\d+)$",
    "wbc3_echo": r"^wc3echo_seed(\d+)$",
    "toph_echo": r"^thecho_seed(\d+)$",
}

def main():
    snap = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(REPO, "outputs/*/eval_results_*.json")):
        task = f.split("/")[-2]
        name = os.path.basename(f)[len("eval_results_"):-len(".json")]
        for config, pat in PATTERNS.items():
            m = re.match(pat, name)
            if m:
                seed = m.group(1) if m.groups() else "0"
                r = json.load(open(f))
                snap[task][config][seed] = {
                    "R": r["avg_reward"], "C": r["avg_cost"],
                    "viol": r.get("constraint_violation_rate", None)}
                break
    # calibration metadata.
    # Parse the tag generically rather than matching a fixed alternation. Any
    # OUT_TAG that appears on disk becomes calfilt_<tag> unless it is one of the
    # explicit aliases below. An earlier fixed list silently dropped the
    # metadata of every tag nobody remembered to add, and mapped two distinct
    # tags onto one key so glob order decided which survived.
    _ALIAS = {"lttv2": "calfilt_ltt_legacy", "lttv2R50": "calfilt_lttR50_legacy"}
    _dropped = set()
    for f in glob.glob(os.path.join(REPO, "outputs/*/calfilt_meta_*.json")):
        task = f.split("/")[-2]
        name = os.path.basename(f)
        m = re.match(r"calfilt_meta_(.+)_seed(\d+)\.json$", name)
        if not m:
            continue
        tag, seed = m.group(1), m.group(2)
        tag = tag[len("calfilt_"):] if tag.startswith("calfilt_") else tag
        key = _ALIAS.get(tag, "calfilt_" + tag)
        if key not in snap[task]:
            _dropped.add(key)
            continue
        if seed in snap[task][key]:
            snap[task][key][seed]["meta"] = json.load(open(f))
    if _dropped:
        print("  [meta] no matching results for tags: " + ", ".join(sorted(_dropped)))
    # Regenerated arms take precedence; legacy (pre-standardization) tags fill
    # in only for tasks that were never regenerated.
    for task, cfgs in snap.items():
        for new, leg in (("calfilt_ltt", "calfilt_ltt_legacy"),
                         ("calfilt_lttR50", "calfilt_lttR50_legacy"),
                         ("calfilt_tier2", "calfilt_tier2_legacy")):
            if leg in cfgs:
                if new not in cfgs or not cfgs[new]:
                    cfgs[new] = cfgs[leg]
                del cfgs[leg]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results_snapshot.json"), "w") as fo:
        json.dump(snap, fo, indent=1)
    n = sum(len(s) for t in snap.values() for s in t.values())
    print(f"Snapshot: {len(snap)} tasks, {n} (config,seed) entries -> {OUT}/results_snapshot.json")

if __name__ == "__main__":
    main()
