import json, os, csv, re
import yaml

LOGS = "/local/home/nstaykov/workspace/mujoco_playground/logs"
QUEUE_DIR = os.path.join(LOGS, "_queue")
OUT = "/local/home/nstaykov/workspace/policy_analyzer/report/appendix"

QUEUES = [
    "pinch_sweep_size_rand-20260622-195014",
    "pinch_sweep_size_rand_sinusoid-20260624-142357",
    "pinch_sweep_size_rand_sinusoid-20260623-133237",
    "downwards_sensor_sweep_120-20260625-124815",
]

class _L(yaml.SafeLoader):
    pass
_L.add_constructor("tag:yaml.org,2002:python/tuple",
                   lambda loader, node: loader.construct_sequence(node))


def parse_log(path):
    """Return (env_cfg, ppo_cfg, cli_overrides) parsed from a run log."""
    with open(path) as f:
        lines = f.readlines()
    # locate section headers
    idx = {}
    for i, ln in enumerate(lines):
        s = ln.rstrip("\n")
        if s == "Environment Config:":
            idx["env"] = i
        elif s == "PPO Training Parameters:":
            idx["ppo"] = i
        elif s.startswith("CLI overrides:"):
            idx["cli"] = i

    def block(start):
        out = []
        for ln in lines[start + 1:]:
            if ln.strip() == "":
                break
            out.append(ln)
        return yaml.load("".join(out), Loader=_L)

    env = block(idx["env"]) if "env" in idx else {}
    ppo = block(idx["ppo"]) if "ppo" in idx else {}
    cli = {}
    if "cli" in idx:
        buf = lines[idx["cli"]].split("CLI overrides:", 1)[1]
        depth = buf.count("{") - buf.count("}")
        j = idx["cli"] + 1
        while depth > 0 and j < len(lines):
            buf += lines[j]
            depth += lines[j].count("{") - lines[j].count("}")
            j += 1
        cli = json.loads(buf.strip())
    return env, ppo, cli


def hidden(sizes):
    if not sizes:
        return ""
    return "-".join(str(int(x)) for x in sizes)


rows = []
for q in QUEUES:
    status = json.load(open(os.path.join(QUEUE_DIR, q, "status.json")))
    for r in status:
        logp = os.path.join(LOGS, os.path.basename(os.path.dirname(r["log"])), os.path.basename(r["log"]))
        logp = os.path.join(LOGS, "_queue", q, os.path.basename(r["log"]))
        env, ppo, cli = parse_log(logp)
        rows.append(dict(queue=q, status=r, env=env, ppo=ppo, cli=cli))

print(f"parsed {len(rows)} runs")


def g(d, *keys, default=""):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def rng(d, *keys):
    v = g(d, *keys, default=None)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return v[0], v[1]
    return "", ""


# ---- common identifier columns ----
def ident(row):
    st = row["status"]
    return dict(
        queue=row["queue"],
        env_name=st["env_name"],
        exp_name=st["exp_name"],
        sensor_bundle=g(row["env"], "sensor_bundle"),
        seed=g(st, "flags", "seed"),
        run_idx=st["idx"],
        result=st.get("result"),
    )

ID_COLS = ["queue", "env_name", "exp_name", "sensor_bundle", "seed", "run_idx", "result"]

# ---------- NETWORK ----------
net_cols = ID_COLS + [
    "policy_hidden_layer_sizes", "value_hidden_layer_sizes",
    "policy_obs_key", "value_obs_key", "normalize_observations",
]
net_rows = []
for row in rows:
    ppo = row["ppo"]
    nf = g(ppo, "network_factory", default={})
    d = ident(row)
    d.update(
        policy_hidden_layer_sizes=hidden(g(nf, "policy_hidden_layer_sizes", default=[])),
        value_hidden_layer_sizes=hidden(g(nf, "value_hidden_layer_sizes", default=[])),
        policy_obs_key=g(nf, "policy_obs_key"),
        value_obs_key=g(nf, "value_obs_key"),
        normalize_observations=g(ppo, "normalize_observations"),
    )
    net_rows.append(d)

# ---------- DOMAIN RANDOMIZATION ----------
dr_cols = ID_COLS + [
    "domain_randomization",
    "cube_size_min", "cube_size_max",
    "cube_mass_dr_min", "cube_mass_dr_max",
    "cube_pos_dr_min", "cube_pos_dr_max",
    "obs_noise_level",
    "obs_noise_scale_joint_pos", "obs_noise_scale_joint_vel",
    "obs_noise_scales_json",
    "perturbation_enable",
    "force_target_sinusoid", "force_target_period",
    "force_target_range_min", "force_target_range_max",
]
dr_rows = []
for row in rows:
    env = row["env"]
    st = row["status"]
    d = ident(row)
    dr_on = g(st, "flags", "domain_randomization", default="")
    # cube_size DR
    cs_min, cs_max = rng(env, "domain_rand", "cube_size")
    if cs_min == "" and st["env_name"] == "TesolloDownwardsRotateZ" and dr_on:
        # baked into env.domain_randomize (cube-size-only, [0.85,1.15]); not printed
        cs_min, cs_max = 0.85, 1.15
    cm_min, cm_max = rng(env, "domain_rand", "cube_mass")
    cp_min, cp_max = rng(env, "domain_rand", "cube_pos")
    ft_min, ft_max = rng(env, "force_target_range")
    scales = g(env, "obs_noise", "scales", default={})
    d.update(
        domain_randomization=dr_on,
        cube_size_min=cs_min, cube_size_max=cs_max,
        cube_mass_dr_min=cm_min, cube_mass_dr_max=cm_max,
        cube_pos_dr_min=cp_min, cube_pos_dr_max=cp_max,
        obs_noise_level=g(env, "obs_noise", "level"),
        obs_noise_scale_joint_pos=g(scales, "joint_pos"),
        obs_noise_scale_joint_vel=g(scales, "joint_vel"),
        obs_noise_scales_json=json.dumps(scales, sort_keys=True) if scales else "",
        perturbation_enable=g(env, "pert_config", "enable"),
        force_target_sinusoid=g(env, "force_target_sinusoid"),
        force_target_period=g(env, "force_target_period"),
        force_target_range_min=ft_min, force_target_range_max=ft_max,
    )
    dr_rows.append(d)

# ---------- TRAINING ----------
tr_cols = ID_COLS + [
    "num_timesteps", "num_envs", "batch_size", "num_minibatches",
    "num_updates_per_batch", "unroll_length", "learning_rate",
    "entropy_cost", "discounting", "reward_scaling", "max_grad_norm",
    "episode_length", "action_repeat", "num_evals", "num_resets_per_eval",
    "training_metrics_steps", "ctrl_dt", "sim_dt", "ema_alpha",
]
tr_rows = []
for row in rows:
    ppo, env = row["ppo"], row["env"]
    d = ident(row)
    d.update(
        num_timesteps=g(ppo, "num_timesteps"),
        num_envs=g(ppo, "num_envs"),
        batch_size=g(ppo, "batch_size"),
        num_minibatches=g(ppo, "num_minibatches"),
        num_updates_per_batch=g(ppo, "num_updates_per_batch"),
        unroll_length=g(ppo, "unroll_length"),
        learning_rate=g(ppo, "learning_rate"),
        entropy_cost=g(ppo, "entropy_cost"),
        discounting=g(ppo, "discounting"),
        reward_scaling=g(ppo, "reward_scaling"),
        max_grad_norm=g(ppo, "max_grad_norm"),
        episode_length=g(ppo, "episode_length"),
        action_repeat=g(ppo, "action_repeat"),
        num_evals=g(ppo, "num_evals"),
        num_resets_per_eval=g(ppo, "num_resets_per_eval"),
        training_metrics_steps=g(ppo, "training_metrics_steps"),
        ctrl_dt=g(env, "ctrl_dt"),
        sim_dt=g(env, "sim_dt"),
        ema_alpha=g(env, "ema_alpha"),
    )
    tr_rows.append(d)


def write_csv(name, cols, data):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in data:
            w.writerow({c: d.get(c, "") for c in cols})
    print("wrote", p, f"({len(data)} rows, {len(cols)} cols)")


write_csv("run_parameters_network.csv", net_cols, net_rows)
write_csv("run_parameters_domain_randomization.csv", dr_cols, dr_rows)
write_csv("run_parameters_training.csv", tr_cols, tr_rows)
