import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

CHART_NAMES = ["VS-ALT-Norm(avg)", "VS-ALT-Norm(top=0.2)", "VS-ALT-Norm(top=0.4)", "HMT", "MaxNorm", "ZEWMAC"]


@dataclass(frozen=True)
class Design:
    settings: tuple = ("identity", "block", "wafer")
    p_values: tuple = (10, 30)                 # identity and block settings
    n_values: tuple = (20, 50)                 # identity and block settings
    lambdas: tuple = (0.1, 0.5, 1.0)
    deltas: tuple = (0.2, 0.4, 0.6, 0.8)
    identity_cases: tuple = (1, 2, 3, 4, 5, 6, 7)
    block_cases: tuple = (1, 2, 4, 7)          # Cases 3, 5 and 6 are not positive definite here
    wafer_n: int = 50
    wafer_deltas: tuple = (1.0,)               # Sigma_OC = Sigma_0 + delta (Sigma_wafer_OC - Sigma_0)

    arl0: int = 200                            # target zero-state in-control ARL
    r_calibration: int = 40000                 # in-control streams used to set the control limits
    calibration_horizon: int = 2000            
    r_in_control: int = 40000                  # streams for the in-control check (delta = 0)
    r_out_of_control: int = 40000              # streams per out-of-control cell
    tau: int = 100                             # change point in the steady state
    max_delay: int = 5000                      # max number of run length allowed

    phi: float = 1.0                           # VS-ALT-Norm threshold constant (Eq. 8)
    eta: float = 4.0                           # VS-ALT-Norm thresholding exponent (Eq. 7)
    q_min: int = 2                             # smallest monitored set

    maxnorm_streams: int = 4000                # Monte Carlo estimate of the MaxNorm moments (Eq. 13)
    maxnorm_burn_in: int = 150
    maxnorm_kept: int = 150
    maxnorm_every: int = 5

    master_seed: int = 20261003
    output_file: str = "VS-ALT-Norm_results.xlsx"


QUICK_DESIGN = replace(          #to check that the code runs
    Design(),
    p_values=(10,), n_values=(10,), lambdas=(0.2, 1.0), deltas=(0.4, 0.8),
    identity_cases=(1, 5), block_cases=(1, 2),
    r_calibration=600, r_in_control=300, r_out_of_control=300, tau=20, maxnorm_streams=200,
    output_file="quick_results.xlsx",
)


def list_configurations(design):   #All combinations of the study (setting, p, n, lambda)
    configurations = []
    for setting in design.settings:
        if setting == "wafer":
            for lam in design.lambdas:
                configurations.append((setting, 5, design.wafer_n, lam))
        else:
            for p in design.p_values:
                for n in design.n_values:
                    for lam in design.lambdas:
                        configurations.append((setting, p, n, lam))
    return configurations


def list_cells(design, setting): #All (state, case, delta) cells of one setting. Case 0 is the in-control check
    if setting == "wafer":
        states, shifts = ["zero"], [(1, delta) for delta in design.wafer_deltas]
    else:
        cases = design.identity_cases if setting == "identity" else design.block_cases
        states, shifts = ["zero", "steady"], [(case, delta) for case in cases for delta in design.deltas]

    cells = []
    for state in states:
        cells.append((state, 0, 0.0))
        for case, delta in shifts:
            cells.append((state, case, delta))
    return cells


def charts_for(p, n, lam): #HMT is left out when lambda = 1 and n <= p
    if lam == 1.0 and n <= p:
        return [name for name in CHART_NAMES if name != "HMT"]
    return list(CHART_NAMES)


# In-control and out-of-control covariance matrices

# Wafer-lapping example 
WAFER_SIGMA_0 = np.array([
    [1.30, 0.46, 0.51, 0.00, 0.00],
    [0.46, 1.30, 0.53, 0.00, 0.00],
    [0.51, 0.53, 1.30, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.30, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.30],
])
WAFER_SIGMA_OC = np.array([
    [1.30, 0.62, 0.63, 0.00, 0.00],
    [0.62, 1.30, 0.68, 0.00, 0.00],
    [0.63, 0.68, 1.30, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.30, -0.55],
    [0.00, 0.00, 0.00, -0.55, 1.30],
])


def in_control_covariance(setting, p):  #sigma0 for each setting
    if setting == "identity":
        return np.eye(p)
    if setting == "block":
        half = p // 2
        i = np.arange(half)
        sigma0 = np.eye(p)
        sigma0[:half, :half] = 1 - np.abs(i[:, None] - i[None, :]) / half
        return sigma0
    if setting == "wafer":
        return WAFER_SIGMA_0.copy()


def shift_matrix(case, delta, p):   #matrix to be added to Sigma_0
    shift = np.zeros((p, p))
    if case == 1:
        np.fill_diagonal(shift, delta)
    elif case == 2:
        shift[0, 0] = delta
    elif case == 3:
        shift[:5, :5] = delta
        np.fill_diagonal(shift[:5, :5], 0.0)
    elif case == 4:
        shift[:3, :3] = delta
    elif case == 5:
        shift[0, 1] = shift[1, 0] = delta
    elif case == 6:
        shift[0, 1] = shift[1, 0] = delta
        shift[0, 0] = shift[1, 1] = delta ** 2
    elif case == 7:
        shift[:2, :2] = delta
        shift[-2:, -2:] = delta
    return shift


def out_of_control_covariance(setting, sigma0, case, delta):  #Build out of control sigma
    if case == 0:
        return sigma0.copy()
    if setting == "wafer":
        return sigma0 + delta * (WAFER_SIGMA_OC - sigma0)

    p = sigma0.shape[0]
    shift = shift_matrix(case, delta, p)
    if setting == "block" and case in (2, 4):
        # In the correlated setting, Cases 2 and 4 are applied at the start of both blocks (Table 1 note).
        shift = shift + np.roll(shift, p // 2, axis=(0, 1))
    return sigma0 + shift


def affected_variables(sigma0, sigma_oc): 
    changed = np.abs(sigma_oc - sigma0) > 1e-12
    return set(np.flatnonzero(changed.any(axis=0)))


def inverse_square_root(matrix):  #Symmetric inverse square root (whitening)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T


# charts

def ewma_variance_factor(lam, t):
    return lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))


class VSALTNormChart:

    def __init__(self, p, n, lam, rule, rho=None, phi=1.0, eta=4.0, q_min=2):
        self.p, self.n, self.lam = p, n, lam
        self.rule, self.rho = rule, rho
        self.phi, self.eta, self.q_min = phi, eta, q_min
        self.theta = np.ones((p, p)) + np.eye(p) #2 on the diagonal, 1 elsewhere.

    def start(self):
        self.D_bar = np.zeros((self.p, self.p))
        self.selected = None

    def update(self, x, t):
        S = np.cov(x, rowvar=False)                                    # Eq. (1)
        D = S - np.eye(self.p)
        self.D_bar = self.lam * D + (1 - self.lam) * self.D_bar        # Eq. (3), recursive form
        c_t = ewma_variance_factor(self.lam, t)
        z = self.D_bar / np.sqrt(self.theta * c_t / (self.n - 1))      # Eq. (4)

        V = np.sum(z ** 2, axis=1)                                     # Eq. (5)
        self.selected = self.select_variables(V)                       # Eq. (6)

        q = len(self.selected)
        tau_q = self.phi * math.sqrt(math.log(q))                      # Eq. (8)
        z_block = z[np.ix_(self.selected, self.selected)]
        z_tilde = z_block * np.maximum(0.0, 1 - np.abs(tau_q / z_block) ** self.eta)   # Eq. (7)
        return np.sum(z_tilde ** 2)                                    # Eq. (9)

    def select_variables(self, V):
        ranking = np.argsort(-V)
        if self.rule == "avg":
            selected = np.flatnonzero(V > V.mean())
        else:
            selected = ranking[:math.ceil(self.rho * self.p)]
        if len(selected) < self.q_min:
            selected = ranking[:self.q_min]
        return np.sort(selected)


def update_mewmc(sigma_hat, x, lam):
    n = x.shape[0]
    return (1 - lam) * sigma_hat + lam * (x.T @ x) / n


class HMTChart:
    def __init__(self, p, n, lam):
        self.p, self.n, self.lam = p, n, lam

    def start(self):
        self.sigma_hat = np.eye(self.p)

    def update(self, x, t):
        self.sigma_hat = update_mewmc(self.sigma_hat, x, self.lam)
        sign, log_det = np.linalg.slogdet(self.sigma_hat)
        return np.trace(self.sigma_hat) - log_det - self.p


def maxnorm_norms(sigma_hat):
    p = sigma_hat.shape[0]
    d = (sigma_hat - np.eye(p))[np.triu_indices(p)]
    return np.sum(d ** 2), np.max(np.abs(d))


class MaxNormChart:

    def __init__(self, p, n, lam, moments):
        self.p, self.n, self.lam = p, n, lam
        self.moments = moments

    def start(self):
        self.sigma_hat = np.eye(self.p)

    def update(self, x, t):
        self.sigma_hat = update_mewmc(self.sigma_hat, x, self.lam)
        T1, T2 = maxnorm_norms(self.sigma_hat)
        m = self.moments
        return max((T1 - m["mean_T1"]) / m["sd_T1"], (T2 - m["mean_T2"]) / m["sd_T2"])


class ZEWMACChart:

    def __init__(self, p, n, lam):
        self.p, self.n, self.lam = p, n, lam

    def start(self):
        self.W = float(self.p)

    def update(self, x, t):
        Z = x[1::2] - x[0::2]                                          # Eq. (16): pairs (2, 1), (4, 3), ...
        Y = 0.5 * np.sum(Z ** 2, axis=1)                               # Eq. (17); Sigma_0^(-1) is already in x
        Y_bar = 2 / self.n * np.sum(Y)
        self.W = self.lam * Y_bar + (1 - self.lam) * self.W            # Eq. (18)
        c_t = ewma_variance_factor(self.lam, t)
        return (self.W - self.p) / math.sqrt(c_t * 4 * self.p / self.n)


def make_charts(chart_names, p, n, lam, design, maxnorm_moments):
    charts = {}
    for name in chart_names:
        if name == "VS-ALT-Norm(avg)":
            chart = VSALTNormChart(p, n, lam, "avg", None, design.phi, design.eta, design.q_min)
        elif name.startswith("VS-ALT-Norm(top="):
            rho = float(name[len("VS-ALT-Norm(top="):-1])
            chart = VSALTNormChart(p, n, lam, "top", rho, design.phi, design.eta, design.q_min)
        elif name == "HMT":
            chart = HMTChart(p, n, lam)
        elif name == "MaxNorm":
            chart = MaxNormChart(p, n, lam, maxnorm_moments)
        elif name == "ZEWMAC":
            chart = ZEWMACChart(p, n, lam)
        chart.start()
        charts[name] = chart
    return charts


def stream_rng(design, purpose, setting, p, n, lam, case, delta, stream): # Random-number generator of one data stream
    purpose_code = {"moments": 0, "calibration": 1, "zero": 2, "steady": 3}[purpose]
    setting_code = {"any": 0, "identity": 1, "block": 2, "wafer": 3}[setting]
    key = [design.master_seed, purpose_code, setting_code, p, n,
           round(100 * lam), case, round(100 * delta), stream]
    return np.random.default_rng(key)


def estimate_maxnorm_moments(p, n, lam, design):
    T1_values, T2_values = [], []
    for stream in range(design.maxnorm_streams):
        rng = stream_rng(design, "moments", "any", p, n, lam, 0, 0.0, stream)
        sigma_hat = np.eye(p)
        for t in range(1, design.maxnorm_burn_in + design.maxnorm_kept + 1):
            x = rng.standard_normal((n, p))
            sigma_hat = update_mewmc(sigma_hat, x, lam)
            if t > design.maxnorm_burn_in and t % design.maxnorm_every == 0:
                T1, T2 = maxnorm_norms(sigma_hat)
                T1_values.append(T1)
                T2_values.append(T2)
    return dict(mean_T1=np.mean(T1_values), sd_T1=np.std(T1_values, ddof=1),
                mean_T2=np.mean(T2_values), sd_T2=np.std(T2_values, ddof=1))


def monitor_stream(charts, limits, chol_before, chol_after, transform, n, change_point, max_delay, rng):  #Run all charts on one data stream until every chart has signalled
    p = chol_before.shape[0]
    signal_time = {name: None for name in charts}
    selected_at_signal = {}

    t = 0
    while t < change_point + max_delay and None in signal_time.values():
        t += 1
        chol = chol_before if t <= change_point else chol_after
        X = rng.standard_normal((n, p)) @ chol.T            # subgroup of n observations
        x = X @ transform                                   # Eq. (10), with mu_0 = 0

        for name, chart in charts.items():
            if signal_time[name] is not None:
                continue
            if chart.update(x, t) > limits[name]:
                signal_time[name] = t
                if isinstance(chart, VSALTNormChart):
                    selected_at_signal[name] = chart.selected

    return signal_time, selected_at_signal


# Control limits

def record_stream(charts, n, p, horizon, rng):
    running_max = {name: -np.inf for name in charts}
    records = {name: ([], []) for name in charts}       # (times, values)

    for t in range(1, horizon + 1):
        x = rng.standard_normal((n, p))                 
        for name, chart in charts.items():
            statistic = chart.update(x, t)
            if statistic > running_max[name]:
                running_max[name] = statistic
                records[name][0].append(t)
                records[name][1].append(statistic)

    return {name: (np.array(times), np.array(values)) for name, (times, values) in records.items()}


def run_length_from_records(times, values, h, horizon): # Time of the first record above h
    k = np.searchsorted(values, h, side="right")      
    return times[k] if k < len(values) else horizon


def arl_from_records(stream_records, h, horizon):
    return np.mean([run_length_from_records(times, values, h, horizon) for times, values in stream_records])


def find_control_limit(stream_records, target_arl, horizon, iterations=60): #Bisection for the limit h with ARL0(h) = target
    low = min(values[0] for times, values in stream_records)      # every stream signals at t = 1
    high = max(values[-1] for times, values in stream_records)    # no stream signals
    for _ in range(iterations):
        middle = (low + high) / 2
        if arl_from_records(stream_records, middle, horizon) < target_arl:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def calibrate(p, n, lam, design):
    chart_names = charts_for(p, n, lam)
    moments = estimate_maxnorm_moments(p, n, lam, design) if "MaxNorm" in chart_names else None

    horizon = design.calibration_horizon
    records = {name: [] for name in chart_names}
    for stream in range(design.r_calibration):
        rng = stream_rng(design, "calibration", "any", p, n, lam, 0, 0.0, stream)
        charts = make_charts(chart_names, p, n, lam, design, moments)
        for name, stream_records in record_stream(charts, n, p, horizon, rng).items():
            records[name].append(stream_records)

    limits, attained_arl0 = {}, {}
    for name in chart_names:
        limits[name] = find_control_limit(records[name], design.arl0, horizon)
        attained_arl0[name] = arl_from_records(records[name], limits[name], horizon)
    print(f"calibrated p = {p}, n = {n}, lambda = {lam:g}", flush=True)
    return dict(charts=chart_names, moments=moments, limits=limits, attained_arl0=attained_arl0)


# performance

def selection_measures(selected, affected):  # Recall, precision, F1 and size
    hits = len(set(selected) & affected)
    recall = hits / len(affected)
    precision = hits / len(selected)
    f1 = 2 * recall * precision / (recall + precision) if hits > 0 else 0.0
    return dict(Recall=recall, Precision=precision, F1=f1, Size=len(selected))


def summarise(run_lengths, max_delay):   #Run length performance summary
    rl = np.array(run_lengths, dtype=float)
    return dict(ARL=rl.mean(), SDRL=rl.std(ddof=1), MRL=np.median(rl), SE=rl.std(ddof=1) / math.sqrt(len(rl)),
                Q05=np.percentile(rl, 5), Q95=np.percentile(rl, 95), N=len(rl),
                censored=int(np.sum(rl >= max_delay)))


def run_cell(task):
    setting, state, p, n, lam, case, delta, calibration, design = task

    sigma0 = in_control_covariance(setting, p)
    sigma_oc = out_of_control_covariance(setting, sigma0, case, delta)
    chol_before = np.linalg.cholesky(sigma0)
    chol_after = np.linalg.cholesky(sigma_oc)
    transform = inverse_square_root(sigma0)
    affected = affected_variables(sigma0, sigma_oc) if case != 0 else None

    change_point = design.tau if state == "steady" else 0
    required = design.r_in_control if case == 0 else design.r_out_of_control
    chart_names = calibration["charts"]
    limits = calibration["limits"]

    run_lengths = {name: [] for name in chart_names}
    diagnosis = {name: [] for name in chart_names}
    streams_used = {name: 0 for name in chart_names}
    false_alarms = {name: 0 for name in chart_names}

    stream = 0
    while any(len(run_lengths[name]) < required for name in chart_names):
        needed = [name for name in chart_names if len(run_lengths[name]) < required]
        charts = make_charts(needed, p, n, lam, design, calibration["moments"])
        rng = stream_rng(design, state, setting, p, n, lam, case, delta, stream)
        signal_time, selected_at_signal = monitor_stream(
            charts, limits, chol_before, chol_after, transform, n, change_point, design.max_delay, rng)
        stream += 1

        for name in needed:
            streams_used[name] += 1
            T = signal_time[name]
            if T is not None and T <= change_point:
                false_alarms[name] += 1
                continue
            if T is None:
                run_lengths[name].append(design.max_delay)
            else:
                run_lengths[name].append(T - change_point)
                if name in selected_at_signal and affected is not None:
                    diagnosis[name].append(selection_measures(selected_at_signal[name], affected))

    rows = []
    for name in chart_names:
        row = dict(Setting=setting, State=state, p=p, n=n, lam=lam, Case=case, delta=delta, Chart=name)
        row.update(summarise(run_lengths[name], design.max_delay))
        row["FA_before_shift"] = false_alarms[name] / streams_used[name] if state == "steady" else np.nan
        for measure in ("Recall", "Precision", "F1", "Size"):
            values = [d[measure] for d in diagnosis[name]]
            row[measure] = np.mean(values) if values else np.nan
        row["UCL"] = limits[name]
        rows.append(row)

    label = "in control" if case == 0 else f"case {case}, delta = {delta:g}"
    print(f"{setting}, {state} state, p = {p}, n = {n}, lambda = {lam:g}, {label}: "
          + ", ".join(f"{row['Chart']} {row['ARL']:.2f}" for row in rows), flush=True)
    return rows


# Excel report

def rmi_table(results): # RMI averaged over delta and then over the out-of-control cases
    configuration = ["Setting", "State", "p", "n", "lam"]
    oc = results[results.Case != 0].copy()
    best_arl = oc.groupby(configuration + ["Case", "delta"], observed=True)["ARL"].transform("min")
    oc["relative"] = (oc.ARL - best_arl) / best_arl
    per_case = oc.groupby(configuration + ["Case", "Chart"], observed=True)["relative"].mean().reset_index()
    rmi = per_case.groupby(configuration + ["Chart"], observed=True)["relative"].mean()
    return rmi.unstack("Chart").reset_index()


def wide_table(results, statistic):
    table = results.pivot_table(index=["Setting", "p", "n", "lam", "Case", "delta"], columns="Chart",
                                values=statistic, observed=True)
    return table.reset_index()


def write_report(rows, calibrations, design):
    results = pd.DataFrame(rows)
    results["Setting"] = pd.Categorical(results["Setting"], list(design.settings), ordered=True)
    results["State"] = pd.Categorical(results["State"], ["zero", "steady"], ordered=True)
    results["Chart"] = pd.Categorical(results["Chart"], CHART_NAMES, ordered=True)
    results = results.sort_values(["Setting", "State", "p", "n", "lam", "Case", "delta", "Chart"])

    limit_rows = []
    for (p, n, lam), calibration in calibrations.items():
        for name in calibration["charts"]:
            limit_rows.append(dict(p=p, n=n, lam=lam, Chart=name, UCL=calibration["limits"][name],
                                   ARL0_calibration=calibration["attained_arl0"][name]))

    sheets = {
        "Design": pd.DataFrame(list(asdict(design).items()), columns=["Item", "Value"]).astype(str),
        "Control limits": pd.DataFrame(limit_rows),
        "In-control check": results[results.Case == 0].pivot_table(
            index=["Setting", "State", "p", "n", "lam"], columns="Chart", values="ARL", observed=True).reset_index(),
    }
    for state, title in (("zero", "Zero state"), ("steady", "Steady state")):
        state_results = results[results.State == state]
        if len(state_results) == 0:
            continue
        sheets[title] = state_results
        sheets[f"{title} ARL"] = wide_table(state_results, "ARL")
        sheets[f"{title} SDRL"] = wide_table(state_results, "SDRL")
    sheets["RMI"] = rmi_table(results)

    with pd.ExcelWriter(design.output_file) as writer:
        for title, table in sheets.items():
            table.to_excel(writer, sheet_name=title, index=False)
    print(f"results saved to {design.output_file}")


# main

def calibrate_task(task):
    p, n, lam, design = task
    return calibrate(p, n, lam, design)


def main():
    parser = argparse.ArgumentParser(description="Simulation study of the VS-ALT-Norm chart")
    parser.add_argument("--quick", action="store_true", help="small run to check that everything works")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    design = QUICK_DESIGN if args.quick else Design()

    configurations = list_configurations(design)
    calibration_keys = sorted({(p, n, lam) for setting, p, n, lam in configurations})

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        calibration_tasks = [(p, n, lam, design) for p, n, lam in calibration_keys]
        calibrations = dict(zip(calibration_keys, pool.map(calibrate_task, calibration_tasks)))
        cell_tasks = []
        for setting, p, n, lam in configurations:
            for state, case, delta in list_cells(design, setting):
                cell_tasks.append((setting, state, p, n, lam, case, delta, calibrations[(p, n, lam)], design))
        rows = [row for cell_rows in pool.map(run_cell, cell_tasks) for row in cell_rows]

    write_report(rows, calibrations, design)


if __name__ == "__main__":
    main()
