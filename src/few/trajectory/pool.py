"""Fan EMRI trajectories out to a small pool of worker processes and collect them.

The trajectory integrator is serial CPU code (DOPR853 + numba + Cython splines) and
holds the GIL, so threads do not help; independent trajectories do parallelise across
PROCESSES. This module runs a batch of inspirals on a pool, ships back a compact
snapshot of each (the call's output plus the integrator's dense-output state), and
serves them to an unchanged waveform generator through a cache that stands in for its
``inspiral_generator``.

Typical use (one batch of template evaluations, e.g. the rows of an information matrix)::

    cache = TrajectoryCache.install(waveform_gen)           # wraps waveform_gen.inspiral_generator
    pool = TrajectoryPool(n_workers=4,
                          inspiral_init_kwargs=inspiral_init_kwargs_from(waveform_gen))
    cache.precompute(wrapper, rows, pool)                   # dry-run capture + fan-out + collect
    templates = [wrapper(*p) for p in rows]                 # every trajectory is a cache hit

``precompute`` runs each evaluation once in CAPTURE mode: the waveform code runs up to its
inspiral call, the exact ``(args, kwargs)`` it would pass are recorded, and a
``BaseException`` unwinds the evaluation (so ``except Exception`` in the caller cannot
swallow it). Whatever the waveform does to its inputs before integrating (spin/sign
conventions, frame conversions in a response wrapper) is therefore replicated exactly.

The executor is pluggable: the default is a ``ProcessPoolExecutor`` with the ``spawn``
start method (safe under an MPI parent: the children never import MPI); any
``concurrent.futures.Executor`` works, e.g. ``mpi4py.futures.MPIPoolExecutor`` for an
MPI fan-out to spare ranks.

Determinism: DOPR853 keeps its step-size controller state (``errOldTemp``,
``previousRejectTemp``) on the stepper ACROSS trajectories, so a serial loop's result
depends on the previous trajectory at the integrator-tolerance level. Workers reset that
state before every trajectory, so a pooled trajectory is bit-identical to one computed by a
freshly built inspiral module, whatever the worker's history (see ``reset_stepper``).

User: the LISA global-fit EMRI likelihood (``lisatools.sources.emri.direct_signal_gen``)
installs a :class:`TrajectoryCache` on its ``FastKerrEccentricEquatorialFlux`` generator
and fans each batch of at least ``EMRI_TRAJ_WORKERS`` rows (8 in production) out to a
:class:`TrajectoryPool` whose executor runs :func:`_worker_run`; smaller batches integrate
serially. Each worker builds its own ``EMRIInspiral`` (so it imports :mod:`few` and loads
the flux file once).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import multiprocessing
import os
import pickle
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

# construction-time keys of EMRIInspiral / TrajectoryBase (everything else is call-time)
_INIT_KEYS = (
    "func",
    "integrate_constants_of_motion",
    "enforce_schwarz_sep",
    "convert_to_pex",
    "rootfind_separatrix",
    "force_backend",
    "buffer_length",
)

# attribute types that _plain_state copies (with None and numpy arrays)
_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


# ----------------------------------------------------------------------------------
# snapshots: what a waveform reads from the inspiral module after the call
# ----------------------------------------------------------------------------------


@dataclasses.dataclass
class InspiralSnapshot:
    """One trajectory: the call output and the integrator state needed to re-serve it.

    Built by :func:`snapshot_inspiral` (in a worker), loaded back by
    :func:`install_snapshot`.

    Attributes:
        key (str): :func:`call_key` of the inspiral call.
        output (tuple): The call's return value as numpy arrays (for ``EMRIInspiral``:
            ``t, p, e, x, Phi_phi, Phi_theta, Phi_r``).
        integrator_state (dict): The integrator's scalar / ``None`` / array attributes,
            with ``trajectory_arr``, ``_integrator_t_cache`` and ``dopr_spline_output`` cut
            to the filled rows.
        func_state (dict): The ODE model's scalar / ``None`` attributes (no arrays).
        dopr_state (dict): The stepper's ``fix_step`` and ``abstol``.
    """

    key: str
    output: tuple
    integrator_state: dict
    func_state: dict
    dopr_state: dict


def reset_stepper(inspiral) -> None:
    """Drop DOPR853's cross-trajectory controller state (a fresh module has none).

    Deletes ``errOldTemp`` and ``previousRejectTemp`` from
    ``inspiral.inspiral_generator.dopr`` if present, so the next trajectory is
    bit-identical to one from a freshly built module.

    Args:
        inspiral (EMRIInspiral): The trajectory module.
    """
    dopr = inspiral.inspiral_generator.dopr
    for name in ("errOldTemp", "previousRejectTemp"):
        if hasattr(dopr, name):
            delattr(dopr, name)


def _plain_state(obj, skip=()) -> dict:
    """Return ``obj``'s scalar, ``None`` and (copied) ``ndarray`` attributes, minus ``skip``."""
    out = {}
    for name, val in vars(obj).items():
        if name in skip:
            continue
        if isinstance(val, _SCALAR_TYPES) or val is None:
            out[name] = val
        elif isinstance(val, np.ndarray):
            out[name] = val.copy()
    return out


def snapshot_inspiral(inspiral, output, key: str = "") -> InspiralSnapshot:
    """Capture ``inspiral``'s post-call state (EMRIInspiral wrapping an Integrate).

    Args:
        inspiral (EMRIInspiral): The module right after the call.
        output (tuple): The call's return value.
        key: :func:`call_key` of the call. Default ``""``.

    Returns:
        InspiralSnapshot: Picklable copy of what a waveform generator reads after the call.
    """
    integ = inspiral.inspiral_generator
    n = int(integ.traj_step)
    state = _plain_state(integ, skip=("trajectory_arr", "_integrator_t_cache", "dopr_spline_output"))
    # only the filled part of the buffers travels
    state["trajectory_arr"] = np.array(integ.trajectory_arr[:n])
    state["_integrator_t_cache"] = np.array(integ._integrator_t_cache[:n])
    state["dopr_spline_output"] = np.array(integ.dopr_spline_output[: max(n - 1, 0)])
    func_state = {k: v for k, v in _plain_state(integ.func).items() if not isinstance(v, np.ndarray)}
    dopr_state = {"fix_step": integ.dopr.fix_step, "abstol": integ.dopr.abstol}
    return InspiralSnapshot(key, tuple(np.array(o) for o in output), state, func_state, dopr_state)


def install_snapshot(inspiral, snap: InspiralSnapshot) -> tuple:
    """Load ``snap`` into ``inspiral`` as if it had just integrated; return the output copy.

    Sets the integrator and model attributes and the stepper's ``fix_step`` / ``abstol``,
    so ``eval_integrator_spline``, ``integrator_spline_phase_coeff``, ... serve the pooled
    trajectory.

    Args:
        inspiral (EMRIInspiral): The module to load into (modified in place).
        snap (InspiralSnapshot): The snapshot.

    Returns:
        tuple: Copies of the snapshot's output arrays.
    """
    integ = inspiral.inspiral_generator
    for name, val in snap.integrator_state.items():
        setattr(integ, name, val.copy() if isinstance(val, np.ndarray) else val)
    for name, val in snap.func_state.items():
        setattr(integ.func, name, val)
    integ.dopr.fix_step = snap.dopr_state["fix_step"]
    integ.dopr.abstol = snap.dopr_state["abstol"]
    return tuple(o.copy() for o in snap.output)


# ----------------------------------------------------------------------------------
# call keys
# ----------------------------------------------------------------------------------


def _norm(v):
    """Reduce a call argument to a hashable, bit-exact tuple (arrays by sha1 of their bytes).

    Floats are keyed by ``float.hex`` (exact bits), ints / bools by ``int``, classes by
    module and qualified name, containers recursively; anything else by ``repr``.
    """
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return ("ndarray", a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).hexdigest())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(_norm(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple(sorted((str(k), _norm(x)) for k, x in v.items())))
    if isinstance(v, type):
        return ("type", v.__module__, v.__qualname__)
    if isinstance(v, (np.floating, float)):
        return ("float", float(v).hex())
    if isinstance(v, (np.integer, int, np.bool_, bool)):
        return ("int", int(v))
    return ("repr", repr(v))


def call_key(args: Sequence, kwargs: dict) -> str:
    """Stable digest of an inspiral call (exact float bits, array contents).

    Args:
        args: Positional arguments of the call.
        kwargs: Keyword arguments of the call.

    Returns:
        str: sha1 hex digest; equal for calls with bit-identical arguments.

    Example:
        >>> from few.trajectory.pool import call_key
        >>> call_key((1e6, 10.0), {"T": 1.0}) == call_key((1e6, 10.0), {"T": 1.0})
        True
    """
    return hashlib.sha1(pickle.dumps((_norm(tuple(args)), _norm(dict(kwargs))))).hexdigest()


# ----------------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------------

# per-process cache of built inspiral modules, keyed by call_key((), init_kwargs)
_WORKER_INSPIRALS: dict = {}


def _build_inspiral(init_kwargs: dict):
    """Build ``EMRIInspiral(**init_kwargs)`` (imported lazily, inside the worker)."""
    from few.trajectory.inspiral import EMRIInspiral

    return EMRIInspiral(**init_kwargs)


def _worker_run(init_kwargs: dict, calls: list) -> list:
    """Run a chunk of inspiral calls in this process (module level: picklable).

    The inspiral module is built once per process and per ``init_kwargs`` (cached in
    ``_WORKER_INSPIRALS``); :func:`reset_stepper` runs before every call. Calling it with
    an empty ``calls`` list only builds the module (used to warm workers up).

    Args:
        init_kwargs: ``EMRIInspiral`` construction kwargs.
        calls: ``(key, args, kwargs)`` triples.

    Returns:
        list: Per call, an :class:`InspiralSnapshot`, or ``(key, "ExcType: message")`` if
        the call raised an ``Exception`` (the real serial call raises it again).
    """
    k = call_key((), init_kwargs)
    insp = _WORKER_INSPIRALS.get(k)
    if insp is None:
        insp = _WORKER_INSPIRALS[k] = _build_inspiral(init_kwargs)
    out = []
    for key, args, kwargs in calls:
        reset_stepper(insp)
        try:
            res = insp(*args, **kwargs)
            out.append(snapshot_inspiral(insp, res, key))
        except Exception as exc:  # the real (serial) call will raise it again
            out.append((key, f"{type(exc).__name__}: {exc}"))
    return out


class TrajectoryPool:
    """A persistent pool that integrates batches of inspiral calls.

    Args:
        n_workers: Worker processes. Default ``None`` means ``max(1, os.cpu_count() // 2)``.
        inspiral_init_kwargs: ``EMRIInspiral`` construction kwargs used in every worker
            (see :func:`inspiral_init_kwargs_from`). Default ``None`` means
            ``{"func": "KerrEccEqFlux"}``.
        executor: An existing ``concurrent.futures.Executor`` to use instead of the
            default spawn-context ``ProcessPoolExecutor`` (not shut down by :meth:`close`).
        chunk: Calls per task (amortises the IPC). Default ``None`` splits each batch
            evenly over ``n_workers``.

    Note:
        The default executor uses the ``spawn`` start method, whose workers re-import the
        parent's ``__main__`` module: in a script, create the pool under
        ``if __name__ == "__main__":``.

    Example:
        .. code-block:: python

            from few.trajectory.pool import TrajectoryPool, call_key

            kw = dict(T=1.0, dt=10.0)
            rows = [(1e6, 1e1, 0.9, 10.0, 0.3, 1.0), (1e6, 1e1, 0.5, 12.0, 0.2, 1.0)]
            with TrajectoryPool(n_workers=2) as pool:
                snaps = pool.run([(call_key(r, kw), r, kw) for r in rows])
            t = snaps[0].output[0]
    """

    def __init__(self, n_workers: Optional[int] = None, inspiral_init_kwargs: Optional[dict] = None,
                 executor: Optional[Executor] = None, chunk: Optional[int] = None):
        self.n_workers = int(n_workers) if n_workers else max(1, (os.cpu_count() or 2) // 2)
        self.init_kwargs = dict(inspiral_init_kwargs or {"func": "KerrEccEqFlux"})
        self.chunk = chunk
        self._own = executor is None
        self._executor = executor or ProcessPoolExecutor(
            max_workers=self.n_workers, mp_context=multiprocessing.get_context("spawn"))

    def run(self, calls: Sequence[tuple]) -> list:
        """Integrate a batch of calls on the pool and wait for all of them.

        Args:
            calls: ``(key, args, kwargs)`` triples (``key`` from :func:`call_key`).

        Returns:
            list: In the order of ``calls``, an :class:`InspiralSnapshot` per successful
            call or ``(key, error_message)`` per failed one; ``[]`` for no calls.
        """
        if not calls:
            return []
        chunk = self.chunk or max(1, -(-len(calls) // self.n_workers))
        parts = [list(calls[i:i + chunk]) for i in range(0, len(calls), chunk)]
        futs = [self._executor.submit(_worker_run, self.init_kwargs, p) for p in parts]
        return [snap for f in futs for snap in f.result()]

    def close(self):
        """Shut the executor down (waiting for running tasks) if the pool created it."""
        if self._own:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def inspiral_init_kwargs_from(waveform_gen) -> dict:
    """EMRIInspiral construction kwargs of a FEW waveform generator's inspiral module.

    Args:
        waveform_gen (object): A FEW waveform generator (reads its ``inspiral_kwargs``).

    Returns:
        dict: The construction-time keys of ``inspiral_kwargs`` (``func``,
        ``integrate_constants_of_motion``, ``enforce_schwarz_sep``, ``convert_to_pex``,
        ``rootfind_separatrix``, ``force_backend``, ``buffer_length``); call-time keys are
        left out.
    """
    kw = getattr(waveform_gen, "inspiral_kwargs", {}) or {}
    return {k: kw[k] for k in _INIT_KEYS if k in kw}


# ----------------------------------------------------------------------------------
# parent side: the cache that stands in for waveform_gen.inspiral_generator
# ----------------------------------------------------------------------------------


class _Captured(BaseException):
    """Unwinds a capture-mode evaluation (BaseException: ``except Exception`` can't eat it)."""


class TrajectoryCache:
    """Stand-in for a waveform generator's ``inspiral_generator``.

    A call whose ``(args, kwargs)`` was precomputed installs that snapshot into the real
    inspiral module (so every attribute and dense-output evaluator the waveform reads is
    the pooled trajectory's) and returns its output; any other call runs the real module.
    Every other attribute is read from (and set on) the real module.

    Args:
        inspiral (EMRIInspiral): The real inspiral module.

    Attributes:
        hits (int): Calls served from the store.
        misses (int): Calls that ran the real module.
        errors (dict): ``key -> error message`` of worker failures from :meth:`precompute`.

    Example:
        .. code-block:: python

            from few.trajectory.pool import (
                TrajectoryCache,
                TrajectoryPool,
                inspiral_init_kwargs_from,
            )

            cache = TrajectoryCache.install(waveform_gen)
            pool = TrajectoryPool(n_workers=4,
                                  inspiral_init_kwargs=inspiral_init_kwargs_from(waveform_gen))
            cache.precompute(lambda *p: waveform_gen(*p, T=1.0, dt=10.0), rows, pool)
            templates = [waveform_gen(*p, T=1.0, dt=10.0) for p in rows]  # cache hits
            cache.clear()
    """

    def __init__(self, inspiral):
        self.__dict__["_inner"] = inspiral
        self.__dict__["_store"] = {}
        self.__dict__["_capture"] = None
        self.__dict__["hits"] = 0
        self.__dict__["misses"] = 0
        self.__dict__["errors"] = {}

    @classmethod
    def install(cls, waveform_gen) -> "TrajectoryCache":
        """Wrap ``waveform_gen.inspiral_generator`` in a cache (idempotent).

        Args:
            waveform_gen (object): A FEW waveform generator; its ``inspiral_generator`` is
                replaced by the cache.

        Returns:
            TrajectoryCache: The new cache, or the one already installed.
        """
        cur = waveform_gen.inspiral_generator
        if isinstance(cur, cls):
            return cur
        cache = cls(cur)
        waveform_gen.inspiral_generator = cache
        return cache

    @staticmethod
    def uninstall(waveform_gen) -> None:
        """Restore the real inspiral module on ``waveform_gen`` if a cache is installed.

        Args:
            waveform_gen (object): The waveform generator passed to :meth:`install`.
        """
        cur = waveform_gen.inspiral_generator
        if isinstance(cur, TrajectoryCache):
            waveform_gen.inspiral_generator = cur._inner

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self._inner, name, value)

    def __call__(self, *args, **kwargs):
        """Serve an inspiral call: record it (capture mode), replay a snapshot, or run it.

        In capture mode (inside :meth:`capture`) the call is recorded and a private
        ``BaseException`` unwinds the evaluation. Otherwise a stored snapshot with the same
        :func:`call_key` is installed into the real module and its output returned (a hit),
        else the real module runs (a miss).

        Args:
            *args: Positional arguments of the inspiral call.
            **kwargs: Keyword arguments of the inspiral call.

        Returns:
            tuple: The inspiral output, ``(t, p, e, x, Phi_phi, Phi_theta, Phi_r)``.
        """
        if self._capture is not None:
            self._capture.append((tuple(args), copy.deepcopy(kwargs)))
            raise _Captured()
        snap = self._store.get(call_key(args, kwargs))
        if snap is not None:
            self.__dict__["hits"] += 1
            return install_snapshot(self._inner, snap)
        self.__dict__["misses"] += 1
        return self._inner(*args, **kwargs)

    def __len__(self):
        return len(self._store)

    def put(self, snap: InspiralSnapshot) -> None:
        """Store a snapshot under its ``key``."""
        self._store[snap.key] = snap

    def clear(self) -> None:
        """Drop every stored snapshot and recorded worker error (``hits`` / ``misses`` stay)."""
        self._store.clear()
        self.errors.clear()

    def capture(self, fn: Callable, arg_rows: Iterable, kwarg_rows: Optional[Iterable] = None) -> list:
        """Run ``fn(*args, **kwargs)`` per row up to its inspiral call; return the calls.

        A row that never reaches the inspiral (raises first, or returns) gives ``None``;
        any ``Exception`` raised by ``fn`` is swallowed.

        Args:
            fn: The evaluation that calls the waveform generator, as ``fn(*row, **kw)``.
            arg_rows: Positional-argument rows.
            kwarg_rows: Keyword-argument dicts, one per row. Default ``None`` means ``{}``.

        Returns:
            list: Per row, the captured ``(args, kwargs)`` of its inspiral call, or ``None``.
        """
        arg_rows = list(arg_rows)
        kwarg_rows = [{}] * len(arg_rows) if kwarg_rows is None else list(kwarg_rows)
        calls = []
        for a, kw in zip(arg_rows, kwarg_rows):
            rec = []
            self.__dict__["_capture"] = rec
            try:
                fn(*a, **kw)
            except _Captured:
                pass
            except Exception:
                pass
            finally:
                self.__dict__["_capture"] = None
            calls.append(rec[0] if rec else None)
        return calls

    def precompute(self, fn: Callable, arg_rows: Iterable, pool: TrajectoryPool,
                   kwarg_rows: Optional[Iterable] = None) -> dict:
        """Capture every row's inspiral call, integrate the new ones on ``pool``, store them.

        Calls already stored, and duplicates within the batch, are integrated once. Worker
        failures are recorded in :attr:`errors`; those rows later run the real module (and
        raise there).

        Args:
            fn: The evaluation, called as ``fn(*row, **kw)`` (see :meth:`capture`).
            arg_rows: Positional-argument rows.
            pool: The pool that integrates the new calls.
            kwarg_rows: Keyword-argument dicts, one per row. Default ``None`` means ``{}``.

        Returns:
            dict: Counts ``rows``, ``captured`` (rows that reached the inspiral),
            ``computed`` (unique new calls sent to the pool) and ``errors`` (worker
            failures).
        """
        calls = self.capture(fn, arg_rows, kwarg_rows)
        todo, seen = [], set()
        for c in calls:
            if c is None:
                continue
            k = call_key(*c)
            if k in self._store or k in seen:
                continue
            seen.add(k)
            todo.append((k, c[0], c[1]))
        n_err = 0
        for res in pool.run(todo):
            if isinstance(res, InspiralSnapshot):
                self.put(res)
            else:
                self.errors[res[0]] = res[1]
                n_err += 1
        return dict(rows=len(calls), captured=sum(c is not None for c in calls), computed=len(todo),
                    errors=n_err)
