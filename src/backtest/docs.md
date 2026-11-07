# Library: backtest

- [Types](#types)
- [Backtest](#backtest-1)
- [Result](#result)
- [Comparison](#comparison)
- [Scoring](#scoring)
- [Pipeline](#pipeline)
- [Leakage](#leakage)
- [Strategies.Trend](#strategiestrend)

## Module backtest

```ocaml
module Backtest: sig .. end
```

Test one trading strategy at a time. The code is Python, written here in OCaml notation. How it behaves is in
`explanation.md`.

---

## Types

```ocaml
type frame
```

A polars `DataFrame`: an `AsOfDate` column plus one float column per series.

```ocaml
type data =
  | Path of string
  | Polars of frame
  | Pandas of pandas_frame
```

A data source. Pass the value directly, without a constructor.

```ocaml
type forecast = frame -> frame
```

`forecast prices` returns the desired direction and size per date and asset.

```ocaml
type period = Dev | Holdout | Full
```

The dates to score: the first 75%, the rest, or all of them.

```ocaml
type scoring = Pdf | Lynx
```

The PDF's formulas, or Lynx's `evaluation.py`.

```ocaml
type key_figures = {
  adjusted_sharpe : float;
  sharpe : float;
  sharpe_ex_slippage : float;
  sharpe_1d_lag : float;
  skewness : float;
  kurtosis : float;
  av_holding : float;
  position_bias : float;
}
```

The figures reported for each strategy.

```ocaml
exception Leakage_error of string
```

Positions depend on future data.

## Backtest

```ocaml
type t
```

One data source and its settings.

```ocaml
val create :
  ?extras:data ->
  ?scoring:scoring ->
  ?delta:float ->
  ?costs:float ->
  ?vol_window:int ->
  ?risk_window:int ->
  ?cap:float option ->
  ?normalize:bool ->
  data -> t
```

`create data` reads `data` and returns a backtest for it. In Python, this is `Backtest(data)`.

- **Raises** `ValueError` if the data has no single date column, or a column isn't numeric.
- **Raises** `TypeError` if `data` isn't a path or frame.

```ocaml
val run : t -> ?period:period -> ?check_leakage:bool -> ?name:string -> forecast -> Result.t
```

`run t f` turns the forecast `f` into positions and scores them.

- **Raises** `Leakage_error` if `f` uses future data.
- **Raises** `ValueError` if the forecast doesn't line up with the prices, or `period` is unknown.

```ocaml
val run_positions :
  t -> ?period:period -> ?check_leakage:bool -> ?name:string -> positions -> Result.t
```

`run_positions t p` scores the positions `p` as they are. `p` is a function returning positions, or a frame.

- **Raises** `Leakage_error` if a function `p` uses future data.
- **Raises** `ValueError` if the positions don't line up with the prices.

```ocaml
val compare : t -> ?period:period -> ?check_leakage:bool -> strategies -> Comparison.t
```

`compare t l` runs every forecast in `l`, a list of functions or a dict of names to functions.

- **Raises** `Leakage_error` if a forecast uses future data.
- **Raises** `ValueError` if `l` is empty, or two functions share a name.

```ocaml
val sweep :
  t -> ?period:period -> ?check_leakage:bool -> forecast -> grid:(string * 'a list) list -> frame
```

`sweep t f ~grid` runs `f` for every combination of parameters, and returns one row per combination. In Python,
each parameter is a keyword argument: `bt.sweep(f, span=[8, 16])`.

- **Raises** `Leakage_error` if `f` uses future data.
- **Raises** `ValueError` if no parameters are given.

## Result

```ocaml
type t = {
  name : string;
  period : period;
  figures : key_figures;
  positions : frame;
  prices : frame;
  scoring : Scoring.t;
}
```

The outcome of a run. Each key figure is also an attribute: `result.sharpe`.

```ocaml
val returns : t -> frame
```

`returns r` is the daily net returns for the period. A property in Python.

```ocaml
val by_year : t -> frame
```

`by_year r` returns `days`, `total_return` and `sharpe` per calendar year.

```ocaml
val by_instrument : t -> frame
```

`by_instrument r` returns `sharpe_ex_slippage`, `av_holding` and `position_bias` per asset.

```ocaml
val to_csv : t -> string -> unit
```

`to_csv r path` saves the positions for all dates to `path`.

```ocaml
val plot : t -> figure
```

`plot r` returns the four-panel dashboard as a matplotlib `Figure`.

## Comparison

```ocaml
type t = {
  results : (string * Result.t) list;
  figures : frame;
  correlations : frame;
}
```

Each strategy's result, a table of key figures, and the correlations of daily net returns.

## Scoring

```ocaml
type t = { kind : scoring; delta : float; costs : float }
```

The scoring formula and its cost setting.

```ocaml
val figures : t -> frame -> frame -> string list -> (string * value) list
```

`figures t positions prices names` calculates the named key figures.

```ocaml
val plot : t -> frame -> frame -> figure
```

`plot t positions prices` returns the dashboard.

## Pipeline

```ocaml
val load_data : data -> frame
```

`load_data source` reads a data source into a standard frame.

- **Raises** `ValueError` if there is no single date column, or a column isn't numeric.
- **Raises** `TypeError` if `source` isn't a path or frame.

```ocaml
val instruments : frame -> string list
```

`instruments f` returns the column names of `f`, without `AsOfDate`.

```ocaml
val vol_scale :
  frame -> frame -> ?vol_window:int -> ?cap:float option -> ?normalize:bool -> unit -> frame
```

`vol_scale forecast prices` scales and caps the forecast, then divides it by asset volatility (equation 2).

- **Raises** `ValueError` if the forecast doesn't line up with the prices.

```ocaml
val risk_scale : frame -> frame -> ?risk_window:int -> unit -> frame
```

`risk_scale positions prices` divides positions by trailing strategy risk (equation 3).

```ocaml
val to_positions :
  frame -> frame -> ?vol_window:int -> ?risk_window:int -> ?cap:float option -> ?normalize:bool ->
  unit -> frame
```

`to_positions forecast prices` applies `vol_scale`, then `risk_scale`.

- **Raises** `ValueError` if the forecast doesn't line up with the prices.

## Leakage

```ocaml
val check_no_leakage : forecast -> frame -> ?extras:frame -> unit -> unit
```

`check_no_leakage f prices` checks that past positions from `f` don't change when later data changes.

- **Raises** `Leakage_error` if they do.

## Strategies.Trend

```ocaml
val forecast : ?spans:int list -> frame -> frame
```

`forecast prices` returns the average vote of moving average crossovers, from -1 to +1.
