import run_reinforcement
from plot import Plot
from preprocess import Preprocess


def fetch_and_process() -> None:
    p = Preprocess()
    df_col = {}
    for sym in p.symbols:
        for t in p.intervals:
            print(f'\nFetching data for {sym}, interval {t}')
            df, fname = p.fetch_kline_data(sym, t)

            print(f'Processing data for {sym}, interval {t}')
            df_col[sym + '_' + t] = p.process_data(df, fname)


def plot() -> None:
    filename = ''
    p = Plot()
    p.plot_evaluation(filename)


if __name__ == '__main__':
    fetch_and_process()

    run_reinforcement.run_rfl()

    plot()
