import vectorbtpro as vbt


def _init_data(market="SET",
               symbols="ADVANC",
               tf="1 day"):
    raw = vbt.TVData.fetch(f"{market}:{symbols}", timeframe=tf).get()
    raw.columns = raw.columns.str.lower()
    raw.index = raw.index.tz_convert(None)
    raw.to_csv(f"{symbols}.csv")


if __name__ == "__main__":
    _init_data()
